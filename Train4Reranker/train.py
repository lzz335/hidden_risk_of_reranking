from torch.optim import AdamW
from Config import TrainConfig, SentencePairDatasetForTwoClass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import json
from tqdm import tqdm
import logging
from datetime import datetime
import os

# Load training data from JSON file
# Expected format: list of objects with 'sentence1', 'sentence2', and 'label' fields
with open("train_corpus2.json", "r", encoding="utf-8") as f:
    data = json.load(f)


class DebertaTwoHeadModel(nn.Module):
    """DeBERTa model with two parallel classification heads for multi-task learning"""

    def __init__(self, pretrained_model):
        """
        Initialize the model with:
        - Pretrained DeBERTa base model
        - Two separate classification heads (for dual-task learning)
        - Xavier initialization for classifier weights
        """
        super().__init__()
        self.deberta = AutoModel.from_pretrained(pretrained_model)
        self.classifier1 = nn.Linear(self.deberta.config.hidden_size, 2)
        self.classifier2 = nn.Linear(self.deberta.config.hidden_size, 2)

        # Weight initialization for better convergence
        nn.init.xavier_normal_(self.classifier1.weight)
        nn.init.xavier_normal_(self.classifier2.weight)
        self.classifier1.bias.data.zero_()
        self.classifier2.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model:
        1. Encode input with DeBERTa
        2. Use [CLS] token representation for classification
        3. Return logits from both classification heads
        """
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier1(cls_output), self.classifier2(cls_output)


class Trainer:
    """Main training orchestrator for the dual-head model"""

    def __init__(self, model, train_loader, val_loader):
        """Initialize trainer with model and data loaders"""
        self.model = model.to(TrainConfig.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_val_acc = 0.0  # Track best validation accuracy
        self.best_model_path = ""  # Path to save best model

        # Create directory for saving models
        os.makedirs("saved_models_nli", exist_ok=True)

        # Prepare class weights for loss functions
        self._prepare_class_weights()

        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=TrainConfig.learning_rate,
            weight_decay=TrainConfig.weight_decay
        )
        self.total_steps = len(train_loader) * TrainConfig.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.total_steps
        )

    def _prepare_class_weights(self):
        """
        Calculate class weights for imbalanced datasets:
        - labels1: 28 positive vs 86 negative samples
        - labels2: 3 positive vs 8 negative samples

        Weighting formula:
        weight = total_samples / (2 * class_count)
        """
        labels1_total = 35892 + 103883  # Total samples for task1
        self.weight1 = torch.tensor([
            labels1_total / (2 * 103883),  # Negative class weight
            labels1_total / (2 * 35892)  # Positive class weight
        ]).to(TrainConfig.device)

        labels2_total = 37959 + 101816  # Total samples for task2
        self.weight2 = torch.tensor([
            labels2_total / (2 * 101816),  # Negative class weight
            labels2_total / (2 * 37959)  # Positive class weight
        ]).to(TrainConfig.device)

        # Initialize loss functions with class weights
        self.criterion1 = nn.CrossEntropyLoss(weight=self.weight1)
        self.criterion2 = nn.CrossEntropyLoss(weight=self.weight2)

    def train_epoch(self):
        """Train model for one epoch with progress tracking"""
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader):
            input_ids = batch['input_ids'].to(TrainConfig.device)
            attention_mask = batch['attention_mask'].to(TrainConfig.device)
            labels1 = batch['labels1'].to(TrainConfig.device)
            labels2 = batch['labels2'].to(TrainConfig.device)

            self.optimizer.zero_grad()
            logits1, logits2 = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Calculate individual task losses
            loss1 = self.criterion1(logits1, labels1)
            loss2 = self.criterion2(logits2, labels2)
            loss = loss1 + loss2  # Total loss is sum of both task losses

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item() * input_ids.size(0)
        return total_loss / len(self.train_loader.dataset)

    def evaluate(self):
        """Evaluate model performance on validation set"""
        self.model.eval()
        total_loss = 0.0
        correct1, total1 = 0, 0
        correct2, total2 = 0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(TrainConfig.device)
                attention_mask = batch['attention_mask'].to(TrainConfig.device)
                labels1 = batch['labels1'].to(TrainConfig.device)
                labels2 = batch['labels2'].to(TrainConfig.device)

                logits1, logits2 = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Calculate losses for validation
                loss1 = self.criterion1(logits1, labels1)
                loss2 = self.criterion2(logits2, labels2)
                total_loss += (loss1 + loss2).item() * input_ids.size(0)

                # Calculate accuracy for each task
                _, preds1 = torch.max(logits1, 1)
                correct1 += (preds1 == labels1).sum().item()
                total1 += labels1.size(0)

                _, preds2 = torch.max(logits2, 1)
                correct2 += (preds2 == labels2).sum().item()
                total2 += labels2.size(0)

        return {
            'loss': total_loss / len(self.val_loader.dataset),
            'acc1': correct1 / total1,
            'acc2': correct2 / total2,
            'avg_acc': (correct1 / total1 + correct2 / total2) / 2  # Average accuracy across tasks
        }

    def save_best_model(self, val_metrics, epoch):
        """Save model checkpoint if validation performance improves"""
        current_acc = val_metrics['avg_acc']
        if current_acc > self.best_val_acc:
            self.best_val_acc = current_acc

            # Remove previous best model file
            if os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)

            # Create timestamped filename for new best model
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.best_model_path = f"saved_models/best_model_epoch{epoch + 1}_acc{current_acc:.4f}_{current_time}.pt"

            # Save model state along with training metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_acc1': val_metrics['acc1'],
                'val_acc2': val_metrics['acc2'],
                'avg_acc': current_acc,
                'val_loss': val_metrics['loss']
            }, self.best_model_path)
            return True
        return False


def prepare_dataloaders(data):
    """Prepare training and validation dataloaders"""
    tokenizer = AutoTokenizer.from_pretrained(TrainConfig.pretrained_model)
    dataset = SentencePairDatasetForTwoClass(data, tokenizer)

    # Split dataset into training and validation sets
    val_size = int(TrainConfig.test_size * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return (
        DataLoader(train_dataset, batch_size=TrainConfig.batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=TrainConfig.batch_size, shuffle=False)
    )


def setup_logging():
    """Configure logging to both file and console output"""
    # Create timestamped log filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{current_time}.log"

    # Configure logging format and handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),  # Write logs to file
            logging.StreamHandler()  # Also output to console
        ]
    )
    return logging.getLogger()


def main(data):
    """Main training function that coordinates all components"""
    logger = setup_logging()

    # Prepare data loaders
    train_loader, val_loader = prepare_dataloaders(data)

    # Initialize model and trainer
    model = DebertaTwoHeadModel(TrainConfig.pretrained_model)
    trainer = Trainer(model, train_loader, val_loader)

    logger.info("Starting training...")
    logger.info(f"Model: {TrainConfig.pretrained_model}")
    logger.info(f"Epochs: {TrainConfig.epochs}")

    # Training loop
    for epoch in range(TrainConfig.epochs):
        train_loss = trainer.train_epoch()
        val_metrics = trainer.evaluate()

        # Save best model if validation performance improves
        is_best = trainer.save_best_model(val_metrics, epoch)

        # Log training progress
        logger.info(f"Epoch {epoch + 1}/{TrainConfig.epochs}")
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Task1 Acc: {val_metrics['acc1']:.4f} | Task2 Acc: {val_metrics['acc2']:.4f}")
        if is_best:
            logger.info(f"New best model saved with average acc: {val_metrics['avg_acc']:.4f}")
        logger.info("")

        # Print progress to console
        print(f"Epoch {epoch + 1}/{TrainConfig.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"Task1 Acc: {val_metrics['acc1']:.4f} | Task2 Acc: {val_metrics['acc2']:.4f}")
        if is_best:
            print(f"New best model saved with average acc: {val_metrics['avg_acc']:.4f}")
        print("")

    # Final logging
    logger.info(f"Training completed. Best validation average accuracy: {trainer.best_val_acc:.4f}")
    logger.info(f"Best model saved at: {trainer.best_model_path}")

    print(f"\nTraining completed. Best validation average accuracy: {trainer.best_val_acc:.4f}")
    print(f"Best model saved at: {trainer.best_model_path}")


if __name__ == "__main__":
    main(data)