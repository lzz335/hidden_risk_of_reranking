from torch.utils.data import Dataset
import torch


class TrainConfig:
    pretrained_model = 'microsoft/deberta-v3-base'
    batch_size = 8
    epochs = 10
    learning_rate = 1e-5
    weight_decay = 0.01
    test_size = 0.1
    max_length = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SentencePairDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.data[idx]["sentence1"],
            self.data[idx]["sentence2"],
            max_length=TrainConfig.max_length,
            padding='max_length',
            truncation=True,
            return_overflowing_tokens=False,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.data[idx]["label"], dtype=torch.long)
        }


class SentencePairDatasetForTwoClass(SentencePairDataset):
    def __init__(self, data, tokenizer):
        super().__init__(data, tokenizer)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.data[idx]["sentence1"],
            self.data[idx]["sentence2"],
            max_length=TrainConfig.max_length,
            padding='max_length',
            truncation=True,
            return_overflowing_tokens=False,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels1': torch.tensor(1 if self.data[idx]["label"] == 0 else 0, dtype=torch.long),
            'labels2': torch.tensor(1 if self.data[idx]["label"] == 1 else 0, dtype=torch.long)
        }
