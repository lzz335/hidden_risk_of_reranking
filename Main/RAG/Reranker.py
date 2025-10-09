# Import necessary libraries for model architecture and tensor operations
from typing import List
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from RAG.Config import TrainConfig

class DebertaTwoHeadModel(nn.Module):
    """DeBERTa model with two classification heads for dual-task learning
    
    This architecture enables the model to learn two related tasks simultaneously,
    improving performance on document re-ranking through multi-task learning.
    """

    def __init__(self, pretrained_model):
        """Initialize the dual-head DeBERTa model
        
        Args:
            pretrained_model: Path or name of the pretrained DeBERTa model
        """
        super().__init__()
        # Load the base DeBERTa model
        self.deberta = AutoModel.from_pretrained(pretrained_model)
        
        # Create two separate classification heads
        # Each head outputs 2 classes (binary classification for different tasks)
        self.classifier1 = nn.Linear(self.deberta.config.hidden_size, 2)
        self.classifier2 = nn.Linear(self.deberta.config.hidden_size, 2)

        # Initialize weights using Xavier initialization for better training stability
        nn.init.xavier_normal_(self.classifier1.weight)
        nn.init.xavier_normal_(self.classifier2.weight)
        # Initialize biases to zero
        self.classifier1.bias.data.zero_()
        self.classifier2.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """Forward pass through the dual-head model
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask for input tokens
            
        Returns:
            Tuple of outputs from both classification heads
        """
        # Process input through the base DeBERTa model
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Extract the [CLS] token representation (used for classification)
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Pass through both classification heads
        return self.classifier1(cls_output), self.classifier2(cls_output)


def combine_documents(documents, scores=None, top_k=5):
    """Combine documents with relevance scores and return formatted top-k string
    
    This function sorts documents by their scores and formats them into a single
    string for use in language model prompts.

    Args:
        documents: List of document texts
        scores: Optional list of relevance scores (default: descending order)
        top_k: Number of top documents to return (default: 5)

    Returns:
        Formatted string containing top-k documents with their positions
    """
    # Generate default scores if not provided (descending order)
    if scores is None:
        scores = [-index / len(documents) for index, _ in enumerate(documents)]
    
    # Validate input dimensions
    assert len(documents) == len(scores)
    assert top_k <= len(documents)

    # Combine documents with scores
    corpus = []
    for doc, score in zip(documents, scores):
        corpus.append({"document": doc, "score": score})
    
    # Sort by score in descending order
    new_corpus = sorted(corpus, key=lambda x: x["score"], reverse=True)
    
    # Format top-k documents into a single string
    query_document = ""
    ids = 0
    # Process from lowest to highest to maintain order in final output
    for i in range(top_k - 1, -1, -1):
        query_document = query_document + "Corpus {}: ".format(ids + 1) + new_corpus[i]["document"] + "\n"
        ids += 1
    return query_document


class ReRankerPipeline:
    """Base class for document re-ranking pipeline
    
    Provides a generic interface for re-ranking retrieved documents using various
    re-ranking methods.

    Attributes:
        reranker_method: Function to calculate document relevance scores
    """

    def __init__(self, reranker_method):
        """Initialize the re-ranking pipeline
        
        Args:
            reranker_method: Function that takes query and documents, returns scores
        """
        self.reranker_method = reranker_method

    def reranker(self, query_data, retrieved_documents):
        """Re-rank documents using the specified method
        
        Args:
            query_data: The search query
            retrieved_documents: List of retrieved documents to re-rank
            
        Returns:
            Formatted string containing top-k re-ranked documents
        """
        return combine_documents(retrieved_documents, self.reranker_method(query_data, retrieved_documents))


class OurReranker:
    """Custom re-ranker using DeBERTa two-head model
    
    Implements a sophisticated re-ranking approach that combines outputs from
    two classification heads using weighted logarithms for improved performance.

    Attributes:
        inference_config: Model inference configuration
        weight: List of weights for combining scores from both heads
        device: Computation device (CPU/GPU)
        inference_model: Loaded dual-head DeBERTa model instance
        tokenizer: Tokenizer for text processing
        max_length: Maximum sequence length for tokenization
    """

    def __init__(self, model_save_path, device='cuda', inference_config=None, weight=None, max_length=512):
        """Initialize the custom re-ranker
        
        Args:
            model_save_path: Path to the saved model checkpoint
            device: Computation device ('cuda' or 'cpu')
            inference_config: Model configuration (default: TrainConfig)
            weight: Weights for combining head outputs (default: [1, -0.5])
            max_length: Maximum token sequence length (default: 512)
        """
        # Set configuration with defaults
        if inference_config is None:
            self.inference_config = TrainConfig
        else:
            self.inference_config = inference_config
            
        # Set weights for combining scores (positive for first head, negative for second)
        if weight is None:
            # 1， -0.5 is the setting in our paper
            self.weight = [1, -0.5]  # Default weights favor first head, penalize second
        else:
            self.weight = weight
            
        # Configure device (fallback to CPU if CUDA unavailable)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.inference_model = None
        
        # Load model and tokenizer
        self.load_model(model_save_path)
        self.tokenizer = AutoTokenizer.from_pretrained(inference_config.pretrained_model)
        self.max_length = max_length

    def load_model(self, model_save_path):
        """Load pre-trained model from specified path
        
        Args:
            model_save_path: Path to the model checkpoint file
        """
        # Initialize the dual-head model
        self.inference_model = DebertaTwoHeadModel(TrainConfig.pretrained_model)
        
        # Load the saved checkpoint
        checkpoint = torch.load(model_save_path, map_location=self.device)
        self.inference_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Optimize for inference: use half precision and eval mode
        self.inference_model.half()  # Convert to FP16 for memory efficiency
        self.inference_model.to(self.device)  # Move model to GPU
        self.inference_model.eval()  # Set to evaluation mode

    def inference(self, input_sentences1: List[str] | str, input_sentences2: List[str]):
        """Run inference on sentence pairs to get relevance scores

        Args:
            input_sentences1: Single query string or list of queries
            input_sentences2: List of documents to compare against queries

        Returns:
            Tuple of probability lists from both model heads:
            - probs1: Probabilities from first classification head
            - probs2: Probabilities from second classification head
        """
        # Handle single query case: duplicate it to match document list length
        if isinstance(input_sentences1, str):
            input_sentences1 = [input_sentences1 for _ in input_sentences2]
            
        # Tokenize the batch of sentence pairs
        inputs = self.tokenizer(
            input_sentences1,
            input_sentences2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True
        )

        # Move input tensors to the same device as the model
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Run inference without gradient computation
        with torch.no_grad():
            logits1, logits2 = self.inference_model(attention_mask=attention_mask, input_ids=input_ids)

        # Apply softmax to convert logits to probabilities and move to CPU
        probs1 = torch.softmax(logits1, dim=-1).cpu().tolist()
        probs2 = torch.softmax(logits2, dim=-1).cpu().tolist()

        return probs1, probs2

    def rerank(self, query_data, retrieved_documents):
        """Calculate re-ranking scores using weighted combination of both heads
        
        Combines the outputs from both classification heads using weighted logarithms.
        This approach provides more nuanced scoring than simple averaging.

        Args:
            query_data: The search query
            retrieved_documents: List of documents to re-rank

        Returns:
            Tuple containing:
                - Combined scores list (weighted log combination)
                - Raw probabilities from first model head
                - Raw probabilities from second model head
        """
        # Get probabilities from both classification heads
        p1, p2 = self.inference(query_data, retrieved_documents)
        
        # Calculate weighted combination using logarithms
        # First head: positive weight, Second head: negative weight
        # Apply minimum threshold of 0.5 to second head to avoid extreme negative values
        combined_scores = [
            torch.log(p1[i][1]) * self.weight[0] + 
            torch.log(p2[i][1] if p2[i][1] > 0.5 else 0.5) * self.weight[1]
            for i in range(len(p1))
        ]
        
        return combined_scores, p1, p2


class RerankerByRerankerModel:
    """Re-ranking using external pre-trained models
    
    Provides a wrapper for using external pre-trained re-ranking models,
    particularly designed for models like Alibaba-NLP/gte-reranker-modernbert-base.

    Attributes:
        reranker_model: Pre-trained model instance for re-ranking
        tokenizer: Tokenizer for text processing
        device: Computation device (CPU/GPU)
    """

    def __init__(self, reranker_model):
        """Initialize with external re-ranking model
        
        Args:
            reranker_model: Pre-trained re-ranking model instance
        """
        self.reranker_model = reranker_model
        self.tokenizer = None  # Lazy loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_tokenizer(self, tokenizer):
        """Set custom tokenizer for text processing
        
        Args:
            tokenizer: Tokenizer instance to use for text preprocessing
        """
        self.tokenizer = tokenizer

    def reranker(self, query_data: str, retrieved_documents: List[str]):
        """Generate relevance scores using external model's compute_score method
        
        Args:
            query_data: The search query
            retrieved_documents: List of documents to score
            
        Returns:
            List of relevance scores for each document
        """
        # Create sentence pairs for the external model
        sentence_pairs = [[query_data, doc] for doc in retrieved_documents]
        # Use the external model's built-in scoring method
        scores = self.reranker_model.compute_score(sentence_pairs)
        return list(scores)

    def reranker2(self, query_data, retrieved_documents):
        """Alternative re-ranking method for specific models
        
        Specifically designed for Alibaba-NLP/gte-reranker-modernbert-base.
        Uses direct model inference with custom tokenizer.

        Args:
            query_data: Search query
            retrieved_documents: List of retrieved documents

        Returns:
            List of similarity scores as Python floats
        """
        # Lazy loading of tokenizer for the specific model
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-reranker-modernbert-base")
            
        # Create sentence pairs
        sentence_pairs = [[query_data, doc] for doc in retrieved_documents]
        
        # Tokenize and prepare inputs
        with torch.no_grad():
            inputs = self.tokenizer(
                sentence_pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            )
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # Get model outputs and extract logits
            scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
            
        # Convert to CPU and return as list
        return scores.to("cpu").numpy().tolist()