# WWW147 Paper Implementation

This repository contains the implementation for the WWW147 paper submission. The project focuses on improving Retrieval-Augmented Generation (RAG) systems through advanced filtering and reranking techniques.

## Repository Structure

```
WWW147/
├── Main/                    # Core implementation modules
│   ├── FilterDataset/       # Dataset preprocessing and filtering
│   ├── Construct Database/  # Database construction for various datasets
│   ├── RAG Pipeline/       # RAG implementation with reranker
│   ├── Empirical Study/    # Analysis and evaluation scripts
│   └── plot_figure.py      # Visualization utilities
├── Train4Reranker/         # Training scripts for the reranker model
├── util/                   # Utility functions and helpers
├── example4dataset/        # Dataset examples and samples
└── example4attention/      # Attention mechanism visualizations
```

## Main Components

### 1. RAG Pipeline (`Main/RAG Pipeline/`)
- **generation.py**: Question answering with context-aware generation
- **Reranker.py**: DeBERTa-based dual-task reranker model
- **MIPS.py**: Maximum Inner Product Search for document retrieval
- **evaluate.py**: Evaluation metrics and scoring functions

### 2. Dataset Construction (`Main/Construct Database/`)
Supports multiple datasets:
- NQ (Natural Questions)
- MuSiQue (Multi-hop Questions)
- MultiHopRAG
- PQA (PubMedQA)

### 3. Reranker Training (`Train4Reranker/`)
- **train.py**: Training script for the dual-head DeBERTa reranker
- **get_label.py**: Label generation for training data

### 4. Utilities (`util/`)
- **filter_nq_set.py**: NQ dataset filtering utilities
- **json_method.py**: JSON processing helpers

## Key Features

- **Dual-Task Reranker**: Implements a DeBERTa-based model with two classification heads for improved document ranking
- **Context-Aware Generation**: Advanced prompting that handles "supportive noise" in retrieved documents
- **Multi-Dataset Support**: Compatible with various QA datasets including multi-hop reasoning tasks
- **Attention Visualization**: Provides insights into the reranker's attention mechanisms

## Datasets

Due to repository size limitations, only a portion of the constructed dataset is provided. The complete dataset will be released upon paper acceptance.

### External Datasets Used:
- [MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue)
- [NQ](https://github.com/google-research-datasets/natural-questions)
- [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa)
- [MultiHopRAG](https://github.com/yixuantt/MultiHop-RAG)
- [PQA](https://github.com/pubmedqa/pubmedqa)

## Requirements

The implementation requires:
- PyTorch
- Transformers (Hugging Face)
- OpenAI API (for GPT models)
- Ollama (for local model inference)
- Standard ML libraries (tqdm, numpy, etc.)

## Usage

1. **Dataset Preparation**: Use scripts in `Main/Construct Database/` to prepare datasets
2. **Training**: Run `Train4Reranker/train.py` to train the reranker model
3. **Inference**: Use `Main/RAG Pipeline/generation.py` for question answering
4. **Evaluation**: Evaluate performance using `Main/RAG Pipeline/evaluate.py`

## Key Implementation Details

### Reranker Architecture
The reranker uses a DeBERTa base model with two parallel classification heads, enabling dual-task learning for improved document ranking performance.

### Context Handling
The generation system is designed to handle "supportive noise" - documents that are semantically relevant but don't contain the exact answer, improving robustness in real-world scenarios.

## License

This project is licensed under the Apache License 2.0 - see the NQ preprocessing files for details.
