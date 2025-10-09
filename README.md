# WWW147 Paper Implementation

This repository contains the implementation for the WWW147 paper submission.


**Note:** This section was originally written by humans.

## Repository Description
Due to the possibility of exposing author identity information when providing cloud storage links during double-blind review, we did not provide cloud storage links. Due to anonymous repository size limitations, we provided the following files:
1. Code files.
2. Supplementary files related to Figure 4 of the paper.
3. All original charts (some of which did not appear in the original paper).
4. Dataset slices and partial data flow slices from the running process for reference by reproducers.

It should be additionally noted that during the code review for revision, we found that in constructing the shots in our Prompt, due to our habits, we unintentionally used cases around us as shots when building 10 shots, which could expose our unit information. Therefore, we deleted 4 of these shots. The complete 10-shot prompt will be provided after the paper is officially published.

## Data Flow Overview
Our repository provides data preprocessing functionality to convert original datasets into a unified format, then:
- For Empirical Study, you need to first build the database for the Musique dataset, then run retrieve.py in the Empirical Study folder to construct retrieval results, then perform reranking, and finally calculate the corresponding distribution information. These distribution informations are plotted into our Figure 3 and Appendix Figure 6.
- For the main experiments, we uniformly perform MIPS operations in batches, then perform reranking, and finally generate results uniformly.

**Note:** Next section was originally written by CLAUDE.

## Repository Structure

```
WWW147/
├── Main/                    # Core implementation modules
│   ├── FilterDataset/       # Dataset preprocessing and filtering
│   ├── Construct Database/  # Database construction for various datasets
│   ├── RAG Pipeline/       # RAG implementation with reranker
│   ├── Empirical Study/    # Analysis for Section 3
│   └── plot_figure.py      # Visualization utilities
│   └── Config.py           # configuration 
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
- MuSiQue
- MultiHopRAG
- PQA (PubMedQA)

### 3. Reranker Training (`Train4Reranker/`)
- **train.py**: Training script for the dual-head DeBERTa reranker
- **get_label.py**: Label generation for training data

### 4. Utilities (`util/`)
- **filter_nq_set.py**: NQ dataset filtering utilities
- **json_method.py**: JSON processing helpers

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


