# WWW147 Paper Implementation - Enhanced Version

This repository contains the enhanced implementation for the WWW147 paper submission with improved modular architecture, comprehensive RAG pipeline, and empirical study capabilities.

## Repository Description
Due to the possibility of exposing author identity information when providing cloud storage links during double-blind review, we did not provide cloud storage links. Due to anonymous repository size limitations, we provided the following files:


- Code files
- Supplementary files related to Figure 4 of the paper
All original charts (some of which did not appear in the original paper)
- Dataset slices and partial data flow slices from the running process for reference by reproducers
- It should be additionally noted that during the code review for revision, we found that in constructing the shots in our Prompt, due to our habits, we unintentionally used cases around us as shots when building 10 shots, which could expose our information. Therefore, we deleted 4 of these shots. The complete 10-shot prompt will be provided after the paper is officially published.

## Repository Structure

```
WWW147/
├── Main/                           # Core implementation modules
│   ├── Config.py                   # Training configuration and dataset classes
│   ├── Prompt.py                   # Noise classification prompts and templates
│   ├── RAG/                        # Refactored RAG implementation
│   │   ├── MIPS.py                 # Maximum Inner Product Search for retrieval
│   │   ├── Reranker.py             # DeBERTa-based dual-task reranker
│   │   ├── RerankerImplement.py    # Multiple reranker model implementations
│   │   ├── generation.py           # Context-aware question answering
│   │   └── evaluate.py             # Evaluation metrics and scoring
│   ├── Empirical Study/            # Analysis and distribution studies
│   │   ├── Empirical-rerank.py     # Empirical reranking analysis
│   │   ├── retrieve.py             # Retrieval experiments
│   │   └── get_distribution.py     # Distribution analysis tools
│   ├── plot_figure.py              # Visualization utilities
├── Train4Reranker/                 # Training scripts for the reranker model
├── util/                           # Enhanced utility functions
│   ├── rag_util.py                 # RAG-specific utilities with English comments
│   └── json_method.py              # JSON processing helpers
├── example4dataset/                # Dataset examples and samples
├── example4attention/              # Attention mechanism visualizations
└── figure/                         # Comprehensive figure collection
    ├── attention.png
    ├── attention_multi_language.png
    ├── empirical study.png
    ├── example_chart.png
    ├── lambda.png
    ├── mips_distribution.png
    ├── pie.png
    ├── reasoning.png
    ├── reranker_distribution.png
    └── why_clip.png
```