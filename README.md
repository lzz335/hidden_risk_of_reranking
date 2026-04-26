# WWW147 Paper Implementation - Enhanced Version

This repository contains the enhanced implementation for the WWW147 paper submission with improved modular architecture, comprehensive RAG pipeline, and empirical study capabilities.

## Notice / Erratum

We sincerely apologize for an issue in the multi-retriever generalization experiment reported in Figure 5 of Appendix E.

In this paper, we study a hidden risk of reranking in RAG systems: although conventional rerankers can increase the proportion of ground-truth documents (GD) among top-ranked results, they may also promote harmful documents (HD), which can mislead the downstream LLM generation. Based on this observation, our method proposes a risk-aware reranking objective that balances the beneficial and harmful impacts of retrieved documents via dual-aspect information gain.

During follow-up work, we found that one experimental setting in Appendix E was incorrect. Specifically, when using Qwen3 for document embedding in the multi-retriever generalization experiment, the instruction should be prepended only to the document side, while no instruction should be added to the query side. However, in our previous implementation, we mistakenly added the instruction to both documents and queries. This likely made the retrieved representations more worse than intended and caused the corresponding results in Figure 5 of Appendix E to be overestimated.

Unfortunately, the source code currently released in this repository is a copy of the version submitted to AAAI 2026. The later code files and intermediate data used for the WWW 2026 submission version, including the additional appendix experiment added after the AAAI rejection, were permanently lost due to a machine failure in November 2025. As a result, the currently available code cannot reproduce this specific appendix experiment.

Based on our subsequent qualitative analysis, we believe that the results of this multi-retriever generalization experiment are likely higher than they should be. We sincerely apologize for the confusion and inconvenience this may cause. This issue concerns the appendix generalization experiment rather than the main motivation of the paper: identifying the hidden risk that reranking may increase HD exposure and proposing a risk-aware reranking framework to mitigate it. We will further examine and clarify this setting in our follow-up work. If you are interested in this issue or related future studies, please feel free to contact us.

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
