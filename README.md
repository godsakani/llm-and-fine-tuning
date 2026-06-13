# Augmented SBERT (AugSBERT)

A hands-on implementation of **Augmented SBERT (AugSBERT)** — a hybrid approach that uses Cross-encoders to label or augment training data and Bi-encoders for fast, scalable inference.

The full walkthrough lives in [`Implementing_Augmented_SBERT_(AugSBERT).ipynb`](Implementing_Augmented_SBERT_(AugSBERT).ipynb).

## Overview

AugSBERT improves Bi-encoder (Sentence-BERT) models by enriching training data in one of three ways, depending on how much labeled data you have:

| Scenario | Labeled data | Strategy |
|----------|--------------|----------|
| **1** | Fully labeled sentence pairs | Augment gold data with MLM-based synonym insertion, then fine-tune a Bi-encoder |
| **2** | Few annotated samples | Train a Cross-encoder on gold data, use it to label silver pairs, then fine-tune a Bi-encoder on gold + silver |
| **3** | No annotated samples | *Planned — section header only in the notebook* |

All implemented scenarios use the [STS Benchmark](https://sbert.net/docs/datasets/stsbenchmark.html) dataset: human-annotated sentence pairs with similarity scores on a 0–5 scale (normalized to 0–1 for training).

## Requirements

Install the core dependencies used in the notebook:

```bash
pip install transformers sentence-transformers datasets
```

The notebook also uses `torch`, `pandas`, `numpy`, `matplotlib`, `seaborn`, and `tqdm` (typically available in a standard ML environment).

> **Note:** The original notebook referenced `nlpaug` for contextual word augmentation, but that library is incompatible with current `transformers` versions. Scenario 1 uses a **custom Masked Language Model (MLM)** augmentation built with `transformers` instead.

## Quick Start

1. Open [`Implementing_Augmented_SBERT_(AugSBERT).ipynb`](Implementing_Augmented_SBERT_(AugSBERT).ipynb) in Jupyter or Google Colab.
2. Run the setup cells to install packages and import libraries.
3. Follow the scenario sections in order (1 → 2).

### Default configuration

| Parameter | Value |
|-----------|-------|
| Backbone model | `bert-base-uncased` |
| Batch size | 32 |
| Epochs | 5 |
| Device | CUDA if available, otherwise CPU |

## Scenario 1: Fully Labeled Dataset

Fine-tune a Sentence-BERT Bi-encoder using gold-standard sentence pairs plus augmented data.

**Pipeline**

1. **Download & prepare data** — STS Benchmark is fetched automatically and split into train (5,749 pairs) and validation (1,500 pairs).
2. **Data augmentation** — A custom `custom_mlm_augment()` function uses `BertForMaskedLM` to insert contextually appropriate words into sentence pairs (replacing the original `nlpaug` approach).
3. **Build SBERT** — `bert-base-uncased` transformer + mean pooling via `sentence-transformers`.
4. **Train** — `CosineSimilarityLoss` with `EmbeddingSimilarityEvaluator` on the validation set.
5. **Compare** — Train on gold-only data vs. gold + augmented data. Augmented training consistently yields better validation metrics.

**Output:** `models/scenario1_model`

## Scenario 2: Few Annotated Samples

When labeled data is limited, use a Cross-encoder to generate weak labels ("silver" data) for additional sentence pairs.

**Pipeline**

1. **Prepare data** — Load STS Benchmark with symmetric pairs (`CrossEncoder(A,B)` = `CrossEncoder(B,A)`).
2. **Train Cross-encoder** — Fine-tune on the small gold training set with `CECorrelationEvaluator`.
3. **Generate silver pairs** — Use a pre-trained Bi-encoder (`bert-base-nli-stsb-mean-tokens`) to retrieve the top-3 most similar sentences per sentence via cosine similarity, skipping duplicates already in the training set.
4. **Label silver pairs** — The trained Cross-encoder assigns similarity scores to the new pairs.
5. **Train Bi-encoder** — Fine-tune on combined gold + silver data with `CosineSimilarityLoss`.
6. **Compare** — Gold-only vs. gold + silver training; the augmented set improves performance by exposing the Bi-encoder to a more diverse set of pairs.

**Outputs:**

- `models/scenario2_model_cross_encoder`
- `models/scenario2_model_bi_encoder`

## Scenario 3: No Annotated Samples

This section is outlined in the notebook but not yet implemented.

## Project Structure

```
.
├── Implementing_Augmented_SBERT_(AugSBERT).ipynb   # Main tutorial notebook
├── README.md
├── stsbenchmark.tsv.gz                             # Downloaded at runtime (Scenario 1)
├── datasets/
│   └── stsbenchmark.tsv.gz                         # Dataset path used in Scenario 2
└── models/
    ├── scenario1_model/                            # Scenario 1 Bi-encoder
    ├── scenario2_model_cross_encoder/              # Scenario 2 Cross-encoder
    └── scenario2_model_bi_encoder/                 # Scenario 2 Bi-encoder
```

## Key Libraries

- **[sentence-transformers](https://www.sbert.net/)** — SBERT models, pooling, losses, and evaluation (`EmbeddingSimilarityEvaluator`, `CrossEncoder`, `CosineSimilarityLoss`)
- **[transformers](https://huggingface.co/docs/transformers)** — BERT backbone and MLM-based augmentation (`BertForMaskedLM`)
- **[datasets](https://huggingface.co/docs/datasets)** — Hugging Face dataset utilities

## References

- [Augmented SBERT paper](https://arxiv.org/abs/2010.08240) — Thakur et al., 2020
- [Sentence-Transformers documentation](https://www.sbert.net/)
- [STS Benchmark dataset](https://sbert.net/docs/datasets/stsbenchmark.html)
