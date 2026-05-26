# Generating SROI^- Ontologies via Knowledge Graph Query Embedding Learning.

This repository contains the implementation of **AConE** (Angle Cone Embedding), a method for complex query answering over knowledge graphs using cone-based geometric embeddings. The model learns to embed entities and relations as points in a high-dimensional space, representing queries as cone structures for efficient reasoning.

## Overview

AConE addresses the challenge of answering complex logical queries over incomplete knowledge graphs (KGs). Rather than storing all possible facts explicitly, this approach learns embeddings that can reason about missing links through geometric representations.

### Key Features
- **Complex Query Support**: Handles 16 types of complex queries including:
  - Path queries (1p, 2p, 3p)
  - Intersection queries (2i, 3i)
  - Union queries (2u-DNF, 2u-DM)
  - Negation queries (2in, 3in, 2u variants)
  - Compositions (ip, pi, inp, pin, pni, up)
  
- **Geometric Embedding**: Uses angle and radius dimensions with cone-based representations for expressive query embeddings
- **Multi-GPU Support**: Configured for distributed training
- **Multiple Datasets**: Pre-configured for NELL-betae and WN18RR-QA benchmarks

## Project Structure

```
├── main.py              # Main training and evaluation script
├── models.py            # Model architecture with AngleScale and other components
├── dataloader.py        # Dataset classes and data loading utilities
├── util.py              # Utility functions (seed setting, tuple conversion, etc.)
├── script.sh            # Example training scripts for different datasets
└── data/
    ├── NELL-betae/      # NELL dataset (63,361 entities, 400 relations)
    │   ├── train.txt
    │   ├── valid.txt
    │   ├── test.txt
    │   ├── stats.txt
    │   ├── train-queries.pkl
    │   ├── valid-queries.pkl
    │   ├── test-queries.pkl
    │   ├── train-answers.pkl
    │   └── [easy/hard answers PKL files]
    └── WN18RR-QA/       # WordNet dataset (40,559 entities, 22 relations)
        ├── train.txt
        ├── valid.txt
        ├── test.txt
        ├── stats.txt
        └── [query and answer files]
```

## Datasets

### NELL-betae
- **Entities**: 63,361
- **Relations**: 400
- Named entity recognition dataset with complex query annotations

### WN18RR-QA
- **Entities**: 40,559
- **Relations**: 22
- WordNet-based knowledge graph with hierarchical structure

## Installation

### Requirements
- Python 3.6+
- PyTorch (with CUDA support recommended)
- NumPy
- TensorBoard (for logging)

### Setup
```bash
# Clone the repository
git clone https://github.com/RoyaHe/AConE.git
cd AConE

# Install dependencies
pip install torch numpy tensorboardX
```

## Usage

### Training and Testing

Use the provided `script.sh` for example training configurations:

```bash
# Train on NELL-betae dataset
CUDA_VISIBLE_DEVICES=1 python main.py --do_train --do_test \
  --data_path ./data/NELL-betae -n 128 -b 512 -d 800 -g 20 -cenr 0.02 \
  --data NELL -lr 0.0001 --max_steps 300001 --cpu_num 0 \
  --valid_steps 60000 --test_batch_size 4 --log_steps 100 \
  --drop 0.2 --tag train --save_checkpoint_steps 3000
```

```bash
# Train on WN18RR-QA dataset
CUDA_VISIBLE_DEVICES=3 python main.py --cuda --do_train --do_test \
  --data_path ./data/WN18RR-QA -n 128 -b 512 -d 800 -g 20 -cenr 0.02 \
  --data wn18rr -lr 0.0001 --max_steps 300001 --cpu_num 0 \
  --valid_steps 60000 --test_batch_size 4 --log_steps 100 \
  --drop 0.2 --tag train --save_checkpoint_steps 3000
```

### Command Line Arguments

Key arguments for `main.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--do_train` | False | Enable training |
| `--do_valid` | False | Enable validation |
| `--do_test` | False | Enable testing |
| `--cuda` | False | Use GPU |
| `--data_path` | None | Path to dataset directory |
| `--data` | None | Dataset name (NELL or wn18rr) |
| `-d, --hidden_dim` | 500 | Embedding dimension |
| `-g, --gamma` | 12.0 | Margin in the loss |
| `-b, --batch_size` | 1024 | Batch size for training |
| `--test_batch_size` | 1 | Batch size for validation/test |
| `-lr, --learning_rate` | 0.0001 | Learning rate |
| `-n, --negative_sample_size` | 128 | Negative samples per query |
| `-cenr, --center_reg` | 0.02 | Center regulation (balances in-cone and out-cone distance) |
| `--max_steps` | 100000 | Maximum training iterations |
| `--valid_steps` | 10000 | Validation interval |
| `--save_checkpoint_steps` | 15000 | Checkpoint save interval |
| `--log_steps` | 100 | Logging interval |
| `--drop` | 0.0 | Dropout rate |
| `--tasks` | all | Tasks to train on (connected by dots) |
| `--seed` | 0 | Random seed |

## Code Components

### models.py
Contains the core model architecture:
- **AngleScale**: Handles angle and radius scaling for embeddings
- Utility functions for complex number conversions (Cartesian ↔ Polar)
- Distance computation functions for cone-based reasoning

### dataloader.py
Dataset utilities:
- **TestDataset**: For evaluation with all entities as negative samples
- **TrainDataset**: For training with sampled negative entities
- **SingledirectionalOneShotIterator**: Efficient iteration over query batches

### main.py
Training and evaluation pipeline:
- Argument parsing and configuration
- Model initialization and training loop
- Validation and testing procedures
- Checkpoint saving and loading
- TensorBoard logging

### util.py
Helper functions:
- `set_global_seed()`: Set random seeds for reproducibility
- `eval_tuple()`: Parse query structures
- `parse_time()`: Timestamp generation
- List/tuple conversion utilities

## Query Types

The model supports the following 16 query structures:

| Query Type | Description | Example |
|-----------|-------------|---------|
| 1p | Single-hop path | e ← r |
| 2p | Two-hop path | e ← r₁ ← r₂ |
| 3p | Three-hop path | e ← r₁ ← r₂ ← r₃ |
| 2i | Two-way intersection | (e₁ ← r₁) ∩ (e₂ ← r₂) |
| 3i | Three-way intersection | (e₁ ← r₁) ∩ (e₂ ← r₂) ∩ (e₃ ← r₃) |
| ip | Intersection then path | ((e₁ ← r₁) ∩ (e₂ ← r₂)) ← r₃ |
| pi | Path then intersection | (e₁ ← r₁ ← r₂) ∩ (e₃ ← r₃) |
| 2in | Two-way intersection with negation | (e₁ ← r₁) ∩ ¬(e₂ ← r₂) |
| 3in | Three-way intersection with negation | (e₁ ← r₁) ∩ (e₂ ← r₂) ∩ ¬(e₃ ← r₃) |
| inp | Intersection with negation then path | ((e₁ ← r₁) ∩ ¬(e₂ ← r₂)) ← r₃ |
| pin | Path then intersection with negation | (e₁ ← r₁ ← r₂) ∩ ¬(e₃ ← r₃) |
| pni | Path with negation then intersection | (e₁ ← r₁ ← ¬r₂) ∩ (e₂ ← r₃) |
| 2u-DNF | Two-way union (DNF form) | (e₁ ← r₁) ∪ (e₂ ← r₂) |
| 2u-DM | Two-way union (De Morgan form) | ¬(¬(e₁ ← r₁) ∩ ¬(e₂ ← r₂)) |
| up-DNF | Union then path (DNF form) | ((e₁ ← r₁) ∪ (e₂ ← r₂)) ← r₃ |
| up-DM | Union then path (De Morgan form) | ¬(¬(e₁ ← r₁ ← r₃) ∩ ¬(e₂ ← r₂ ← r₃)) |

## Training Details

- **Loss Function**: Margin-based loss with negative sampling
- **Optimization**: Adam or SGD optimizer (configurable)
- **Regularization**: 
  - Center regulation to balance cone geometry
  - Dropout for generalization
  - Gradient clipping (default: 0.5)
- **Evaluation Metrics**: MRR (Mean Reciprocal Rank) and Hits@K

## Performance

The model is evaluated on:
- **Mean Reciprocal Rank (MRR)**: Measures ranking quality of answers
- **Hits@1, Hits@3, Hits@10**: Measures if correct answer appears in top-K predictions

## Output

During training, the model logs:
- Loss values at regular intervals
- Validation metrics every `--valid_steps` iterations
- Checkpoint files every `--save_checkpoint_steps` iterations
- TensorBoard summaries for visualization

## Citation

If you use this code, please cite the relevant papers (cite BetaE or the original AConE paper if available).

## License

This project is provided as-is for research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

This repo contains the official Pytorch implementation of [Generating SROI^- Ontologies via Knowledge Graph Query Embedding Learning.](https://arxiv.org/abs/2407.09212)

 
