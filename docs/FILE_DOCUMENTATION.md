# FLASH-IDS Codebase Documentation

This document provides a comprehensive overview of every file in the Flash-IDS repository and its purpose within the intrusion detection system.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Root Files](#root-files)
3. [Dataset Notebooks](#dataset-notebooks)
4. [Data Files](#data-files)
5. [Trained Weights](#trained-weights)
6. [System Architecture](#system-architecture)
7. [Setup Instructions](#setup-instructions)

---

## Project Overview

**FLASH** (Fast Learning Attack Scanner via Hashing) is an intrusion detection system developed by researchers at IEEE Symposium on Security and Privacy 2024. It uses **Graph Neural Networks (GNN)** combined with **Word2Vec semantic embeddings** to detect attacks in provenance graphs generated from system audit logs.

The system operates by:
1. Converting system audit logs into directed graphs (nodes = processes/files, edges = actions)
2. Generating semantic embeddings via Word2Vec from action sequences
3. Training GraphSAGE models for node classification
4. Using multi-snapshot ensemble voting for inference
5. Applying two-hop graph propagation to expand detected alerts

---

## Root Files

### `README.md`
The main documentation file. Contains:
- Project introduction and paper link
- Prerequisites (Jupyter Notebook)
- Installation instructions
- Dataset access links (DARPA OpTC, E3, Streamspot, Unicorn)
- Code structure explanation
- Citation information (BibTeX)

### `requirements.txt`
Pinned Python dependencies for the project:

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 1.3.5 | Data manipulation |
| numpy | 1.23.1 | Numerical computing |
| scikit-learn | 1.1.1 | Machine learning utilities |
| torch | 1.12.0+cu113 | Deep learning framework |
| torch-geometric | 2.1.0 | Graph neural networks |
| xgboost | 0.90 | Gradient boosting classifier |
| gensim | 4.3.0 | Word2Vec embeddings |
| torch-cluster | 1.6.0 | Graph clustering ops |
| torch-scatter | 2.0.9 | Scatter operations |
| torch-sparse | 0.6.15 | Sparse tensor ops |
| torch-spline-conv | 1.2.1 | Spline convolution |
| networkx | 3.0 | Graph utilities |

### `.DS_Store`
macOS system file (can be ignored/deleted)

---

## Dataset Notebooks

Each Jupyter notebook handles one dataset: parsing, training, and evaluation.

### `Cadets.ipynb`
- **Dataset**: DARPA E3 Cadets (node-level)
- **Source**: Google Drive
- **Approach**: Online GNN + Word2Vec (no decoupled mode due to missing node attributes)
- **Features**: Uses embedded positional encoder + Word2Vec
- **Classes**: 6 node types (PROCESS, FILE, SOCKET, PIPE, NETFLOW, DIR)

### `OpTC.ipynb`
- **Dataset**: DARPA OpTC (graph-level)
- **Source**: GitHub (FiveDirections/OpTC-data)
- **Approach**: XGBoost hybrid + offline embeddings
- **Model**: `xgb.pkl` classifier

### `Fivedirections.ipynb`
- **Dataset**: DARPA Five Directions
- **Approach**: Online GNN + Word2Vec
- **Weights**: 13 GNN snapshots (0-12) + Word2Vec model

### `Theia.ipynb`
- **Dataset**: DARPA Theia
- **Approach**: Online GNN + Word2Vec
- **Weights**: 20 GNN snapshots (0-19) + Word2Vec model

### `Trace.ipynb`
- **Dataset**: DARPA Trace
- **Approach**: Online GNN + Word2Vec
- **Weights**: 20 GNN snapshots (0-19) + Word2Vec model

### `Streamspot.ipynb`
- **Dataset**: Streamspot (graph-level, bipartite)
- **Source**: GitHub (sbustreamspot/sbustreamspot-data)
- **Approach**: Graph-level classification
- **Weights**: Single model (`lstreamspot.pth`) + Word2Vec

### `Unicorn.ipynb`
- **Dataset**: Unicorn (Shellshock APT)
- **Source**: GitHub (margoseltzer/shellshock-apt)
- **Approach**: Custom GNN variant
- **Weights**: 20 numbered snapshots + base model

### `utils.ipynb`
Shared utility functions used across all notebooks:

| Function | Purpose |
|----------|---------|
| `Split_logs_into_chunks()` | Split DataFrame into graph chunks |
| `Modify_Structure()` | Combine attack + benign data |
| `train_xgb_model()` | Train XGBoost classifier |
| `train_gnn()` | Train GNN model |
| `Validate()` | Calculate confidence scores |

---

## Data Files

Located in `data_files/` directory.

### `cadets.json`
- **Format**: JSON array
- **Content**: Ground truth malicious node UUIDs
- **Purpose**: Evaluation labels for Cadets dataset
- **Size**: ~514KB

### `fivedirections.json`
- **Format**: JSON array
- **Content**: Ground truth malicious node IDs
- **Purpose**: Evaluation labels for Five Directions dataset

### `theia.json`
- **Format**: JSON array
- **Content**: Ground truth malicious node IDs
- **Purpose**: Evaluation labels for Theia dataset

### `trace.json`
- **Format**: JSON array
- **Content**: Ground truth malicious node IDs
- **Purpose**: Evaluation labels for Trace dataset

### `optc.txt`
- **Format**: Tab-delimited text
- **Content**: Parsed edge list (src_id, src_type, dst_id, dst_type, edge_type, timestamp)
- **Purpose**: Pre-processed graph edges for OpTC dataset

### `emb_store.json`
- **Format**: JSON dictionary
- **Content**: Pre-computed embeddings for Windows system files
- **Purpose**: Embedding lookup for file path semantic features

---

## Trained Weights

Located in `trained_weights/` directory, organized by dataset.

### `trained_weights/cadets/`
| File | Purpose |
|------|---------|
| `word2vec_cadets_E3.model` | Trained Word2Vec model (window=5, dim=30) |
| `lword2vec_gnn_cadets{0-21}_E3.pth` | 22 GNN model snapshots for ensemble voting |

### `trained_weights/optc/`
| File | Purpose |
|------|---------|
| `gnn_temp.pth` | Intermediate GNN embeddings |
| `xgb.pkl` | Trained XGBoost classifier |

### `trained_weights/fivedirections/`
| File | Purpose |
|------|---------|
| `word2vec_five_E3.model` | Word2Vec model |
| `lword2vec_gnn_five{0-12}_E3.pth` | 13 GNN snapshots |

### `trained_weights/theia/`
| File | Purpose |
|------|---------|
| `word2vec_theia_E3.model` | Word2Vec model |
| `lword2vec_gnn_theia{0-19}_E3.pth` | 20 GNN snapshots |

### `trained_weights/trace/`
| File | Purpose |
|------|---------|
| `word2vec_trace_E3.model` | Word2Vec model |
| `lword2vec_gnn_trace{0-19}_E3.pth` | 20 GNN snapshots |

### `trained_weights/streamspot/`
| File | Purpose |
|------|---------|
| `streamspot.model` | Word2Vec model |
| `lstreamspot.pth` | Graph-level classifier |

### `trained_weights/unicorn/`
| File | Purpose |
|------|---------|
| `unicorn.model` | Word2Vec model |
| `unicorn.pth` | Base GNN model |
| `unicorn{0-19}.pth` | 20 numbered snapshots |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW LOG FILES                                 │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PARSING LAYER                                     │
│  - Extract UUIDs, node types                                        │
│  - Build edge lists (src, dst, edge_type, timestamp)                    │
│  - Output: Tab-delimited .txt files                                  │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 FEATURE EXTRACTION                                    │
│  ┌──────────────────────┐    ┌──────────────────────┐          │
│  │   WORD2VEC          │    │   POSITIONAL         │          │
│  │   Semantic Embeddings │    │   Encoder          │          │
│  │   (window=5,dim=30)│    │   (time-based)     │          │
│  └──────────────────────┘    └──────────────────────┘          │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  GNN TRAINING                                       │
│  Model: GraphSAGE (2 layers, 32 hidden, dropout=0.5)                │
│  Training: Confidence-based progressive filtering                      │
│  Output: 22 model snapshots per dataset                            │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  INFERENCE                                           │
│  1. Load 22 model snapshots                                        │
│  2. Ensemble voting (majority vote)                               │
│  3. Two-hop propagation                                          │
│  4. Calculate PR/F1/DR against ground truth                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Algorithms

1. **Confidence-Based Filtering**: During training, correctly classified nodes are masked out progressively, focusing learning on harder examples.

2. **Multi-Snapshot Ensemble**: 22 models vote on test nodes. A node is flagged malicious if majority agree OR confidence ≥ 0.9.

3. **Two-Hop Propagation**: Detected alerts are expanded to include neighboring nodes within 2 graph hops.

---

## Setup Instructions

### Option 1: pip (Original)

```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: uv (Recommended)

```bash
# Create virtual environment
uv venv .venv --python 3.9

# Install core dependencies
uv pip install pandas==1.3.5 numpy==1.23.1 scikit-learn==1.1.1 xgboost==0.90 gensim==4.3.0 networkx==3.0

# Install PyTorch with CUDA 11.3
uv pip install torch==1.12.0+cu113 --index-url https://download.pytorch.org/whl/cu113

# Install PyTorch Geometric
uv pip install torch-geometric==2.1.0 torch-cluster==1.6.0+pt112cu113 torch-scatter==2.0.9 torch-sparse==0.6.15+pt112cu113 torch-spline-conv==1.2.1 --index-url https://data.pyg.org/whl/torch-1.12.0+cu113

# Install Jupyter
uv pip install jupyter notebook
```

### Running the Notebooks

```bash
# Activate environment
source .venv/bin/activate

# Start Jupyter
jupyter notebook
```

Open any dataset notebook (e.g., `Cadets.ipynb`) and run all cells. Results appear at the end of execution.

---

## File Summary Table

| Category | Count | Description |
|----------|-------|------------|
| Root files | 3 | README, requirements.txt, .DS_Store |
| Notebooks | 8 | Dataset-specific evaluation code |
| Data files | 6 | Ground truth + embeddings |
| Trained weights | ~150 | Models across 7 datasets |

---

## Citation

```bibtex
@proceedings{flash2024,
  title = {FLASH: A Comprehensive Approach to Intrusion Detection via Provenance Graph Representation Learning},
  author = {Rehman, Mati Ur and Ahmadi, Hadi and Hassan, Wajih Ul},
  booktitle = {IEEE Symposium on Security and Privacy (S&P)},
  year = {2024}
}
```