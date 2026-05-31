# Flash-IDS Project Reference

Complete reference for every file, folder, and dependency in the project.

---

## Project Overview

**Flash-IDS** (Fast Learning Attack Scanner via Hashing) — Intrusion detection using provenance graph representation learning on DARPA CADETS and other CDM-format datasets. Combines Word2Vec embeddings with GraphSAGE GNN ensembles for anomaly detection on system-call provenance graphs.

**Reference paper:** IEEE S&P 2024

---

## Directory Map

```
Flash-IDS/
├── cadets_sampled/         ← Sampled CDM data (1% of DARPA E3)
├── data_files/             ← Ground truth labels + precomputed embeddings
├── docs/                   ← Auto-generated documentation
├── drift_windows/          ← Chronological windows for drift analysis
├── results/                ← Benchmark + drift report output
├── trained_weights/        ← All trained model artifacts
│   ├── cadets/
│   ├── fivedirections/
│   ├── theia/
│   ├── trace/
│   ├── optc/
│   ├── streamspot/
│   └── unicorn/
├── benchmarks/             ← Placeholder (empty)
├── *.py                    ← Scripts (see below)
├── *.ipynb                 ← Jupyter notebooks (one per dataset)
├── *.txt                   ← Parsed edge TSVs
└── *.json                  ← Experiment output / config
```

---

## Directory Details

### `cadets_sampled/` — Sampled CADETS CDM Data

**Files:**
| File | Size | Lines | Content |
|---|---|---|---|
| `train.jsonl` | 39.9 MB | 54,236 | Training CDM records (Events + entities) |
| `test.jsonl` | 23.8 MB | 35,956 | Test CDM records |
| `train.jsonl.txt` | — | 29,992 | Parsed train edges (auto-generated) |
| `test.jsonl.txt` | — | 13,584 | Parsed test edges (auto-generated) |
| `sample_stats.json` | — | — | Sampling metadata (record counts, timestamps) |

**CDM JSONL format:** One JSON object per line. Each object has a `datum` key containing a CDM18 record type (e.g., `com.bbn.tc.schema.avro.cdm18.Event`). Event records contain `type` (action), `subject` (actor UUID), `predicateObject` (object UUID), `predicateObject2`, `timestampNanos`, and optional `properties.map.exec`.

**Source:** 1% sample of the full DARPA E3 CADETS dataset (~5M total records → ~54K sampled). Split time-based at timestamp `1522794779711385638`.

**Used by:** `run_train.py`, `run_experiments.py`, `Cadets_script.py`, `Cadets.ipynb`

---

### `data_files/` — Ground Truth & Embeddings

**Files:**
| File | Content | Type |
|---|---|---|
| `cadets.json` | `["proc_1", ..., "proc_10"]` | **Synthetic** GT (doesn't match real UUIDs) |
| `fivedirections.json` | 50+ DARPA E3 UUIDs | Real GT for FiveDirections |
| `theia.json` | ~50 DARPA E3 UUIDs | Real GT for Theia |
| `trace.json` | ~50 DARPA E3 UUIDs | Real GT for Trace |
| `optc.txt` | 658 UUIDs (one per line) | Real GT for OpTC |
| `emb_store.json` | Pre-computed 20-dim path embeddings | Windows system file paths |

**How ground truth works:** GT files contain a JSON list of malicious node UUIDs. During evaluation, detected nodes are intersected with GT to compute TP/FP/FN/TN. Two-hop propagation expands TPs along graph edges for alert correlation.

**Used by:** `run_train.py` (evaluation step), `run_experiments.py` (benchmark metrics), notebooks.

---

### `drift_windows/` — Drift Analysis Slices

**Files:** `window_0.tsv` through `window_3.tsv` (4 files × 7,498 rows each)

Each file is a chronological slice of `cadets_train.txt` split into equal quarters by row count. Used by `flash_drift.py` to compute distribution drift (PSI, KL, JS) between reference (W0) and subsequent windows.

**Format:** Same 6-column tab-separated edge TSV as `cadets_train.txt`.

**Used by:** `flash_drift.py`, `run_experiments.py --mode drift`

---

### `results/` — Experiment Output

**Files:**
| File | Content |
|---|---|
| `benchmark_results.json` | 6-run benchmark (3 variants × 2 seeds). Metrics per run: precision, recall, f1, fpr, tpr, train_time, infer_time, node counts |
| `drift_report.json` | 4-window drift analysis. Per window: PSI scores (per feature), JS divergence, novelty rate |

**Benchmark JSON schema:**
```json
{
  "name": "FLASH Embedding Comparison",
  "results": [
    {
      "variant": "word2vec",
      "seed": 42,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "fpr": 0.0,
      "tpr": 0.0,
      "train_time_s": 0.6,
      "infer_time_s": 0.1,
      "n_train_nodes": 15071,
      "n_test_nodes": 7417,
      "n_snapshots": 2,
      "extra": { "n_detected": 1105 }
    }
  ]
}
```

**Drift JSON schema:**
```json
[
  {
    "window": "W0",
    "n_test": 7498,
    "psi_scores": { "actor_type": 0.0, "object_type": 0.0, "action": 0.0 },
    "novel_token_rate": 0.0
  }
]
```

**Used by:** Analysis/reporting (consumed externally).

---

### `trained_weights/` — Model Artifacts

Each subdirectory corresponds to a dataset. Contains trained Word2Vec models and GNN snapshot ensembles.

#### `trained_weights/cadets/` — 35 files
| Pattern | Count | Example |
|---|---|---|
| `word2vec_cadets_E3.model` | 1 | gensim Word2Vec (dim=30, window=5, vocab=83) |
| `lword2vec_gnn_cadets{N}_E3.pth` | 22 | GNN snapshots 0–21 |
| `bench_{variant}_s{seed}_snap{N}.pth` | 12 | Benchmark variant models (word2vec/random/token_mean × seeds 42/43 × 2 snaps) |

**Training:** Word2Vec trained for 300 epochs on node feature sequences. GNN is a 2-layer GraphSAGE (SAGEConv) with hidden=32, dropout=0.5, trained with confidence-based progressive node masking. Each snapshot removes nodes that the ensemble already classifies correctly with high confidence.

#### `trained_weights/fivedirections/` — 14 files
Word2Vec + 13 GNN snapshots for the FiveDirections dataset.

#### `trained_weights/theia/` — 21 files
Word2Vec + 20 GNN snapshots for the Theia dataset.

#### `trained_weights/trace/` — 21 files
Word2Vec + 20 GNN snapshots for the Trace dataset.

#### `trained_weights/optc/` — 2 files
- `gnn_temp.pth`: Intermediate GNN embeddings
- `xgb.pkl`: Trained XGBoost classifier (OpTC hybrid GNN+XGBoost approach)

#### `trained_weights/streamspot/` — 2 files
- `streamspot.model`: Word2Vec for Streamspot
- `lstreamspot.pth`: Graph-level classifier

#### `trained_weights/unicorn/` — 22 files
- `unicorn.model`: Word2Vec
- `unicorn.pth`: Base GNN
- `unicorn{0..19}.pth`: 20 snapshots

---

## Script Reference

### `run_train.py` (1,023 lines) — **Primary Training Pipeline**

**Imports:** torch, torch-geometric, pandas, numpy, sklearn, gensim, orjson, rich

**Key classes:**
| Class | Description |
|---|---|
| `GCN` | 2-layer GraphSAGE (SAGEConv → ReLU → Dropout → SAGEConv → Softmax). Configurable hidden=32, dropout=0.5 |
| `PositionalEncoder` | Sinusoidal positional encoding for sequence order |
| `EpochSaver` | gensim callback: saves Word2Vec model after each epoch |
| `EpochLogger` | gensim callback: logs epoch progress |

**Entry function:** `run_training_pipeline()` — validates JSONL, samples/splits, parses CDM→TSV, enriches attributes, builds graph, trains Word2Vec, trains 22 GNN snapshots, evaluates on test set.

**Execution modes:**
- CLI: `python run_train.py` (guarded by `if __name__ == '__main__'`)
- Import: `import run_train as P` and call `P.run_training_pipeline()`

**Config block** (lines 55–96): Edit these knobs before running:
- `Train`, `USE_SYNTHETIC` — mode flags
- `TRAIN_JSONL_PATHS`, `TEST_JSONL_PATHS` — CDM input files
- `NUM_SNAPSHOTS`, `W2V_EPOCHS`, `GNN_HIDDEN`, `GNN_LR`, etc. — hyperparameters
- `LABEL_MAP` — 6 CDM node types → class indices

---

### `run_experiments.py` (339 lines) — **Experiment Orchestrator**

**Modes:**
| CLI | Function | What it does |
|---|---|---|
| `--mode embed_parity` | `run_embedding_parity()` | Tests all embedding providers produce correct vector shapes |
| `--mode drift` | `run_drift_analysis()` | Chronological drift + retrain policy on cadets_train.txt |
| `--mode benchmark` | `run_benchmark()` | Compares word2vec/random/token_mean variants with real GNN training |
| `--mode all` | All three | Full experiment suite |

**Imports:** `flash_embed`, `flash_benchmark`, `flash_drift`, `run_train` (as P), torch, torch-geometric, pandas, numpy

**Benchmark flow:** For each (variant, seed) config: build graph from TSV → embed nodes with provider → train GNN snapshots → run ensemble inference → compute metrics vs ground truth.

---

### `flash_benchmark.py` (262 lines) — **Benchmark Harness**

**Classes:**
| Class | Purpose |
|---|---|
| `BenchResult` (dataclass) | Single run: precision, recall, f1, fpr, tpr, train_time, infer_time, node counts |
| `BenchRunConfig` (dataclass) | Configuration: variant, seed, data_split, paths, hyperparams |
| `BenchmarkSuite` | Orchestrates configs through train/infer callbacks, produces leaderboard DataFrame |

**Key functions:**
- `compute_metrics(tp, fp, fn, tn)` → precision, recall, f1, fpr, tpr
- `two_hop_propagation()` — replicates notebook's alert expansion logic
- `build_time_split_configs()` — creates chronologically-split configs for drift evaluation

**Dependencies:** pandas, numpy, json, dataclasses

---

### `flash_drift.py` (234 lines) — **Drift Detection**

**Classes:**
| Class | Purpose |
|---|---|
| `DriftReport` (dataclass) | Per-window drift: PSI scores, KL/JS divergence, novelty rate, novelty by type |
| `RetrainPolicy` | Configurable thresholds (PSI>0.2, novelty>0.15, F1 drop>0.05) for retrain triggering |

**Key functions:**
| Function | Description |
|---|---|
| `load_edge_tsv()` | Load 6-col TSV into DataFrame |
| `extract_feature_profiles()` | Build categorical distribution profiles for PSI computation |
| `psi()` | Population Stability Index |
| `kl_divergence()` | Kullback-Leibler divergence |
| `js_divergence()` | Jensen-Shannon divergence |
| `compute_novelty_rate()` | Ratio of tokens in window not seen in reference |
| `build_time_windows()` | Split TSV into N chronological chunks |
| `analyze_drift()` | Full drift analysis across windows |
| `should_retrain()` | Apply RetrainPolicy thresholds to DriftReport |

**Dependencies:** pandas, numpy, scipy.stats, dataclasses

---

### `flash_embed.py` (196 lines) — **Embedding Providers**

**Providers (all implement `embed(document) -> np.ndarray`):**
| Provider | Description |
|---|---|
| `Word2VecProvider` | Loads gensim model; averages word vectors + positional encoding |
| `RandomProvider` | Random Gaussian vector (for baselines/smoke tests) |
| `TokenMeanProvider` | Static lookup table; bag-of-tokens mean (needs `fit()`) |
| `LLMEmbedProvider` | Sentence-transformer backend (optional; requires `sentence-transformers` package) |

**Factory:** `get_provider(name, vector_size=30, **kwargs)` returns the appropriate provider.

**Dependencies:** numpy, gensim (for Word2Vec), sentence-transformers (optional for LLM)

---

### Other Scripts

| Script | Lines | Purpose |
|---|---|---|
| `Cadets_script.py` | 1,289 | Full Cadets pipeline (nbconvert-friendly; no `__name__` guard). Same logic as `run_train.py` |
| `run_headless.py` | 52 | Headless runner for `Cadets_script.py` (strips IPython magics) |
| `run_training.py` | 51 | Original nbconvert-based runner: `jupyter nbconvert --execute Cadets.ipynb` |
| `run_training_clean.py` | 52 | Clean wrapper: removes stale TSVs, then imports `Cadets_script.py` via `importlib` |

---

## Notebooks

| Notebook | Dataset | Purpose |
|---|---|---|
| `Cadets.ipynb` | DARPA CADETS | Main pipeline: training, evaluation, ground-truth comparison |
| `Fivedirections.ipynb` | DARPA FiveDirections | Same pipeline for FiveDirections E3 |
| `OpTC.ipynb` | DARPA OpTC | OpTC pipeline with XGBoost hybrid |
| `streamspot.ipynb` | Streamspot | Streamspot pipeline |
| `Theia.ipynb` | DARPA Theia | Theia pipeline |
| `Trace.ipynb` | DARPA Trace | Trace pipeline |
| `unicorn.ipynb` | Unicorn | Unicorn pipeline |
| `utils.ipynb` | — | Shared utility functions used across notebooks |

All notebooks replicate the same core pipeline: CDM JSONL → edge TSV → graph → Word2Vec → GNN ensemble → evaluation.

---

## Edge TSV Format

Used by: `cadets_train.txt`, `cadets_test.txt`, `train.jsonl.txt`, `test.jsonl.txt`, all `drift_windows/*.tsv`

```
actorID\tactor_type\tobjectID\tobject_type\taction\ttimestamp
72FB0406...\tSUBJECT_PROCESS\t8A44F292...\tFILE_OBJECT_FILE\tEVENT_READ\t1522794779711385638
```

- **actorID / objectID:** UUID strings (DARPA CDM18 format)
- **actor_type / object_type:** CDM node types (SUBJECT_PROCESS, FILE_OBJECT_FILE, etc.)
- **action:** CDM event types (EVENT_READ, EVENT_WRITE, EVENT_EXECUTE, etc.)
- **timestamp:** Nanosecond epoch timestamps

---

## Dependencies

| Package | Version | Used For |
|---|---|---|
| `torch` | 1.12.0+cu113 | GNN training, tensors, GPU support |
| `torch-geometric` | 2.1.0 | GraphSAGE, NeighborLoader, Data objects |
| `pandas` | 1.3.5 | TSV/DataFrame operations |
| `numpy` | 1.23.1 | Numerical arrays, random seeds |
| `scikit-learn` | 1.1.1 | LabelEncoder, t-SNE, class weights |
| `gensim` | 4.3.0 | Word2Vec training and inference |
| `orjson` | (latest) | Fast CDM JSONL parsing |
| `rich` | (latest) | Console output, progress bars, logging |
| `xgboost` | 0.90 | OpTC hybrid classifier |
| `networkx` | 3.0 | Graph utilities |
| `matplotlib` | (latest) | t-SNE visualizations |
| `sentence-transformers` | (optional) | LLM embedding provider |

---

## System Architecture Diagram

```
CDM JSONL (train.jsonl / test.jsonl)
    │
    ▼
┌─────────────────────────────┐
│   Parsing Layer             │  process_data() → UUID→type map
│   (process_data +           │  process_edges() → extract edges from Events
│    process_edges)           │  Output: 6-col tab-separated edge TSV
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Graph Construction        │  prepare_graph():
│   (prepare_graph)           │  • Features: [exec, action, path]
│                             │  • Labels: 6 CDM node types
│                             │  • Edges: directed (actor → object)
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Embedding                 │  Word2Vec (gensim) + PositionalEncoder
│   (Word2Vec + PosEnc)       │  window=5, dim=30, epochs=300
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   GNN Ensemble Training     │  GCN (2-layer SAGEConv)
│   (22 snapshots)            │  Confidence-based progressive masking
│                             │  22 epochs, one snapshot per epoch
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Inference                 │  Multi-snapshot ensemble voting
│   (ensemble voting)         │  Majority vote OR conf ≥ 0.9
│                             │  Two-hop alert propagation
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Evaluation                │  ground_truth.json → TP/FP/FN/TN
│   (helper)                  │  → Precision, Recall, F1, FPR, TPR
└─────────────────────────────┘
```

### Experiment Pipeline

```
run_experiments.py --mode all
    ├── embed_parity    → Verify provider shapes (random ✓, token_mean ✓, word2vec ✓)
    ├── drift           → Chronological PSI/KL/JS on cadets_train.txt → drift_report.json
    └── benchmark       → For each (variant, seed): build graph → embed → train GNN → infer → save
```

---

## Entry Points Quick Reference

| Command | What it does |
|---|---|
| `python run_train.py` | Full Cadets training pipeline |
| `python run_experiments.py --mode all` | All experiments (parity + drift + benchmark) |
| `python run_experiments.py --mode benchmark` | Embedding comparison benchmark only |
| `python run_experiments.py --mode drift` | Drift analysis only |
| `python run_experiments.py --mode embed_parity` | Embedding parity check only |
