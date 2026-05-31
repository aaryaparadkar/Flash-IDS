# Flash-IDS CADETS Flow: End-to-End

## Overview

This document explains the complete execution flow used to run Flash-IDS on CADETS data, including training, benchmarking, and drift analysis.

---

## 1) Inputs and Data Sources

### Primary data
- `cadets_sampled/train.jsonl`
- `cadets_sampled/test.jsonl`

These are CDM JSONL files (one JSON object per line), containing Event records and object/process records (DARPA E3 format, 1% sample of the full Cadets dataset).

### Intermediate TSVs (generated/used)
- `cadets_train.txt`
- `cadets_test.txt`

Schema:
`actorID, actor_type, objectID, object_type(object), action, timestamp`

---

## 2) Training Pipeline (`run_train.py`)

The training pipeline runs in this order:

1. **Validate** input JSONL files (check CDM Event records)
2. **Parse** CDM records into edge TSV (extract UUIDs, node types, timestamps)
3. **Build** node/edge graph representation via `prepare_graph()`
4. **Enrich** rows with optional `exec`/`path` attributes from raw CDM
5. **Train** Word2Vec embeddings (gensim, vector_size=30, window=5, 300 epochs)
6. **Train** GraphSAGE GCN ensemble (2-layer SAGEConv, hidden=32, dropout=0.5, 22 snapshots with confidence-based progressive node masking)
7. **Run** inference + evaluation pass on the test graph (multi-snapshot ensemble voting)

### Artifacts generated
- Word2Vec → `trained_weights/cadets/word2vec_cadets_E3.model`
- 22 GNN snapshots → `trained_weights/cadets/lword2vec_gnn_cadets{0..21}_E3.pth`

---

## 3) Key Fixes Applied

### A) Synthetic mode bug
- A second `USE_SYNTHETIC = True` assignment later in the file was overriding the config-block value, forcing random embeddings instead of real Word2Vec.
- **Fix:** Removed the duplicate assignment.

### B) Device mismatch (CPU vs CUDA)
- Snapshot masking and evaluation had mixed-device indexing issues (`cond` tensor on CUDA, `subg.n_id` on CPU).
- **Fix:** Aligned boolean masks, node index tensors, and assignment tensors to the same device.

### C) Attribute merge was dropping edges
- Enrichment used `inner` join which dropped edge rows that didn't find matching attributes.
- **Fix:** Changed to `left` join, preserving all base edge rows.

### D) First-run Word2Vec loading
- Module-level `Word2Vec.load()` would crash on first run when no model existed yet.
- **Fix:** Graceful fallback when model file doesn't exist (`w2vmodel = None`).

---

## 4) Benchmarking (`run_experiments.py --mode benchmark`)

`run_benchmark()` was originally a smoke-test path using dummy train/infer functions. It has been replaced with real train/infer logic:

- Loads real graph data from TSV
- Uses real embedding providers (`flash_embed.py`):
  - `word2vec` — gensim Word2Vec embeddings
  - `random` — random embeddings (baseline)
  - `token_mean` — static lookup table, bag-of-tokens mean
- Trains real GNN snapshots per variant and seed
- Runs real ensemble inference
- Stores results in `results/benchmark_results.json`

### Fair evaluation rules

The benchmark now avoids two common F1-inflation bugs:

- Anomaly decisions are made from model scores (`1 - confidence` plus entropy), not from `pred == true_label` during inference.
- Thresholds are selected on the chronological validation tail only when enough validation GT nodes are present. Otherwise the benchmark uses a train-score quantile fallback and records `threshold_source` in the result JSON.
- Test recall is computed only against GT nodes present in the evaluated test graph. `recall_ceiling` remains in `extra` as a dataset-coverage diagnostic.
- Two-hop propagation can expand detections around true-positive alerts, but it no longer converts ground-truth neighbors into true positives when the detector did not alert.

To build a split from raw CDM JSONL without biased line sampling:

```bash
python flash_sampling.py \
  --input /path/to/ta1-cadets-e3-official.json \
  --out-dir cadets_sampled_fair \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --sample-rate 0.01 \
  --seed 42
```

Then benchmark that split:

```bash
CADETS_SAMPLED_DIR=cadets_sampled_fair \
python run_experiments.py --mode benchmark --embed-mode baseline
```

### Benchmark matrix
| Variant | Seeds | Snapshots |
|---|---|---|
| word2vec | 42, 43 | 2 each |
| random | 42, 43 | 2 each |
| token_mean | 42, 43 | 2 each |

---

## 5) Drift Analysis (`run_experiments.py --mode drift`)

- Splits `cadets_train.txt` into 4 chronological windows (7,498 rows each)
- Computes PSI, KL-divergence, JS-divergence, and novelty rates per window vs reference (W0)
- Applies `RetrainPolicy` thresholds: PSI > 0.2, novelty > 0.15, F1 drop > 0.05
- Output: `results/drift_report.json`

---

## 6) Why Precision/Recall/F1 Are Currently Zero

The benchmark runs real inference, but metrics show zeros because:
- `data_files/cadets.json` contains **synthetic** ground truth (`proc_1`..`proc_10`)
- Real CADETS runs produce UUID-style node IDs, so there's no overlap
- Without matching GT labels, `two_hop_propagation()` can't compute PR/F1

To fix: replace `data_files/cadets.json` with a JSON list of real CADETS malicious UUIDs.

---

## 7) Output Inventory

### Models
| Artifact | Location |
|---|---|
| Word2Vec | `trained_weights/cadets/word2vec_cadets_E3.model` |
| GNN snapshots (22) | `trained_weights/cadets/lword2vec_gnn_cadets{0..21}_E3.pth` |
| Benchmark snapshots (12) | `trained_weights/cadets/bench_{variant}_s{seed}_snap{idx}.pth` |

### Reports
| Report | Location |
|---|---|
| Benchmark results | `results/benchmark_results.json` |
| Drift report | `results/drift_report.json` |

---

## 8) Recommended Next Steps

1. Add real CADETS ground truth UUIDs to `data_files/cadets.json`
2. Re-run: `python run_experiments.py --mode benchmark`
3. Increase benchmark snapshots from 2 to 22 for production comparison
4. Compare embedding variants using true PR/F1

---

## 9) One-Command Reproduction

```bash
# Full pipeline
python run_train.py

# All experiments (embed parity + drift + benchmark)
python run_experiments.py --mode all
```

Use the project virtual environment interpreter if needed.
