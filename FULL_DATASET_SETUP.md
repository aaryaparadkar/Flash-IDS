# Full Dataset Training — Flash-IDS

Run everything from the repo root on the target machine.

---

## 1. System Prep

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv build-essential git
```

Check for NVIDIA GPU (skipped if no GPU):

```bash
nvidia-smi
```

---

## 2. Python Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install numpy pandas scikit-learn scipy gensim rich tqdm orjson requests
```

If **CUDA GPU** available:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
```

If **CPU only**:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
```

---

## 3. Place Raw CADETS Data

Create the raw data directory:

```bash
mkdir -p ~/datasets/cadets_raw
```

If multiple JSON shards:

```bash
cat ~/datasets/cadets_raw/ta1-cadets-e3-official.json* > ~/datasets/cadets_raw/cadets_full.jsonl
```

If a single raw JSONL file:

```bash
cp ~/datasets/cadets_raw/ta1-cadets-e3-official.json ~/datasets/cadets_raw/cadets_full.jsonl
```

---

## 4. Create Full Fair Split

Uses the entire dataset without downsampling (train 70% / val 15% / test 15%):

```bash
source .venv/bin/activate

python flash_sampling.py \
  --input ~/datasets/cadets_raw/cadets_full.jsonl \
  --out-dir ~/datasets/cadets_fair_full \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --sample-rate 1.0 \
  --seed 42
```

---

## 5. Run Raw-Feature Tabular Benchmark

Fastest first pass — SGD logistic regression on extracted CDM features.

```bash
source .venv/bin/activate

python run_experiments.py \
  --mode tabular \
  --sampled-dir ~/datasets/cadets_fair_full
```

Results written to:

```
results/benchmark_tabular.json
```

---

## 6. Run GNN Benchmark

Slower — trains a graph neural network on the full temporal graph.

```bash
source .venv/bin/activate

CADETS_SAMPLED_DIR=~/datasets/cadets_fair_full \
python run_experiments.py \
  --mode benchmark \
  --embed-mode baseline \
  --validation-ratio 0.2 \
  --fallback-quantile 0.995 \
  --min-validation-gt 20
```

Results written to:

```
results/benchmark_baseline.json
```

---

## 7. Optional — Hugging Face Embeddings

Only run if you have a Hugging Face token available.

```bash
export HF_TOKEN=hf_your_token_here

CADETS_SAMPLED_DIR=~/datasets/cadets_fair_full \
python run_experiments.py \
  --mode benchmark \
  --embed-mode hf \
  --hf-model BAAI/bge-small-en-v1.5 \
  --hf-batch-size 64 \
  --hf-max-length 128
```

---

## Provision Checklist

| Item | Expectation |
|------|-------------|
| Disk | 1 TB SSD — sufficient for full CADETS dataset |
| RAM  | 36 GB — fine for tabular; GNN may be tight |
| GPU  | Strongly recommended for GNN training |
| Split | Use `--sample-rate 1.0` (full) |
| Test | Only trust metrics from `test` sections in results JSON |

If the full GNN run crashes due to memory, reduce batch sizes in `run_train.py` or run the tabular benchmark first.
