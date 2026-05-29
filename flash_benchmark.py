"""
Benchmark harness for FLASH model variants.

Provides a uniform interface to:
  - Run model variants through a fixed evaluation matrix
  - Log metrics (Precision, Recall, F1, PR-AUC, FPR, latency, memory)
  - Produce a comparison leaderboard with confidence intervals
"""

import os
import json
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Sequence, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger("flash-benchmark")


@dataclass
class BenchResult:
    variant: str
    seed: int
    data_split: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    fpr: float = 0.0
    tpr: float = 0.0
    pr_auc: float = 0.0
    roc_auc: float = 0.0
    train_time_s: float = 0.0
    infer_time_s: float = 0.0
    peak_memory_mb: float = 0.0
    n_train_nodes: int = 0
    n_test_nodes: int = 0
    n_snapshots: int = 0
    extra: Dict = field(default_factory=dict)


@dataclass
class BenchRunConfig:
    """Describes one run in the evaluation matrix."""
    variant: str
    data_split: str
    seed: int
    train_path: str
    test_path: str
    gt_path: Optional[str] = None
    num_snapshots: int = 22
    vector_size: int = 30
    extra_cfg: Dict = field(default_factory=dict)


def compute_metrics(tp, fp, fn, tn):
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, fpr, tpr


def score_auc(ids: Sequence[str], scores: Sequence[float], ground_truth_set):
    """Compute PR-AUC/ROC-AUC for node-level anomaly scores when possible."""
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        return 0.0, 0.0

    y_true = np.array([1 if node_id in ground_truth_set else 0 for node_id in ids], dtype=np.int32)
    y_score = np.array(scores, dtype=np.float32)
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0
    return float(average_precision_score(y_true, y_score)), float(roc_auc_score(y_true, y_score))


def detections_at_threshold(ids: Sequence[str], scores: Sequence[float], threshold: float):
    """Return node IDs with anomaly score at or above a fixed threshold."""
    return {node_id for node_id, score in zip(ids, scores) if score >= threshold}


def optimize_threshold(
    ids: Sequence[str],
    scores: Sequence[float],
    ground_truth_set,
    all_ids_set,
    eval_edges,
    eval_mapp,
) -> Tuple[Optional[float], Dict]:
    """Pick the best F1 threshold on validation data only.

    Returns None when validation labels have no overlap with the validation graph;
    callers should then use an unsupervised fallback threshold.
    """
    gt_overlap = set(ids).intersection(ground_truth_set)
    if not gt_overlap:
        return None, {"reason": "no_validation_gt_overlap", "gt_overlap": 0}

    score_array = np.array(scores, dtype=np.float32)
    candidates = np.unique(score_array)
    if len(candidates) > 512:
        candidates = np.quantile(score_array, np.linspace(0.0, 1.0, 512))
        candidates = np.unique(candidates)

    best = {
        "threshold": float(candidates[0]),
        "precision": 0.0,
        "recall": 0.0,
        "f1": -1.0,
        "fpr": 0.0,
        "tpr": 0.0,
        "gt_overlap": len(gt_overlap),
    }
    for threshold in candidates:
        detected = detections_at_threshold(ids, scores, float(threshold))
        p, r, f1_val, fpr, tpr = two_hop_propagation(
            detected, ground_truth_set, all_ids_set, eval_edges, eval_mapp
        )
        if f1_val > best["f1"]:
            best.update({
                "threshold": float(threshold),
                "precision": p,
                "recall": r,
                "f1": f1_val,
                "fpr": fpr,
                "tpr": tpr,
            })

    return best["threshold"], best


def two_hop_propagation(detected_set, ground_truth_set, all_ids_set, eval_edges, eval_mapp):
    """Replicate the FLASH helper/two-hop logic from the notebook."""
    ground_truth_set = ground_truth_set.intersection(all_ids_set)
    tp = detected_set.intersection(ground_truth_set)
    fp = detected_set - ground_truth_set
    fn = ground_truth_set - detected_set
    tn = all_ids_set - (ground_truth_set | detected_set)

    two_hop_gt = set()
    two_hop_tp = set()
    for edge in zip(eval_edges[0], eval_edges[1]):
        src_mapped = eval_mapp[edge[0]] if edge[0] < len(eval_mapp) else str(edge[0])
        dst_mapped = eval_mapp[edge[1]] if edge[1] < len(eval_mapp) else str(edge[1])
        if src_mapped in ground_truth_set or dst_mapped in ground_truth_set:
            two_hop_gt.add(src_mapped)
            two_hop_gt.add(dst_mapped)
        if src_mapped in tp or dst_mapped in tp:
            two_hop_tp.add(src_mapped)
            two_hop_tp.add(dst_mapped)

    tp_expanded = tp.union(fn.intersection(two_hop_tp))
    fp_after = fp - two_hop_gt
    fn_after = fn - two_hop_tp
    return compute_metrics(len(tp_expanded), len(fp_after), len(fn_after), len(tn))


class BenchmarkSuite:
    """Orchestrates multiple BenchRunConfig runs and aggregates results."""

    def __init__(self, name="FLASH Benchmark"):
        self.name = name
        self.results: List[BenchResult] = []
        self.configs: List[BenchRunConfig] = []

    def add_config(self, cfg: BenchRunConfig):
        self.configs.append(cfg)

    def add_configs(self, cfgs: List[BenchRunConfig]):
        self.configs.extend(cfgs)

    def run(self, train_fn: Callable, infer_fn: Callable,
            train_kwargs: Optional[Dict] = None,
            infer_kwargs: Optional[Dict] = None):
        """Run all configs through train_fn / infer_fn."""
        train_kwargs = train_kwargs or {}
        infer_kwargs = infer_kwargs or {}

        for cfg in self.configs:
            logger.info(f"Running: {cfg.variant} | split={cfg.data_split} | seed={cfg.seed}")
            np.random.seed(cfg.seed)

            result = BenchResult(
                variant=cfg.variant,
                seed=cfg.seed,
                data_split=cfg.data_split,
            )

            # ── Train ──
            t0 = time.time()
            try:
                train_out = train_fn(
                    train_path=cfg.train_path,
                    test_path=cfg.test_path,
                    num_snapshots=cfg.num_snapshots,
                    vector_size=cfg.vector_size,
                    seed=cfg.seed,
                    **train_kwargs,
                )
                train_time = time.time() - t0
                result.train_time_s = train_time
                result.n_train_nodes = train_out.get("n_train_nodes", 0)
                result.n_snapshots = train_out.get("n_snapshots", 0)
            except Exception as e:
                logger.error(f"Training failed for {cfg.variant}: {e}")
                self.results.append(result)
                continue

            # ── Inference ──
            t0 = time.time()
            try:
                infer_out = infer_fn(
                    test_path=cfg.test_path,
                    num_snapshots=cfg.num_snapshots,
                    vector_size=cfg.vector_size,
                    model_prefix=train_out.get("model_prefix", ""),
                    **infer_kwargs,
                )
                infer_time = time.time() - t0
                result.infer_time_s = infer_time
                result.n_test_nodes = infer_out.get("n_test_nodes", 0)

                # Compute metrics
                detected = infer_out.get("detected_ids", set())
                all_ids = infer_out.get("all_ids", set())
                gt = set()
                if cfg.gt_path and os.path.exists(cfg.gt_path):
                    with open(cfg.gt_path) as f:
                        gt = set(json.load(f))

                if gt:
                    gt_overlap = len(gt.intersection(all_ids))
                    recall_ceiling = (gt_overlap / len(gt)) if gt else 0.0
                    result.extra["gt_size"] = len(gt)
                    result.extra["gt_overlap_test_ids"] = gt_overlap
                    result.extra["recall_ceiling"] = recall_ceiling
                    p, r, f1_val, fpr, tpr = two_hop_propagation(
                        detected, gt, all_ids,
                        infer_out.get("edges", [[], []]),
                        infer_out.get("mapp", []),
                    )
                    result.precision = p
                    result.recall = r
                    result.f1 = f1_val
                    result.fpr = fpr
                    result.tpr = tpr

                result.extra["n_detected"] = len(detected)

            except Exception as e:
                logger.error(f"Inference failed for {cfg.variant}: {e}")

            self.results.append(result)

        return self.results

    def leaderboard(self, group_by="variant") -> pd.DataFrame:
        """Aggregate results into a comparison leaderboard."""
        if not self.results:
            return pd.DataFrame()

        records = []
        for r in self.results:
            d = asdict(r)
            d.pop("extra", None)
            records.append(d)

        df = pd.DataFrame(records)
        group_cols = [group_by] if group_by else []
        if group_by:
            agg = df.groupby(group_cols).agg(["mean", "std", "count"])
            return agg
        return df

    def save(self, path: str):
        """Save results to JSON."""
        data = {
            "name": self.name,
            "results": [asdict(r) for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Benchmark results saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load results from JSON."""
        with open(path) as f:
            data = json.load(f)
        suite = cls(name=data["name"])
        suite.results = [BenchResult(**r) for r in data["results"]]
        return suite


# ── Helper: build time-split configs from a single JSONL ─────────────────

def build_time_split_configs(
    jsonl_path: str,
    variant: str,
    gt_path: Optional[str] = None,
    num_windows: int = 4,
    seeds: List[int] = None,
    num_snapshots: int = 2,
) -> List[BenchRunConfig]:
    """Create configs for chronologically-split windows (drift evaluation)."""
    if seeds is None:
        seeds = [42]

    with open(jsonl_path) as f:
        total_lines = sum(1 for _ in f)

    configs = []
    window_size = total_lines // num_windows

    for i in range(num_windows - 1):
        train_start = 0
        train_end = (i + 1) * window_size
        test_start = train_end
        test_end = min(test_start + window_size, total_lines)

        split_name = f"T{i}_trainT0-T{i}_testT{i+1}"
        for seed in seeds:
            configs.append(BenchRunConfig(
                variant=f"{variant}_{split_name}",
                data_split=split_name,
                seed=seed,
                train_path=jsonl_path,
                test_path=jsonl_path,
                gt_path=gt_path,
                num_snapshots=num_snapshots,
                extra_cfg={
                    "train_lines": (train_start, train_end),
                    "test_lines": (test_start, test_end),
                },
            ))
    return configs
