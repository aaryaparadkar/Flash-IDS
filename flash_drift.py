"""
Drift detection and temporal resilience analysis for FLASH models.

Provides:
  - Chronological window splitting
  - Feature distribution drift metrics (PSI, KL, JS divergence)
  - Performance decay tracking over time
  - Retrain-trigger policy helpers
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict, field

logger = logging.getLogger("flash-drift")


@dataclass
class DriftReport:
    """Container for drift analysis results."""
    window_name: str
    psi_scores: Dict[str, float] = field(default_factory=dict)
    kl_divergence: Dict[str, float] = field(default_factory=dict)
    js_divergence: Dict[str, float] = field(default_factory=dict)
    novel_token_rate: float = 0.0
    novelty_by_type: Dict[str, float] = field(default_factory=dict)
    f1_score: float = 0.0
    fpr: float = 0.0
    recall: float = 0.0
    n_test: int = 0


# ── Feature extraction from edge TSV ─────────────────────────────────────

def load_edge_tsv(path: str) -> pd.DataFrame:
    """Load FLASH edge TSV into DataFrame."""
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["actorID", "actor_type", "objectID",
                            "object_type", "action", "timestamp"])
    df = df.dropna()
    df["timestamp"] = df["timestamp"].astype(str)
    return df


def extract_feature_profiles(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Build categorical distribution profiles for drift measurement."""
    profiles = {}
    for col in ["actor_type", "object_type", "action"]:
        counts = df[col].value_counts(normalize=True)
        profiles[col] = counts
    # Token-level: unique exec/path tokens
    profiles["_n_unique_actors"] = np.array([df["actorID"].nunique()])
    profiles["_n_unique_objects"] = np.array([df["objectID"].nunique()])
    return profiles


# ── Drift Metrics ────────────────────────────────────────────────────────

def psi(expected: pd.Series, actual: pd.Series, epsilon=1e-8) -> float:
    """Population Stability Index between two categorical distributions."""
    all_cats = set(list(expected.index) + list(actual.index))
    p = np.array([expected.get(c, epsilon) for c in all_cats])
    q = np.array([actual.get(c, epsilon) for c in all_cats])
    p = p / p.sum()
    q = q / q.sum()
    return np.sum((p - q) * np.log(p / q))


def kl_divergence(p, q, epsilon=1e-8):
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return np.sum(p * np.log(p / q))


def js_divergence(p, q, epsilon=1e-8):
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def compute_feature_psi(ref_profile: Dict, window_profile: Dict) -> Dict[str, float]:
    """Compute PSI for each categorical feature."""
    scores = {}
    for key in ref_profile:
        if key.startswith("_"):
            continue
        ref_series = ref_profile[key]
        win_series = window_profile[key]
        all_cats = set(list(ref_series.index) + list(win_series.index))
        p = np.array([ref_series.get(c, 1e-8) for c in all_cats])
        q = np.array([win_series.get(c, 1e-8) for c in all_cats])
        p = p / p.sum()
        q = q / q.sum()
        scores[key] = np.sum((p - q) * np.log(p / q))
    return scores


# ── Novelty detection ────────────────────────────────────────────────────

def compute_novelty_rate(
    ref_df: pd.DataFrame,
    window_df: pd.DataFrame,
) -> Tuple[float, Dict[str, float]]:
    """Fraction of tokens/patterns in window that never appeared in ref."""
    ref_tokens = set()
    for col in ["actorID", "objectID", "action"]:
        ref_tokens.update(ref_df[col].unique())

    total = 0
    novel = 0
    novelty_by_type = {}
    for col in ["actorID", "objectID", "action"]:
        vals = set(window_df[col].unique())
        n_total = len(vals)
        n_novel = len(vals - ref_tokens)
        novelty_by_type[col] = n_novel / n_total if n_total > 0 else 0.0
        total += n_total
        novel += n_novel

    overall_rate = novel / total if total > 0 else 0.0
    return overall_rate, novelty_by_type


# ── Chronological split builder ──────────────────────────────────────────

def build_time_windows(
    tsv_path: str,
    num_windows: int = 4,
    output_dir: str = "drift_windows",
) -> List[Dict]:
    """Split TSV into chronological windows and return window metadata."""
    os.makedirs(output_dir, exist_ok=True)
    df = load_edge_tsv(tsv_path)
    df = df.sort_values(by="timestamp")
    total = len(df)
    window_size = total // num_windows

    windows = []
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size if i < num_windows - 1 else total
        wdf = df.iloc[start:end]
        path = os.path.join(output_dir, f"window_{i}.tsv")
        wdf.to_csv(path, sep="\t", index=False, header=False)
        windows.append({
            "idx": i,
            "path": path,
            "n_rows": len(wdf),
            "time_range": (str(wdf["timestamp"].iloc[0]),
                           str(wdf["timestamp"].iloc[-1])),
        })
        logger.info(f"Window {i}: {len(wdf)} rows -> {path}")
    return windows


# ── Full drift analysis on one TSV ───────────────────────────────────────

def analyze_drift(
    tsv_path: str,
    num_windows: int = 4,
    reference_window_idx: int = 0,
) -> List[DriftReport]:
    """Run full drift analysis across chronological windows of a TSV."""
    windows = build_time_windows(tsv_path, num_windows)
    ref_df = load_edge_tsv(windows[reference_window_idx]["path"])
    ref_profile = extract_feature_profiles(ref_df)

    reports = []
    for w in windows:
        wdf = load_edge_tsv(w["path"])
        win_profile = extract_feature_profiles(wdf)

        psi_scores = compute_feature_psi(ref_profile, win_profile)
        novel_rate, novelty_by_type = compute_novelty_rate(ref_df, wdf)

        report = DriftReport(
            window_name=f"W{w['idx']}",
            psi_scores=psi_scores,
            novel_token_rate=novel_rate,
            novelty_by_type=novelty_by_type,
            n_test=w["n_rows"],
        )

        # Per-feature JS divergence
        for key in ref_profile:
            if key.startswith("_"):
                continue
            ref_vals = ref_profile[key]
            win_vals = win_profile[key]
            all_cats = set(list(ref_vals.index) + list(win_vals.index))
            p = np.array([ref_vals.get(c, 1e-8) for c in all_cats])
            q = np.array([win_vals.get(c, 1e-8) for c in all_cats])
            p = p / p.sum()
            q = q / q.sum()
            report.js_divergence[key] = float(js_divergence(p, q))
            report.kl_divergence[key] = float(kl_divergence(p, q))

        reports.append(report)

    return reports


# ── Retrain trigger policy ───────────────────────────────────────────────

@dataclass
class RetrainPolicy:
    psi_threshold: float = 0.2
    novel_token_threshold: float = 0.15
    f1_drop_threshold: float = 0.05
    min_window_size: int = 1000

    def should_retrain(self, report: DriftReport,
                       baseline_f1: float) -> Tuple[bool, str]:
        reasons = []
        for feature, psi_val in report.psi_scores.items():
            if psi_val > self.psi_threshold:
                reasons.append(f"PSI({feature})={psi_val:.3f}>{self.psi_threshold}")
        if report.novel_token_rate > self.novel_token_threshold:
            reasons.append(f"novelty={report.novel_token_rate:.3f}>{self.novel_token_threshold}")
        if baseline_f1 > 0 and report.f1_score > 0:
            drop = baseline_f1 - report.f1_score
            if drop > self.f1_drop_threshold:
                reasons.append(f"F1_drop={drop:.3f}>{self.f1_drop_threshold}")
        if report.n_test < self.min_window_size:
            reasons.append(f"window_too_small={report.n_test}<{self.min_window_size}")

        should = len(reasons) > 0
        return should, "; ".join(reasons) if reasons else "ok"
