#!/usr/bin/env python3
"""Evaluate F1 over CADETS test timeline windows with window-specific GT.

This fixes the low-F1 issue caused by splitting cadets_train.txt, which has very
little malicious ground-truth coverage. The model trains on cadets_train.txt and
is evaluated on chronological windows from cadets_test.txt, where the CADETS
ground truth is actually present.

By default, windows are equal-duration timestamp windows. That is the most
literal "timeline" split: each point represents a real time period rather than
the same number of log rows.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_train as P  # noqa: E402
from flash_benchmark import two_hop_propagation  # noqa: E402
from flash_drift import compute_feature_psi, compute_novelty_rate, extract_feature_profiles  # noqa: E402
from flash_embed import get_provider  # noqa: E402


WINDOWS = [
    ("early", "Early sample", "W1"),
    ("middle", "Middle sample", "W2"),
    ("late", "Late sample", "W3"),
]


def load_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["actorID", "actor_type", "objectID", "object_type", "action", "timestamp"],
    )
    df = df.dropna()
    df.sort_values("timestamp", inplace=True)
    return df


def write_window_metadata(out_dir: Path, windows: list[pd.DataFrame], mode: str) -> None:
    metadata = {
        "window_mode": mode,
        "windows": [],
    }
    for (window_name, timeline, _), df in zip(WINDOWS, windows):
        metadata["windows"].append(
            {
                "window": window_name,
                "timeline": timeline,
                "rows": int(len(df)),
                "start_timestamp_ns": int(df["timestamp"].min()) if len(df) else None,
                "end_timestamp_ns": int(df["timestamp"].max()) if len(df) else None,
            }
        )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def write_test_windows(test_df: pd.DataFrame, out_dir: Path, mode: str) -> list[pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    test_df = test_df.reset_index(drop=True).copy()

    if mode == "rows":
        windows = [df.copy() for df in np.array_split(test_df, len(WINDOWS))]
    elif mode == "time":
        start = int(test_df["timestamp"].min())
        end = int(test_df["timestamp"].max())
        step = (end - start) / len(WINDOWS)
        windows = []
        for idx in range(len(WINDOWS)):
            lower = start + idx * step
            upper = start + (idx + 1) * step
            if idx == len(WINDOWS) - 1:
                mask = (test_df["timestamp"] >= lower) & (test_df["timestamp"] <= end)
            else:
                mask = (test_df["timestamp"] >= lower) & (test_df["timestamp"] < upper)
            windows.append(test_df.loc[mask].copy())
    else:
        raise ValueError(f"Unknown window mode: {mode}")

    for (window_name, _, _), df in zip(WINDOWS, windows):
        df.to_csv(out_dir / f"{window_name}.tsv", sep="\t", header=False, index=False)
    write_window_metadata(out_dir, windows, mode)
    return [df.copy() for df in windows]


def prepare_df_for_graph(df: pd.DataFrame) -> pd.DataFrame:
    graph_df = df.rename(columns={"object_type": "object"}).copy()
    P._ensure_exec_path_columns(graph_df)
    return graph_df


def build_graph(df: pd.DataFrame, embed_provider) -> tuple[Data, list[str], list[list[int]]]:
    phrases, labels, edges, mapp = P.prepare_graph(prepare_df_for_graph(df))
    nodes = np.array([embed_provider.infer(x) for x in phrases], dtype=np.float32)
    graph = Data(
        x=torch.tensor(nodes, dtype=torch.float).to(P.device),
        y=torch.tensor(labels, dtype=torch.long).to(P.device),
        edge_index=torch.tensor(edges, dtype=torch.long).to(P.device),
    )
    graph.n_id = torch.arange(graph.num_nodes, device=P.device)
    return graph, mapp, edges


def class_weights(labels) -> torch.Tensor:
    labels = np.asarray(labels)
    n_classes = len(P.LABEL_MAP)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float).to(P.device)


def train_snapshots(graph: Data, seed: int, num_snapshots: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = P.GCN(P.W2V_VECTOR_SIZE, len(P.LABEL_MAP)).to(P.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=P.GNN_LR, weight_decay=P.GNN_WEIGHT_DECAY)
    criterion = CrossEntropyLoss(weight=class_weights(graph.y.cpu().numpy()))
    mask = torch.ones(graph.num_nodes, dtype=torch.bool, device=P.device)
    snapshots = []

    for snap_idx in range(num_snapshots):
        if not mask.any():
            break

        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = criterion(out[mask], graph.y[mask])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            out = model(graph.x, graph.edge_index)
            sorted_vals, indices = out.sort(dim=1, descending=True)
            conf = (sorted_vals[:, 0] - sorted_vals[:, 1]) / sorted_vals[:, 0].clamp(min=1e-8)
            conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
            pred = indices[:, 0]
            learned = ((pred == graph.y) | (conf >= 0.9)) & mask
            mask[learned] = False

        snapshots.append(copy.deepcopy(model.state_dict()))
        print(f"snapshot {snap_idx}/{num_snapshots - 1}: loss={loss.item():.4f}, remaining={int(mask.sum().item())}")

    return model, snapshots


def infer_detected(model, snapshots, graph: Data, mapp: list[str]):
    flag = torch.ones(graph.num_nodes, dtype=torch.bool, device=P.device)
    all_conf = []
    with torch.no_grad():
        for state in snapshots:
            model.load_state_dict(state)
            model.eval()
            out = model(graph.x, graph.edge_index)
            sorted_vals, indices = out.sort(dim=1, descending=True)
            conf = (sorted_vals[:, 0] - sorted_vals[:, 1]) / sorted_vals[:, 0].clamp(min=1e-8)
            conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
            all_conf.append(conf.detach().cpu().numpy())
            pred = indices[:, 0]
            flag[pred == graph.y] = False

    detected_idx = torch.where(flag)[0].tolist()
    detected = set(mapp[i] for i in detected_idx if i < len(mapp))
    conf_arr = np.concatenate(all_conf) if all_conf else np.array([], dtype=np.float32)
    return detected, {
        "anomaly_rate": float(len(detected_idx) / graph.num_nodes) if graph.num_nodes else 0.0,
        "mean_confidence": float(conf_arr.mean()) if len(conf_arr) else 0.0,
        "low_confidence_rate": float((conf_arr < 0.25).mean()) if len(conf_arr) else 0.0,
    }


def drift_summary(train_df: pd.DataFrame, test_df: pd.DataFrame):
    ref_profile = extract_feature_profiles(train_df)
    test_profile = extract_feature_profiles(test_df)
    psi_scores = compute_feature_psi(ref_profile, test_profile)
    novel_rate, novelty_by_type = compute_novelty_rate(train_df, test_df)
    return psi_scores, novel_rate, novelty_by_type


def should_retrain(psi_scores: dict[str, float], novelty_by_type: dict[str, float]):
    reasons = []
    if psi_scores.get("action", 0.0) > 0.2:
        reasons.append(f"PSI(action)={psi_scores.get('action', 0.0):.3f}>0.2")
    if psi_scores.get("object_type", 0.0) > 0.2:
        reasons.append(f"PSI(object_type)={psi_scores.get('object_type', 0.0):.3f}>0.2")
    if novelty_by_type.get("action", 0.0) > 0.15:
        reasons.append(f"action_novelty={novelty_by_type.get('action', 0.0):.3f}>0.15")
    return bool(reasons), "; ".join(reasons) if reasons else "ok"


def window_ground_truth(global_gt: set[str], mapp: list[str]) -> set[str]:
    return global_gt & set(mapp)


def evaluate_row(
    strategy: str,
    train_label: str,
    window_name: str,
    timeline: str,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    model,
    snapshots,
    embed_provider,
    global_gt: set[str],
    retrain_triggered: bool,
    retrain_reason: str,
) -> dict[str, str]:
    psi_scores, novel_rate, novelty_by_type = drift_summary(train_df, test_df)
    test_graph, test_mapp, test_edges = build_graph(test_df, embed_provider)
    detected, online_stats = infer_detected(model, snapshots, test_graph, test_mapp)
    all_ids = set(test_mapp)
    gt = window_ground_truth(global_gt, test_mapp)

    precision = recall = f1 = fpr = tpr = 0.0
    if gt:
        precision, recall, f1, fpr, tpr = two_hop_propagation(detected, gt, all_ids, test_edges, test_mapp)

    return {
        "timeline": timeline,
        "window": window_name,
        "strategy": strategy,
        "train_windows": train_label,
        "psi_action": f"{psi_scores.get('action', 0.0):.12f}",
        "novel_token_rate": f"{novel_rate:.12f}",
        "precision": f"{precision:.12f}",
        "recall": f"{recall:.12f}",
        "f1": f"{f1:.12f}",
        "fpr": f"{fpr:.12f}",
        "tpr": f"{tpr:.12f}",
        "anomaly_rate": f"{online_stats['anomaly_rate']:.12f}",
        "mean_confidence": f"{online_stats['mean_confidence']:.12f}",
        "low_confidence_rate": f"{online_stats['low_confidence_rate']:.12f}",
        "gt_present": str(len(gt)),
        "gt_total": str(len(global_gt)),
        "retrain_triggered": str(bool(retrain_triggered)),
        "retrain_reason": retrain_reason,
    }


def evaluate_model(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    model,
    snapshots,
    embed_provider,
    global_gt: set[str],
) -> dict[str, float]:
    psi_scores, novel_rate, novelty_by_type = drift_summary(train_df, test_df)
    test_graph, test_mapp, test_edges = build_graph(test_df, embed_provider)
    detected, online_stats = infer_detected(model, snapshots, test_graph, test_mapp)
    gt = window_ground_truth(global_gt, test_mapp)

    precision = recall = f1 = fpr = tpr = 0.0
    if gt:
        precision, recall, f1, fpr, tpr = two_hop_propagation(
            detected,
            gt,
            set(test_mapp),
            test_edges,
            test_mapp,
        )

    return {
        "psi_action": float(psi_scores.get("action", 0.0)),
        "novel_token_rate": float(novel_rate),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "anomaly_rate": float(online_stats["anomaly_rate"]),
        "mean_confidence": float(online_stats["mean_confidence"]),
        "low_confidence_rate": float(online_stats["low_confidence_rate"]),
        "gt_present": len(gt),
        "gt_total": len(global_gt),
    }


def row_from_metrics(
    strategy: str,
    train_label: str,
    window_name: str,
    timeline: str,
    metrics: dict[str, float],
    retrain_triggered: bool,
    retrain_reason: str,
    policy_conditions: str = "",
    current_validation_f1: str = "",
    candidate_validation_f1: str = "",
) -> dict[str, str]:
    return {
        "timeline": timeline,
        "window": window_name,
        "strategy": strategy,
        "train_windows": train_label,
        "psi_action": f"{metrics['psi_action']:.12f}",
        "novel_token_rate": f"{metrics['novel_token_rate']:.12f}",
        "precision": f"{metrics['precision']:.12f}",
        "recall": f"{metrics['recall']:.12f}",
        "f1": f"{metrics['f1']:.12f}",
        "fpr": f"{metrics['fpr']:.12f}",
        "tpr": f"{metrics['tpr']:.12f}",
        "anomaly_rate": f"{metrics['anomaly_rate']:.12f}",
        "mean_confidence": f"{metrics['mean_confidence']:.12f}",
        "low_confidence_rate": f"{metrics['low_confidence_rate']:.12f}",
        "gt_present": str(metrics["gt_present"]),
        "gt_total": str(metrics["gt_total"]),
        "retrain_triggered": str(bool(retrain_triggered)),
        "retrain_reason": retrain_reason,
        "policy_conditions": policy_conditions,
        "current_validation_f1": current_validation_f1,
        "candidate_validation_f1": candidate_validation_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tsv", default="cadets_train.txt")
    parser.add_argument("--test-tsv", default="cadets_test.txt")
    parser.add_argument("--gt", default="data_files/cadets.json")
    parser.add_argument("--window-dir", default="test_timeline_windows")
    parser.add_argument(
        "--window-mode",
        choices=["time", "rows"],
        default="time",
        help="Split cadets_test.txt by equal timestamp duration or equal row count.",
    )
    parser.add_argument("--output", default="results/cadets_test_timeline_f1_with_drift_policy.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-snapshots", type=int, default=6)
    args = parser.parse_args()

    train_df = load_tsv(ROOT / args.train_tsv)
    test_df = load_tsv(ROOT / args.test_tsv)
    test_windows = write_test_windows(test_df, ROOT / args.window_dir, args.window_mode)
    global_gt = set(json.loads((ROOT / args.gt).read_text(encoding="utf-8")))

    embed_provider = get_provider("word2vec", model_path=P.W2V_MODEL_PATH, vector_size=P.W2V_VECTOR_SIZE)
    print(f"Training static baseline on {args.train_tsv} ({len(train_df):,} rows)")
    train_graph, _, _ = build_graph(train_df, embed_provider)
    static_model, static_snapshots = train_snapshots(train_graph, args.seed, args.num_snapshots)

    rows = []
    for (window_name, timeline, _), window_df in zip(WINDOWS, test_windows):
        metrics = evaluate_model(
            window_df,
            train_df,
            static_model,
            static_snapshots,
            embed_provider,
            global_gt,
        )
        row = row_from_metrics(
            "Static baseline",
            "cadets_train",
            window_name,
            timeline,
            metrics,
            False,
            "n/a",
        )
        rows.append(row)
        print(f"static -> {window_name}: F1={float(row['f1']):.3f}, GT={row['gt_present']}/{row['gt_total']}")

    current_train_df = train_df.copy()
    current_model = static_model
    current_snapshots = static_snapshots
    current_train_label = "cadets_train"
    for idx, ((window_name, timeline, _), window_df) in enumerate(zip(WINDOWS, test_windows)):
        psi_scores, _, novelty_by_type = drift_summary(current_train_df, window_df)
        trigger, reason = should_retrain(psi_scores, novelty_by_type)
        metrics = evaluate_model(
            window_df,
            current_train_df,
            current_model,
            current_snapshots,
            embed_provider,
            global_gt,
        )
        row = row_from_metrics(
            "Naive drift policy",
            current_train_label,
            window_name,
            timeline,
            metrics,
            trigger,
            reason,
        )
        rows.append(row)
        print(f"naive policy -> {window_name}: F1={float(row['f1']):.3f}, retrain={trigger}, {reason}")

        if trigger and idx < len(test_windows) - 1:
            current_train_df = pd.concat([train_df] + test_windows[: idx + 1], ignore_index=True)
            current_train_label = "cadets_train+" + "+".join(w[0] for w in WINDOWS[: idx + 1])
            print(f"Retraining naive policy model on {current_train_label} ({len(current_train_df):,} rows)")
            graph, _, _ = build_graph(current_train_df, embed_provider)
            current_model, current_snapshots = train_snapshots(graph, args.seed, args.num_snapshots)

    gated_train_df = train_df.copy()
    gated_model = static_model
    gated_snapshots = static_snapshots
    gated_train_label = "cadets_train"
    for idx, ((window_name, timeline, _), window_df) in enumerate(zip(WINDOWS, test_windows)):
        psi_scores, _, novelty_by_type = drift_summary(gated_train_df, window_df)
        trigger, reason = should_retrain(psi_scores, novelty_by_type)
        current_metrics = evaluate_model(
            window_df,
            gated_train_df,
            gated_model,
            gated_snapshots,
            embed_provider,
            global_gt,
        )

        accepted = False
        policy_conditions = f"drift={trigger}"
        current_validation_f1 = ""
        candidate_validation_f1 = ""
        retrain_reason = reason if trigger else "no drift trigger"

        if trigger and idx < len(test_windows) - 1:
            candidate_train_df = pd.concat([train_df] + test_windows[: idx + 1], ignore_index=True)
            candidate_train_label = "cadets_train+" + "+".join(w[0] for w in WINDOWS[: idx + 1])
            print(f"Training gated candidate on {candidate_train_label} ({len(candidate_train_df):,} rows)")
            candidate_graph, _, _ = build_graph(candidate_train_df, embed_provider)
            candidate_model, candidate_snapshots = train_snapshots(candidate_graph, args.seed, args.num_snapshots)
            candidate_metrics = evaluate_model(
                window_df,
                candidate_train_df,
                candidate_model,
                candidate_snapshots,
                embed_provider,
                global_gt,
            )
            current_validation_f1 = f"{current_metrics['f1']:.12f}"
            candidate_validation_f1 = f"{candidate_metrics['f1']:.12f}"
            policy_conditions = (
                f"drift=True; current_val_f1={current_metrics['f1']:.3f}; "
                f"candidate_val_f1={candidate_metrics['f1']:.3f}; "
                f"current_val_fpr={current_metrics['fpr']:.3f}; "
                f"candidate_val_fpr={candidate_metrics['fpr']:.3f}"
            )
            accepted = (
                candidate_metrics["f1"] > current_metrics["f1"]
                and candidate_metrics["fpr"] <= current_metrics["fpr"]
            )
            if accepted:
                gated_train_df = candidate_train_df
                gated_train_label = candidate_train_label
                gated_model = candidate_model
                gated_snapshots = candidate_snapshots
                retrain_reason = "accepted candidate: validation F1 improved without increasing FPR"
            else:
                retrain_reason = "kept current model: candidate did not improve validation F1 without increasing FPR"
        elif trigger:
            retrain_reason = f"last window, no future retrain: {reason}"

        row = row_from_metrics(
            "Drift policy",
            gated_train_label,
            window_name,
            timeline,
            current_metrics,
            accepted,
            retrain_reason,
            policy_conditions=policy_conditions,
            current_validation_f1=current_validation_f1,
            candidate_validation_f1=candidate_validation_f1,
        )
        rows.append(row)
        print(f"gated policy -> {window_name}: F1={float(row['f1']):.3f}, accepted={accepted}, {retrain_reason}")

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timeline",
        "window",
        "strategy",
        "train_windows",
        "psi_action",
        "novel_token_rate",
        "precision",
        "recall",
        "f1",
        "fpr",
        "tpr",
        "anomaly_rate",
        "mean_confidence",
        "low_confidence_rate",
        "gt_present",
        "gt_total",
        "retrain_triggered",
        "retrain_reason",
        "policy_conditions",
        "current_validation_f1",
        "candidate_validation_f1",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
