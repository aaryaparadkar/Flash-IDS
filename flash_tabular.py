"""Tabular provenance baseline for fair CADETS evaluation.

This model uses only train labels for fitting, validation labels for threshold
selection, and test labels for the final report.
"""

import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


EVENT_KEY = "com.bbn.tc.schema.avro.cdm18.Event"
UUID_KEY = "com.bbn.tc.schema.avro.cdm18.UUID"


def _path_tokens(path: str):
    if not path:
        return []
    parts = [part for part in str(path).lower().split("/") if part]
    tokens = []
    for i in range(1, min(len(parts), 4) + 1):
        tokens.append("path:" + "/".join(parts[:i]))
    if parts:
        tokens.append("base:" + parts[-1])
    tokens.extend("seg:" + part for part in parts)
    return tokens


def load_raw_node_features(jsonl_path: str, ground_truth: Iterable[str]):
    gt = set(ground_truth)
    features: Dict[str, Counter] = defaultdict(Counter)

    with open(jsonl_path) as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("datum", {}).get(EVENT_KEY)
            if not event:
                continue

            subject = (event.get("subject") or {}).get(UUID_KEY)
            objects = []
            for field in ("predicateObject", "predicateObject2"):
                value = event.get(field)
                if isinstance(value, dict) and value.get(UUID_KEY):
                    objects.append(value[UUID_KEY])

            props = (event.get("properties") or {}).get("map", {}) or {}
            action = str(event.get("type", ""))
            exec_name = str(props.get("exec", "")).lower()
            event_name = (event.get("name") or {}).get("string", "")
            paths = []
            for field in ("predicateObjectPath", "predicateObject2Path"):
                value = event.get(field)
                if isinstance(value, dict) and value.get("string"):
                    paths.append(value["string"])

            for role, node_id in [("subject", subject)] + [("object", obj) for obj in objects]:
                if not node_id:
                    continue
                row = features[node_id]
                row["bias"] = 1.0
                row[f"role:{role}"] += 1.0
                row[f"action:{role}:{action}"] += 1.0
                row[f"action:{action}"] += 1.0
                row[f"event_name:{event_name}"] += 1.0
                row[f"exec:{exec_name}"] += 1.0
                row[f"exec_role:{role}:{exec_name}"] += 1.0
                for key in ("ppid", "uid", "euid", "gid", "return_value", "ret_fd1"):
                    if key in props:
                        row[f"prop:{key}"] += 1.0
                for path in paths:
                    for token in _path_tokens(path):
                        row[token] += 1.0
                        row[f"{role}:{token}"] += 1.0

    ids = sorted(features)
    dicts = [dict(features[node_id]) for node_id in ids]
    labels = np.array([1 if node_id in gt else 0 for node_id in ids], dtype=np.int32)
    return ids, dicts, labels


def _best_threshold(scores, labels):
    best = {"threshold": float(scores.max()) if len(scores) else 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in np.unique(scores):
        pred = scores >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
    return best


def _metrics(scores, labels, threshold):
    pred = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    if len(np.unique(labels)) > 1:
        pr_auc = float(average_precision_score(labels, scores))
        roc_auc = float(roc_auc_score(labels, scores))
    else:
        pr_auc = 0.0
        roc_auc = 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "n_detected": int(pred.sum()),
    }


def run_tabular_benchmark(sampled_dir: str, gt_path: str, out_path: str, alpha_values=None):
    alpha_values = alpha_values or [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    with open(gt_path) as handle:
        gt = set(json.load(handle))

    train_ids, train_dicts, train_y = load_raw_node_features(os.path.join(sampled_dir, "train.jsonl"), gt)
    val_ids, val_dicts, val_y = load_raw_node_features(os.path.join(sampled_dir, "val.jsonl"), gt)
    test_ids, test_dicts, test_y = load_raw_node_features(os.path.join(sampled_dir, "test.jsonl"), gt)

    vectorizer = FeatureHasher(n_features=2 ** 18, input_type="dict", alternate_sign=False)
    train_x = vectorizer.transform(train_dicts)
    val_x = vectorizer.transform(val_dicts)
    test_x = vectorizer.transform(test_dicts)

    results = []
    for alpha in alpha_values:
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=float(alpha),
            class_weight="balanced",
            max_iter=200,
            tol=1e-4,
            random_state=42,
        )
        model.fit(train_x, train_y)
        val_scores = model.decision_function(val_x)
        test_scores = model.decision_function(test_x)
        threshold = _best_threshold(val_scores, val_y)
        test_metrics = _metrics(test_scores, test_y, threshold["threshold"])
        results.append({
            "variant": "raw_tabular_sgd_logreg",
            "alpha": float(alpha),
            "validation": threshold,
            "test": test_metrics,
            "n_train_nodes": len(train_ids),
            "n_val_nodes": len(val_ids),
            "n_test_nodes": len(test_ids),
            "n_train_positive": int(train_y.sum()),
            "n_val_positive": int(val_y.sum()),
            "n_test_positive": int(test_y.sum()),
            "n_features": int(train_x.shape[1]),
        })

    payload = {"name": "FLASH Raw Tabular Benchmark", "sampled_dir": sampled_dir, "results": results}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return payload
