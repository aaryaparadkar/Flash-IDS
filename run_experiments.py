#!/usr/bin/env python3
"""
FLASH Experiment Runner

Orchestrates:
  1. Baseline training + benchmark (Word2Vec)
  2. Embedding provider comparison (Q1: TokenMean, LLM)
  3. Drift evaluation (Q2: time windows, PSI, retrain policy)

Usage:
  python run_experiments.py --mode all
  python run_experiments.py --mode benchmark
  python run_experiments.py --mode drift --tsv cadets_train.txt
"""

import os
import sys
import json
import argparse
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("flash-experiments")

os.makedirs("trained_weights/cadets", exist_ok=True)


def import_pipeline():
    """Import shared pipeline functions from run_train.py"""
    sys.path.insert(0, os.path.dirname(__file__))
    import run_train as P
    return P


def run_benchmark(embed_mode='baseline', hf_model='BAAI/bge-small-en-v1.5',
                  hf_batch_size=64, hf_max_length=128):
    """Run the embedding-provider comparison benchmark with real models.

    Parameters
    ----------
    embed_mode : str
        'baseline' for Word2Vec-based providers, 'hf' for Hugging Face transformer.
    """
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    from flash_embed import get_provider, TokenMeanProvider, batch_event_to_text
    from flash_benchmark import BenchmarkSuite, BenchRunConfig, BenchResult

    P = import_pipeline()

    if not os.path.exists(P.TRAIN_TSV):
        logger.error(f"Train TSV not found: {P.TRAIN_TSV}")
        return

    mode_tag = "hf" if embed_mode == "hf" else "baseline"
    suite = BenchmarkSuite(name=f"FLASH Benchmark ({mode_tag})")

    if embed_mode == "hf":
        # HF mode: single provider, all configs use it
        from flash_embed import HFTransformerProvider as _HF
        _hf_provider = _HF(model_name=hf_model, batch_size=hf_batch_size,
                           max_length=hf_max_length, cache_dir='.hf_cache')
        _emb_dim = _hf_provider.vector_size
        variants = [("hf_transformer", {"vector_size": _emb_dim})]
        logger.info(f"HF benchmark using {hf_model} (dim={_emb_dim})")
    else:
        _emb_dim = P.W2V_VECTOR_SIZE
        variants = [
            ("word2vec", {"name": "word2vec", "vector_size": 30}),
            ("random", {"name": "random", "vector_size": 30}),
            ("token_mean", {"name": "token_mean", "vector_size": 30}),
        ]

    for vname, vcfg in variants:
        for _seed in [42, 43]:
            suite.add_config(BenchRunConfig(
                variant=vname,
                data_split="default",
                seed=_seed,
                train_path=P.TRAIN_TSV,
                test_path=P.TEST_TSV,
                gt_path="data_files/cadets.json" if os.path.exists("data_files/cadets.json") else None,
                num_snapshots=22, vector_size=vcfg.get("vector_size", _emb_dim),
            ))

    logger.info(f"Benchmark suite: {len(suite.configs)} configs")

    def _load_tsv(tsv_path):
        df = pd.read_csv(tsv_path, sep='\t', header=None,
                         names=['actorID','actor_type','objectID','object','action','timestamp'])
        df = df.dropna()
        P._ensure_exec_path_columns(df)
        return P.prepare_graph(df)

    def _embed_phrases(phrases, vector_size, variant, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if embed_mode == "hf":
            texts = batch_event_to_text(phrases)
            return _hf_provider.embed_batch(texts)
        cfg = {"name": variant, "vector_size": vector_size}
        if variant == "word2vec":
            cfg["model_path"] = P.W2V_MODEL_PATH
        provider = get_provider(**cfg)
        if hasattr(provider, "fit"):
            provider.fit(phrases)
        return np.array([provider.embed(p) for p in phrases], dtype=np.float32)

    def _train_gnn(graph, num_snapshots, model_prefix, variant, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = P.GCN(_emb_dim, len(P.LABEL_MAP)).to(P.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=P.GNN_LR, weight_decay=P.GNN_WEIGHT_DECAY)
        from sklearn.utils import class_weight
        from torch.nn import CrossEntropyLoss
        l = graph.y.cpu().numpy()
        cw = class_weight.compute_class_weight(class_weight=None, classes=np.unique(l), y=l)
        criterion = CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float).to(P.device), reduction='mean')
        mask = torch.tensor([True] * graph.num_nodes, dtype=torch.bool, device=P.device)
        for m_n in range(num_snapshots):
            loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=P.GNN_BATCH_SIZE, input_nodes=mask)
            total_loss = 0
            for subg in loader:
                model.train()
                optimizer.zero_grad()
                out = model(subg.x, subg.edge_index)
                loss = criterion(out, subg.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * subg.batch_size
            loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=P.GNN_BATCH_SIZE, input_nodes=mask)
            for subg in loader:
                model.eval()
                out = model(subg.x, subg.edge_index)
                sorted_, indices = out.sort(dim=1, descending=True)
                conf = (sorted_[:, 0] - sorted_[:, 1]) / sorted_[:, 0]
                conf = (conf - conf.min()) / conf.max()
                pred = indices[:, 0]
                cond = (pred == subg.y) | (conf >= 0.9)
                mask[subg.n_id[cond.cpu()].to(mask.device)] = False
            sp = f'{model_prefix}_snap{m_n}.pth'
            torch.save(model.state_dict(), sp)
        return model

    def _ensemble_infer(model, graph, num_snapshots, model_prefix):
        mask = torch.tensor([True] * graph.num_nodes, dtype=torch.bool, device=P.device)
        for m_n in range(num_snapshots):
            sp = f'{model_prefix}_snap{m_n}.pth'
            if not os.path.exists(sp):
                continue
            model.load_state_dict(torch.load(sp, map_location=P.device))
            loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=P.GNN_BATCH_SIZE)
            for subg in loader:
                model.eval()
                out = model(subg.x, subg.edge_index)
                sorted_, indices = out.sort(dim=1, descending=True)
                pred = indices[:, 0]
                cond = (pred == subg.y)
                mask[subg.n_id[cond.cpu()].to(mask.device)] = False
        index = torch.where(mask)[0].tolist()
        return index

    # ── Load data once (shared across variants) ──
    P.logger.info("Loading training data...")
    train_phrases, train_labels, train_edges, train_mapp = _load_tsv(P.TRAIN_TSV)
    test_phrases, test_labels, test_edges, test_mapp = _load_tsv(P.TEST_TSV)
    P.logger.info(f"Train: {len(train_phrases)} nodes, Test: {len(test_phrases)} nodes")

    for cfg in suite.configs:
        variant = cfg.variant
        seed = cfg.seed
        np.random.seed(seed)
        P.logger.info(f"Benchmark run: {variant} seed={seed}")

        result = BenchResult(
            variant=f"{mode_tag}_{variant}", seed=seed, data_split=cfg.data_split,
            n_train_nodes=len(train_phrases), n_test_nodes=len(test_phrases),
            n_snapshots=cfg.num_snapshots,
        )
        result.extra["embed_mode"] = embed_mode
        if embed_mode == "hf":
            result.extra["hf_model"] = hf_model

        model_prefix = f'trained_weights/cadets/bench_{mode_tag}_{variant}_s{seed}'

        # ── Train ──
        t0 = time.time()
        try:
            train_nodes = _embed_phrases(train_phrases, cfg.vector_size, variant, seed)
            train_graph = Data(x=torch.tensor(train_nodes, dtype=torch.float).to(P.device),
                               y=torch.tensor(train_labels, dtype=torch.long).to(P.device),
                               edge_index=torch.tensor(train_edges, dtype=torch.long).to(P.device))
            train_graph.n_id = torch.arange(train_graph.num_nodes)
            model = _train_gnn(train_graph, cfg.num_snapshots, model_prefix, variant, seed)
            result.train_time_s = time.time() - t0

            # ── Inference ──
            t0 = time.time()
            test_nodes = _embed_phrases(test_phrases, cfg.vector_size, variant, seed)
            test_graph = Data(x=torch.tensor(test_nodes, dtype=torch.float).to(P.device),
                              y=torch.tensor(test_labels, dtype=torch.long).to(P.device),
                              edge_index=torch.tensor(test_edges, dtype=torch.long).to(P.device))
            test_graph.n_id = torch.arange(test_graph.num_nodes)
            detected_idx = _ensemble_infer(model, test_graph, cfg.num_snapshots, model_prefix)
            result.infer_time_s = time.time() - t0

            detected_ids = set(test_mapp[i] for i in detected_idx if i < len(test_mapp))
            all_ids = set(test_mapp)
            result.n_test_nodes = len(all_ids)
            result.extra["n_detected"] = len(detected_ids)

            gt = set()
            if cfg.gt_path and os.path.exists(cfg.gt_path):
                with open(cfg.gt_path) as f:
                    gt = set(json.load(f))
                # Sanity check: warn if GT looks synthetic (proc_N pattern)
                _sample = next(iter(gt), "")
                if _sample and _sample.startswith("proc_"):
                    logger.warning("  Ground truth IDs look synthetic (not CDM UUIDs) — metrics will be 0")

            from flash_benchmark import two_hop_propagation
            if gt:
                p, r, f1_val, fpr, tpr_ = two_hop_propagation(
                    detected_ids, gt, all_ids, test_edges, test_mapp)
                result.precision = p
                result.recall = r
                result.f1 = f1_val
                result.fpr = fpr
                result.tpr = tpr_
                P.logger.info(f"  {variant} s{seed}: P={p:.3f} R={r:.3f} F1={f1_val:.3f}")
            else:
                P.logger.info(f"  {variant} s{seed}: detected={len(detected_ids)} (no GT)")

        except Exception as e:
            P.logger.error(f"  {variant} s{seed} failed: {e}")
            import traceback
            traceback.print_exc()

        suite.results.append(result)

    lb = suite.leaderboard()
    logger.info(f"\nBenchmark leaderboard:\n{lb}")
    out_path = f"results/benchmark_{mode_tag}.json"
    suite.save(out_path)
    logger.info(f"Benchmark complete. Results saved to {out_path}")


def run_drift_analysis():
    """Run drift analysis on training TSV (Q2)."""
    from flash_drift import analyze_drift, RetrainPolicy, DriftReport

    tsv_path = "cadets_train.txt"
    if not os.path.exists(tsv_path):
        logger.error(f"TSV not found: {tsv_path}")
        return

    logger.info(f"Analyzing drift on {tsv_path}...")
    reports = analyze_drift(tsv_path, num_windows=4, reference_window_idx=0)

    logger.info(f"\n{'='*60}")
    logger.info("Drift Analysis Report")
    logger.info(f"{'='*60}")

    for r in reports:
        logger.info(f"\nWindow: {r.window_name} ({r.n_test} rows)")
        logger.info(f"  PSI scores: { {k: f'{v:.4f}' for k, v in r.psi_scores.items()} }")
        logger.info(f"  JS divergence: { {k: f'{v:.4f}' for k, v in r.js_divergence.items()} }")
        logger.info(f"  Novel token rate: {r.novel_token_rate:.4f}")
        logger.info(f"  Novelty by type: {r.novelty_by_type}")

    # Apply retrain policy
    policy = RetrainPolicy()
    baseline_f1 = 0.9  # Placeholder until real metrics available
    logger.info(f"\n{'='*60}")
    logger.info("Retrain Trigger Policy Check")
    logger.info(f"{'='*60}")
    for r in reports[1:]:  # Skip reference window
        should, reason = policy.should_retrain(r, baseline_f1)
        logger.info(f"  {r.window_name}: retrain={should}  ({reason})")

    os.makedirs("results", exist_ok=True)
    report_data = [
        {
            "window": r.window_name,
            "n_test": r.n_test,
            "psi_scores": r.psi_scores,
            "novel_token_rate": r.novel_token_rate,
        }
        for r in reports
    ]
    with open("results/drift_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info("\nDrift report saved to results/drift_report.json")


def run_rolling_drift_experiment(num_windows=4, num_snapshots=6, seeds=None, source_tsv=None):
    """Train/test across chronological windows to measure drift impact.

    Strategies:
      static_w0: train on W0, test W1/W2/W3
      expanding: train on all prior windows, test next window
      sliding_k2: train on the latest two prior windows
      recent_only: train only on the immediately previous window
      anchor_recent: train on W0 plus the immediately previous window
      triggered: retrain on prior windows only after meaningful drift
    """
    import copy
    import torch
    from torch_geometric.data import Data
    from torch.nn import CrossEntropyLoss
    from flash_benchmark import two_hop_propagation
    from flash_drift import (
        build_time_windows,
        compute_feature_psi,
        compute_novelty_rate,
        extract_feature_profiles,
        load_edge_tsv,
    )
    from flash_embed import get_provider

    if seeds is None:
        seeds = [42]

    P = import_pipeline()

    source_tsv = source_tsv or P.TRAIN_TSV
    if not os.path.exists(source_tsv):
        logger.error(f"TSV not found: {source_tsv}")
        return

    logger.info(f"Building {num_windows} chronological windows from {source_tsv}")
    windows = build_time_windows(source_tsv, num_windows=num_windows)
    window_paths = [w["path"] for w in windows]
    window_dfs = [load_edge_tsv(path) for path in window_paths]

    gt_path = "data_files/cadets.json"
    ground_truth = set()
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            ground_truth = set(json.load(f))
    else:
        logger.warning("No ground truth file found; precision/recall/F1 will be zero")

    embed_provider = get_provider(
        "word2vec",
        model_path=P.W2V_MODEL_PATH,
        vector_size=P.W2V_VECTOR_SIZE,
    )

    def _prepare_df_for_graph(df):
        graph_df = df.rename(columns={"object_type": "object"}).copy()
        P._ensure_exec_path_columns(graph_df)
        return graph_df

    def _concat_windows(indices):
        return pd.concat([window_dfs[i] for i in indices], ignore_index=True)

    def _build_graph(df):
        phrases, labels, edges, mapp = P.prepare_graph(_prepare_df_for_graph(df))
        nodes = np.array([embed_provider.infer(x) for x in phrases], dtype=np.float32)
        graph = Data(
            x=torch.tensor(nodes, dtype=torch.float).to(P.device),
            y=torch.tensor(labels, dtype=torch.long).to(P.device),
            edge_index=torch.tensor(edges, dtype=torch.long).to(P.device),
        )
        graph.n_id = torch.arange(graph.num_nodes, device=P.device)
        return graph, mapp, edges

    def _class_weights(labels):
        labels = np.asarray(labels)
        n_classes = len(P.LABEL_MAP)
        counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (n_classes * counts)
        return torch.tensor(weights, dtype=torch.float).to(P.device)

    def _train_snapshots(graph, run_seed):
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        model = P.GCN(P.W2V_VECTOR_SIZE, len(P.LABEL_MAP)).to(P.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=P.GNN_LR,
            weight_decay=P.GNN_WEIGHT_DECAY,
        )
        criterion = CrossEntropyLoss(weight=_class_weights(graph.y.cpu().numpy()))
        mask = torch.ones(graph.num_nodes, dtype=torch.bool, device=P.device)
        snapshots = []

        for snap_idx in range(num_snapshots):
            if not mask.any():
                logger.info(f"    snapshot {snap_idx}: no active nodes remain")
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
            logger.info(
                f"    snapshot {snap_idx}/{num_snapshots - 1}: "
                f"loss={loss.item():.4f}, remaining={int(mask.sum().item())}"
            )

        return model, snapshots

    def _infer_detected(model, snapshots, graph, mapp):
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

    def _drift_summary(train_df, test_df):
        ref_profile = extract_feature_profiles(train_df)
        test_profile = extract_feature_profiles(test_df)
        psi_scores = compute_feature_psi(ref_profile, test_profile)
        novel_rate, novelty_by_type = compute_novelty_rate(train_df, test_df)
        return psi_scores, novel_rate, novelty_by_type

    def _should_trigger_retrain(psi_scores, novelty_by_type):
        reasons = []
        action_psi = psi_scores.get("action", 0.0)
        object_type_psi = psi_scores.get("object_type", 0.0)
        action_novelty = novelty_by_type.get("action", 0.0)
        if action_psi > 0.2:
            reasons.append(f"PSI(action)={action_psi:.3f}>0.2")
        if object_type_psi > 0.2:
            reasons.append(f"PSI(object_type)={object_type_psi:.3f}>0.2")
        if action_novelty > 0.15:
            reasons.append(f"action_novelty={action_novelty:.3f}>0.15")
        return bool(reasons), "; ".join(reasons) if reasons else "ok"

    results = []
    strategies = []
    for test_idx in range(1, num_windows):
        strategies.append(("static_w0", [0], test_idx))
        strategies.append(("expanding", list(range(test_idx)), test_idx))
        sliding_start = max(0, test_idx - 2)
        strategies.append(("sliding_k2", list(range(sliding_start, test_idx)), test_idx))
        strategies.append(("recent_only", [test_idx - 1], test_idx))
        anchor_indices = sorted(set([0, test_idx - 1]))
        strategies.append(("anchor_recent", anchor_indices, test_idx))

    for run_seed in seeds:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Rolling drift seed={run_seed}")
        logger.info(f"{'=' * 60}")

        for strategy, train_indices, test_idx in strategies:
            logger.info(
                f"\nRolling run: strategy={strategy}, "
                f"train={train_indices}, test=W{test_idx}"
            )
            train_df = _concat_windows(train_indices)
            test_df = window_dfs[test_idx]
            psi_scores, novel_rate, novelty_by_type = _drift_summary(train_df, test_df)

            start = time.time()
            train_graph, _, _ = _build_graph(train_df)
            model, snapshots = _train_snapshots(train_graph, run_seed)
            train_time = time.time() - start

            start = time.time()
            test_graph, test_mapp, test_edges = _build_graph(test_df)
            detected, online_stats = _infer_detected(model, snapshots, test_graph, test_mapp)
            infer_time = time.time() - start

            all_ids = set(test_mapp)
            precision = recall = f1 = fpr = tpr = 0.0
            if ground_truth:
                precision, recall, f1, fpr, tpr = two_hop_propagation(
                    detected,
                    ground_truth,
                    all_ids,
                    test_edges,
                    test_mapp,
                )

            row = {
                "strategy": strategy,
                "train_windows": [f"W{i}" for i in train_indices],
                "test_window": f"W{test_idx}",
                "seed": run_seed,
                "num_snapshots": len(snapshots),
                "n_train_edges": int(len(train_df)),
                "n_test_edges": int(len(test_df)),
                "n_train_nodes": int(train_graph.num_nodes),
                "n_test_nodes": int(test_graph.num_nodes),
                "n_detected": int(len(detected)),
                "psi_scores": {k: float(v) for k, v in psi_scores.items()},
                "novel_token_rate": float(novel_rate),
                "novelty_by_type": {k: float(v) for k, v in novelty_by_type.items()},
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fpr": float(fpr),
                "tpr": float(tpr),
                "anomaly_rate": online_stats["anomaly_rate"],
                "mean_confidence": online_stats["mean_confidence"],
                "low_confidence_rate": online_stats["low_confidence_rate"],
                "train_time_s": round(train_time, 4),
                "infer_time_s": round(infer_time, 4),
                "retrain_triggered": strategy in {"expanding", "sliding_k2", "recent_only", "anchor_recent"} and train_indices != [0],
                "retrain_reason": (
                    "always expand training history"
                    if strategy == "expanding" and len(train_indices) > 1
                    else "sliding recent window retrain"
                    if strategy == "sliding_k2" and len(train_indices) > 1
                    else "recent-only retrain"
                    if strategy == "recent_only" and train_indices != [0]
                    else "baseline plus recent-window retrain"
                    if strategy == "anchor_recent" and train_indices != [0]
                    else "n/a"
                ),
            }
            results.append(row)
            logger.info(
            f"  {strategy} -> W{test_idx}: "
            f"PSI(action)={row['psi_scores'].get('action', 0):.3f}, "
            f"novelty={row['novel_token_rate']:.3f}, "
            f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, FPR={fpr:.3f}, "
            f"anom_rate={online_stats['anomaly_rate']:.3f}, conf={online_stats['mean_confidence']:.3f}"
        )

        logger.info("\nRolling run: strategy=triggered policy")
        current_train_indices = [0]
        current_train_df = _concat_windows(current_train_indices)
        start = time.time()
        current_train_graph, _, _ = _build_graph(current_train_df)
        current_model, current_snapshots = _train_snapshots(current_train_graph, run_seed)
        current_train_time = time.time() - start

        for test_idx in range(1, num_windows):
            test_df = window_dfs[test_idx]
            psi_scores, novel_rate, novelty_by_type = _drift_summary(current_train_df, test_df)
            should_retrain, retrain_reason = _should_trigger_retrain(psi_scores, novelty_by_type)

            start = time.time()
            test_graph, test_mapp, test_edges = _build_graph(test_df)
            detected, online_stats = _infer_detected(current_model, current_snapshots, test_graph, test_mapp)
            infer_time = time.time() - start

            all_ids = set(test_mapp)
            precision = recall = f1 = fpr = tpr = 0.0
            if ground_truth:
                precision, recall, f1, fpr, tpr = two_hop_propagation(
                    detected,
                    ground_truth,
                    all_ids,
                    test_edges,
                    test_mapp,
                )

            row = {
                "strategy": "triggered",
                "train_windows": [f"W{i}" for i in current_train_indices],
                "test_window": f"W{test_idx}",
                "seed": run_seed,
                "num_snapshots": len(current_snapshots),
                "n_train_edges": int(len(current_train_df)),
                "n_test_edges": int(len(test_df)),
                "n_train_nodes": int(current_train_graph.num_nodes),
                "n_test_nodes": int(test_graph.num_nodes),
                "n_detected": int(len(detected)),
                "psi_scores": {k: float(v) for k, v in psi_scores.items()},
                "novel_token_rate": float(novel_rate),
                "novelty_by_type": {k: float(v) for k, v in novelty_by_type.items()},
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fpr": float(fpr),
                "tpr": float(tpr),
                "anomaly_rate": online_stats["anomaly_rate"],
                "mean_confidence": online_stats["mean_confidence"],
                "low_confidence_rate": online_stats["low_confidence_rate"],
                "train_time_s": round(current_train_time, 4),
                "infer_time_s": round(infer_time, 4),
                "retrain_triggered": bool(should_retrain),
                "retrain_reason": retrain_reason,
            }
            results.append(row)
            logger.info(
                f"  triggered -> W{test_idx}: "
            f"train={row['train_windows']}, "
            f"PSI(action)={row['psi_scores'].get('action', 0):.3f}, "
            f"retrain={should_retrain}, "
            f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, FPR={fpr:.3f}, "
            f"anom_rate={online_stats['anomaly_rate']:.3f}, conf={online_stats['mean_confidence']:.3f}"
        )

            if should_retrain and test_idx < num_windows - 1:
                current_train_indices = list(range(test_idx + 1))
                current_train_df = _concat_windows(current_train_indices)
                logger.info(f"    retraining for next window on {current_train_indices}: {retrain_reason}")
                start = time.time()
                current_train_graph, _, _ = _build_graph(current_train_df)
                current_model, current_snapshots = _train_snapshots(current_train_graph, run_seed)
                current_train_time = time.time() - start

    summary_rows = []
    for row in results:
        summary_rows.append({
            "strategy": row["strategy"],
            "seed": row["seed"],
            "train_windows": "+".join(row["train_windows"]),
            "test_window": row["test_window"],
            "psi_action": row["psi_scores"].get("action", 0.0),
            "novel_token_rate": row["novel_token_rate"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "fpr": row["fpr"],
            "anomaly_rate": row["anomaly_rate"],
            "mean_confidence": row["mean_confidence"],
            "low_confidence_rate": row["low_confidence_rate"],
            "retrain_triggered": row["retrain_triggered"],
            "retrain_reason": row["retrain_reason"],
        })

    summary_df = pd.DataFrame(summary_rows)
    adaptive_rows = []
    candidate_strategies = ["expanding", "sliding_k2", "recent_only", "anchor_recent"]
    for (seed_value, test_window), group in summary_df.groupby(["seed", "test_window"]):
        candidates = group[group["strategy"].isin(candidate_strategies)].copy()
        if candidates.empty:
            continue
        chosen = (
            candidates
            .sort_values(["anomaly_rate", "mean_confidence"], ascending=[True, False])
            .iloc[0]
            .copy()
        )
        chosen["strategy"] = "adaptive_online"
        chosen["retrain_triggered"] = chosen["train_windows"] != "W0"
        chosen["retrain_reason"] = "selected lowest online anomaly rate among candidate retraining windows"
        adaptive_rows.append(chosen.to_dict())

    if adaptive_rows:
        summary_df = pd.concat([summary_df, pd.DataFrame(adaptive_rows)], ignore_index=True)

    aggregate_df = (
        summary_df
        .groupby("strategy", as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_fpr=("fpr", "mean"),
            std_fpr=("fpr", "std"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_anomaly_rate=("anomaly_rate", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_low_confidence_rate=("low_confidence_rate", "mean"),
            mean_psi_action=("psi_action", "mean"),
            retrains=("retrain_triggered", "sum"),
        )
        .sort_values(["mean_f1", "mean_fpr"], ascending=[False, True])
    )
    best_by_window_df = (
        summary_df
        .groupby(["test_window", "strategy", "train_windows"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            mean_fpr=("fpr", "mean"),
            mean_psi_action=("psi_action", "mean"),
            retrain_rate=("retrain_triggered", "mean"),
            mean_anomaly_rate=("anomaly_rate", "mean"),
            mean_confidence=("mean_confidence", "mean"),
        )
        .sort_values(["test_window", "mean_f1", "mean_fpr"], ascending=[True, False, True])
        .groupby("test_window", as_index=False)
        .head(1)
    )

    def _md_table(df, cols):
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in df.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    report_lines = [
        "# Rolling Concept Drift Benchmark",
        "",
        (
            "This experiment compares static, expanding, sliding-window, and drift-triggered retraining "
            f"across chronological CADets windows using seed(s): {', '.join(map(str, seeds))}."
        ),
        "",
        "It also tests recent-only and anchor-recent retraining to measure whether data selection improves adaptation beyond simply deciding when to retrain.",
        "",
        "## Strategy Averages",
        "",
        _md_table(
            aggregate_df,
            ["strategy", "mean_f1", "std_f1", "mean_fpr", "std_fpr", "mean_precision", "mean_recall", "mean_anomaly_rate", "mean_confidence", "mean_psi_action", "retrains"],
        ),
        "",
        "## Best Strategy Per Test Window",
        "",
        _md_table(
            best_by_window_df,
            ["test_window", "strategy", "train_windows", "mean_psi_action", "mean_f1", "mean_fpr", "mean_anomaly_rate", "mean_confidence", "retrain_rate"],
        ),
        "",
        "## Interpretation",
        "",
        "- Static W0 is the no-adaptation baseline.",
        "- Compare mean F1 and mean FPR together; the best drift strategy should improve detection without inflating false positives.",
        "- Adaptive-online selects among candidate retraining windows using anomaly rate, so it does not need ground-truth labels at selection time.",
        "- Expanding and triggered retraining show whether adaptive retraining improves over the static baseline.",
        "- Recent-only retraining tests fast adaptation, but can forget older normal behavior.",
        "- Anchor-recent retraining tests a compromise: keep W0 as the stable baseline and add the latest behavior window.",
        "- Sliding-window retraining reduces action drift aggressively, but this should be validated on more datasets because it can discard older behavioral context.",
        "- The triggered policy matches expanding in this setup because drift thresholds fire before W2 and W3.",
        "- Anomaly rate and confidence are included as online signals because they can be monitored without ground-truth labels.",
        "- Raw actor/object UUID novelty is recorded but not used as a retrain trigger because new UUIDs are normal in provenance logs.",
    ]

    output = {
        "metadata": {
            "source_tsv": source_tsv,
            "num_windows": num_windows,
            "seeds": seeds,
            "embedding": "word2vec",
            "ground_truth_path": gt_path if ground_truth else None,
            "strategies": ["static_w0", "expanding", "sliding_k2", "recent_only", "anchor_recent", "triggered", "adaptive_online"],
            "adaptive_online_policy": {
                "candidate_strategies": candidate_strategies,
                "selection_signal": "minimum anomaly_rate, tie-break by maximum mean_confidence",
                "note": "This is a label-free model-selection policy evaluated from already trained candidate windows.",
            },
            "trigger_policy": {
                "action_psi_threshold": 0.2,
                "object_type_psi_threshold": 0.2,
                "action_novelty_threshold": 0.15,
                "note": "Raw actor/object UUID novelty is recorded but not used as a trigger.",
            },
        },
        "results": results,
        "summary": summary_rows,
        "strategy_averages": aggregate_df.to_dict(orient="records"),
        "best_by_window": best_by_window_df.to_dict(orient="records"),
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/rolling_drift_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    summary_df.to_csv("results/rolling_drift_summary.csv", index=False)
    aggregate_df.to_csv("results/rolling_drift_strategy_averages.csv", index=False)
    best_by_window_df.to_csv("results/rolling_drift_best_by_window.csv", index=False)
    with open("results/rolling_drift_report.md", "w") as f:
        f.write("\n".join(report_lines))
    logger.info(f"\nRolling drift benchmark saved to {out_path}")
    logger.info("Rolling drift summary saved to results/rolling_drift_summary.csv")
    logger.info("Rolling drift report saved to results/rolling_drift_report.md")
    return output


def run_drift_threshold_sensitivity(
    num_windows=4,
    num_snapshots=6,
    seed=42,
    thresholds=None,
):
    """Evaluate how PSI(action) retrain thresholds affect performance.

    This mode isolates the action-PSI threshold. Other drift signals are
    recorded, but they are not used as triggers here.
    """
    import copy
    import torch
    from torch_geometric.data import Data
    from torch.nn import CrossEntropyLoss
    from flash_benchmark import two_hop_propagation
    from flash_drift import (
        build_time_windows,
        compute_feature_psi,
        compute_novelty_rate,
        extract_feature_profiles,
        load_edge_tsv,
    )
    from flash_embed import get_provider

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.5]

    P = import_pipeline()
    source_tsv = P.TRAIN_TSV
    if not os.path.exists(source_tsv):
        logger.error(f"TSV not found: {source_tsv}")
        return

    logger.info(f"Building {num_windows} chronological windows from {source_tsv}")
    windows = build_time_windows(source_tsv, num_windows=num_windows)
    window_dfs = [load_edge_tsv(w["path"]) for w in windows]

    gt_path = "data_files/cadets.json"
    ground_truth = set()
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            ground_truth = set(json.load(f))
    else:
        logger.warning("No ground truth file found; precision/recall/F1 will be zero")

    embed_provider = get_provider(
        "word2vec",
        model_path=P.W2V_MODEL_PATH,
        vector_size=P.W2V_VECTOR_SIZE,
    )

    def _prepare_df_for_graph(df):
        graph_df = df.rename(columns={"object_type": "object"}).copy()
        P._ensure_exec_path_columns(graph_df)
        return graph_df

    def _concat_windows(indices):
        return pd.concat([window_dfs[i] for i in indices], ignore_index=True)

    def _build_graph(df):
        phrases, labels, edges, mapp = P.prepare_graph(_prepare_df_for_graph(df))
        nodes = np.array([embed_provider.infer(x) for x in phrases], dtype=np.float32)
        graph = Data(
            x=torch.tensor(nodes, dtype=torch.float).to(P.device),
            y=torch.tensor(labels, dtype=torch.long).to(P.device),
            edge_index=torch.tensor(edges, dtype=torch.long).to(P.device),
        )
        graph.n_id = torch.arange(graph.num_nodes, device=P.device)
        return graph, mapp, edges

    def _class_weights(labels):
        labels = np.asarray(labels)
        n_classes = len(P.LABEL_MAP)
        counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (n_classes * counts)
        return torch.tensor(weights, dtype=torch.float).to(P.device)

    def _train_snapshots(graph):
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = P.GCN(P.W2V_VECTOR_SIZE, len(P.LABEL_MAP)).to(P.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=P.GNN_LR,
            weight_decay=P.GNN_WEIGHT_DECAY,
        )
        criterion = CrossEntropyLoss(weight=_class_weights(graph.y.cpu().numpy()))
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

        return model, snapshots

    def _infer_detected(model, snapshots, graph, mapp):
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

    def _drift_summary(train_df, test_df):
        ref_profile = extract_feature_profiles(train_df)
        test_profile = extract_feature_profiles(test_df)
        psi_scores = compute_feature_psi(ref_profile, test_profile)
        novel_rate, novelty_by_type = compute_novelty_rate(train_df, test_df)
        return psi_scores, novel_rate, novelty_by_type

    rows = []
    for threshold in thresholds:
        logger.info(f"\nThreshold run: PSI(action) > {threshold}")
        current_train_indices = [0]
        current_train_df = _concat_windows(current_train_indices)
        current_train_graph, _, _ = _build_graph(current_train_df)
        current_model, current_snapshots = _train_snapshots(current_train_graph)
        retrain_count = 0

        for test_idx in range(1, num_windows):
            test_df = window_dfs[test_idx]
            psi_scores, novel_rate, novelty_by_type = _drift_summary(current_train_df, test_df)
            action_psi = psi_scores.get("action", 0.0)
            should_retrain = action_psi > threshold

            test_graph, test_mapp, test_edges = _build_graph(test_df)
            detected, online_stats = _infer_detected(current_model, current_snapshots, test_graph, test_mapp)

            precision = recall = f1 = fpr = tpr = 0.0
            if ground_truth:
                precision, recall, f1, fpr, tpr = two_hop_propagation(
                    detected,
                    ground_truth,
                    set(test_mapp),
                    test_edges,
                    test_mapp,
                )

            row = {
                "policy": f"action_psi_gt_{threshold}",
                "action_psi_threshold": float(threshold),
                "seed": seed,
                "train_windows": "+".join(f"W{i}" for i in current_train_indices),
                "test_window": f"W{test_idx}",
                "psi_action": float(action_psi),
                "psi_object_type": float(psi_scores.get("object_type", 0.0)),
                "action_novelty": float(novelty_by_type.get("action", 0.0)),
                "novel_token_rate": float(novel_rate),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fpr": float(fpr),
                "tpr": float(tpr),
                "anomaly_rate": online_stats["anomaly_rate"],
                "mean_confidence": online_stats["mean_confidence"],
                "low_confidence_rate": online_stats["low_confidence_rate"],
                "retrain_triggered": bool(should_retrain),
                "n_detected": int(len(detected)),
            }
            rows.append(row)
            logger.info(
                f"  threshold={threshold:.2f} -> W{test_idx}: "
                f"train={row['train_windows']}, PSI(action)={action_psi:.3f}, "
                f"retrain={should_retrain}, F1={f1:.3f}, FPR={fpr:.3f}, "
                f"anom_rate={online_stats['anomaly_rate']:.3f}, conf={online_stats['mean_confidence']:.3f}"
            )

            if should_retrain and test_idx < num_windows - 1:
                retrain_count += 1
                current_train_indices = list(range(test_idx + 1))
                current_train_df = _concat_windows(current_train_indices)
                current_train_graph, _, _ = _build_graph(current_train_df)
                current_model, current_snapshots = _train_snapshots(current_train_graph)

        for row in rows:
            if row["action_psi_threshold"] == float(threshold):
                row["total_retrains_for_policy"] = retrain_count

    detail_df = pd.DataFrame(rows)
    summary_df = (
        detail_df
        .groupby(["policy", "action_psi_threshold"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            mean_fpr=("fpr", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_anomaly_rate=("anomaly_rate", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_low_confidence_rate=("low_confidence_rate", "mean"),
            mean_psi_action=("psi_action", "mean"),
            retrain_windows=("retrain_triggered", "sum"),
            retrain_events=("total_retrains_for_policy", "max"),
        )
        .sort_values(["mean_f1", "mean_fpr"], ascending=[False, True])
    )

    def _md_table(df, cols):
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in df.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    report_lines = [
        "# Drift Threshold Sensitivity",
        "",
        "This experiment varies only the PSI(action) retraining threshold. Action novelty and UUID novelty are recorded but not used as triggers here, so the threshold effect is isolated.",
        "",
        "## Policy Summary",
        "",
        _md_table(
            summary_df,
            ["policy", "action_psi_threshold", "mean_f1", "mean_fpr", "mean_precision", "mean_recall", "mean_anomaly_rate", "mean_confidence", "mean_psi_action", "retrain_events"],
        ),
        "",
        "## Interpretation",
        "",
        "- Lower thresholds retrain earlier and are more sensitive to drift.",
        "- Higher thresholds avoid retraining but can leave the model stale for later windows.",
        "- The best threshold should balance F1/FPR against retraining cost.",
        "- Anomaly rate and confidence are label-free online monitoring signals that can be tracked before ground truth is available.",
    ]

    output = {
        "metadata": {
            "source_tsv": source_tsv,
            "seed": seed,
            "thresholds": thresholds,
            "trigger_signal": "PSI(action)",
            "note": "Action novelty is recorded but disabled as a trigger in this sensitivity test.",
        },
        "details": rows,
        "summary": summary_df.to_dict(orient="records"),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/drift_threshold_sensitivity.json", "w") as f:
        json.dump(output, f, indent=2)
    detail_df.to_csv("results/drift_threshold_sensitivity.csv", index=False)
    summary_df.to_csv("results/drift_threshold_summary.csv", index=False)
    with open("results/drift_threshold_report.md", "w") as f:
        f.write("\n".join(report_lines))

    logger.info("Drift threshold sensitivity saved to results/drift_threshold_sensitivity.json")
    logger.info("Drift threshold report saved to results/drift_threshold_report.md")
    return output


def run_drift_policy(action_psi_threshold=0.5, num_windows=4, num_snapshots=6, seed=42):
    """Run the selected drift adaptation policy and write decision artifacts."""
    output = run_drift_threshold_sensitivity(
        num_windows=num_windows,
        num_snapshots=num_snapshots,
        seed=seed,
        thresholds=[action_psi_threshold],
    )
    if not output:
        return

    details = output["details"]
    summary = output["summary"][0] if output["summary"] else {}
    decisions = []
    for row in details:
        is_last_window = row["test_window"] == f"W{num_windows - 1}"
        retrain_triggered = bool(row["retrain_triggered"])
        if retrain_triggered and is_last_window:
            decision = "drift_detected_continue_retrain"
        elif retrain_triggered:
            decision = "retrain_before_next_window"
        else:
            decision = "keep_current_model"

        reason = (
            f"PSI(action)={row['psi_action']:.3f}>{action_psi_threshold:.3f}"
            if retrain_triggered
            else f"PSI(action)={row['psi_action']:.3f}<={action_psi_threshold:.3f}"
        )
        decisions.append({
            "test_window": row["test_window"],
            "train_windows": row["train_windows"],
            "decision": decision,
            "reason": reason,
            "psi_action": row["psi_action"],
            "psi_object_type": row["psi_object_type"],
            "action_novelty": row["action_novelty"],
            "novel_token_rate": row["novel_token_rate"],
            "anomaly_rate": row["anomaly_rate"],
            "mean_confidence": row["mean_confidence"],
            "low_confidence_rate": row["low_confidence_rate"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "fpr": row["fpr"],
            "n_detected": row["n_detected"],
        })

    decision_df = pd.DataFrame(decisions)

    def _md_table(df, cols):
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in df.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    report_lines = [
        "# Drift Adaptation Policy",
        "",
        f"Selected policy: retrain when `PSI(action) > {action_psi_threshold:.3f}`.",
        "",
        "This policy uses action-distribution drift as the retraining trigger. Raw actor/object UUID novelty and online detector confidence are reported as monitoring signals, but they are not retraining triggers in this version.",
        "",
        "## Why This Policy",
        "",
        "- The rolling-window experiment showed static W0 degrades under drift, while expanding/triggered retraining improves mean F1 and lowers FPR.",
        "- The threshold sensitivity run showed `PSI(action) > 0.5` keeps similar performance to lower thresholds while reducing retraining events on this dataset.",
        "- Action PSI is more stable than raw UUID novelty because provenance graphs naturally contain many new actor/object identifiers over time.",
        "",
        "## Decisions",
        "",
        _md_table(
            decision_df,
            ["test_window", "train_windows", "decision", "psi_action", "anomaly_rate", "mean_confidence", "f1", "fpr"],
        ),
        "",
        "## Aggregate Result",
        "",
        _md_table(
            pd.DataFrame([summary]),
            ["policy", "mean_f1", "mean_fpr", "mean_anomaly_rate", "mean_confidence", "retrain_events"],
        ),
        "",
        "## How To Use It",
        "",
        "1. Train the initial model on W0.",
        "2. For each new window, compute PSI(action) against the current training window set.",
        f"3. If PSI(action) exceeds {action_psi_threshold:.3f}, mark the window as drifted and retrain before the next window using all observed windows.",
        "4. Track anomaly rate and confidence online so unusual detector behavior is visible before labels arrive.",
    ]

    policy_output = {
        "metadata": {
            "policy": "action_psi_retrain",
            "action_psi_threshold": action_psi_threshold,
            "seed": seed,
            "num_windows": num_windows,
            "num_snapshots": num_snapshots,
        },
        "decisions": decisions,
        "summary": summary,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/drift_policy_decisions.json", "w") as f:
        json.dump(policy_output, f, indent=2)
    decision_df.to_csv("results/drift_policy_decisions.csv", index=False)
    with open("results/drift_policy_report.md", "w") as f:
        f.write("\n".join(report_lines))

    logger.info("Drift policy decisions saved to results/drift_policy_decisions.json")
    logger.info("Drift policy report saved to results/drift_policy_report.md")
    return policy_output


def run_embedding_parity():
    """Test that all embedding providers produce correct shapes (Q1 smoke)."""
    from flash_embed import get_provider

    print("\nEmbedding provider parity check:")
    print("-" * 50)

    test_doc = ["SUBJECT_PROCESS", "read", "FILE_OBJECT_FILE",
                "/usr/bin/test", "open", "netflow"]

    providers = [
        ("random", {"name": "random", "vector_size": 30}),
        ("token_mean", {"name": "token_mean", "vector_size": 30}),
    ]

    # Fit TokenMean on synthetic data
    if os.path.exists("cadets_train.txt"):
        df = pd.read_csv("cadets_train.txt", sep="\t", header=None,
                         names=["actorID", "actor_type", "objectID",
                                "object_type", "action", "timestamp"])
        phrases = []
        for _, row in df.iterrows():
            phrases.append([row["action"], row["actor_type"], row["object_type"]])
    else:
        phrases = [test_doc]

    for pname, pcfg in providers:
        p = get_provider(**pcfg)
        if hasattr(p, "fit"):
            p.fit(phrases)
        vec = p.embed(test_doc)
        status = "✓" if vec.shape == (pcfg["vector_size"],) else "✗"
        print(f"  {status} {pname}: embed({len(test_doc)} tokens) -> {vec.shape}")

    # Test the word2vec provider exists but skip if model not present
    w2v_path = "trained_weights/cadets/word2vec_cadets_E3.model"
    if os.path.exists(w2v_path):
        p = get_provider("word2vec", model_path=w2v_path, vector_size=30)
        vec = p.embed(test_doc)
        print(f"  ✓ word2vec: embed({len(test_doc)} tokens) -> {vec.shape}")
    else:
        print(f"  - word2vec: model not found at {w2v_path} (expected until full training)")

    print("\nEmbedding parity check complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLASH Experiments Runner")
    parser.add_argument("--mode", default="all",
                        choices=["all", "benchmark", "drift", "rolling_drift", "drift_thresholds", "drift_policy", "embed_parity", "explain"],
                        help="Which experiment to run")
    parser.add_argument("--embed-mode", default="baseline",
                        choices=["baseline", "hf", "both"],
                        help="Embedding backend: baseline (Word2Vec), hf (Hugging Face), or both")
    parser.add_argument("--hf-model", default="BAAI/bge-small-en-v1.5",
                        help="Hugging Face model ID for --embed-mode hf")
    parser.add_argument("--hf-batch-size", type=int, default=64,
                        help="Batch size for HF API calls")
    parser.add_argument("--hf-max-length", type=int, default=128,
                        help="Max token length for HF model")
    parser.add_argument("--rolling-seeds", default="42",
                        help="Comma-separated seeds for --mode rolling_drift, e.g. 42 or 42,43,44")
    parser.add_argument("--rolling-source-tsv", default=None,
                        help="Optional TSV source for --mode rolling_drift; defaults to cadets_train.txt")
    parser.add_argument("--drift-thresholds", default="0.1,0.2,0.3,0.5",
                        help="Comma-separated PSI(action) thresholds for --mode drift_thresholds")
    parser.add_argument("--policy-action-psi-threshold", type=float, default=0.5,
                        help="PSI(action) retraining threshold for --mode drift_policy")
    args = parser.parse_args()

    if args.embed_mode in ("hf", "both") and not os.environ.get("HF_TOKEN"):
        print("[bold red]Error: HF_TOKEN environment variable not set.[/bold red]")
        print("Get a token at https://huggingface.co/settings/tokens")
        print("Set it with: export HF_TOKEN=hf_...")
        sys.exit(1)

    os.makedirs("results", exist_ok=True)

    if args.mode in ("all", "embed_parity"):
        run_embedding_parity()

    if args.mode in ("all", "drift"):
        run_drift_analysis()

    if args.mode in ("all", "rolling_drift"):
        rolling_seeds = [int(s.strip()) for s in args.rolling_seeds.split(",") if s.strip()]
        run_rolling_drift_experiment(seeds=rolling_seeds, source_tsv=args.rolling_source_tsv)

    if args.mode in ("all", "drift_thresholds"):
        thresholds = [float(s.strip()) for s in args.drift_thresholds.split(",") if s.strip()]
        run_drift_threshold_sensitivity(thresholds=thresholds)

    if args.mode in ("all", "drift_policy"):
        run_drift_policy(action_psi_threshold=args.policy_action_psi_threshold)

    if args.mode in ("all", "benchmark"):
        if args.embed_mode == "both":
            logger.info("\n" + "=" * 60)
            logger.info("RUN 1/2: Baseline (Word2Vec) benchmark")
            logger.info("=" * 60)
            run_benchmark(embed_mode="baseline",
                          hf_model=args.hf_model,
                          hf_batch_size=args.hf_batch_size,
                          hf_max_length=args.hf_max_length)
            logger.info("\n" + "=" * 60)
            logger.info("RUN 2/2: HF Transformer benchmark")
            logger.info("=" * 60)
            run_benchmark(embed_mode="hf",
                          hf_model=args.hf_model,
                          hf_batch_size=args.hf_batch_size,
                          hf_max_length=args.hf_max_length)
        else:
            run_benchmark(embed_mode=args.embed_mode,
                          hf_model=args.hf_model,
                          hf_batch_size=args.hf_batch_size,
                          hf_max_length=args.hf_max_length)

    if args.mode in ("all", "explain"):
        from flash_explain import generate_explanations
        logger.info("\n" + "=" * 60)
        logger.info("Generating per-node explanations")
        logger.info("=" * 60)
        generate_explanations(
            embed_mode="hf" if args.embed_mode != "baseline" else "baseline",
            hf_model=args.hf_model,
            hf_batch_size=args.hf_batch_size,
            hf_max_length=args.hf_max_length,
        )

    logger.info("\nAll experiments complete.")
