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
                        choices=["all", "benchmark", "drift", "embed_parity", "explain"],
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
