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
import shutil
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


def ensure_tsvs_current(P):
    """Regenerate edge TSVs when the sampled JSONL split is newer.

    Benchmarks consume cadets_train.txt/cadets_test.txt, while the improved
    sampler writes JSONL. This guard prevents benchmarking stale low-coverage
    TSVs after regenerating the sampled data.
    """
    sampled_dir = os.path.dirname(P.TRAIN_JSONL_PATHS[0])
    val_jsonl = os.path.join(sampled_dir, "val.jsonl")
    val_tsv = "cadets_val.txt" if os.path.exists(val_jsonl) else None
    pairs = [
        (P.TRAIN_JSONL_PATHS[0], P.TRAIN_TSV),
        (P.TEST_JSONL_PATHS[0], P.TEST_TSV),
    ]
    if val_tsv:
        pairs.append((val_jsonl, val_tsv))
    force_refresh = bool(os.environ.get("CADETS_SAMPLED_DIR"))
    for jsonl_path, tsv_path in pairs:
        if not os.path.exists(jsonl_path):
            continue
        needs_refresh = (
            force_refresh
            or tsv_path == val_tsv
            or not os.path.exists(tsv_path)
            or os.path.getmtime(jsonl_path) > os.path.getmtime(tsv_path)
        )
        if not needs_refresh:
            continue

        logger.info(f"Refreshing {tsv_path} from {jsonl_path}")
        id_map = P.process_data(jsonl_path)
        P.process_edges(jsonl_path, id_map)
        generated = f"{jsonl_path}.txt"
        if not os.path.exists(generated):
            raise RuntimeError(f"Expected processed TSV not found: {generated}")
        shutil.copyfile(generated, tsv_path)
    return val_tsv


def run_benchmark(embed_mode='baseline', hf_model='BAAI/bge-small-en-v1.5',
                  hf_batch_size=64, hf_max_length=128,
                  validation_ratio=0.2, fallback_quantile=0.995,
                  min_validation_gt=20, llm_filter=None):
    """Run the embedding-provider comparison benchmark with real models.

    Parameters
    ----------
    embed_mode : str
        'baseline' for Word2Vec-based providers, 'hf' for Hugging Face transformer.
    """
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    from flash_embed import get_provider, batch_event_to_text
    from flash_benchmark import (
        BenchmarkSuite,
        BenchRunConfig,
        BenchResult,
        detections_at_threshold,
        optimize_threshold,
        score_auc,
        two_hop_propagation,
    )

    P = import_pipeline()
    val_tsv = ensure_tsvs_current(P)

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

    def _read_tsv(tsv_path):
        df = pd.read_csv(tsv_path, sep='\t', header=None,
                         names=['actorID','actor_type','objectID','object','action','timestamp'])
        df = df.dropna()
        P._ensure_exec_path_columns(df)
        df.sort_values(by='timestamp', ascending=True, inplace=True)
        return df

    def _prepare_graph(df):
        return P.prepare_graph(df)

    def _split_train_validation(df):
        if validation_ratio <= 0 or validation_ratio >= 0.5 or len(df) < 10:
            return df, df.iloc[0:0].copy()
        split_idx = int(len(df) * (1.0 - validation_ratio))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def _build_provider(variant, vector_size, seed, train_phrases):
        rng_state = np.random.get_state()
        np.random.seed(seed)
        torch.manual_seed(seed)
        if embed_mode == "hf":
            return _hf_provider
        cfg = {"name": variant, "vector_size": vector_size}
        if variant == "word2vec":
            cfg["model_path"] = P.W2V_MODEL_PATH
        provider = get_provider(**cfg)
        if hasattr(provider, "fit"):
            provider.fit(train_phrases)
        np.random.set_state(rng_state)
        return provider

    def _embed_phrases(phrases, provider):
        if embed_mode == "hf":
            texts = batch_event_to_text(phrases)
            return provider.embed_batch(texts)
        return np.array([provider.embed(p) for p in phrases], dtype=np.float32)

    def _as_probabilities(out):
        row_sums = out.detach().sum(dim=1)
        if out.detach().min().item() >= 0 and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
            return out
        return torch.softmax(out, dim=1)

    def _train_gnn(graph, num_snapshots, model_prefix, variant, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = P.GCN(_emb_dim, len(P.LABEL_MAP)).to(P.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=P.GNN_LR, weight_decay=P.GNN_WEIGHT_DECAY)
        from sklearn.utils import class_weight
        from torch.nn import CrossEntropyLoss
        l = graph.y.cpu().numpy()
        cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(l), y=l)
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
                probs = _as_probabilities(out)
                sorted_, indices = probs.sort(dim=1, descending=True)
                conf = (sorted_[:, 0] - sorted_[:, 1]) / (sorted_[:, 0] + 1e-12)
                conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-12)
                pred = indices[:, 0]
                cond = (pred == subg.y) | (conf >= 0.9)
                mask[subg.n_id[cond.cpu()].to(mask.device)] = False
            sp = f'{model_prefix}_snap{m_n}.pth'
            torch.save(model.state_dict(), sp)
        return model

    def _ensemble_scores(model, graph, num_snapshots, model_prefix):
        scores = torch.zeros(graph.num_nodes, dtype=torch.float, device=P.device)
        counts = torch.zeros(graph.num_nodes, dtype=torch.float, device=P.device)
        for m_n in range(num_snapshots):
            sp = f'{model_prefix}_snap{m_n}.pth'
            if not os.path.exists(sp):
                continue
            model.load_state_dict(torch.load(sp, map_location=P.device))
            loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=P.GNN_BATCH_SIZE)
            for subg in loader:
                model.eval()
                out = model(subg.x, subg.edge_index)
                probs = _as_probabilities(out)
                max_prob = probs.max(dim=1).values
                entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
                entropy = entropy / np.log(probs.shape[1])
                anomaly_score = 0.7 * (1.0 - max_prob) + 0.3 * entropy
                node_ids = subg.n_id.to(P.device)
                scores[node_ids] += anomaly_score.detach()
                counts[node_ids] += 1.0
        counts = torch.clamp(counts, min=1.0)
        return (scores / counts).detach().cpu().numpy()

    # ── Load data once (shared across variants) ──
    P.logger.info("Loading training data...")
    full_train_df = _read_tsv(P.TRAIN_TSV)
    if val_tsv and os.path.exists(val_tsv):
        train_df = full_train_df
        val_df = _read_tsv(val_tsv)
    else:
        train_df, val_df = _split_train_validation(full_train_df)
    test_df = _read_tsv(P.TEST_TSV)
    train_phrases, train_labels, train_edges, train_mapp = _prepare_graph(train_df)
    val_phrases, val_labels, val_edges, val_mapp = _prepare_graph(val_df) if len(val_df) else ([], [], [[], []], [])
    test_phrases, test_labels, test_edges, test_mapp = _prepare_graph(test_df)
    P.logger.info(
        f"Train: {len(train_phrases)} nodes, Val: {len(val_phrases)} nodes, Test: {len(test_phrases)} nodes"
    )

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
            provider = _build_provider(variant, cfg.vector_size, seed, train_phrases)
            train_nodes = _embed_phrases(train_phrases, provider)
            train_graph = Data(x=torch.tensor(train_nodes, dtype=torch.float).to(P.device),
                               y=torch.tensor(train_labels, dtype=torch.long).to(P.device),
                               edge_index=torch.tensor(train_edges, dtype=torch.long).to(P.device))
            train_graph.n_id = torch.arange(train_graph.num_nodes)
            model = _train_gnn(train_graph, cfg.num_snapshots, model_prefix, variant, seed)
            result.train_time_s = time.time() - t0

            gt = set()
            if cfg.gt_path and os.path.exists(cfg.gt_path):
                with open(cfg.gt_path) as f:
                    gt = set(json.load(f))
                _sample = next(iter(gt), "")
                if _sample and _sample.startswith("proc_"):
                    logger.warning("  Ground truth IDs look synthetic (not CDM UUIDs) — metrics are invalid")

            # ── Validation calibration + inference ──
            t0 = time.time()
            train_scores = _ensemble_scores(model, train_graph, cfg.num_snapshots, model_prefix)

            threshold = float(np.quantile(train_scores, fallback_quantile))
            threshold_source = f"train_quantile_{fallback_quantile:.3f}"
            val_stats = {}
            if len(val_phrases):
                val_nodes = _embed_phrases(val_phrases, provider)
                val_graph = Data(x=torch.tensor(val_nodes, dtype=torch.float).to(P.device),
                                 y=torch.tensor(val_labels, dtype=torch.long).to(P.device),
                                 edge_index=torch.tensor(val_edges, dtype=torch.long).to(P.device))
                val_graph.n_id = torch.arange(val_graph.num_nodes)
                val_scores = _ensemble_scores(model, val_graph, cfg.num_snapshots, model_prefix)
                if gt:
                    val_gt_overlap = len(set(val_mapp).intersection(gt))
                    tuned_threshold, val_stats = optimize_threshold(
                        val_mapp, val_scores, gt, set(val_mapp), val_edges, val_mapp
                    )
                    if tuned_threshold is not None and val_gt_overlap >= min_validation_gt:
                        threshold = tuned_threshold
                        threshold_source = "validation_f1"
                    elif tuned_threshold is not None:
                        val_stats["reason"] = "validation_gt_overlap_below_minimum"
                        val_stats["min_validation_gt"] = min_validation_gt

            test_nodes = _embed_phrases(test_phrases, provider)
            test_graph = Data(x=torch.tensor(test_nodes, dtype=torch.float).to(P.device),
                              y=torch.tensor(test_labels, dtype=torch.long).to(P.device),
                              edge_index=torch.tensor(test_edges, dtype=torch.long).to(P.device))
            test_graph.n_id = torch.arange(test_graph.num_nodes)
            test_scores = _ensemble_scores(model, test_graph, cfg.num_snapshots, model_prefix)
            result.infer_time_s = time.time() - t0

            detected_ids = detections_at_threshold(test_mapp, test_scores, threshold)
            all_ids = set(test_mapp)
            result.extra["n_detected_raw"] = len(detected_ids)

            # ── LLM post-filter (GNN mode) ──
            if llm_filter is not None and len(test_df):
                ctx_dict = LLMFilter.build_context_from_tsv(test_df, test_mapp, full_train_df)
                filtered_ids, llm_stats = llm_filter.filter(
                    test_mapp, test_scores.tolist(), ctx_dict, threshold
                )
                detected_ids = set(filtered_ids)
                result.extra["llm_filter_stats"] = llm_stats
                result.extra["n_detected"] = len(detected_ids)
            else:
                result.extra["n_detected"] = len(detected_ids)

            result.n_test_nodes = len(all_ids)
            result.extra["threshold"] = threshold
            result.extra["threshold_source"] = threshold_source
            result.extra["validation"] = val_stats

            if gt:
                gt_overlap = len(gt.intersection(all_ids))
                recall_ceiling = gt_overlap / len(gt) if gt else 0.0
                result.extra["gt_size"] = len(gt)
                result.extra["gt_overlap_test_ids"] = gt_overlap
                result.extra["recall_ceiling"] = recall_ceiling
                result.pr_auc, result.roc_auc = score_auc(test_mapp, test_scores, gt)
                p, r, f1_val, fpr, tpr_ = two_hop_propagation(
                    detected_ids, gt, all_ids, test_edges, test_mapp)
                result.precision = p
                result.recall = r
                result.f1 = f1_val
                result.fpr = fpr
                result.tpr = tpr_
                parts = [f"  {variant} s{seed}: P={p:.3f} R={r:.3f} F1={f1_val:.3f}"]
                if result.extra.get("llm_filter_stats"):
                    ls = result.extra["llm_filter_stats"]
                    parts.append(
                        f"[LLM: rev={ls.get('n_reviewed',0)} kept={ls.get('n_kept',0)} "
                        f"sup={ls.get('n_suppressed',0)} final={ls.get('n_final',0)}]"
                    )
                P.logger.info(" ".join(parts))
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
                        choices=["all", "benchmark", "tabular", "drift", "embed_parity", "explain"],
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
    parser.add_argument("--validation-ratio", type=float, default=0.2,
                        help="Chronological tail of train TSV used only for threshold calibration")
    parser.add_argument("--fallback-quantile", type=float, default=0.995,
                        help="Unsupervised train-score quantile used when validation has no GT overlap")
    parser.add_argument("--min-validation-gt", type=int, default=20,
                        help="Minimum validation GT overlap required before supervised threshold tuning")
    parser.add_argument("--sampled-dir", default=None,
                        help="Directory containing fair train/val/test JSONL for --mode tabular")
    parser.add_argument("--llm-filter", action="store_true",
                        help="Enable LLM post-filter for false positive reduction")
    parser.add_argument("--llm-model", default="mistral-small-latest",
                        help="Mistral model ID (default: mistral-small-latest)")
    parser.add_argument("--llm-api-key", default=None,
                        help="Mistral API key (falls back to MISTRAL_API_KEY env var)")
    parser.add_argument("--llm-review-margin", type=float, default=0.15,
                        help="Score range above threshold to send for LLM review (default: 0.15)")
    parser.add_argument("--llm-max-candidates", type=int, default=500,
                        help="Maximum borderline candidates to review via LLM (default: 500)")
    parser.add_argument("--llm-cache-path", default="results/llm_filter_cache.jsonl",
                        help="Cache file for LLM responses (default: results/llm_filter_cache.jsonl)")
    args = parser.parse_args()

    if args.embed_mode in ("hf", "both") and not os.environ.get("HF_TOKEN"):
        print("[bold red]Error: HF_TOKEN environment variable not set.[/bold red]")
        print("Get a token at https://huggingface.co/settings/tokens")
        print("Set it with: export HF_TOKEN=hf_...")
        sys.exit(1)

    os.makedirs("results", exist_ok=True)

    # ── LLM filter setup ──
    llm_filter = None
    if args.llm_filter:
        from flash_llm_filter import MistralClient, LLMFilter

        try:
            _client = MistralClient(model=args.llm_model, api_key=args.llm_api_key)
            llm_filter = LLMFilter(
                client=_client,
                review_margin=args.llm_review_margin,
                max_candidates=args.llm_max_candidates,
                cache_path=args.llm_cache_path,
            )
            logger.info("LLM filter enabled: model=%s margin=%.2f max=%d",
                        args.llm_model, args.llm_review_margin, args.llm_max_candidates)
        except ValueError as e:
            logger.warning("LLM filter disabled: %s", e)

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
                           hf_max_length=args.hf_max_length,
                           validation_ratio=args.validation_ratio,
                           fallback_quantile=args.fallback_quantile,
                           min_validation_gt=args.min_validation_gt,
                           llm_filter=llm_filter)
            logger.info("\n" + "=" * 60)
            logger.info("RUN 2/2: HF Transformer benchmark")
            logger.info("=" * 60)
            run_benchmark(embed_mode="hf",
                           hf_model=args.hf_model,
                           hf_batch_size=args.hf_batch_size,
                           hf_max_length=args.hf_max_length,
                           validation_ratio=args.validation_ratio,
                           fallback_quantile=args.fallback_quantile,
                           min_validation_gt=args.min_validation_gt,
                           llm_filter=llm_filter)
        else:
            run_benchmark(embed_mode=args.embed_mode,
                           hf_model=args.hf_model,
                           hf_batch_size=args.hf_batch_size,
                           hf_max_length=args.hf_max_length,
                           validation_ratio=args.validation_ratio,
                           fallback_quantile=args.fallback_quantile,
                           min_validation_gt=args.min_validation_gt,
                           llm_filter=llm_filter)

    if args.mode in ("all", "tabular"):
        from flash_tabular import run_tabular_benchmark
        sampled_dir = args.sampled_dir or os.environ.get("CADETS_SAMPLED_DIR")
        if not sampled_dir:
            raise SystemExit("--sampled-dir or CADETS_SAMPLED_DIR is required for tabular benchmark")
        payload = run_tabular_benchmark(
            sampled_dir=sampled_dir,
            gt_path="data_files/cadets.json",
            out_path="results/benchmark_tabular.json",
            llm_filter=llm_filter,
        )
        best = max(payload["results"], key=lambda row: row["test"]["f1"])
        logger.info(
            "Tabular best: alpha=%s P=%.3f R=%.3f F1=%.3f PR-AUC=%.3f ROC-AUC=%.3f",
            best["alpha"], best["test"]["precision"], best["test"]["recall"],
            best["test"]["f1"], best["test"]["pr_auc"], best["test"]["roc_auc"],
        )
        if "llm_filtered" in best:
            lf = best["llm_filtered"]
            ls = best["llm_filter_stats"]
            logger.info(
                "  LLM-filtered: P=%.3f R=%.3f F1=%.3f | reviewed=%d kept=%d suppressed=%d final=%d",
                lf["precision"], lf["recall"], lf["f1"],
                ls.get("n_reviewed", 0), ls.get("n_kept", 0),
                ls.get("n_suppressed", 0), ls.get("n_final", 0),
            )

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
