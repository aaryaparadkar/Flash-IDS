"""
Per-node explanation generator for FLASH GNN ensemble.

Runs inference on the test graph with all snapshots, collects per-node
prediction trajectories, and exports a structured JSON for LLM reasoning.
"""

import os
import sys
import json
import logging
import torch
import numpy as np
from collections import Counter
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

sys.path.insert(0, os.path.dirname(__file__))
import run_train as P

logger = logging.getLogger("flash-explain")

# Reverse label map for readable class names
ID_TO_LABEL = {v: k for k, v in P.LABEL_MAP.items()}

EXPLAIN_OUTPUT = "results/explain_results.json"


def generate_explanations(
    train_tsv="cadets_train.txt",
    test_tsv="cadets_test.txt",
    raw_test_path="cadets_sampled/test.jsonl",
    num_snapshots=22,
    embed_mode="hf",
    hf_model="BAAI/bge-small-en-v1.5",
    hf_batch_size=64,
    hf_max_length=128,
    output_path=EXPLAIN_OUTPUT,
):
    """Run ensemble inference and generate per-node explanations.

    Returns
    -------
    list[dict]
        One entry per node, with prediction trajectories and metadata.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load test data ──
    test_df = P.pd.read_csv(
        test_tsv, sep="\t", header=None,
        names=["actorID", "actor_type", "objectID", "object", "action", "timestamp"]
    )
    test_df = test_df.dropna()
    test_df.sort_values(by="timestamp", ascending=True, inplace=True)
    test_df = P.add_attributes(test_df, raw_test_path)
    eval_phrases, eval_labels, eval_edges, eval_mapp = P.prepare_graph(test_df)
    logger.info("Test graph: %d nodes, %d edges", len(eval_phrases), len(eval_edges))

    # ── Embed ──
    if embed_mode == "hf":
        from flash_embed import get_provider as _gp, batch_event_to_text
        provider = _gp(
            "hf", model_name=hf_model,
            batch_size=hf_batch_size, max_length=hf_max_length,
            cache_dir=".hf_cache"
        )
        texts = batch_event_to_text(eval_phrases)
        node_embs = provider.embed_batch(texts)
    else:
        node_embs = np.array([P.infer(x) for x in eval_phrases])

    eval_graph = Data(
        x=torch.tensor(node_embs, dtype=torch.float).to(device),
        y=torch.tensor(eval_labels, dtype=torch.long).to(device),
        edge_index=torch.tensor(eval_edges, dtype=torch.long).to(device),
    )
    eval_graph.n_id = torch.arange(eval_graph.num_nodes, device=device)

    # ── Ensemble inference ──
    model = P.GCN(node_embs.shape[1], len(P.LABEL_MAP)).to(device)
    num_nodes = eval_graph.num_nodes

    # Track per-snapshot predictions: [snapshots, nodes]
    snap_preds = torch.zeros((num_snapshots, num_nodes), dtype=torch.long, device=device)
    snap_conf = torch.zeros((num_snapshots, num_nodes), dtype=torch.float, device=device)
    snap_correct = torch.zeros((num_snapshots, num_nodes), dtype=torch.bool, device=device)
    snaps_loaded = 0

    for m_n in range(num_snapshots):
        snap_path = f"{P.GNN_SNAPSHOT_PREFIX}{m_n}_E3.pth"
        if not os.path.exists(snap_path):
            logger.warning("Snapshot %s not found, skipping", snap_path)
            continue
        model.load_state_dict(torch.load(snap_path, map_location=device, weights_only=True))
        loader = NeighborLoader(eval_graph, num_neighbors=[-1, -1], batch_size=P.GNN_BATCH_SIZE)
        for subg in loader:
            model.eval()
            out = model(subg.x, subg.edge_index)
            sorted_vals, indices = out.sort(dim=1, descending=True)
            conf = (sorted_vals[:, 0] - sorted_vals[:, 1]) / sorted_vals[:, 0].clamp(min=1e-8)
            conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
            pred = indices[:, 0]
            nid = subg.n_id
            snap_preds[snaps_loaded, nid] = pred
            snap_conf[snaps_loaded, nid] = conf
            snap_correct[snaps_loaded, nid] = (pred == subg.y)
        snaps_loaded += 1

    # Trim unused snapshot dimension
    snap_preds = snap_preds[:snaps_loaded]
    snap_conf = snap_conf[:snaps_loaded]
    snap_correct = snap_correct[:snaps_loaded]

    # ── Anomaly flag (original criterion) ──
    # Flagged if NO snapshot predicted correctly
    never_correct = (~snap_correct.any(dim=0))

    # ── Build explanations ──
    from flash_embed import event_to_text

    explanations = []
    for node_idx in range(num_nodes):
        uuid = eval_mapp[node_idx]
        preds = snap_preds[:, node_idx].tolist()
        confs = snap_conf[:, node_idx].tolist()
        corrects = snap_correct[:, node_idx].tolist()

        vote = Counter(preds).most_common()
        consensus_class = vote[0][0] if vote else -1
        consensus_frac = vote[0][1] / snaps_loaded if vote else 0

        # Mean confidence for the majority prediction
        majority_conf = np.mean(
            [confs[i] for i, p in enumerate(preds) if p == consensus_class]
        ) if snaps_loaded > 0 else 0.0

        # Event text
        feat_list = eval_phrases[node_idx] if node_idx < len(eval_phrases) else []
        ev_text = event_to_text(feat_list)

        entry = {
            "node_index": int(node_idx),
            "uuid": uuid,
            "event_text": ev_text,
            "raw_features": feat_list,
            "true_label": int(eval_labels[node_idx]),
            "true_label_name": ID_TO_LABEL.get(int(eval_labels[node_idx]), "UNKNOWN"),
            "anomaly_flag": bool(never_correct[node_idx].item()),
            "ensemble": {
                "num_snapshots": snaps_loaded,
                "num_correct": int(snap_correct[:, node_idx].sum().item()),
                "consensus_class": int(consensus_class),
                "consensus_label": ID_TO_LABEL.get(int(consensus_class), "UNKNOWN"),
                "consensus_fraction": round(consensus_frac, 4),
                "mean_confidence_majority": round(float(majority_conf), 4),
                "vote_distribution": [
                    {"class": int(c), "label": ID_TO_LABEL.get(int(c), "UNKNOWN"), "count": int(n)}
                    for c, n in vote
                ],
            },
            "per_snapshot": [
                {
                    "snapshot": i,
                    "predicted_class": int(preds[i]),
                    "predicted_label": ID_TO_LABEL.get(int(preds[i]), "UNKNOWN"),
                    "confidence": round(float(confs[i]), 4),
                    "correct": bool(corrects[i]),
                }
                for i in range(snaps_loaded)
            ],
        }

        # Graph context (neighbors)
        neighbors = []
        ei = eval_edges
        for e in range(len(ei[0])):
            if ei[0][e] == node_idx:
                neighbors.append(int(ei[1][e]))
            elif ei[1][e] == node_idx:
                neighbors.append(int(ei[0][e]))
        entry["neighbor_indices"] = neighbors[:20]
        entry["num_neighbors"] = len(neighbors)

        # Feature attribution: which feature dimension contributed most
        emb = node_embs[node_idx]
        if isinstance(emb, np.ndarray):
            top_feats = np.argsort(-np.abs(emb))[:5].tolist()
            entry["top_embedding_dims"] = top_feats
            entry["embedding_norm"] = round(float(np.linalg.norm(emb)), 4)

        explanations.append(entry)

    # ── Flagged-only summary ──
    flagged = [e for e in explanations if e["anomaly_flag"]]
    flagged_uuids = [e["uuid"] for e in flagged]
    logger.info("Flagged %d / %d nodes as anomalous", len(flagged), num_nodes)
    logger.info("Sample flagged UUIDs: %s", flagged_uuids[:5])

    # ── Save ──
    output = {
        "metadata": {
            "total_nodes": num_nodes,
            "num_flagged": len(flagged),
            "num_snapshots_loaded": snaps_loaded,
            "embed_mode": embed_mode,
            "hf_model": hf_model if embed_mode == "hf" else None,
        },
        "flagged_nodes": flagged,
        "all_nodes": explanations,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Explanations saved to %s", output_path)
    return output


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--embed-mode", choices=["baseline", "hf"], default="hf")
    p.add_argument("--hf-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--hf-batch-size", type=int, default=64)
    p.add_argument("--hf-max-length", type=int, default=128)
    p.add_argument("--output", default=EXPLAIN_OUTPUT)
    args = p.parse_args()
    generate_explanations(
        embed_mode=args.embed_mode,
        hf_model=args.hf_model,
        hf_batch_size=args.hf_batch_size,
        hf_max_length=args.hf_max_length,
        output_path=args.output,
    )
