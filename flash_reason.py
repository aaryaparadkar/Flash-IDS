"""
Natural-language explanation generator for FLASH-IDS anomalies.

Reads per-node inference data from flash_explain.py output and produces
human-readable alert reports.  Supports two modes:

  template   — deterministic template-based explanations (no dependencies)
  llm        — LLM-generated explanations via Mistral API or local transformers

Usage:
  python flash_reason.py                              (template mode, default)
  python flash_reason.py --mode llm                    (Mistral API, needs MISTRAL_API_KEY)
  python flash_reason.py --mode llm --model mistral-large-latest
  python flash_reason.py --mode llm --backend local    (local transformers)
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("flash-reason")

INPUT_PATH = "results/explain_results.json"
OUTPUT_PATH = "results/explain_nl.json"

ID_TO_LABEL = {
    0: "SUBJECT_PROCESS", 1: "FILE_OBJECT_FILE",
    2: "FILE_OBJECT_UNIX_SOCKET", 3: "UnnamedPipeObject",
    4: "NetFlowObject", 5: "FILE_OBJECT_DIR",
}


def _risk_label(consensus_frac, num_correct, num_snapshots):
    if num_correct == 0:
        return "high"
    if consensus_frac < 0.5:
        return "medium"
    return "low"


def _confidence_word(mean_conf):
    if mean_conf < 0.1:
        return "very low"
    if mean_conf < 0.25:
        return "low"
    if mean_conf < 0.5:
        return "moderate"
    return "high"


def template_explain(node):
    """Build a human-readable explanation from structured data without an LLM."""
    ev_text = node.get("event_text", "unknown")
    uuid = node.get("uuid", "unknown")
    true_label = node.get("true_label_name", "unknown")
    ens = node.get("ensemble", {})
    consensus_label = ens.get("consensus_label", "unknown")
    consensus_frac = ens.get("consensus_fraction", 0)
    num_correct = ens.get("num_correct", 0)
    num_snaps = ens.get("num_snapshots", 1)
    mean_conf = ens.get("mean_confidence_majority", 0)
    vote_dist = ens.get("vote_distribution", [])
    num_neighbors = node.get("num_neighbors", 0)

    risk = _risk_label(consensus_frac, num_correct, num_snaps)
    conf_word = _confidence_word(mean_conf)

    # Build vote summary
    total = sum(v.get("count", 0) for v in vote_dist)
    vote_lines = []
    for v in vote_dist:
        pct = v["count"] / total * 100 if total > 0 else 0
        vote_lines.append(f"  - {v['label']}: {v['count']}/{total} ({pct:.0f}%)")

    lines = []
    lines.append(f"Node: {uuid}")
    lines.append(f"Risk Level: {risk.upper()}")
    lines.append(f"")
    lines.append(f"Event: {ev_text}")
    lines.append(f"")
    lines.append(f"Analysis:")
    lines.append(f"  The true type is '{true_label}' but the ensemble never agreed with this "
                 f"classification across any of the {num_snaps} snapshots.")
    lines.append(f"  The consensus prediction is '{consensus_label}' ({consensus_frac*100:.0f}% "
                 f"of snapshots agree).")
    lines.append(f"  Mean confidence for the majority prediction is {conf_word} ({mean_conf:.3f}).")
    lines.append(f"")
    lines.append(f"Snapshot Vote Distribution:")
    for vl in vote_lines:
        lines.append(vl)
    lines.append(f"")
    lines.append(f"Graph Context: {num_neighbors} neighbor(s) in the event graph.")

    if risk == "high":
        lines.append(f"")
        lines.append(f"Recommendation: Investigate this node. The model consistently fails to "
                     f"classify its behavior, which may indicate an unusual or anomalous process.")
    elif risk == "medium":
        lines.append(f"")
        lines.append(f"Recommendation: Review this node. Ensemble disagreement suggests "
                     f"ambiguous behavior worth attention.")

    return "\n".join(lines)


def _build_llm_prompt(node):
    """Build a structured prompt for LLM-based explanation."""
    ev_text = node.get("event_text", "unknown")
    uuid = node.get("uuid", "unknown")
    true_label = node.get("true_label_name", "unknown")
    ens = node.get("ensemble", {})
    consensus_label = ens.get("consensus_label", "unknown")
    consensus_frac = ens.get("consensus_fraction", 0)
    num_correct = ens.get("num_correct", 0)
    num_snaps = ens.get("num_snapshots", 1)
    mean_conf = ens.get("mean_confidence_majority", 0)
    vote_dist = ens.get("vote_distribution", [])
    num_neighbors = node.get("num_neighbors", 0)

    return f"""You are a cybersecurity analyst. Explain the following anomaly alert.

Event: {ev_text}
True label: {true_label}
Ensemble snapshots: {num_snaps}
Times classified correctly: {num_correct}
Consensus prediction: {consensus_label} ({consensus_frac*100:.0f}% agreement)
Mean confidence: {mean_conf:.3f}
Vote distribution: {json.dumps(vote_dist)}
Neighbors in graph: {num_neighbors}

Provide:
1. A short (1-sentence) summary of what happened
2. Risk level (low/medium/high)
3. A brief explanation of why this is anomalous
4. A recommended action"""


def _call_mistral_api(prompt, model, api_key):
    """Call Mistral API chat endpoint with retry on rate limit."""
    import requests as _req
    import time as _time
    api_url = "https://api.mistral.ai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.3,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        resp = _req.post(api_url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        if resp.status_code == 429:
            wait = 2 ** (attempt + 1)
            logger.warning("    Rate limited, retrying in %ds...", wait)
            _time.sleep(wait)
            continue
        raise RuntimeError(f"Mistral API error {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("Mistral API: rate limit exceeded after retries")


def _call_local_llm(prompt, model_name, _cache={}):
    """Call local transformers model for text generation."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if "model" not in _cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading local model: %s on %s", model_name, device)
        _cache["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        _cache["model"] = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        _cache["device"] = device

    tokenizer = _cache["tokenizer"]
    model = _cache["model"]
    device = _cache["device"]

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=300, temperature=0.3, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _process_node(node, i, mode, backend, model, api_key):
    """Process a single node for explanation."""
    uuid = node.get("uuid", f"node_{i}")
    logger.info("  [%d] %s ...", i + 1, uuid)

    if mode == "template":
        explanation = template_explain(node)
    else:
        prompt = _build_llm_prompt(node)
        try:
            if backend == "mistral":
                explanation = _call_mistral_api(prompt, model, api_key)
            elif backend == "local":
                explanation = _call_local_llm(prompt, model)
            else:
                explanation = "[Unknown backend]"
        except Exception as e:
            explanation = f"[LLM error: {e}]"
            logger.warning("    LLM call failed: %s", e)

    return {
        "node_index": node.get("node_index"),
        "uuid": uuid,
        "risk_level": _risk_label(
            node.get("ensemble", {}).get("consensus_fraction", 0),
            node.get("ensemble", {}).get("num_correct", 0),
            node.get("ensemble", {}).get("num_snapshots", 1),
        ),
        "explanation": explanation,
    }


def generate_reasons(mode="template", model="mistral-large-latest",
                     backend="mistral", input_path=INPUT_PATH, output_path=OUTPUT_PATH,
                     max_workers=2):
    """Generate natural-language explanations for flagged nodes."""

    if not os.path.exists(input_path):
        logger.error("Input file not found: %s. Run flash_explain.py first.", input_path)
        return

    with open(input_path) as f:
        data = json.load(f)

    flagged = data.get("flagged_nodes", [])
    logger.info("Generating explanations for %d flagged nodes (mode=%s, backend=%s)", len(flagged), mode, backend)

    api_key = None
    if mode == "llm" and backend == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            logger.error("MISTRAL_API_KEY not set. Mistral backend requires an API key.")
            return
        logger.info("Using Mistral model: %s", model)

    reasons = []
    if mode == "template":
        for i, node in enumerate(flagged):
            reasons.append(_process_node(node, i, mode, backend, model, api_key))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_node, node, i, mode, backend, model, api_key): i
                for i, node in enumerate(flagged)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    reasons.append(result)
                except Exception as e:
                    logger.error("    Node %d failed: %s", idx, e)

    reasons.sort(key=lambda r: r.get("node_index", 0) or 0)

    output = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "mode": mode,
            "backend": backend,
            "model": model if mode == "llm" else "template",
            "total_flagged": len(flagged),
            "total_explanations": len(reasons),
        },
        "explanations": reasons,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Explanations saved to %s", output_path)

    # Also write a human-readable summary
    summary_path = output_path.replace(".json", ".md")
    with open(summary_path, "w") as f:
        f.write("# FLASH-IDS Anomaly Explanations\n\n")
        f.write(f"Generated: {output['metadata']['generated_at']}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Backend: {backend}\n")
        f.write(f"Flagged nodes: {len(flagged)}\n\n")
        f.write("---\n\n")
        for r in reasons:
            f.write(f"## {r['uuid']}  — Risk: {r['risk_level'].upper()}\n\n")
            f.write(r["explanation"])
            f.write("\n\n---\n\n")

    logger.info("Human-readable summary saved to %s", summary_path)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FLASH-IDS Explanation Generator")
    p.add_argument("--mode", choices=["template", "llm"], default="template",
                   help="Explanation mode: template (deterministic) or llm (Mistral API)")
    p.add_argument("--backend", choices=["mistral", "local"], default="mistral",
                   help="LLM backend: mistral (API) or local (transformers)")
    p.add_argument("--model", default="mistral-large-latest",
                   help="Model ID for LLM mode (Mistral model name or local HF model)")
    p.add_argument("--input", default=INPUT_PATH)
    p.add_argument("--output", default=OUTPUT_PATH)
    args = p.parse_args()
    generate_reasons(mode=args.mode, model=args.model, backend=args.backend,
                     input_path=args.input, output_path=args.output)
