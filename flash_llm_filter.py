"""LLM post-filter for false positive reduction using Mistral API."""

import json
import hashlib
import time
import logging
import re
import os
from typing import Dict, List, Optional, Tuple
from collections import Counter

import requests
import numpy as np

logger = logging.getLogger("flash-llm-filter")

SYSTEM_PROMPT = (
    "You are a cybersecurity analyst reviewing system audit provenance data. "
    "Determine if an entity flagged as anomalous is likely malicious or benign "
    "based ONLY on the context provided.\n\n"
    "Respond with EXACTLY this JSON structure (no extra text):\n"
    "{\n"
    '  "keep_positive": true,\n'
    '  "confidence": 0.85,\n'
    '  "category": "likely_malicious",\n'
    '  "malicious_indicators": ["indicator1"],\n'
    '  "benign_explanation": null,\n'
    '  "reason": "Brief justification."\n'
    "}\n\n"
    "Rules:\n"
    "- Keep the alert unless there is a clear benign explanation.\n"
    "- Malicious indicators: shell execution, sensitive file writes, privilege changes, "
    "unusual network activity, rare tool/path combinations, credential access.\n"
    "- Benign indicators: routine cron jobs, package managers, log rotation, "
    "standard system administration, expected service behavior.\n"
    "- Never use ground truth labels. Never invent facts outside provided context."
)

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


class MistralClient:
    """Thin wrapper around the Mistral chat API with JSON mode."""

    def __init__(self, model: str = "mistral-small-latest", api_key: Optional[str] = None, timeout: int = 60):
        self.model = model
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "MISTRAL_API_KEY not set. Pass api_key= or set the MISTRAL_API_KEY environment variable."
            )
        self.timeout = timeout

    def chat(self, messages: List[Dict], temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(3):
            resp = requests.post(
                MISTRAL_API_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning("Mistral API rate limited, retrying in %ds...", wait)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Mistral API error {resp.status_code}: {resp.text[:300]}")
        raise RuntimeError("Mistral API: rate limit exceeded after 3 retries")

    def check_available(self) -> bool:
        try:
            resp = requests.get(
                "https://api.mistral.ai/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False


def extract_tabular_context(node_feat: dict, train_feat_keys: set) -> dict:
    """Build human-readable context dict from a raw tabular Counter of features."""
    ctx = {}
    roles = Counter()
    actions = Counter()
    execs = Counter()
    paths = Counter()
    novel = []

    for key, val in node_feat.items():
        if key == "bias":
            continue
        if key.startswith("role:"):
            roles[key.split(":", 1)[1]] = int(val)
        elif key.startswith("action:"):
            actions[key] = int(val)
        elif key.startswith("exec:"):
            execs[key] = int(val)
        elif any(key.startswith(p) for p in ("path:", "seg:", "base:")):
            paths[key] = int(val)
        if key not in train_feat_keys:
            novel.append(key)

    if roles:
        ctx["role_counts"] = dict(roles.most_common(5))
    if actions:
        ctx["actions"] = [k for k, _ in actions.most_common(10)]
    if execs:
        ctx["execs"] = [k for k, _ in execs.most_common(5)]
    if paths:
        ctx["paths"] = [k for k, _ in paths.most_common(10)]
    if novel:
        ctx["novel_features"] = novel[:20]
    return ctx


def build_node_context(node_id: str, ctx: dict) -> str:
    """Format a per-node context dict into a prompt text block."""
    parts = [f"Node ID: {node_id}"]
    if roles := ctx.get("role_counts"):
        parts.append(f"Roles: {roles}")
    if actions := ctx.get("actions"):
        parts.append(f"Actions: {actions}")
    if execs := ctx.get("execs"):
        parts.append(f"Executables: {execs}")
    if paths := ctx.get("paths"):
        parts.append(f"Paths: {paths}")
    if novel := ctx.get("novel_features"):
        parts.append(f"Novel/unusual features: {novel}")
    return "\n".join(parts)


def _parse_json_response(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[^{}]*"keep_positive"\s*:\s*(?:true|false)[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


class LLMFilter:
    """Reviews borderline ML detections by querying Mistral API.

    Parameters
    ----------
    client : MistralClient
    review_margin : float
        Score range above threshold that is considered borderline:
        [threshold, threshold + margin).  Scores at or above the upper
        bound are auto-kept; scores below threshold are ignored.
    max_candidates : int
        Maximum number of candidates to send to the LLM (highest-score
        borderline items are reviewed first).
    delay_s : float
        Delay between API calls to avoid rate limits.
    cache_path : str or None
        Path to a JSONL cache file for LLM responses.
    """

    def __init__(
        self,
        client: MistralClient,
        review_margin: float = 0.15,
        max_candidates: int = 500,
        delay_s: float = 0.2,
        cache_path: Optional[str] = None,
    ):
        self.client = client
        self.review_margin = review_margin
        self.max_candidates = max_candidates
        self.delay_s = delay_s
        self.cache_path = cache_path
        self._cache: Dict[str, dict] = {}
        self._load_cache()

    def _hash(self, node_id: str, context_str: str, score: float, threshold: float) -> str:
        raw = json.dumps([node_id, context_str, round(score, 4), round(threshold, 4)], sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _load_cache(self):
        if not self.cache_path:
            return
        try:
            with open(self.cache_path) as f:
                for line in f:
                    entry = json.loads(line)
                    self._cache[entry["_hash"]] = entry
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_cache(self, hash_key: str, entry: dict):
        if not self.cache_path:
            return
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"_hash": hash_key, **entry}) + "\n")
        self._cache[hash_key] = entry

    def review_one(self, node_id: str, context_str: str, score: float, threshold: float) -> Tuple[bool, dict]:
        hash_key = self._hash(node_id, context_str, score, threshold)
        if hash_key in self._cache:
            cached = self._cache[hash_key]
            return cached.get("keep_positive", True), cached

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Review this entity flagged as anomalous "
                    f"(score={score:.4f}, threshold={threshold:.4f}):\n\n"
                    f"{context_str}\n\nRespond with JSON only."
                ),
            },
        ]
        try:
            raw = self.client.chat(messages)
            parsed = _parse_json_response(raw)
            if parsed is None:
                logger.warning("Unparseable Mistral response for %s, keeping alert", node_id)
                parsed = {
                    "keep_positive": True,
                    "confidence": 0.5,
                    "category": "parse_fallback",
                    "malicious_indicators": [],
                    "benign_explanation": "llm_parse_failed",
                    "reason": "Defaulting to keep — LLM response not parseable",
                }
            entry = {**parsed, "_raw": raw}
            self._save_cache(hash_key, entry)
            time.sleep(self.delay_s)
            return entry.get("keep_positive", True), entry
        except Exception as e:
            logger.error("Mistral review failed for %s: %s", node_id, e)
            return True, {"keep_positive": True, "error": str(e), "category": "error_fallback"}

    def filter(
        self,
        ids: List[str],
        scores: List[float],
        context_dict: Dict[str, dict],
        threshold: float,
    ) -> Tuple[List[str], dict]:
        """Review borderline detections and return the final positive set.

        Parameters
        ----------
        ids : all node IDs in the split (not just candidates)
        scores : per-node anomaly scores (parallel to ids)
        context_dict : node_id -> context dict (from extract_tabular_context)
        threshold : ML model threshold

        Returns
        -------
        final_positives : list of node IDs kept after LLM review
        stats : dict with review statistics
        """
        upper = threshold + self.review_margin
        borderline = [(nid, sc) for nid, sc in zip(ids, scores) if threshold <= sc < upper]
        borderline.sort(key=lambda x: -x[1])
        borderline = borderline[: self.max_candidates]

        kept: List[str] = []
        suppressed: List[str] = []
        stats: dict = {
            "n_borderline": len(borderline),
            "n_reviewed": 0,
            "n_kept": 0,
            "n_suppressed": 0,
        }

        for node_id, score in borderline:
            ctx = context_dict.get(node_id, {})
            context_str = build_node_context(node_id, ctx)
            keep, _ = self.review_one(node_id, context_str, score, threshold)
            stats["n_reviewed"] += 1
            if keep:
                kept.append(node_id)
                stats["n_kept"] += 1
            else:
                suppressed.append(node_id)
                stats["n_suppressed"] += 1

        auto_kept = [nid for nid, sc in zip(ids, scores) if sc >= upper]
        final_positives = auto_kept + kept

        stats.update(
            {
                "n_auto_kept": len(auto_kept),
                "n_final": len(final_positives),
                "suppressed_preview": suppressed[:20],
            }
        )
        return final_positives, stats

    @staticmethod
    def build_context_from_tsv(
        df,
        test_ids: List[str],
        train_df,
    ) -> Dict[str, dict]:
        """Build per-node context dicts from a TSV DataFrame for GNN mode."""
        train_actions = set(train_df["action"].unique()) if train_df is not None else set()

        context: Dict[str, dict] = {}
        for node_id in test_ids:
            rows = df[(df["actorID"] == node_id) | (df["objectID"] == node_id)]
            if rows.empty:
                context[node_id] = {}
                continue

            roles = []
            actions = []
            execs = set()
            paths = set()
            for _, row in rows.iterrows():
                if row["actorID"] == node_id:
                    roles.append("subject")
                if row["objectID"] == node_id:
                    roles.append("object")
                actions.append(str(row.get("action", "")))
                ex = str(row.get("exec", "") or row.get("exec_path", "") or "")
                if ex:
                    execs.add(ex)
                obj_path = str(row.get("object", "") or "")
                if obj_path and obj_path.startswith("/"):
                    paths.add(obj_path)

            ctx: dict = {}
            role_counts = Counter(roles)
            ctx["role_counts"] = dict(role_counts.most_common())

            action_counter = Counter(actions)
            ctx["actions"] = [a for a, _ in action_counter.most_common(10)]

            if execs:
                ctx["execs"] = list(execs)[:5]

            if paths:
                ctx["paths"] = list(paths)[:10]

            novel = [a for a in action_counter if a not in train_actions]
            if novel:
                ctx["novel_features"] = novel[:10]

            context[node_id] = ctx

        return context
