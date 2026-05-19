"""
Pluggable embedding providers for FLASH.

Replaces Word2Vec with alternative backends (token encoders, LLM embeddings)
while preserving the `infer(document) -> np.ndarray` interface expected by
the GNN pipeline.
"""

import numpy as np
import hashlib
import os
import torch
import requests
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface for embedding providers.  `embed()` is the core call;
    `infer()` adds the optional positional-encoder step timed by document
    length, matching the original pipeline contract.
    """

    def __init__(self, vector_size=30, max_len=100000):
        self.vector_size = vector_size
        self._encoder = PositionalEncoder(vector_size, max_len) if hasattr(self, "_use_pe") else None

    @abstractmethod
    def embed(self, document):
        """Return a single vector for *document* (list of tokens)."""

    def infer(self, document):
        vec = self.embed(document)
        if self._encoder is not None and len(document) < 100000:
            t = torch.tensor(vec, dtype=torch.float).unsqueeze(0)
            t = self._encoder.embed(t)
            vec = t.squeeze(0).detach().cpu().numpy()
        return vec


# ── Positional encoder (shared) ──────────────────────────────────────────

class PositionalEncoder:
    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x):
        return x + self.pe[:x.size(0)]


# ── Backend: Word2Vec (baseline) ─────────────────────────────────────────

import math


class Word2VecProvider(EmbeddingProvider):
    """Original FLASH Word2Vec embedding provider."""

    def __init__(self, model_path, vector_size=30, max_len=100000):
        from gensim.models import Word2Vec
        self.w2v = Word2Vec.load(model_path)
        self.vector_size = vector_size
        self._encoder = PositionalEncoder(vector_size, max_len)
        self._max_len = max_len

    def embed(self, document):
        word_embeddings = [self.w2v.wv[word] for word in document if word in self.w2v.wv]
        if not word_embeddings:
            return np.zeros(self.vector_size).astype(np.float32)
        emb = np.mean(word_embeddings, axis=0)
        return emb.astype(np.float32)

    def infer(self, document):
        vec = self.embed(document)
        t = torch.tensor(vec, dtype=torch.float).unsqueeze(0)
        if len(document) < self._max_len:
            t = self._encoder.embed(t)
        return t.squeeze(0).detach().cpu().numpy()


# ── Backend: Random (synthetic/smoke) ────────────────────────────────────

class RandomProvider(EmbeddingProvider):
    """Random embeddings — for smoke tests and ablation baselines."""

    def __init__(self, vector_size=30):
        self.vector_size = vector_size

    def embed(self, document):
        return np.random.randn(self.vector_size).astype(np.float32)


# ── Backend: Static token lookup (bag-of-tokens mean) ────────────────────

class TokenMeanProvider(EmbeddingProvider):
    """Learns a static embedding table from training documents.
    Useful as a lightweight baseline between Word2Vec and full LLM embeddings.
    """

    def __init__(self, vocab_size=5000, vector_size=30):
        self.vector_size = vector_size
        self.vocab_size = vocab_size
        self.table = np.random.randn(vocab_size, vector_size).astype(np.float32) * 0.1
        self._token_to_id = {}
        self._next_id = 0

    def fit(self, documents):
        """Build vocabulary from training documents."""
        for doc in documents:
            for token in doc:
                if token not in self._token_to_id and self._next_id < self.vocab_size:
                    self._token_to_id[token] = self._next_id
                    self._next_id += 1

    def _token_id(self, token):
        return self._token_to_id.get(token, 0)

    def embed(self, document):
        if not document:
            return np.zeros(self.vector_size).astype(np.float32)
        indices = [self._token_id(t) for t in document]
        vecs = self.table[indices]
        return vecs.mean(axis=0)


# ── Event text canonicalization ───────────────────────────────────────────

def event_to_text(features):
    """Convert a node's feature list (from prepare_graph) to semantic text.

    The feature list from prepare_graph is typically:
      [exec_path, action, optional_path]

    Returns a canonical text string for transformer embedding.
    """
    parts = []
    if features and len(features) > 0:
        exec_path = str(features[0]) if features[0] else ""
        if exec_path:
            parts.append(f"executable:{exec_path}")
    if features and len(features) > 1:
        action = str(features[1]) if features[1] else ""
        if action:
            parts.append(f"action:{action}")
    if features and len(features) > 2:
        path = str(features[2]) if features[2] else ""
        if path:
            parts.append(f"path:{path}")
    text = " ".join(parts) if parts else "unknown_event"
    return text


def batch_event_to_text(features_list):
    """Convert a list of feature lists to canonical text strings."""
    return [event_to_text(f) for f in features_list]


# ── Backend: LLM / transformer embeddings (pluggable) ────────────────────

class LLMEmbedProvider(EmbeddingProvider):
    """Wrapper for local or API-based LLM embedders.

    Example usage with a local sentence-transformer model::

        provider = LLMEmbedProvider(
            model_name='all-MiniLM-L6-v2',
            vector_size=384,
            cache_dir='.embed_cache'
        )
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", vector_size=384,
                 device="cpu", cache_dir=None, use_positional_encoder=True):
        self.model_name = model_name
        self.vector_size = vector_size
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._cache = {}
        if use_positional_encoder:
            self._encoder = PositionalEncoder(vector_size)
            self._use_pe = True
        else:
            self._encoder = None
            self._use_pe = False
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            raise ImportError("sentence-transformers not installed. "
                              "Install with: pip install sentence-transformers")

    def _cache_key(self, doc):
        return hashlib.md5(" ".join(doc).encode()).hexdigest()

    def embed(self, document):
        if not document:
            return np.zeros(self.vector_size).astype(np.float32)
        key = self._cache_key(document)
        if key in self._cache:
            return self._cache[key]
        # Join tokens into a single text for sentence-transformers
        text = " ".join(document)
        emb = self._model.encode(text, convert_to_numpy=True)
        emb = emb.astype(np.float32)
        if self.cache_dir:
            self._cache[key] = emb
        return emb


# ── Backend: Hugging Face Inference API (cloud) ───────────────────────────

class HFTransformerProvider(EmbeddingProvider):
    """Cloud-based transformer embeddings via Hugging Face Inference API.

    Requires environment variable HF_TOKEN set with a valid Hugging Face
    API token. Embeds text via the hosted inference endpoint, with optional
    on-disk caching and exponential backoff retry.

    Parameters
    ----------
    model_name : str
        Hugging Face model ID (e.g. 'sentence-transformers/all-MiniLM-L6-v2')
    vector_size : int
        Embedding dimension (inferred from model if not given)
    batch_size : int
        Number of texts to embed per API call
    max_length : int
        Maximum token length for the model
    cache_dir : str or None
        Directory for on-disk embedding cache
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 vector_size=None, batch_size=64, max_length=128,
                 cache_dir=".hf_cache"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        self._session = None
        self._headers = None
        self._cache = {}
        self._load_cache()

        # If vector_size not given, use default for common models
        if vector_size is None:
            self.vector_size = self._default_vector_size(model_name)
        else:
            self.vector_size = vector_size

    def _default_vector_size(self, model_name):
        known = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "intfloat/e5-small-v2": 384,
            "intfloat/e5-base-v2": 768,
        }
        for key, dim in known.items():
            if key in model_name:
                return dim
        return 384

    def _load_cache(self):
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.cache_dir, "hf_cache.npy")
            if os.path.exists(cache_path):
                try:
                    self._cache = np.load(cache_path, allow_pickle=True).item()
                except Exception:
                    self._cache = {}

    def _save_cache(self):
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, "hf_cache.npy")
            try:
                np.save(cache_path, self._cache, allow_pickle=True)
            except Exception:
                pass

    def _ensure_session(self):
        if self._session is None:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError(
                    "HF_TOKEN environment variable not set. "
                    "Get a token at https://huggingface.co/settings/tokens"
                )
            self._session = requests.Session()
            self._headers = {"Authorization": f"Bearer {token}"}
            self._api_url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}"

    def _cache_key(self, text):
        raw = f"{self.model_name}:{self.max_length}:{text}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _embed_texts(self, texts):
        """Call HF Inference API with exponential backoff."""
        import time as _t
        self._ensure_session()
        payload = {
            "inputs": texts,
            "options": {
                "wait_for_model": True,
                "use_cache": True,
            },
        }
        if self.max_length:
            payload["parameters"] = {"max_length": self.max_length}

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = self._session.post(
                    self._api_url,
                    headers=self._headers,
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        return np.array(data, dtype=np.float32)
                    raise RuntimeError(f"Unexpected API response: {data}")
                elif resp.status_code == 503 and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    _t.sleep(wait)
                    continue
                else:
                    raise RuntimeError(
                        f"HF API error {resp.status_code}: {resp.text[:200]}"
                    )
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    _t.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError(f"HF API failed after {max_retries} retries")

    def embed(self, document):
        """Embed a single document (list of tokens) into a vector."""
        text = " ".join(document) if isinstance(document, list) else str(document)
        ck = self._cache_key(text)
        if ck in self._cache:
            return self._cache[ck]
        vecs = self._embed_texts([text])
        emb = vecs[0]
        self._cache[ck] = emb
        self._save_cache()
        return emb

    def embed_batch(self, documents):
        """Embed a list of documents (list of token-lists).

        Returns a numpy array of shape (len(documents), vector_size).
        """
        texts_to_fetch = []
        results = [None] * len(documents)
        for i, doc in enumerate(documents):
            text = " ".join(doc) if isinstance(doc, list) else str(doc)
            ck = self._cache_key(text)
            if ck in self._cache:
                results[i] = self._cache[ck]
            else:
                texts_to_fetch.append((i, text))

        if texts_to_fetch:
            for start in range(0, len(texts_to_fetch), self.batch_size):
                batch = texts_to_fetch[start:start + self.batch_size]
                batch_texts = [t for _, t in batch]
                vecs = self._embed_texts(batch_texts)
                for j, (idx, orig_text) in enumerate(batch):
                    if j < len(vecs):
                        emb = np.array(vecs[j], dtype=np.float32)
                        results[idx] = emb
                        ck = self._cache_key(orig_text)
                        self._cache[ck] = emb
                self._save_cache()
        return np.array(results, dtype=np.float32)


# ── Factory ──────────────────────────────────────────────────────────────

def get_provider(name="word2vec", **kwargs):
    providers = {
        "random": RandomProvider,
        "word2vec": Word2VecProvider,
        "token_mean": TokenMeanProvider,
        "llm": LLMEmbedProvider,
        "hf": HFTransformerProvider,
    }
    cls = providers.get(name)
    if cls is None:
        raise ValueError(f"Unknown provider {name}. Options: {list(providers.keys())}")
    return cls(**kwargs)
