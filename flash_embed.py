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


# ── Factory ──────────────────────────────────────────────────────────────

def get_provider(name="word2vec", **kwargs):
    providers = {
        "random": RandomProvider,
        "word2vec": Word2VecProvider,
        "token_mean": TokenMeanProvider,
        "llm": LLMEmbedProvider,
    }
    cls = providers.get(name)
    if cls is None:
        raise ValueError(f"Unknown provider {name}. Options: {list(providers.keys())}")
    return cls(**kwargs)
