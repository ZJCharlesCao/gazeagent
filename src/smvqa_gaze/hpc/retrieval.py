from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .config import RETRIEVER_ID, atomic_save_npy


class Contriever:
    """facebook/contriever-msmarco with mean pooling and inner-product scores,
    as in Video-RAG. Text embeddings are cached on disk by content hash."""

    def __init__(self, cache_dir: Path, model_id: str = RETRIEVER_ID, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        import torch

        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors.append(pooled.float().cpu().numpy())
        return np.concatenate(vectors) if vectors else np.zeros((0, 768), dtype=np.float32)

    def encode(self, texts: list[str], cache_name: str | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        if cache_name is None:
            return self._encode(texts)
        digest = hashlib.sha256("\x1f".join(texts).encode("utf-8")).hexdigest()[:16]
        path = self.cache_dir / f"{cache_name}_{digest}.npy"
        if path.is_file():
            return np.load(path)
        vectors = self._encode(texts)
        atomic_save_npy(path, vectors)
        return vectors


class TextIndex:
    """Exact inner-product search (FAISS IndexFlatIP) with causal filtering."""

    def __init__(self, entries: list[dict[str, Any]], vectors: np.ndarray) -> None:
        import faiss

        self.entries = entries
        self.vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.index = faiss.IndexFlatIP(self.vectors.shape[1] if len(self.vectors) else 768)
        if len(self.vectors):
            self.index.add(self.vectors)

    def scores(self, query: np.ndarray) -> np.ndarray:
        if not len(self.entries):
            return np.zeros(0, dtype=np.float32)
        scores, ids = self.index.search(query.reshape(1, -1).astype(np.float32), len(self.entries))
        output = np.empty(len(self.entries), dtype=np.float32)
        output[ids[0]] = scores[0]
        return output

    def search(self, query: np.ndarray, k: int, allowed: np.ndarray) -> list[tuple[dict[str, Any], float]]:
        scores = self.scores(query)
        candidates = np.flatnonzero(allowed)
        if not len(candidates):
            return []
        top = candidates[np.argsort(-scores[candidates])[:k]]
        return [(self.entries[i], float(scores[i])) for i in top]
