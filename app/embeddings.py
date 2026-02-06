from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.settings import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


def embed_text(text: str) -> list[float]:
    if not text.strip():
        return [0.0] * settings.embed_dim
    v = _model().encode([text], normalize_embeddings=True)[0]
    v = np.asarray(v, dtype=np.float32)
    if v.shape[0] != settings.embed_dim:
        raise ValueError(f"Embedding dim mismatch: got {v.shape[0]}, expected {settings.embed_dim}")
    return v.tolist()
