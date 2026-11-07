"""Fixed-point encoding and pseudorandom masks for secure aggregation.

Vectors live in the ring Z_(2^64). Floats are scaled by 2^SCALE_BITS, rounded and
stored as two's-complement uint64, so additions wrap modulo 2^64 and negatives work.
"""
from __future__ import annotations

import hashlib

import numpy as np

SCALE_BITS = 20
SCALE = float(1 << SCALE_BITS)


def encode(x: np.ndarray, clip: float = 16.0) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -clip, clip)
    return np.round(x * SCALE).astype(np.int64).view(np.uint64)


def decode(u: np.ndarray) -> np.ndarray:
    return u.astype(np.uint64).view(np.int64).astype(np.float64) / SCALE


def prg(seed: bytes, length: int) -> np.ndarray:
    """Deterministic uniform uint64 vector expanded from a 32-byte seed."""
    key = int.from_bytes(hashlib.sha256(seed).digest(), "big")
    rng = np.random.Generator(np.random.PCG64(key))
    return rng.integers(0, np.iinfo(np.uint64).max, size=length, dtype=np.uint64, endpoint=True)
