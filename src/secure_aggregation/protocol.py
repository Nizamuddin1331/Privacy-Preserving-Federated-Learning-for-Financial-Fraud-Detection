"""Pairwise-masking secure aggregation (simplified Bonawitz et al., CCS 2017).

Round structure
  1. Each client generates an X25519 key pair and publishes the public key.
  2. Every pair (i, j) derives a shared seed via ECDH + HKDF-SHA256.
  3. Client i uploads  y_i = enc(x_i) + sum_{j>i} PRG(s_ij) - sum_{j<i} PRG(s_ij)  (mod 2^64).
     Masks cancel in the sum, so the server learns only sum_i x_i, never any x_i.
  4. If client d drops *after* key exchange, each survivor i reveals s_id and the
     server removes the leftover masks.

Simplifications versus the full protocol (documented in the report): no Shamir
secret sharing of seeds, no double-masking, honest-but-curious server, and the
dropout-recovery step reveals pairwise seeds of dropped clients only.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .crypto_utils import decode, encode, prg

_RAW = dict(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)


class SecAggClient:
    def __init__(self, client_id: int):
        self.id = client_id
        self._sk = X25519PrivateKey.generate()
        self.public_key: bytes = self._sk.public_key().public_bytes(**_RAW)
        self._seeds: Dict[int, bytes] = {}

    def derive_seeds(self, round_id: int, peer_keys: Dict[int, bytes]) -> None:
        for j, pk in peer_keys.items():
            if j == self.id:
                continue
            shared = self._sk.exchange(X25519PublicKey.from_public_bytes(pk))
            lo, hi = sorted((self.id, j))
            info = f"secagg|r{round_id}|{lo}|{hi}".encode()
            self._seeds[j] = HKDF(hashes.SHA256(), 32, None, info).derive(shared)

    def mask(self, x: np.ndarray, clip: float = 16.0) -> np.ndarray:
        y = encode(x, clip)
        for j, seed in self._seeds.items():
            m = prg(seed, len(y))
            y = y + m if self.id < j else y - m   # uint64 arithmetic wraps mod 2^64
        return y

    def reveal_seeds_for(self, dropped: Sequence[int]) -> Dict[int, bytes]:
        return {d: self._seeds[d] for d in dropped if d in self._seeds}


class SecAggServer:
    def __init__(self, vector_len: int):
        self.n = vector_len

    def aggregate(self, masked: Dict[int, np.ndarray], survivors_seeds: Dict[int, Dict[int, bytes]] | None = None,
                  dropped: Sequence[int] = ()) -> np.ndarray:
        """Sum masked uploads; strip leftover masks of dropped clients. Returns float sum."""
        if not masked:
            raise ValueError("No uploads to aggregate")
        total = np.zeros(self.n, dtype=np.uint64)
        for y in masked.values():
            total = total + y
        for i, seeds in (survivors_seeds or {}).items():
            for d, seed in seeds.items():
                m = prg(seed, self.n)
                # survivor i added +m if i<d else -m; remove that contribution
                total = total - m if i < d else total + m
        return decode(total)


def secure_sum(vectors: Dict[int, np.ndarray], round_id: int = 0, dropped_after_masking: Sequence[int] = (),
               clip: float = 16.0) -> np.ndarray:
    """Convenience wrapper simulating one full SecAgg round for the given client vectors."""
    ids = sorted(vectors)
    clients = {i: SecAggClient(i) for i in ids}
    keys = {i: c.public_key for i, c in clients.items()}
    for c in clients.values():
        c.derive_seeds(round_id, keys)
    uploads = {i: clients[i].mask(vectors[i], clip) for i in ids if i not in dropped_after_masking}
    seeds = {i: clients[i].reveal_seeds_for(dropped_after_masking) for i in uploads} if dropped_after_masking else None
    n = len(next(iter(vectors.values())))
    return SecAggServer(n).aggregate(uploads, seeds, dropped_after_masking)
