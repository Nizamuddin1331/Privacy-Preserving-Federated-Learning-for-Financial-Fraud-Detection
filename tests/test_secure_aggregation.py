import numpy as np
import pytest

from src.secure_aggregation.crypto_utils import decode, encode, prg
from src.secure_aggregation.protocol import SecAggClient, SecAggServer, secure_sum


def test_fixed_point_roundtrip_including_negatives():
    x = np.array([-3.25, 0.0, 1e-4, 7.5])
    assert np.allclose(decode(encode(x)), x, atol=2e-6)


def test_prg_is_deterministic_and_seed_dependent():
    assert np.array_equal(prg(b"a" * 32, 10), prg(b"a" * 32, 10))
    assert not np.array_equal(prg(b"a" * 32, 10), prg(b"b" * 32, 10))


def test_masks_cancel_so_server_gets_exact_sum():
    rng = np.random.default_rng(0)
    vecs = {i: rng.normal(size=500) for i in range(5)}
    got = secure_sum(vecs, round_id=1)
    assert np.allclose(got, sum(vecs.values()), atol=1e-4)


def test_individual_upload_reveals_nothing_about_input():
    rng = np.random.default_rng(1)
    x = {i: rng.normal(size=2000) for i in range(3)}
    cs = {i: SecAggClient(i) for i in x}
    keys = {i: c.public_key for i, c in cs.items()}
    for c in cs.values():
        c.derive_seeds(0, keys)
    masked = cs[0].mask(x[0]).view(np.int64).astype(np.float64)
    assert abs(np.corrcoef(masked, x[0])[0, 1]) < 0.1            # uncorrelated with the plaintext
    assert np.std(masked) > 1e15                                  # uniform over the 64-bit ring


def test_dropout_recovery():
    rng = np.random.default_rng(2)
    vecs = {i: rng.normal(size=300) for i in range(5)}
    got = secure_sum(vecs, 3, dropped_after_masking=(1, 4))
    assert np.allclose(got, vecs[0] + vecs[2] + vecs[3], atol=1e-4)


def test_server_rejects_empty_round():
    with pytest.raises(ValueError):
        SecAggServer(10).aggregate({})
