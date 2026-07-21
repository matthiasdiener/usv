"""Tests for the opt-in JAX timer backend.

The real timing path needs JAX installed (GPU or CPU); it is skipped otherwise.
"""

from __future__ import annotations

import pytest

from usv import do_bench, get_timer
from usv.timer import _JaxTimer


def test_jax_not_in_auto(monkeypatch):
    # "auto" must never pick JAX, even if JAX is importable.
    import usv.timer as timer

    monkeypatch.setattr(timer, "_TorchTimer", lambda: (_ for _ in ()).throw(RuntimeError()))
    tm = get_timer("auto")
    assert not isinstance(tm, _JaxTimer)


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_timer("nope")


def test_jax_timer_times():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    x = jnp.arange(1024, dtype=jnp.float32)
    fn = jax.jit(lambda: jnp.sum(x * x))
    m = do_bench(fn, warmup=2, iters=8, timer="jax")
    assert m.n == 8 and m.median >= 0.0
