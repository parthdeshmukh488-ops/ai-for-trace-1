"""Activation storage, and the budget check that fires before the allocator does.

The point of the budget is that a capture which will not fit says so, naming
the sites responsible, instead of the process being killed with no indication
of which site was to blame.  So the tests are mostly about the error message:
it is the feature.
"""

from __future__ import annotations

import pytest
import torch

from circuitry.util.store import (
    ActivationBudgetExceeded,
    ActivationStore,
    ShuffleBuffer,
    bytes_for,
)


def test_bytes_for_matches_hand_computation():
    assert bytes_for((8, 512, 12, 768), torch.float32) == 8 * 512 * 12 * 768 * 4
    assert bytes_for((2, 3), torch.float64) == 48
    assert bytes_for((), torch.float32) == 4


def test_bytes_for_rejects_an_unknown_dtype():
    with pytest.raises(ValueError, match="byte width"):
        bytes_for((2,), torch.complex64)


def test_bytes_for_rejects_a_negative_dimension():
    with pytest.raises(ValueError, match="negative dimension"):
        bytes_for((2, -1), torch.float32)


def test_store_and_retrieve():
    store = ActivationStore()
    tensor = torch.zeros(4, 4)
    store.store("blocks.0.resid_post", tensor)
    assert store["blocks.0.resid_post"] is tensor
    assert "blocks.0.resid_post" in store
    assert len(store) == 1
    assert list(store) == ["blocks.0.resid_post"]


def test_a_second_write_to_one_site_raises():
    """Last-write-wins would hide a duplicated hook or a non-unique site name."""
    store = ActivationStore()
    store.store("s", torch.zeros(2))
    with pytest.raises(KeyError, match="already captured"):
        store.store("s", torch.zeros(2))


def test_missing_site_error_lists_what_is_present():
    store = ActivationStore()
    store.store("blocks.0.mlp.post", torch.zeros(2))
    with pytest.raises(KeyError) as excinfo:
        store["blocks.0.mlp.pre"]
    message = str(excinfo.value)
    assert "blocks.0.mlp.pre" in message
    assert "blocks.0.mlp.post" in message


def test_missing_site_error_on_an_empty_store_says_so():
    with pytest.raises(KeyError, match="nothing"):
        ActivationStore()["anything"]


def test_budget_blocks_a_store_that_would_exceed_it():
    store = ActivationStore(budget_bytes=64)
    store.store("fits", torch.zeros(8, dtype=torch.float32))  # 32 bytes
    with pytest.raises(ActivationBudgetExceeded) as excinfo:
        store.store("does_not", torch.zeros(16, dtype=torch.float32))  # 64 more
    error = excinfo.value
    assert error.budget == 64
    assert error.required == 96
    assert error.breakdown == {"does_not": 64}
    assert "over by" in str(error)


def test_check_plan_refuses_before_anything_is_allocated():
    """The whole point: the capture is rejected without allocating a byte of it."""
    store = ActivationStore(budget_bytes=1024)
    plan = {
        "blocks.0.attn.result": ((8, 512, 12, 768), torch.float32),
        "blocks.0.resid_post": ((8, 512, 768), torch.float32),
    }
    with pytest.raises(ActivationBudgetExceeded) as excinfo:
        store.check_plan(plan)
    assert len(store) == 0
    assert store.used_bytes == 0
    # The largest site is named first, which is the one worth dropping.
    assert "blocks.0.attn.result" in str(excinfo.value)


def test_check_plan_returns_the_requirement_when_it_fits():
    store = ActivationStore(budget_bytes=10_000)
    required = store.check_plan({"a": ((10,), torch.float32), "b": ((10,), torch.float32)})
    assert required == 80


def test_check_plan_accounts_for_what_is_already_stored():
    store = ActivationStore(budget_bytes=100)
    store.store("already", torch.zeros(20, dtype=torch.float32))  # 80 bytes
    with pytest.raises(ActivationBudgetExceeded):
        store.check_plan({"more": ((10,), torch.float32)})  # 40 more


def test_error_names_only_the_largest_few_sites():
    """A capture of 200 sites must not produce a 200-item error message."""
    store = ActivationStore(budget_bytes=1)
    plan = {f"site_{i}": ((i + 1,), torch.float32) for i in range(200)}
    with pytest.raises(ActivationBudgetExceeded) as excinfo:
        store.check_plan(plan)
    message = str(excinfo.value)
    assert message.count("site_") <= 5
    assert "site_199" in message  # the largest


def test_no_budget_means_no_limit():
    store = ActivationStore()
    store.store("big", torch.zeros(1000))
    assert store.budget_bytes is None
    assert store.used_bytes == 4000


def test_negative_budget_raises():
    with pytest.raises(ValueError, match="budget_bytes"):
        ActivationStore(budget_bytes=-1)


def test_detach_preserves_values_and_drops_the_graph():
    store = ActivationStore()
    base = torch.ones(3, requires_grad=True)
    store.store("live", base * 2)
    detached = store.detach()
    assert torch.equal(detached["live"], store["live"])
    assert not detached["live"].requires_grad
    assert store["live"].requires_grad


def test_items_and_keys():
    store = ActivationStore()
    store.store("a", torch.zeros(1))
    assert list(store.keys()) == ["a"]
    assert [name for name, _ in store.items()] == ["a"]


def test_repr_reports_usage_and_budget():
    store = ActivationStore(budget_bytes=4096)
    store.store("a", torch.zeros(256, dtype=torch.float32))
    text = repr(store)
    assert "1 sites" in text
    assert "KiB" in text


def test_shuffle_buffer_decorrelates_its_stream():
    """The confound this buffer exists to remove, measured rather than assumed.

    Activations arrive as a stream of forward passes, so consecutive rows come
    from consecutive tokens of the same document and are strongly correlated.
    A dictionary trained on that learns document structure instead of
    features, and its reconstruction loss says nothing about it.

    The stream below is deliberately worse than a real one: each chunk is a
    single "document" with a large constant offset, so consecutive rows within
    a chunk are nearly identical.  A buffer that shuffles should break the
    correlation between successive rows of the batches it yields.
    """
    import torch

    chunks = [
        torch.full((32, 4), float(doc)) + 0.01 * torch.arange(32)[:, None]
        for doc in range(40)
    ]

    def correlation(rows: torch.Tensor) -> float:
        a = rows[:-1, 0]
        b = rows[1:, 0]
        if float(a.std()) == 0 or float(b.std()) == 0:
            return 1.0
        return float(torch.corrcoef(torch.stack([a, b]))[0, 1].abs())

    # Unshuffled, straight off the stream: successive rows are the same document.
    raw = torch.cat(chunks)[:256]
    assert correlation(raw) > 0.99

    buffer = ShuffleBuffer(
        iter(chunks), capacity=512, batch_size=64,
        generator=torch.Generator().manual_seed(0),
    )
    batches = [next(buffer) for _ in range(4)]
    shuffled = torch.cat(batches)
    assert correlation(shuffled) < 0.3, correlation(shuffled)
    assert buffer.rows_seen > 0


def test_shuffle_buffer_refuses_a_capacity_below_its_batch():
    """It would do no shuffling at all, which is the one thing it is for."""
    import torch

    with pytest.raises(ValueError, match="no shuffling"):
        ShuffleBuffer(iter([torch.zeros(8, 2)]), capacity=16, batch_size=32)


def test_shuffle_buffer_stops_when_the_stream_runs_out():
    import torch

    buffer = ShuffleBuffer(
        iter([torch.zeros(40, 2)]), capacity=32, batch_size=16,
        generator=torch.Generator().manual_seed(0),
    )
    drawn = list(buffer)
    assert 0 < len(drawn) <= 3
    assert buffer.exhausted


def test_human_readable_sizes_scale_to_tebibytes():
    """Exercises the top of the unit ladder, which a realistic budget never reaches."""
    store = ActivationStore(budget_bytes=1)
    with pytest.raises(ActivationBudgetExceeded) as excinfo:
        store.check_plan({"vast": ((1 << 41,), torch.float32)})
    assert "TiB" in str(excinfo.value)
