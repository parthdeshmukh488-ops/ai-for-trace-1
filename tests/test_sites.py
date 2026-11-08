"""The site table: names, shapes, index spaces, and expansion that cannot be empty."""

from __future__ import annotations

import pytest

from circuitry.model import sites


def test_all_sites_is_in_forward_order(tiny_config):
    names = sites.all_sites(tiny_config)
    assert names[0] == "embed"
    assert names[-1] == "logits"
    assert names.index("blocks.0.resid_pre") < names.index("blocks.0.attn.q")
    assert names.index("blocks.0.attn.pattern") < names.index("blocks.0.attn.z")
    assert names.index("blocks.0.resid_post") < names.index("blocks.1.resid_pre")


def test_expensive_sites_are_excluded_by_default(tiny_config):
    assert "blocks.0.attn.result" not in sites.all_sites(tiny_config)
    assert "blocks.0.attn.result" in sites.all_sites(tiny_config, include_expensive=True)


def test_a_wildcard_cannot_pull_in_an_expensive_site(tiny_config):
    """Reaching for attn.* must not quietly allocate n_heads times the memory."""
    got = sites.resolve("blocks.0.attn.*", tiny_config)
    assert "blocks.0.attn.result" not in got
    assert "blocks.0.attn.pattern" in got


def test_an_expensive_site_named_explicitly_is_honoured(tiny_config):
    assert sites.resolve("blocks.0.attn.result", tiny_config) == ("blocks.0.attn.result",)


def test_wildcard_with_include_expensive(tiny_config):
    got = sites.resolve("blocks.0.attn.*", tiny_config, include_expensive=True)
    assert "blocks.0.attn.result" in got


def test_glob_with_no_match_raises(tiny_config):
    """A silently-empty expansion is how an experiment measures the empty set."""
    with pytest.raises(sites.SiteNotFound, match="matched no site"):
        sites.resolve("blocks.*.attn.nonsense", tiny_config)


def test_no_match_error_hints_at_the_expensive_guard(tiny_config):
    with pytest.raises(sites.SiteNotFound, match="include_expensive"):
        sites.resolve("*.attn.result", tiny_config)


def test_layer_ranges_expand(tiny_config):
    assert sites.resolve("blocks.0-1.mlp.post", tiny_config) == (
        "blocks.0.mlp.post",
        "blocks.1.mlp.post",
    )


def test_reversed_range_raises(tiny_config):
    with pytest.raises(sites.SiteNotFound, match="wrong way round"):
        sites.resolve("blocks.1-0.mlp.post", tiny_config)


def test_resolve_deduplicates_and_orders(tiny_config):
    got = sites.resolve(["blocks.1.mlp.out", "blocks.0.mlp.out", "blocks.*.mlp.out"], tiny_config)
    assert got == ("blocks.0.mlp.out", "blocks.1.mlp.out")


def test_resolve_with_no_patterns_raises(tiny_config):
    with pytest.raises(sites.SiteNotFound, match="no patterns"):
        sites.resolve([], tiny_config)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("embed", ("pos",)),
        ("blocks.0.resid_post", ("pos",)),
        ("blocks.0.ln1.scale", ("pos",)),
        ("blocks.0.attn.z", ("pos", "head")),
        ("blocks.0.attn.result", ("pos", "head")),
        ("blocks.0.attn.pattern", ("head", "query", "key")),
        ("blocks.0.mlp.post", ("pos", "neuron")),
        ("logits", ("pos", "vocab")),
    ],
)
def test_index_spaces(name, expected):
    assert sites.index_space(name) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("embed", (3, 7, 16)),
        ("blocks.0.ln1.scale", (3, 7, 1)),
        ("blocks.0.attn.q", (3, 7, 2, 8)),
        ("blocks.0.attn.pattern", (3, 2, 7, 7)),
        ("blocks.0.attn.result", (3, 7, 2, 16)),
        ("blocks.0.mlp.pre", (3, 7, 32)),
        ("logits", (3, 7, 20)),
    ],
)
def test_shapes(tiny_config, name, expected):
    assert sites.shape(name, tiny_config, 3, 7) == expected


def test_axis_dim_is_uniform_across_layouts():
    """One rule for every site: index-space axes are the dims after batch, in order."""
    assert sites.axis_dim("blocks.0.resid_post", "pos") == 1
    assert sites.axis_dim("blocks.0.attn.z", "pos") == 1
    assert sites.axis_dim("blocks.0.attn.z", "head") == 2
    assert sites.axis_dim("blocks.0.attn.pattern", "head") == 1
    assert sites.axis_dim("blocks.0.attn.pattern", "query") == 2
    assert sites.axis_dim("blocks.0.attn.pattern", "key") == 3


def test_axis_dim_rejects_an_axis_the_site_does_not_have():
    with pytest.raises(sites.SiteNotFound, match="no axis"):
        sites.axis_dim("blocks.0.resid_post", "head")


def test_unknown_leaf_names_the_alternatives():
    with pytest.raises(sites.SiteNotFound, match="resid_post"):
        sites.parse("blocks.0.attn.probs")


def test_unknown_global_site_raises():
    with pytest.raises(sites.SiteNotFound, match="not a site name"):
        sites.parse("final_logits")


def test_out_of_range_layer_raises(tiny_config):
    with pytest.raises(sites.SiteNotFound, match="2 layers"):
        sites.shape("blocks.5.resid_post", tiny_config, 1, 1)
    with pytest.raises(sites.SiteNotFound, match="2 layers"):
        sites.validate(["blocks.5.resid_post"], tiny_config)


def test_parse_round_trip(tiny_config):
    for name in sites.all_sites(tiny_config, include_expensive=True):
        layer, leaf = sites.parse(name)
        assert name == (f"blocks.{layer}.{leaf}" if layer is not None else leaf)
