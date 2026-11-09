"""Components, the edge space, and scoring a circuit against a known one."""

from __future__ import annotations

import json

import pytest

from circuitry import ModelConfig
from circuitry.attrib.circuit import (
    Circuit,
    Component,
    Edge,
    all_components,
    all_edges,
    component_of_node,
)
from circuitry.symbolic.truth import Node


def test_component_output_sites():
    assert Component.embed().output_site == "embed"
    assert Component.attn(3, 2).output_site == "blocks.3.attn.z"
    assert Component.mlp(1).output_site == "blocks.1.mlp.out"
    assert Component.logits(12).output_site == "logits"


def test_a_head_is_identified_by_index_not_by_slicing():
    """attn.z rather than attn.out, so one head can be patched without the rest."""
    assert Component.attn(0, 1).write_index == {"head": 1}
    assert Component.mlp(0).write_index == {}


def test_edge_space_sizes(tiny_config):
    assert len(all_components(tiny_config)) == 1 + tiny_config.n_layers * (
        tiny_config.n_heads + 1
    ) + 1
    # Every edge is a real dependency the architecture permits.
    assert len(all_edges(tiny_config)) == 46


def test_edge_space_is_the_precision_denominator():
    """It should be stated, not implicit in a loop, because precision depends on it."""
    cfg = ModelConfig(
        n_layers=12, n_heads=12, d_model=768, d_mlp=3072, d_vocab=50257, n_ctx=1024
    )
    assert len(all_edges(cfg)) == 32491


def test_an_mlp_can_read_its_own_layer_s_attention(tiny_config):
    edges = all_edges(tiny_config)
    assert Edge(Component.attn(1, 0), Component.mlp(1), "mlp_in") in edges
    # ... but a head cannot read its own layer's attention.
    assert Edge(Component.attn(1, 0), Component.attn(1, 1), "v") not in edges


def test_a_head_cannot_read_a_later_layer(tiny_config):
    edges = set(all_edges(tiny_config))
    assert Edge(Component.attn(1, 0), Component.attn(0, 0), "v") not in edges


def test_unknown_port_rejected():
    with pytest.raises(ValueError, match="port must be"):
        Edge(Component.embed(), Component.mlp(0), "value")


def test_a_port_that_does_not_match_its_receiver_is_rejected():
    """Not a nicety: the ablation engine looks edges up by (layer, port, head).

    An edge whose port does not match its target would never be found there,
    so a typo would quietly drop an edge from a circuit instead of raising.
    """
    with pytest.raises(ValueError, match="ports are"):
        Edge(Component.embed(), Component.attn(1, 0), "mlp_in")
    with pytest.raises(ValueError, match="ports are"):
        Edge(Component.embed(), Component.mlp(1), "q")
    with pytest.raises(ValueError, match="ports are"):
        Edge(Component.embed(), Component.logits(4), "v")


def test_the_embedding_takes_no_edges_and_the_sink_gives_none():
    with pytest.raises(ValueError, match="no inputs"):
        Edge(Component.mlp(0), Component.embed(), "resid")
    with pytest.raises(ValueError, match="no output"):
        Edge(Component.logits(4), Component.mlp(0), "mlp_in")


def test_component_of_node_round_trips():
    assert component_of_node(Node("embed")) == Component.embed()
    assert component_of_node(Node("blocks.2.attn.z", (("head", 3),))) == Component.attn(2, 3)
    assert component_of_node(Node("blocks.1.mlp.out")) == Component.mlp(1)


def test_component_of_node_rejects_the_logit_sink():
    with pytest.raises(ValueError, match="no layer"):
        component_of_node(Node("logits"))


def test_circuit_of_closes_over_edge_endpoints():
    edge = Edge(Component.embed(), Component.mlp(0), "mlp_in")
    circuit = Circuit.of(edges=[edge])
    assert Component.embed() in circuit.components
    assert Component.mlp(0) in circuit.components


def test_set_algebra(tiny_config):
    a = Circuit.of(edges=[Edge(Component.embed(), Component.mlp(0), "mlp_in")])
    b = Circuit.of(edges=[Edge(Component.embed(), Component.mlp(1), "mlp_in")])
    assert len((a | b).edges) == 2
    assert len((a & b).edges) == 0
    assert len((a - b).edges) == 1
    assert (a | b).jaccard(a) == pytest.approx(0.5)


def test_scoring_reports_nodes_and_edges_separately():
    """They disagree when the components are right and the wiring is not."""
    truth = Circuit.of(
        edges=[
            Edge(Component.embed(), Component.mlp(0), "mlp_in"),
            Edge(Component.mlp(0), Component.mlp(1), "mlp_in"),
        ]
    )
    # Same three components, one edge wrong.
    found = Circuit.of(
        edges=[
            Edge(Component.embed(), Component.mlp(0), "mlp_in"),
            Edge(Component.embed(), Component.mlp(1), "mlp_in"),
        ]
    )
    scores = found.score_against(truth)
    assert scores["node_precision"] == pytest.approx(1.0)
    assert scores["node_recall"] == pytest.approx(1.0)
    assert scores["edge_precision"] == pytest.approx(0.5)
    assert scores["edge_recall"] == pytest.approx(0.5)


def test_scoring_an_empty_circuit_is_zero_not_an_error():
    truth = Circuit.of(edges=[Edge(Component.embed(), Component.mlp(0), "mlp_in")])
    scores = Circuit.empty().score_against(truth)
    assert scores["edge_precision"] == 0.0
    assert scores["edge_recall"] == 0.0
    assert scores["edge_f1"] == 0.0


def test_complete_circuit_has_perfect_recall_and_poor_precision(tiny_config):
    truth = Circuit.of(edges=[Edge(Component.embed(), Component.mlp(0), "mlp_in")])
    scores = Circuit.complete(tiny_config).score_against(truth)
    assert scores["edge_recall"] == pytest.approx(1.0)
    assert scores["edge_precision"] < 0.05


def test_from_ground_truth_converts_a_compiled_graph():
    from circuitry import compile_program
    from circuitry.symbolic import programs

    program, vocab = programs.build("sort")
    compiled = compile_program(program, vocab, 4, verify=False)
    circuit = Circuit.from_ground_truth(compiled.graph, compiled.cfg)
    assert len(circuit.components) >= 3
    assert len(circuit.edges) >= 3
    assert circuit.score_against(circuit)["edge_f1"] == pytest.approx(1.0)


def test_serialisation_round_trips_through_json():
    circuit = Circuit.of(
        edges=[Edge(Component.attn(0, 1), Component.mlp(1), "mlp_in")], label="x"
    )
    payload = json.loads(circuit.to_json())
    assert payload["label"] == "x"
    assert payload["edges"][0]["source"] == "a0.h1"
    assert "a0.h1 -mlp_in-> m1" in circuit.describe()
