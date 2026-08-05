from pathlib import Path

import pytest

from cakecast import (
    Observation,
    chain_closure,
    load_evidence,
    load_graph,
    pool_by_node,
    posterior_mean,
)

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "data" / "graphs" / "cake_graph_v1.yaml"
EVIDENCE = ROOT / "data" / "evidence" / "seed_evidence.csv"


@pytest.fixture(scope="module")
def graph():
    return load_graph(GRAPH)


@pytest.fixture(scope="module")
def observations():
    return load_evidence(EVIDENCE)


def test_graph_loads(graph):
    assert graph.version == "0.1.0"
    assert len(graph.tasks) == 7
    assert len(graph.steps) == 25


def test_goal_children_match_tasks(graph):
    assert {t.id for t in graph.tasks} == {
        "t1_plan", "t2_source", "t3_sponge",
        "t4_icing", "t5_assemble", "t6_package", "t7_deliver",
    }


def test_every_latent_step_declares_a_proxy(graph):
    for step in graph.latent_steps:
        assert step.proxy, f"{step.id} is latent with no proxy"


def test_step_ids_are_unique(graph):
    ids = [s.id for s in graph.steps]
    assert len(ids) == len(set(ids))


def test_evidence_loads(observations):
    assert len(observations) == 30
    assert all(0.0 <= o.belief_mean <= 1.0 for o in observations)


def test_beta_params_recover_the_mean():
    obs = Observation("s_x", "expert", "e1", 0.8, 10, "2026-08-05")
    a, b = obs.beta_params
    assert posterior_mean(a, b) == pytest.approx(0.8)
    assert a + b == pytest.approx(10)


def test_pooling_adds_pseudocounts():
    obs = [
        Observation("s_x", "expert", "e1", 0.8, 10, "2026-08-05"),
        Observation("s_x", "expert", "e2", 0.6, 10, "2026-08-05"),
    ]
    a, b = pool_by_node(obs, prior=(0.0, 0.0))["s_x"]
    assert a + b == pytest.approx(20)
    assert posterior_mean(a, b) == pytest.approx(0.7)


def test_rejects_out_of_range_mean():
    with pytest.raises(ValueError):
        Observation("s_x", "expert", "e1", 1.4, 10, "2026-08-05")


def test_rejects_nonpositive_strength():
    with pytest.raises(ValueError):
        Observation("s_x", "expert", "e1", 0.5, 0, "2026-08-05")


def test_closure_is_a_probability(graph, observations):
    result = chain_closure(graph, pool_by_node(observations), n_samples=2000)
    assert 0.0 <= result.mean <= 1.0
    assert result.q05 <= result.mean <= result.q95
    assert len(result.weakest_steps) == 5


def test_closure_falls_with_chain_length(graph, observations):
    """A longer chain of independent steps cannot be more likely to close."""
    priors = pool_by_node(observations)
    full = chain_closure(graph, priors, n_samples=4000, seed=1).mean
    assert full <= max(posterior_mean(*priors.get(s.id, (1.0, 1.0))) for s in graph.steps)


# Known drift between v1 graph and seed evidence — see docs/DRIFT.md.
# When v2 is cut and the two reconcile, flip these to assert equality.
EXPECTED_ORPHANED = {
    "s_allergen_check", "s_spot_wrong_stock", "s_substitute", "s_oven",
    "s_icing_consistency", "s_buy_icing", "s_qc", "s_confirm",
}
EXPECTED_UNCOVERED = {"s_check_stock", "s_bake", "s_icing_texture"}


def test_drift_is_exactly_as_documented(graph, observations):
    evidence_ids = set(pool_by_node(observations))
    assert evidence_ids - graph.step_ids == EXPECTED_ORPHANED
    assert graph.step_ids - evidence_ids == EXPECTED_UNCOVERED
