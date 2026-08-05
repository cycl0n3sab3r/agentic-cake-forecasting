"""Belief network over the task graph.

Stage 1 (here): analytic Monte Carlo over independent per-step Beta posteriors,
with no dependency beyond numpy. Good enough to get numbers on the board and to
sanity-check the graph.

Stage 2 (not yet written): a Pyro model that drops the independence assumption —
shared latent capability factors across steps, source-reliability parameters, and
proper handling of latent steps observed only through proxies. The interface below
is meant to survive that swap.

Caveat worth reading before believing any number out of here: 25 serial steps
multiplied under an independence assumption drives closure probability to ~1e-8
on the seed priors. That is a property of the modelling choice, not a finding.
Real agents retry, recover, and have correlated competence across steps — all
three push the true number up by orders of magnitude. Treat `chain_closure` as a
lower bound and a pipeline smoke test until the Pyro model lands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evidence import posterior_mean
from .graph import TaskGraph


@dataclass
class ClosureResult:
    """Probability that every step in the chain succeeds."""

    mean: float
    q05: float
    q95: float
    weakest_steps: list[tuple[str, float]]

    @property
    def log10_mean(self) -> float:
        return float(np.log10(self.mean)) if self.mean > 0 else float("-inf")

    def __str__(self) -> str:
        lines = [
            f"P(chain closes) = {self.mean:.3e}  "
            f"(90% CI {self.q05:.3e} – {self.q95:.3e})",
            f"                = 10^{self.log10_mean:.2f}",
            "Weakest steps:",
        ]
        lines += [f"  {sid:<22} {p:.3f}" for sid, p in self.weakest_steps]
        return "\n".join(lines)


def chain_closure(
    graph: TaskGraph,
    priors: dict[str, tuple[float, float]],
    n_samples: int = 20_000,
    seed: int = 0,
    top_k: int = 5,
) -> ClosureResult:
    """Sample per-step success probabilities and multiply along the chain.

    Steps absent from `priors` fall back to Beta(1, 1) — maximally uncertain,
    which will visibly tank the closure probability. That is intentional: an
    un-elicited step should look expensive, not free.
    """
    rng = np.random.default_rng(seed)
    step_ids = [s.id for s in graph.steps]

    draws = np.ones((n_samples, len(step_ids)))
    for i, sid in enumerate(step_ids):
        a, b = priors.get(sid, (1.0, 1.0))
        draws[:, i] = rng.beta(a, b, size=n_samples)

    closure = draws.prod(axis=1)

    means = [
        (sid, posterior_mean(*priors.get(sid, (1.0, 1.0)))) for sid in step_ids
    ]
    means.sort(key=lambda kv: kv[1])

    return ClosureResult(
        mean=float(closure.mean()),
        q05=float(np.quantile(closure, 0.05)),
        q95=float(np.quantile(closure, 0.95)),
        weakest_steps=means[:top_k],
    )
