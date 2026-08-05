"""Loading evidence rows and pooling them into per-node Beta priors."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = {
    "node_id",
    "source_type",
    "source_id",
    "belief_mean",
    "belief_strength",
    "recorded_at",
    "note",
}


@dataclass(frozen=True)
class Observation:
    """One recorded belief about one node, from one source."""

    node_id: str
    source_type: str
    source_id: str
    belief_mean: float
    belief_strength: float
    recorded_at: str
    note: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.belief_mean <= 1.0:
            raise ValueError(
                f"{self.node_id}: belief_mean {self.belief_mean} outside [0, 1]"
            )
        if self.belief_strength <= 0:
            raise ValueError(
                f"{self.node_id}: belief_strength must be positive, "
                f"got {self.belief_strength}"
            )

    @property
    def beta_params(self) -> tuple[float, float]:
        """(alpha, beta) for this observation alone.

        Strength is a pseudo-count: mean 0.9 at strength 10 carries the weight of
        roughly ten observations, nine of them successes.
        """
        a = self.belief_mean * self.belief_strength
        b = (1.0 - self.belief_mean) * self.belief_strength
        return a, b


class EvidenceError(ValueError):
    """Raised when an evidence file is malformed."""


def load_evidence(path: str | Path) -> list[Observation]:
    path = Path(path)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - cols
        if missing:
            raise EvidenceError(f"{path.name}: missing columns {sorted(missing)}")

        observations: list[Observation] = []
        for lineno, row in enumerate(reader, start=2):
            try:
                observations.append(
                    Observation(
                        node_id=row["node_id"].strip(),
                        source_type=row["source_type"].strip(),
                        source_id=row["source_id"].strip(),
                        belief_mean=float(row["belief_mean"]),
                        belief_strength=float(row["belief_strength"]),
                        recorded_at=row["recorded_at"].strip(),
                        note=(row.get("note") or "").strip(),
                    )
                )
            except (ValueError, KeyError) as exc:
                raise EvidenceError(f"{path.name} line {lineno}: {exc}") from exc

    return observations


def pool_by_node(
    observations: list[Observation],
    prior: tuple[float, float] = (1.0, 1.0),
) -> dict[str, tuple[float, float]]:
    """Pool observations per node into a single Beta(alpha, beta).

    Naive conjugate pooling: pseudo-counts add. This treats every source as
    independent and equally trustworthy, which is wrong in the ways you would
    expect — correlated experts get double-counted, and a confident model guess
    can outweigh a real eval. Source weighting belongs here later.
    """
    pooled: dict[str, list[float]] = defaultdict(lambda: list(prior))
    for obs in observations:
        a, b = obs.beta_params
        pooled[obs.node_id][0] += a
        pooled[obs.node_id][1] += b
    return {node: (ab[0], ab[1]) for node, ab in pooled.items()}


def posterior_mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)
