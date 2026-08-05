#!/usr/bin/env python3
"""Validate a task graph and cross-check it against an evidence file.

    python scripts/validate_graph.py data/graphs/cake_graph_v1.yaml \
                                     data/evidence/seed_evidence.csv

Exits non-zero if the graph is malformed or if graph and evidence disagree about
which nodes exist. Run in CI so the two files cannot drift apart silently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cakecast import (  # noqa: E402
    EvidenceError,
    GraphError,
    load_evidence,
    load_graph,
    pool_by_node,
    posterior_mean,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("evidence", type=Path, nargs="?")
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="report node mismatches as warnings instead of failing",
    )
    args = parser.parse_args()

    try:
        graph = load_graph(args.graph)
    except GraphError as exc:
        print(f"FAIL  {exc}")
        return 1

    print(f"Graph   {args.graph.name}  v{graph.version}")
    print(f"        {len(graph.tasks)} tasks, {len(graph.steps)} steps, "
          f"{len(graph.latent_steps)} latent")

    if args.evidence is None:
        print("OK      graph structure valid")
        return 0

    try:
        observations = load_evidence(args.evidence)
    except EvidenceError as exc:
        print(f"FAIL  {exc}")
        return 1

    priors = pool_by_node(observations)
    print(f"Evidence {args.evidence.name}  {len(observations)} rows, "
          f"{len(priors)} distinct nodes")

    evidence_ids = set(priors)
    orphaned = sorted(evidence_ids - graph.step_ids)
    uncovered = sorted(graph.step_ids - evidence_ids)

    if orphaned:
        print(f"\n  {len(orphaned)} evidence node(s) not present in the graph:")
        for nid in orphaned:
            print(f"    {nid:<22} mean {posterior_mean(*priors[nid]):.2f}")
    if uncovered:
        print(f"\n  {len(uncovered)} graph step(s) with no evidence:")
        for sid in uncovered:
            print(f"    {sid:<22} {graph.step(sid).label}")

    if orphaned or uncovered:
        if args.allow_drift:
            print("\nWARN    graph and evidence disagree (--allow-drift set)")
            return 0
        print("\nFAIL    graph and evidence disagree about which nodes exist")
        return 1

    print("\nOK      graph and evidence agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
