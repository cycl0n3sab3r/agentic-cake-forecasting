# agentic-cake-forecasting

A toy end-to-end testbed for forecasting when AI agents can autonomously complete a
long, physically-grounded task chain — here, baking and delivering a vegan cake.

The cake is a stand-in. The point is the machinery: a task graph of atomic steps, a
Bayesian belief network over per-step autonomy probabilities, evidence accumulated
from expert elicitation and evals, and a front end for inspecting the result.

## Layout

```
data/
  graphs/        task-graph definitions (YAML), versioned by filename
  evidence/      per-node belief observations (CSV), append-only
src/cakecast/    Python package: graph loading, evidence, BBN model
scripts/         one-shot utilities (validation, import, export)
tests/           pytest suite
frontend/        web UI (not yet started)
docs/            design notes
.github/         CI
```

## Data model

**Graph.** A goal decomposes into tasks, each of which decomposes into steps. A step
is the atomic unit of belief: *can an AI system do this one step autonomously to a
professional standard?* Steps are flagged `measurable: true` (directly evaluable) or
`measurable: false` (latent — only observable through a stated `proxy`).

**Evidence.** Each row is one observation about one node:

| column | meaning |
|---|---|
| `node_id` | step ID it attaches to |
| `source_type` | `model_guess`, `expert`, `eval`, ... |
| `source_id` | which model / which expert / which eval run |
| `belief_mean` | central estimate of P(step succeeds autonomously) |
| `belief_strength` | confidence as a pseudo-count — higher is tighter |
| `recorded_at` | ISO date |
| `note` | free text |

`(mean, strength)` maps to a Beta prior as `Beta(mean·strength, (1−mean)·strength)`,
so `strength` reads directly as "worth about this many observations". Multiple rows
for one node are pooled.

Evidence is **append-only**: to revise a belief, add a new row rather than editing an
old one. History is the audit trail.

## Known issue: v1 graph / seed evidence drift

`seed_evidence.csv` was recorded against a later graph than `cake_graph_v1.yaml`.
Eight evidence node IDs have no matching step, and three steps have no evidence.
Some are renames, some are genuinely new nodes. See `docs/DRIFT.md`.

`scripts/validate_graph.py` fails on this by design. Resolve it by cutting a `v2`
graph that reconciles the two, then keep CI green from there.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/validate_graph.py data/graphs/cake_graph_v1.yaml data/evidence/seed_evidence.csv
pytest
```

## Roadmap

- [ ] Reconcile graph/evidence drift; cut `cake_graph_v2.yaml`
- [ ] Pyro BBN: per-step Beta posteriors, chain closure probability for the goal
- [ ] Expert elicitation intake → evidence rows
- [ ] Move evidence from CSV to a real database once row count justifies it
- [ ] Front end: graph view, per-node posteriors, sensitivity to individual steps
