# agentic-cake-forecasting

A toy end-to-end testbed for forecasting when AI agents can autonomously complete a
long, physically-grounded task chain; for now... baking and delivering a vegan cake.

We're making a vegan cake as a trivial example to test our method: a task graph of atomic steps, a
Bayesian belief network over per-step autonomous capability probabilities, evidence accumulated
from expert elicitation and evals, and a front end for interacting with the result.

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
`measurable: false` (i.e., latent, only observable through a stated `proxy`).

**Evidence.** Each row is one observation about one node:

| column | meaning |
|---|---|
| `node_id` | step ID it attaches to, e.g., mix ingredients step |
| `source_type` | `model_guess`, `expert`, `eval`, ... |
| `source_id` | which model / which expert / which eval run |
| `belief_mean` | central estimate of P(step succeeds autonomously) |
| `belief_strength` | confidence as a pseudo-count |
| `recorded_at` | ISO date |
| `note` | free text |

`(mean, strength)` maps to a Beta prior as `Beta(mean·strength, (1−mean)·strength)`,
so `strength` reads directly as "worth about this many observations". Multiple rows
for one node are pooled to get more robust estimates.

Evidence is **append-only**: to revise a belief, add a new row rather than editing an
old one (I reason this let's us keep track better).
