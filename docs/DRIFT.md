# Graph / evidence drift at v1

`seed_evidence.csv` carries 30 rows. `cake_graph_v1.yaml` declares 25 steps. Only 22
node IDs appear in both. The evidence file appears to have been recorded against a
later, richer graph than the one committed as v1.

## Evidence nodes with no step in v1

| node_id | belief_mean | reading |
|---|---|---|
| `s_oven` | 0.70 | almost certainly a rename of `s_bake` |
| `s_icing_consistency` | 0.40 | almost certainly a rename of `s_icing_texture` |
| `s_spot_wrong_stock` | 0.55 | almost certainly a rename of `s_check_stock` |
| `s_allergen_check` | 0.85 | genuinely new — vegan order, so this matters a lot |
| `s_substitute` | 0.75 | genuinely new — recovery step after a sourcing failure |
| `s_buy_icing` | 0.85 | genuinely new — looks like a *fallback branch*, not a serial step |
| `s_qc` | 0.75 | genuinely new — inspection before packaging |
| `s_confirm` | 0.95 | genuinely new — confirm delivery with the customer |

## v1 steps with no evidence

`s_check_stock`, `s_bake`, `s_icing_texture` — all three are the other half of the
rename pairs above. Once renames are applied, coverage is complete for the serial
steps and only the five new nodes need eliciting.

## Reconciliation plan

1. Cut `cake_graph_v2.yaml` adopting the evidence file's names as canonical, since
   the evidence is the thing that's expensive to re-gather.
2. Add `s_allergen_check` to `t1_plan`, `s_substitute` to `t2_source`,
   `s_qc` to `t5_assemble`, `s_confirm` to `t7_deliver`.
3. Decide how `s_buy_icing` composes. If it's a fallback for a failed
   `s_whip`/`s_icing_consistency`, it is **not** a serial step and must not be
   multiplied into the chain — it's a disjunction, and modelling it as serial will
   understate the closure probability. This is the one that needs a real decision
   rather than a rename.
4. Add a `renamed_from` field to steps in v2 so old evidence rows still resolve.
5. Once green, drop `--allow-drift` from the CI workflow.

## Why this matters beyond bookkeeping

The seed evidence is all `model_guess` from a single source at one point in time,
so the pooled priors are correlated in a way the current naive pooling ignores. Any
closure number produced before real elicitation should be read as a smoke test of
the pipeline, not a forecast.
