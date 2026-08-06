# data

`graphs/` are task-graph definitions. Version in the filename, we should never edit a released version in place. `graph_version` inside the file must match.

`evidence/` is append-only observation rows. To revise a belief, add a new row with a
later `recorded_at`; do not edit or delete old rows.

Eventually we want to use a database not a CSV, i.e., a single `observations` table
with the same columns plus a surrogate key.
