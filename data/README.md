# data

`graphs/` — task-graph definitions. Version in the filename, never edit a released
version in place; cut a new file. `graph_version` inside the file must match.

`evidence/` — append-only observation rows. To revise a belief, add a new row with a
later `recorded_at`; do not edit or delete old rows. The history is the audit trail.

Once evidence outgrows CSV, the migration target is a single `observations` table
with the same columns plus a surrogate key. Keep the CSV loader working as an
importer.
