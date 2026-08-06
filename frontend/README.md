# frontend

Not started yet.

Intended shape: a static site that reads exported JSON from the Python side
(graph + posteriors) and renders the task graph with per-node beliefs, so there
is no server to run and it can go on GitHub Pages or similar.

Suggested first step: add `scripts/export_json.py` writing to `frontend/public/`,
then pick a framework once we have some data.
