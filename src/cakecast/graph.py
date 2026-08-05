"""Loading and structural validation of task graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Step:
    """An atomic unit of belief: one thing an agent either can or cannot do."""

    id: str
    label: str
    measurable: bool
    proxy: str | None = None
    task_id: str = ""


@dataclass(frozen=True)
class Task:
    id: str
    label: str
    steps: tuple[Step, ...]


@dataclass
class TaskGraph:
    version: str
    title: str
    goal_id: str
    goal_label: str
    tasks: tuple[Task, ...]
    notes: str = ""
    _by_id: dict[str, Step] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {s.id: s for t in self.tasks for s in t.steps}

    @property
    def steps(self) -> tuple[Step, ...]:
        return tuple(s for t in self.tasks for s in t.steps)

    @property
    def step_ids(self) -> set[str]:
        return set(self._by_id)

    def step(self, step_id: str) -> Step:
        return self._by_id[step_id]

    @property
    def latent_steps(self) -> tuple[Step, ...]:
        return tuple(s for s in self.steps if not s.measurable)


class GraphError(ValueError):
    """Raised when a graph file is structurally invalid."""


def load_graph(path: str | Path) -> TaskGraph:
    """Parse a graph YAML file into a TaskGraph, raising GraphError on bad structure."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    for key in ("graph_version", "goal", "tasks"):
        if key not in raw:
            raise GraphError(f"{path.name}: missing top-level key {key!r}")

    goal = raw["goal"]
    tasks: list[Task] = []
    seen_step_ids: set[str] = set()

    for t in raw["tasks"]:
        steps: list[Step] = []
        for s in t.get("steps", []):
            sid = s["id"]
            if sid in seen_step_ids:
                raise GraphError(f"{path.name}: duplicate step id {sid!r}")
            seen_step_ids.add(sid)

            measurable = bool(s.get("measurable", False))
            proxy = s.get("proxy")
            if not measurable and not proxy:
                raise GraphError(
                    f"{path.name}: step {sid!r} is latent but declares no proxy"
                )
            steps.append(
                Step(
                    id=sid,
                    label=s["label"],
                    measurable=measurable,
                    proxy=proxy,
                    task_id=t["id"],
                )
            )
        tasks.append(Task(id=t["id"], label=t["label"], steps=tuple(steps)))

    graph = TaskGraph(
        version=raw["graph_version"],
        title=raw.get("title", ""),
        goal_id=goal["id"],
        goal_label=goal.get("label", ""),
        tasks=tuple(tasks),
        notes=raw.get("notes", "") or "",
    )

    declared_children = set(goal.get("children", []))
    actual_tasks = {t.id for t in tasks}
    if declared_children and declared_children != actual_tasks:
        missing = declared_children - actual_tasks
        extra = actual_tasks - declared_children
        raise GraphError(
            f"{path.name}: goal.children does not match tasks "
            f"(missing from tasks: {sorted(missing)}; "
            f"not listed under goal: {sorted(extra)})"
        )

    return graph
