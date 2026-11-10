"""Declared lifecycles: the state machines this package persists, in one place.

Several long-running things here move through a small set of statuses: a
remote job, a fan-out run, a local fine-tune run, a gateway training job.
Each used to state its statuses, its terminal set and its legal moves only
implicitly, as string literals and frozensets repeated across the modules
that read and write them. Two copies drifting apart is how a cancelled run
came to be reported as ``unknown``.

A :class:`Lifecycle` states one machine as data. The modules that own a
status derive their constants from it, so there is one definition to change;
``stateset-agents lifecycles`` prints the same data for a person or an agent
deciding what a status means and what can follow it.

Declaring a lifecycle does not add enforcement. Where a writer already
refused illegal moves (``finetune_workspace.transition_state``) it still
does, by its own rules; :meth:`Lifecycle.require` is there for new code.

This module imports nothing from the package, so anything may import it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = [
    "FANOUT_RUN",
    "FINETUNE_RUN",
    "GATEWAY_TRAINING_JOB",
    "LIFECYCLES",
    "REMOTE_JOB",
    "Lifecycle",
    "get_lifecycle",
]


@dataclass(frozen=True)
class Lifecycle:
    """One state machine: its states, where it starts, ends, and may move.

    ``states`` maps each status string to a one-line meaning. ``transitions``
    maps a status to the statuses that may follow it; a status missing from it
    has no successors. ``terminal`` statuses are where polling stops. A
    terminal status can still have successors when the owner allows a recorded
    outcome to be corrected; ``settled`` is the subset that never changes
    again. ``owner`` names the module that writes the status and ``stored_in``
    says where a reader finds it.
    """

    name: str
    summary: str
    states: Mapping[str, str]
    initial: str
    terminal: frozenset[str]
    transitions: Mapping[str, frozenset[str]]
    owner: str
    stored_in: str

    def __post_init__(self) -> None:
        known = set(self.states)
        if self.initial not in known:
            raise ValueError(f"{self.name}: initial state {self.initial!r} is unknown")
        unknown = (set(self.terminal) | set(self.transitions)) - known
        for targets in self.transitions.values():
            unknown |= set(targets) - known
        if unknown:
            raise ValueError(f"{self.name}: undeclared states {sorted(unknown)}")
        reachable, frontier = {self.initial}, [self.initial]
        while frontier:
            for target in self.transitions.get(frontier.pop(), ()):
                if target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        if reachable != known:
            raise ValueError(
                f"{self.name}: unreachable states {sorted(known - reachable)}"
            )
        if not self.terminal:
            raise ValueError(f"{self.name}: a lifecycle needs a terminal state")
        object.__setattr__(self, "states", MappingProxyType(dict(self.states)))
        object.__setattr__(
            self,
            "transitions",
            MappingProxyType({k: frozenset(v) for k, v in self.transitions.items()}),
        )

    @property
    def all_states(self) -> frozenset[str]:
        return frozenset(self.states)

    @property
    def active(self) -> frozenset[str]:
        """Statuses a poller should keep watching."""
        return self.all_states - self.terminal

    @property
    def settled(self) -> frozenset[str]:
        """Terminal statuses with no successor: the outcome is permanent."""
        return frozenset(s for s in self.terminal if not self.transitions.get(s))

    def is_terminal(self, status: object) -> bool:
        return status in self.terminal

    def next_states(self, status: str) -> frozenset[str]:
        """Statuses that may follow ``status`` (empty for an unknown one)."""
        return self.transitions.get(status, frozenset())

    def can_transition(self, current: str, new: str) -> bool:
        """Whether ``current`` may become ``new``. Staying put is always legal."""
        return current == new or new in self.next_states(current)

    def require(self, current: str, new: str) -> None:
        """Raise ``ValueError`` naming the legal moves when this one is not."""
        if current not in self.states:
            raise ValueError(
                f"{self.name}: unknown status {current!r}; "
                f"known: {', '.join(sorted(self.states))}"
            )
        if not self.can_transition(current, new):
            allowed = ", ".join(sorted(self.next_states(current))) or "nothing"
            raise ValueError(
                f"{self.name}: {current!r} cannot become {new!r}; "
                f"it can become: {allowed}"
            )

    def describe(self) -> dict[str, Any]:
        """Plain data for JSON output, sorted so it is stable across runs."""
        return {
            "name": self.name,
            "summary": self.summary,
            "owner": self.owner,
            "stored_in": self.stored_in,
            "initial": self.initial,
            "states": [
                {
                    "status": status,
                    "meaning": meaning,
                    "terminal": status in self.terminal,
                    "settled": status in self.settled,
                    "next": sorted(self.next_states(status)),
                }
                for status, meaning in self.states.items()
            ],
        }


def _moves(**table: tuple[str, ...]) -> dict[str, frozenset[str]]:
    return {state: frozenset(targets) for state, targets in table.items()}


REMOTE_JOB = Lifecycle(
    name="remote-job",
    summary="A fine-tune job submitted to a remote provider (train-remote).",
    states={
        "pending": "Submitted; the provider has not started it.",
        "running": "The provider is running it.",
        "succeeded": "Finished; artifacts can be fetched.",
        "failed": "Finished with an error.",
        "cancelled": "Stopped on request.",
    },
    initial="pending",
    terminal=frozenset({"succeeded", "failed", "cancelled"}),
    transitions=_moves(
        pending=("running", "succeeded", "failed", "cancelled"),
        running=("succeeded", "failed", "cancelled"),
    ),
    owner="stateset_agents.remote.job",
    stored_in="JobStatus, returned by each executor's status()",
)

FANOUT_RUN = Lifecycle(
    name="fanout-run",
    summary="One run (an arm and a seed) of a fan-out.",
    states={
        "pending": "Waiting for a slot, or waiting to retry on a fresh sandbox.",
        "running": "An attempt is in progress.",
        "succeeded": "The workload finished and its result was collected.",
        "failed": "The workload failed, or its attempts ran out.",
        "cancelled": "Stopped on request.",
    },
    initial="pending",
    terminal=frozenset({"succeeded", "failed", "cancelled"}),
    transitions=_moves(
        pending=("running", "failed", "cancelled"),
        # Back to pending when an attempt is lost to preemption or an
        # infrastructure failure and another attempt remains.
        running=("pending", "succeeded", "failed", "cancelled"),
    ),
    owner="stateset_agents.fanout.orchestrator",
    stored_in="<run-dir>/runs/<run_id>/state.json",
)

_FINETUNE_STATES = (
    "prepared",
    "queued",
    "starting",
    "running",
    "interrupted",
    "succeeded",
    "failed",
    "cancelled",
)
_FINETUNE_OPEN = ("prepared", "queued", "starting", "running", "interrupted")

FINETUNE_RUN = Lifecycle(
    name="finetune-run",
    summary="A local fine-tune run in a workspace (finetune, the studio).",
    states={
        "prepared": "The run directory and plan exist; no worker has started.",
        "queued": "A worker is waiting for the device.",
        "starting": "A worker has started and is loading.",
        "running": "Training is in progress.",
        "interrupted": "The worker stopped without an outcome; it can be resumed.",
        "succeeded": "Training finished and the adapter was written.",
        "failed": "Training ended with an error.",
        "cancelled": "Stopped on request.",
    },
    initial="prepared",
    terminal=frozenset({"interrupted", "succeeded", "failed", "cancelled"}),
    # What transition_state allows: any status that is not a final outcome may
    # move anywhere; a success not yet confirmed by its artifacts may still be
    # replaced by a failure or a cancellation; failed and cancelled are fixed.
    transitions={
        **{
            state: frozenset(s for s in _FINETUNE_STATES if s != state)
            for state in _FINETUNE_OPEN
        },
        "succeeded": frozenset({"failed", "cancelled"}),
    },
    owner="stateset_agents.finetune_workspace",
    stored_in="<workspace>/runs/<run_id>/state.json",
)

GATEWAY_TRAINING_JOB = Lifecycle(
    name="gateway-training-job",
    summary="A training job submitted to the API gateway (POST /training).",
    states={
        "queued": "Accepted; waiting for a worker (durable queue only).",
        "running": "Training is in progress.",
        "cancelling": "A cancel was requested; the job is stopping.",
        "completed": "Finished successfully.",
        "failed": "Finished with an error.",
        "cancelled": "Stopped on request.",
    },
    initial="queued",
    terminal=frozenset({"completed", "failed", "cancelled"}),
    transitions=_moves(
        queued=("running", "failed", "cancelled"),
        running=("cancelling", "completed", "failed", "cancelled"),
        cancelling=("completed", "failed", "cancelled"),
    ),
    owner="stateset_agents.api.services.training_service",
    stored_in="the job record returned by GET /training/{job_id}",
)

LIFECYCLES: Mapping[str, Lifecycle] = MappingProxyType(
    {
        machine.name: machine
        for machine in (REMOTE_JOB, FANOUT_RUN, FINETUNE_RUN, GATEWAY_TRAINING_JOB)
    }
)


def get_lifecycle(name: str) -> Lifecycle:
    """The lifecycle called ``name``, or ``ValueError`` listing the known ones."""
    try:
        return LIFECYCLES[name]
    except KeyError:
        raise ValueError(
            f"unknown lifecycle {name!r}; known: {', '.join(sorted(LIFECYCLES))}"
        ) from None
