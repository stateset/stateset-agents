"""Staging deployment lifecycle with durable idempotency and explicit rollback."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from stateset_agents.evaluation.agent_runs import content_hash

from .storage import private_directory, read_json, workspace_lock, write_json
from .workflow import Experiment


class DeploymentProvider(Protocol):
    """Provider methods must honor idempotency keys across client restarts."""

    evidence_mode: str

    def create(self, checkpoint: str, key: str) -> dict[str, Any]:
        """Create one staging replica, or return the previously accepted deployment."""
        ...

    def get(self, deployment_id: str) -> dict[str, Any]:
        """Return current checkpoint, endpoint, phase and replica readiness."""
        ...

    def stop(self, deployment_id: str, key: str) -> None:
        """Request scale-to-zero idempotently; completion must be monitored."""
        ...


class RiverDeploymentProvider:
    """Adapter for River's gated team deployment API; caller owns its SDK client."""

    evidence_mode = "live"

    def __init__(self, client: Any) -> None:
        self.client = client

    @staticmethod
    def _snapshot(deployment: Any) -> dict[str, Any]:
        replicas = deployment.replicas
        ready = (
            deployment.phase == "ready"
            and bool(replicas)
            and all(r.desired > 0 and r.ready >= r.desired for r in replicas.values())
        )
        return {
            "id": deployment.id,
            "checkpoint": deployment.checkpoint,
            "endpoint": deployment.base_url,
            "phase": deployment.phase,
            "ready": ready,
        }

    def create(self, checkpoint: str, key: str) -> dict[str, Any]:
        """Submit once with a durable key; readiness is checked separately."""
        return self._snapshot(
            self.client.create_deployment(
                checkpoint, unified_replicas=1, idempotency_key=key, wait=False
            )
        )

    def get(self, deployment_id: str) -> dict[str, Any]:
        """Treat degraded and zero-replica deployments as unready."""
        return self._snapshot(self.client.get_deployment(deployment_id))

    def stop(self, deployment_id: str, key: str) -> None:
        """Request zero replicas without claiming the workers have stopped yet."""
        self.client.scale_on_target(
            deployment_id, unified_replicas=0, idempotency_key=key
        )


class Deployment:
    """Private staging deployment and append-only endpoint routing journal.

    Applications must read ``endpoint()`` to use this routing decision. This
    journal does not change an external load balancer. Rollback records a healthy
    previous endpoint before requesting candidate scale-to-zero. No background
    monitor runs unless the operator schedules ``monitor()``.
    """

    def __init__(self, root: Path) -> None:
        self.root = private_directory(root)

    @classmethod
    def prepare(
        cls,
        root: Path,
        experiment: Experiment,
        *,
        previous_deployment_id: str | None = None,
    ) -> Deployment:
        """Prepare a reviewable staging plan without making provider requests."""
        approval = experiment.verified_approval()
        deployment = cls(root)
        if previous_deployment_id is not None and (
            not isinstance(previous_deployment_id, str)
            or not previous_deployment_id.strip()
        ):
            raise ValueError("Previous deployment identity must be nonempty")
        with workspace_lock(deployment.root):
            if any(p.name != ".workflow.lock" for p in deployment.root.iterdir()):
                raise ValueError("Deployment workspace must be empty")
            write_json(
                deployment.root / "plan.json",
                {
                    "schema_version": 1,
                    "experiment": str(experiment.root),
                    "approval_sha256": content_hash(approval),
                    "checkpoint": approval["checkpoint"],
                    "evidence_mode": approval["evidence_mode"],
                    "scope": "staging_only",
                    "previous_deployment_id": previous_deployment_id,
                    "replicas": 1,
                    "cost_usd": None,
                    "billing_limit_enforced": False,
                },
            )
        return deployment

    def _plan(self, provider: DeploymentProvider) -> dict[str, Any]:
        plan = read_json(self.root / "plan.json")
        approval = Experiment(Path(plan["experiment"])).verified_approval()
        if (
            plan.get("schema_version") != 1
            or plan.get("scope") != "staging_only"
            or plan.get("replicas") != 1
            or plan.get("approval_sha256") != content_hash(approval)
            or plan.get("checkpoint") != approval["checkpoint"]
            or provider.evidence_mode != approval["evidence_mode"]
            or plan.get("evidence_mode") != approval["evidence_mode"]
        ):
            raise ValueError(
                "Deployment must match the verified staging approval and evidence mode"
            )
        return dict(plan)

    def _retained_plan(self, provider: DeploymentProvider) -> dict[str, Any]:
        # Cleanup must remain possible after source/evidence changes invalidate approval.
        plan = read_json(self.root / "plan.json")
        if (
            plan.get("schema_version") != 1
            or plan.get("scope") != "staging_only"
            or plan.get("evidence_mode") != provider.evidence_mode
        ):
            raise ValueError("Provider does not match the retained staging deployment")
        return dict(plan)

    @staticmethod
    def _validate(
        snapshot: dict[str, Any],
        *,
        deployment_id: str | None = None,
        checkpoint: str | None = None,
    ) -> dict[str, Any]:
        if (
            not isinstance(snapshot, dict)
            or not isinstance(snapshot.get("id"), str)
            or not snapshot["id"]
            or type(snapshot.get("ready")) is not bool
        ):
            raise ValueError("Invalid deployment status")
        if deployment_id is not None and snapshot["id"] != deployment_id:
            raise ValueError("Provider returned a different deployment")
        if checkpoint is not None and snapshot.get("checkpoint") != checkpoint:
            raise ValueError("Provider returned a different checkpoint")
        if not isinstance(snapshot.get("checkpoint"), str) or not snapshot[
            "checkpoint"
        ].startswith("river://"):
            raise ValueError("Missing deployment checkpoint")
        if snapshot["ready"]:
            url = urlsplit(snapshot.get("endpoint", ""))
            if (
                snapshot.get("phase") != "ready"
                or url.scheme != "https"
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError(
                    "Serving deployment requires a credential-free HTTPS endpoint"
                )
        return {
            key: snapshot.get(key)
            for key in ("id", "checkpoint", "endpoint", "phase", "ready")
        }

    def deploy(
        self, provider: DeploymentProvider, *, reconcile: bool = False
    ) -> dict[str, Any]:
        """Create once; explicit reconciliation reuses the same provider key."""
        with workspace_lock(self.root):
            plan = self._plan(provider)
            path = self.root / "create-attempt.json"
            if (self.root / "created.json").exists():
                created = read_json(self.root / "created.json")
                return self._validate(
                    provider.get(created["id"]),
                    deployment_id=created["id"],
                    checkpoint=plan["checkpoint"],
                )
            if path.exists():
                attempt = read_json(path)
                if not reconcile:
                    raise ValueError(
                        "Creation outcome unknown; explicitly reconcile the retained key"
                    )
                if attempt["plan_sha256"] != content_hash(plan):
                    raise ValueError("Creation attempt belongs to a different plan")
            else:
                if reconcile:
                    raise ValueError("No creation attempt to reconcile")
                attempt = {"plan_sha256": content_hash(plan), "key": str(uuid.uuid4())}
                write_json(path, attempt)
            snapshot = self._validate(
                provider.create(plan["checkpoint"], attempt["key"]),
                checkpoint=plan["checkpoint"],
            )
            if snapshot["id"] == plan["previous_deployment_id"]:
                raise ValueError("Candidate and previous deployment must differ")
            write_json(self.root / "created.json", snapshot)
            return snapshot

    def _routes(self) -> list[Path]:
        return sorted(self.root.glob("route-*.json"))

    def _route(self, snapshot: dict[str, Any], reason: str) -> None:
        write_json(
            self.root / f"route-{len(self._routes()):06d}.json",
            {"deployment": snapshot, "reason": reason},
        )

    def activate(self, provider: DeploymentProvider) -> dict[str, Any]:
        """Route this SDK's endpoint reader to a healthy candidate once."""
        with workspace_lock(self.root):
            plan = self._plan(provider)
            if (self.root / "stop-attempt.json").exists():
                raise ValueError("Candidate shutdown was already requested")
            if self._routes():
                raise ValueError(
                    "A route already exists; activation cannot overwrite rollback"
                )
            created = read_json(self.root / "created.json")
            candidate = self._validate(
                provider.get(created["id"]),
                deployment_id=created["id"],
                checkpoint=plan["checkpoint"],
            )
            if not candidate["ready"]:
                raise ValueError("Candidate is not ready")
            self._route(candidate, "activate")
            return candidate

    def endpoint(self) -> str:
        """Read the last committed local route; contains no inference credential."""
        with workspace_lock(self.root):
            routes = self._routes()
            if not routes:
                raise ValueError("No active endpoint")
            snapshot = self._validate(read_json(routes[-1])["deployment"])
            if not snapshot["ready"]:
                raise ValueError("No serving route")
            return str(snapshot["endpoint"])

    def _stop(self, provider: DeploymentProvider, deployment_id: str) -> None:
        path = self.root / "stop-attempt.json"
        if path.exists():
            attempt = read_json(path)
            if attempt["id"] != deployment_id:
                raise ValueError("Stop attempt belongs to another deployment")
        else:
            attempt = {"id": deployment_id, "key": str(uuid.uuid4())}
            write_json(path, attempt)
        if not (self.root / "stop-requested.json").exists():
            provider.stop(deployment_id, attempt["key"])
            write_json(
                self.root / "stop-requested.json",
                {"id": deployment_id, "state": "requested_not_confirmed"},
            )

    def rollback(self, provider: DeploymentProvider) -> dict[str, Any]:
        """Route to the healthy previous deployment, then request candidate stop."""
        with workspace_lock(self.root):
            plan = self._retained_plan(provider)
            previous_id = plan["previous_deployment_id"]
            if not previous_id:
                raise ValueError("Rollback requires a previous deployment")
            previous = self._validate(
                provider.get(previous_id), deployment_id=previous_id
            )
            if not previous["ready"]:
                raise ValueError("Previous deployment is not ready; route is unchanged")
            created = read_json(self.root / "created.json")
            if created["id"] == previous_id:
                raise ValueError("Refusing to stop the rollback deployment")
            routes = self._routes()
            if not routes or read_json(routes[-1])["reason"] != "rollback":
                self._route(previous, "rollback")
            self._stop(provider, created["id"])
            return previous

    def stop(self, provider: DeploymentProvider) -> None:
        """Request candidate shutdown; use monitor to confirm provider state."""
        with workspace_lock(self.root):
            plan = self._retained_plan(provider)
            created = read_json(self.root / "created.json")
            if created["id"] == plan["previous_deployment_id"]:
                raise ValueError("Refusing to stop the rollback deployment")
            routes = self._routes()
            if routes and read_json(routes[-1])["deployment"]["id"] == created["id"]:
                self._route({**created, "ready": False, "phase": "stopping"}, "stop")
            self._stop(provider, created["id"])

    def monitor(
        self, provider: DeploymentProvider, *, rollback_on_failure: bool = False
    ) -> dict[str, Any]:
        """Retain one readiness observation; optionally roll back an active failure."""
        with workspace_lock(self.root):
            plan = self._retained_plan(provider)
            created = read_json(self.root / "created.json")
            snapshot = self._validate(
                provider.get(created["id"]),
                deployment_id=created["id"],
                checkpoint=plan["checkpoint"],
            )
            write_json(self.root / f"observation-{uuid.uuid4().hex}.json", snapshot)
            routes = self._routes()
            active = (
                bool(routes)
                and read_json(routes[-1])["deployment"]["id"] == created["id"]
            )
        if rollback_on_failure and active and not snapshot["ready"]:
            self.rollback(provider)
        return snapshot
