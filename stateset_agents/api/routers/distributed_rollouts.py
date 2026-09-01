"""Authenticated HTTP transport for distributed rollout workers."""

from __future__ import annotations

from typing import Annotated, Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from stateset_agents.training.async_rollouts import (
    AsyncRolloutClosed,
    AsyncRolloutError,
    AsyncRolloutTimeout,
    RolloutRecord,
)
from stateset_agents.training.distributed_rollouts import (
    DistributedRolloutControlPlane,
    WorkerCapacityError,
    WorkerLease,
    WorkerLeaseError,
    WorkerLeaseExpired,
)
from stateset_agents.training.policy_artifacts import (
    PolicyArtifact,
    PolicyArtifactUnavailable,
)
from stateset_agents.utils.credentials import credential_fingerprint

from ..auth import AuthenticatedUser
from ..dependencies import get_strict_current_user

router = APIRouter(prefix="/api/v1/rollouts", tags=["distributed-rollouts"])

WorkerId = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
]
LeaseId = Annotated[str, Field(min_length=1, max_length=128)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LeaseRequest(_StrictModel):
    """Request carrying the current worker-generation lease."""

    lease_id: LeaseId


class RolloutValue(_StrictModel):
    """Wire representation of one policy-versioned rollout."""

    rollout_id: Annotated[str, Field(min_length=1, max_length=512)]
    policy_version: Annotated[int, Field(ge=0)]
    sampler_log_probs: Annotated[list[float], Field(min_length=1, max_length=1_000_000)]
    payload: dict[str, Any]
    policy_artifact_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = (
        None
    )


class SubmitRolloutRequest(LeaseRequest):
    """Lease and rollout submitted by a remote worker."""

    rollout: RolloutValue
    timeout_seconds: Annotated[float, Field(gt=0, le=300)] | None = None


class PolicyArtifactResponse(_StrictModel):
    """Content-addressed weights assigned to a remote worker."""

    policy_version: int
    uri: str
    sha256: str
    size_bytes: int
    published_at: float


class WorkerLeaseResponse(_StrictModel):
    """Renewable worker lease and exact sampling assignment."""

    worker_id: str
    lease_id: str
    generation: int
    policy_version: int
    issued_at: float
    expires_at: float
    artifact: PolicyArtifactResponse | None = None


class SubmitRolloutResponse(_StrictModel):
    """Admission result after lease and queue checks."""

    accepted: bool


class WorkerHealthResponse(_StrictModel):
    """Lease-ID-free worker health record."""

    worker_id: str
    generation: int
    policy_version: int
    lease_seconds_remaining: float


class WorkerHealthListResponse(_StrictModel):
    """Live workers visible to an administrator."""

    workers: list[WorkerHealthResponse]


class DistributedStatsResponse(_StrictModel):
    """Auditable distributed control-plane counters."""

    stats: dict[str, int]


async def get_rollout_control_plane(request: Request) -> DistributedRolloutControlPlane:
    """Return the training-owned control plane or report not-ready."""
    control_plane = getattr(
        request.app.state, "distributed_rollout_control_plane", None
    )
    if not isinstance(control_plane, DistributedRolloutControlPlane):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Distributed rollout control plane is not configured",
        )
    return control_plane


async def require_rollout_worker(
    user: AuthenticatedUser = Depends(get_strict_current_user),
) -> AuthenticatedUser:
    """Require a credential authorized to operate rollout workers."""
    if not {"rollout_worker", "trainer", "admin"}.intersection(user.roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Required rollout_worker, trainer, or admin",
        )
    return user


async def require_rollout_admin(
    user: AuthenticatedUser = Depends(get_strict_current_user),
) -> AuthenticatedUser:
    """Require administrative access to fleet-wide worker state."""
    if "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Required admin",
        )
    return user


def _scoped_worker_id(user: AuthenticatedUser, worker_id: str) -> str:
    """Bind an external worker ID to its authenticated principal."""
    owner = credential_fingerprint(user.user_id)
    return f"{owner}:{worker_id}"


def _lease_response(
    lease: WorkerLease,
    external_worker_id: str,
    artifact: PolicyArtifact | None,
) -> WorkerLeaseResponse:
    return WorkerLeaseResponse(
        worker_id=external_worker_id,
        lease_id=lease.lease_id,
        generation=lease.generation,
        policy_version=lease.policy_version,
        issued_at=lease.issued_at,
        expires_at=lease.expires_at,
        artifact=(
            PolicyArtifactResponse(**artifact.to_dict())
            if artifact is not None
            else None
        ),
    )


def _raise_transport_error(exc: Exception) -> NoReturn:
    if isinstance(exc, WorkerCapacityError):
        raise HTTPException(
            status_code=429, detail=str(exc), headers={"Retry-After": "1"}
        ) from exc
    if isinstance(exc, WorkerLeaseExpired):
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    if isinstance(exc, PolicyArtifactUnavailable):
        raise HTTPException(
            status_code=503, detail=str(exc), headers={"Retry-After": "1"}
        ) from exc
    if isinstance(exc, WorkerLeaseError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, AsyncRolloutTimeout):
        raise HTTPException(
            status_code=503, detail=str(exc), headers={"Retry-After": "1"}
        ) from exc
    if isinstance(exc, AsyncRolloutClosed):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, AsyncRolloutError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    raise exc


@router.post("/workers/{worker_id}/register", response_model=WorkerLeaseResponse)
async def register_worker(
    worker_id: WorkerId,
    user: AuthenticatedUser = Depends(require_rollout_worker),
    control: DistributedRolloutControlPlane = Depends(get_rollout_control_plane),
) -> WorkerLeaseResponse:
    """Register a remote worker and fence any prior process generation."""
    try:
        lease = await control.register(_scoped_worker_id(user, worker_id))
    except (WorkerCapacityError, WorkerLeaseError, PolicyArtifactUnavailable) as exc:
        _raise_transport_error(exc)
    artifact = await control.policy_artifact(lease.policy_version)
    return _lease_response(lease, worker_id, artifact)


@router.post("/workers/{worker_id}/heartbeat", response_model=WorkerLeaseResponse)
async def heartbeat_worker(
    worker_id: WorkerId,
    body: LeaseRequest,
    user: AuthenticatedUser = Depends(require_rollout_worker),
    control: DistributedRolloutControlPlane = Depends(get_rollout_control_plane),
) -> WorkerLeaseResponse:
    """Renew a worker lease and receive its latest policy assignment."""
    try:
        lease = await control.heartbeat(
            _scoped_worker_id(user, worker_id), body.lease_id
        )
    except (WorkerLeaseError, PolicyArtifactUnavailable) as exc:
        _raise_transport_error(exc)
    artifact = await control.policy_artifact(lease.policy_version)
    return _lease_response(lease, worker_id, artifact)


@router.post("/workers/{worker_id}/submit", response_model=SubmitRolloutResponse)
async def submit_rollout(
    worker_id: WorkerId,
    body: SubmitRolloutRequest,
    user: AuthenticatedUser = Depends(require_rollout_worker),
    control: DistributedRolloutControlPlane = Depends(get_rollout_control_plane),
) -> SubmitRolloutResponse:
    """Submit a rollout through lease, policy, lag, and deduplication gates."""
    value = body.rollout
    try:
        record = RolloutRecord(
            rollout_id=value.rollout_id,
            policy_version=value.policy_version,
            sampler_log_probs=tuple(value.sampler_log_probs),
            payload=value.payload,
            policy_artifact_sha256=value.policy_artifact_sha256,
        )
        accepted = await control.submit(
            _scoped_worker_id(user, worker_id),
            body.lease_id,
            record,
            timeout_seconds=body.timeout_seconds,
        )
    except (ValueError, WorkerLeaseError, AsyncRolloutError) as exc:
        if isinstance(exc, ValueError):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _raise_transport_error(exc)
    return SubmitRolloutResponse(accepted=accepted)


@router.delete("/workers/{worker_id}", status_code=204)
async def unregister_worker(
    worker_id: WorkerId,
    body: LeaseRequest,
    user: AuthenticatedUser = Depends(require_rollout_worker),
    control: DistributedRolloutControlPlane = Depends(get_rollout_control_plane),
) -> Response:
    """Release a worker without allowing stale owners to remove replacements."""
    try:
        await control.unregister(_scoped_worker_id(user, worker_id), body.lease_id)
    except WorkerLeaseError as exc:
        _raise_transport_error(exc)
    return Response(status_code=204)


@router.get("/workers", response_model=WorkerHealthListResponse)
async def list_worker_health(
    _user: AuthenticatedUser = Depends(require_rollout_admin),
    control: DistributedRolloutControlPlane = Depends(get_rollout_control_plane),
) -> WorkerHealthListResponse:
    """Return lease-ID-free fleet health to administrators."""
    workers = await control.health()
    return WorkerHealthListResponse(
        workers=[WorkerHealthResponse(**vars(worker)) for worker in workers]
    )


@router.get("/stats", response_model=DistributedStatsResponse)
async def get_distributed_stats(
    _user: AuthenticatedUser = Depends(require_rollout_admin),
    control: DistributedRolloutControlPlane = Depends(get_rollout_control_plane),
) -> DistributedStatsResponse:
    """Return fleet-wide admission and lifecycle counters."""
    return DistributedStatsResponse(stats=(await control.stats()).to_dict())


__all__ = ["get_rollout_control_plane", "router"]
