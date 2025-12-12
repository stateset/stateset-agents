"""
API Versioning Strategy for StateSet Agents

This module provides a comprehensive API versioning system with:
- Semantic version tracking
- Deprecation warnings and sunset dates
- Version negotiation via headers
- Automatic migration helpers
- Change log generation

Usage:
    from api.versioning import APIVersion, VersionedRouter, deprecate

    router = VersionedRouter(current_version=APIVersion.V2)

    @router.get("/training")
    @deprecate(sunset="2025-06-01", replacement="/api/v2/training")
    async def training_v1():
        ...
"""

import functools
import logging
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple, Type, Union

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ============================================================================
# Version Definitions
# ============================================================================


class APIVersion(str, Enum):
    """Supported API versions."""

    V1 = "v1"
    V2 = "v2"

    @classmethod
    def latest(cls) -> "APIVersion":
        """Get the latest stable API version."""
        return cls.V2

    @classmethod
    def from_string(cls, version: str) -> "APIVersion":
        """Parse version string into APIVersion enum.

        Args:
            version: Version string like "v1", "v2", or "2"

        Returns:
            Corresponding APIVersion

        Raises:
            ValueError: If version string is not recognized
        """
        # Normalize input
        version = version.lower().strip()
        if not version.startswith("v"):
            version = f"v{version}"

        try:
            return cls(version)
        except ValueError:
            valid = [v.value for v in cls]
            raise ValueError(
                f"Unknown API version '{version}'. Valid versions: {valid}"
            )


@dataclass
class VersionInfo:
    """Information about a specific API version.

    Attributes:
        version: The API version
        release_date: When this version was released
        status: Current status (stable, deprecated, sunset)
        sunset_date: When this version will stop working (if deprecated)
        changelog_url: Link to version changelog
        breaking_changes: List of breaking changes from previous version
    """

    version: APIVersion
    release_date: date
    status: str = "stable"  # stable, deprecated, sunset
    sunset_date: Optional[date] = None
    changelog_url: Optional[str] = None
    breaking_changes: List[str] = field(default_factory=list)
    new_features: List[str] = field(default_factory=list)

    def is_deprecated(self) -> bool:
        """Check if this version is deprecated."""
        return self.status == "deprecated"

    def is_sunset(self) -> bool:
        """Check if this version has been sunset."""
        if self.status == "sunset":
            return True
        if self.sunset_date and date.today() >= self.sunset_date:
            return True
        return False

    def days_until_sunset(self) -> Optional[int]:
        """Get days until sunset (None if not deprecated)."""
        if not self.sunset_date:
            return None
        delta = self.sunset_date - date.today()
        return max(0, delta.days)


# Version registry
VERSION_INFO: Dict[APIVersion, VersionInfo] = {
    APIVersion.V1: VersionInfo(
        version=APIVersion.V1,
        release_date=date(2024, 1, 1),
        status="deprecated",
        sunset_date=date(2025, 6, 1),
        changelog_url="https://docs.stateset.ai/changelog/v1",
        breaking_changes=[],
        new_features=[
            "Initial release",
            "Training endpoints",
            "Conversation API",
        ],
    ),
    APIVersion.V2: VersionInfo(
        version=APIVersion.V2,
        release_date=date(2024, 6, 1),
        status="stable",
        changelog_url="https://docs.stateset.ai/changelog/v2",
        breaking_changes=[
            "Changed training response format",
            "Renamed 'prompt' to 'messages' in conversation API",
            "Authentication now required by default",
        ],
        new_features=[
            "Streaming responses",
            "Structured outputs",
            "Function calling",
            "Improved error messages",
        ],
    ),
}


# ============================================================================
# Deprecation Utilities
# ============================================================================


@dataclass
class DeprecationNotice:
    """Notice about a deprecated endpoint or feature.

    Attributes:
        message: Human-readable deprecation message
        sunset_date: When the feature will be removed
        replacement: New endpoint/feature to use instead
        migration_guide_url: Link to migration documentation
    """

    message: str
    sunset_date: Optional[date] = None
    replacement: Optional[str] = None
    migration_guide_url: Optional[str] = None

    def to_header_value(self) -> str:
        """Format as HTTP Deprecation header value."""
        parts = [f'@{self.sunset_date.isoformat()}' if self.sunset_date else "true"]
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "message": self.message,
            "sunset_date": self.sunset_date.isoformat() if self.sunset_date else None,
            "replacement": self.replacement,
            "migration_guide": self.migration_guide_url,
        }


def deprecate(
    sunset: Optional[str] = None,
    replacement: Optional[str] = None,
    message: Optional[str] = None,
    migration_guide: Optional[str] = None,
) -> Callable:
    """Decorator to mark an endpoint as deprecated.

    Adds deprecation headers to responses and logs warnings.

    Args:
        sunset: Date when endpoint will be removed (YYYY-MM-DD)
        replacement: New endpoint to use instead
        message: Custom deprecation message
        migration_guide: URL to migration documentation

    Example:
        @router.get("/old-endpoint")
        @deprecate(
            sunset="2025-06-01",
            replacement="/api/v2/new-endpoint",
            migration_guide="https://docs.example.com/migrate"
        )
        async def old_endpoint():
            ...
    """

    def decorator(func: Callable) -> Callable:
        # Parse sunset date
        sunset_date = None
        if sunset:
            try:
                sunset_date = datetime.strptime(sunset, "%Y-%m-%d").date()
            except ValueError:
                logger.warning(f"Invalid sunset date format: {sunset}")

        # Create deprecation notice
        notice = DeprecationNotice(
            message=message
            or f"This endpoint is deprecated"
            + (f" and will be removed on {sunset}" if sunset else ""),
            sunset_date=sunset_date,
            replacement=replacement,
            migration_guide_url=migration_guide,
        )

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Log deprecation warning
            logger.warning(
                f"Deprecated endpoint called: {func.__name__}",
                extra={
                    "endpoint": func.__name__,
                    "sunset_date": sunset,
                    "replacement": replacement,
                },
            )

            # Check if past sunset date
            if notice.sunset_date and date.today() >= notice.sunset_date:
                raise HTTPException(
                    status_code=410,
                    detail={
                        "error": "endpoint_sunset",
                        "message": f"This endpoint was removed on {notice.sunset_date}",
                        "replacement": replacement,
                        "migration_guide": migration_guide,
                    },
                )

            # Call original function
            response = await func(*args, **kwargs)

            # Add deprecation headers if it's a Response object
            if isinstance(response, Response):
                response.headers["Deprecation"] = notice.to_header_value()
                if replacement:
                    response.headers["Link"] = f'<{replacement}>; rel="successor-version"'
                if notice.sunset_date:
                    response.headers["Sunset"] = notice.sunset_date.isoformat()

            return response

        # Store deprecation info on function
        wrapper._deprecation_notice = notice
        return wrapper

    return decorator


# ============================================================================
# Version Negotiation
# ============================================================================


async def get_requested_version(
    request: Request,
    accept_version: Optional[str] = Header(None, alias="Accept-Version"),
    x_api_version: Optional[str] = Header(None, alias="X-API-Version"),
) -> APIVersion:
    """Extract requested API version from request headers.

    Checks headers in order of priority:
    1. Accept-Version header
    2. X-API-Version header
    3. URL path prefix
    4. Default to latest version

    Args:
        request: FastAPI request
        accept_version: Accept-Version header value
        x_api_version: X-API-Version header value

    Returns:
        Requested API version
    """
    # Check Accept-Version header first
    if accept_version:
        try:
            return APIVersion.from_string(accept_version)
        except ValueError as e:
            logger.warning(f"Invalid Accept-Version header: {e}")

    # Check X-API-Version header
    if x_api_version:
        try:
            return APIVersion.from_string(x_api_version)
        except ValueError as e:
            logger.warning(f"Invalid X-API-Version header: {e}")

    # Try to extract from URL path
    path = request.url.path
    for version in APIVersion:
        if f"/api/{version.value}/" in path or path.startswith(f"/{version.value}/"):
            return version

    # Default to latest
    return APIVersion.latest()


def add_version_headers(response: Response, version: APIVersion) -> Response:
    """Add version information headers to response.

    Args:
        response: Response to modify
        version: Current API version

    Returns:
        Modified response with version headers
    """
    info = VERSION_INFO.get(version)

    response.headers["X-API-Version"] = version.value
    response.headers["X-API-Latest-Version"] = APIVersion.latest().value

    if info:
        if info.is_deprecated():
            response.headers["Deprecation"] = "true"
            if info.sunset_date:
                response.headers["Sunset"] = info.sunset_date.isoformat()
                days = info.days_until_sunset()
                if days is not None and days < 30:
                    response.headers["Warning"] = (
                        f'299 - "This API version will be sunset in {days} days"'
                    )

    return response


# ============================================================================
# Versioned Router
# ============================================================================


class VersionedRouter(APIRouter):
    """APIRouter with built-in versioning support.

    Automatically handles version negotiation, deprecation headers,
    and version-specific route registration.

    Example:
        router = VersionedRouter(
            current_version=APIVersion.V2,
            prefix="/api/v2"
        )

        @router.get("/training")
        async def get_training(version: APIVersion = Depends(get_requested_version)):
            return {"version": version.value}
    """

    def __init__(
        self,
        current_version: APIVersion = APIVersion.latest(),
        **kwargs,
    ):
        """Initialize versioned router.

        Args:
            current_version: Version this router serves
            **kwargs: Additional APIRouter arguments
        """
        # Set default prefix if not provided
        if "prefix" not in kwargs:
            kwargs["prefix"] = f"/api/{current_version.value}"

        super().__init__(**kwargs)
        self.current_version = current_version
        self.version_info = VERSION_INFO.get(current_version)

        # Add version middleware
        self._version_endpoints: Dict[str, Set[APIVersion]] = {}

    def add_version_route(
        self,
        path: str,
        endpoint: Callable,
        versions: Optional[List[APIVersion]] = None,
        **kwargs,
    ):
        """Add a route with version constraints.

        Args:
            path: URL path
            endpoint: Endpoint function
            versions: API versions this endpoint supports
            **kwargs: Additional route arguments
        """
        versions = versions or [self.current_version]

        # Register endpoint for versions
        if path not in self._version_endpoints:
            self._version_endpoints[path] = set()
        self._version_endpoints[path].update(versions)

        # Wrap endpoint to add version headers
        @functools.wraps(endpoint)
        async def versioned_endpoint(
            request: Request,
            *args,
            **kw,
        ):
            version = await get_requested_version(request)

            # Check if version is supported for this endpoint
            if path in self._version_endpoints:
                if version not in self._version_endpoints[path]:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "version_not_supported",
                            "message": f"Version {version.value} not supported for this endpoint",
                            "supported_versions": [
                                v.value for v in self._version_endpoints[path]
                            ],
                        },
                    )

            # Call original endpoint
            response = await endpoint(request, *args, **kw)

            # Add version headers
            if isinstance(response, Response):
                add_version_headers(response, version)

            return response

        # Register route
        self.add_api_route(path, versioned_endpoint, **kwargs)


# ============================================================================
# Migration Helpers
# ============================================================================


@dataclass
class MigrationStep:
    """A single step in a version migration.

    Attributes:
        from_version: Source version
        to_version: Target version
        field_renames: Fields that were renamed {old: new}
        field_removals: Fields that were removed
        field_additions: Fields that were added
        transform_functions: Custom transformation functions
    """

    from_version: APIVersion
    to_version: APIVersion
    field_renames: Dict[str, str] = field(default_factory=dict)
    field_removals: List[str] = field(default_factory=list)
    field_additions: Dict[str, Any] = field(default_factory=dict)
    transform_functions: Dict[str, Callable] = field(default_factory=dict)


class RequestMigrator:
    """Helps migrate requests between API versions.

    Useful for accepting old request formats and transforming
    them to the current format.

    Example:
        migrator = RequestMigrator()
        migrator.add_step(MigrationStep(
            from_version=APIVersion.V1,
            to_version=APIVersion.V2,
            field_renames={"prompt": "messages"},
        ))

        # In endpoint:
        request_data = migrator.migrate(old_data, APIVersion.V1, APIVersion.V2)
    """

    def __init__(self):
        self.steps: Dict[tuple, MigrationStep] = {}

    def add_step(self, step: MigrationStep) -> None:
        """Add a migration step."""
        key = (step.from_version, step.to_version)
        self.steps[key] = step

    def migrate(
        self,
        data: Dict[str, Any],
        from_version: APIVersion,
        to_version: APIVersion,
    ) -> Dict[str, Any]:
        """Migrate data from one version to another.

        Args:
            data: Request/response data to migrate
            from_version: Current data version
            to_version: Target version

        Returns:
            Migrated data

        Raises:
            ValueError: If no migration path exists
        """
        if from_version == to_version:
            return data

        # Find migration path
        path = self._find_migration_path(from_version, to_version)
        if not path:
            raise ValueError(
                f"No migration path from {from_version.value} to {to_version.value}"
            )

        # Apply each step
        result = data.copy()
        for step in path:
            result = self._apply_step(result, step)

        return result

    def _find_migration_path(
        self,
        from_version: APIVersion,
        to_version: APIVersion,
    ) -> List[MigrationStep]:
        """Find migration steps to get from one version to another."""
        # Simple case: direct path exists
        key = (from_version, to_version)
        if key in self.steps:
            return [self.steps[key]]
        # Multi-hop search using BFS over available steps
        from collections import deque

        adjacency: Dict[APIVersion, List[Tuple[APIVersion, MigrationStep]]] = {}
        for (src, dst), step in self.steps.items():
            adjacency.setdefault(src, []).append((dst, step))

        queue: Deque[APIVersion] = deque([from_version])
        # Map version -> (previous_version, step_used_to_reach)
        prev: Dict[APIVersion, Tuple[Optional[APIVersion], Optional[MigrationStep]]] = {
            from_version: (None, None)
        }

        while queue:
            current = queue.popleft()
            for nxt, step in adjacency.get(current, []):
                if nxt in prev:
                    continue
                prev[nxt] = (current, step)
                if nxt == to_version:
                    queue.clear()
                    break
                queue.append(nxt)

        if to_version not in prev:
            return []

        # Reconstruct path
        path: List[MigrationStep] = []
        cur: APIVersion = to_version
        while True:
            parent, step = prev[cur]
            if parent is None or step is None:
                break
            path.append(step)
            cur = parent
        path.reverse()
        return path

    def _apply_step(
        self, data: Dict[str, Any], step: MigrationStep
    ) -> Dict[str, Any]:
        """Apply a single migration step."""
        result = data.copy()

        # Apply field renames
        for old_name, new_name in step.field_renames.items():
            if old_name in result:
                result[new_name] = result.pop(old_name)

        # Remove deprecated fields
        for field_name in step.field_removals:
            result.pop(field_name, None)

        # Add new fields with defaults
        for field_name, default_value in step.field_additions.items():
            if field_name not in result:
                result[field_name] = default_value

        # Apply custom transforms
        for field_name, transform in step.transform_functions.items():
            if field_name in result:
                result[field_name] = transform(result[field_name])

        return result


# ============================================================================
# Version Info Endpoint
# ============================================================================


def create_version_info_router() -> APIRouter:
    """Create router with version information endpoints.

    Returns:
        Router with /versions endpoint
    """
    router = APIRouter(tags=["Version Info"])

    @router.get("/versions")
    async def list_versions() -> Dict[str, Any]:
        """Get information about all API versions."""
        versions = []
        for version, info in VERSION_INFO.items():
            versions.append({
                "version": version.value,
                "release_date": info.release_date.isoformat(),
                "status": info.status,
                "sunset_date": info.sunset_date.isoformat() if info.sunset_date else None,
                "days_until_sunset": info.days_until_sunset(),
                "changelog_url": info.changelog_url,
                "breaking_changes": info.breaking_changes,
                "new_features": info.new_features,
            })

        return {
            "current_version": APIVersion.latest().value,
            "versions": versions,
        }

    @router.get("/versions/{version}")
    async def get_version_info(version: str) -> Dict[str, Any]:
        """Get detailed information about a specific API version."""
        try:
            api_version = APIVersion.from_string(version)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        info = VERSION_INFO.get(api_version)
        if not info:
            raise HTTPException(
                status_code=404,
                detail=f"No information available for version {version}",
            )

        return {
            "version": api_version.value,
            "release_date": info.release_date.isoformat(),
            "status": info.status,
            "is_latest": api_version == APIVersion.latest(),
            "sunset_date": info.sunset_date.isoformat() if info.sunset_date else None,
            "days_until_sunset": info.days_until_sunset(),
            "changelog_url": info.changelog_url,
            "breaking_changes": info.breaking_changes,
            "new_features": info.new_features,
        }

    return router


__all__ = [
    "APIVersion",
    "VersionInfo",
    "VERSION_INFO",
    "DeprecationNotice",
    "deprecate",
    "get_requested_version",
    "add_version_headers",
    "VersionedRouter",
    "MigrationStep",
    "RequestMigrator",
    "create_version_info_router",
]
