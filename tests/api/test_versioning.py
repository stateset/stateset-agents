"""Tests for API version migration helpers."""

from enum import Enum

import pytest

from api.versioning import APIVersion, MigrationStep, RequestMigrator


class DummyVersion(str, Enum):
    """Local enum to simulate multiple API versions in tests."""

    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


def test_multi_hop_migration_applies_steps() -> None:
    migrator = RequestMigrator()
    migrator.add_step(
        MigrationStep(
            from_version=DummyVersion.V1,  # type: ignore[arg-type]
            to_version=DummyVersion.V2,  # type: ignore[arg-type]
            field_renames={"prompt": "messages"},
        )
    )
    migrator.add_step(
        MigrationStep(
            from_version=DummyVersion.V2,  # type: ignore[arg-type]
            to_version=DummyVersion.V3,  # type: ignore[arg-type]
            field_additions={"new_field": 123},
        )
    )

    data = {"prompt": "hello"}
    migrated = migrator.migrate(
        data, DummyVersion.V1, DummyVersion.V3  # type: ignore[arg-type]
    )
    assert migrated == {"messages": "hello", "new_field": 123}


def test_migration_no_path_raises() -> None:
    migrator = RequestMigrator()
    migrator.add_step(
        MigrationStep(
            from_version=APIVersion.V1,
            to_version=APIVersion.V2,
        )
    )
    with pytest.raises(ValueError):
        migrator.migrate({}, APIVersion.V2, APIVersion.V1)

