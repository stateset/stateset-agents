"""Lock consistency ignores generated metadata but not resolved packages."""

from pathlib import Path

import pytest

from scripts.check_lockfile_bodies import lock_body, main


def test_lock_body_ignores_generator_header_only(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text(
        "# Python 3.11\n# pip-compile\n\nnumpy==2.4.6\n    # via project\n"
    )
    second.write_text(
        "# Python 3.12\n# pip-compile --no-index\n\nnumpy==2.4.6\n    # via project\n"
    )
    assert lock_body(first) == lock_body(second)

    second.write_text("# Python 3.12\n\nnumpy==2.4.7\n    # via project\n")
    assert lock_body(first) != lock_body(second)


def test_lock_body_rejects_empty_lock(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("# generated\n")
    with pytest.raises(ValueError, match="no resolved packages"):
        lock_body(path)


@pytest.mark.parametrize("pairs", [1, 2, 3])
def test_every_requested_lock_pair_is_checked(tmp_path: Path, pairs: int) -> None:
    paths = []
    for index in range(pairs):
        for side in ("committed", "regenerated"):
            path = tmp_path / f"{index}-{side}.txt"
            path.write_text("# metadata\nnumpy==2.4.6\n")
            paths.append(str(path))
    assert main(paths) == 0
    Path(paths[-1]).write_text("# metadata\nnumpy==2.4.7\n")
    assert main(paths) == 1


@pytest.mark.parametrize("paths", [[], ["unpaired"]])
def test_lock_comparison_rejects_incomplete_pairs(paths: list[str]) -> None:
    with pytest.raises(SystemExit, match="expected pairs"):
        main(paths)
