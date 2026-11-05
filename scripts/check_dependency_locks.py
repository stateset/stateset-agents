"""Resolve dependency locks without rewriting their checkout files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.check_lockfile_bodies import lock_body

ROOT = Path(__file__).resolve().parents[1]
LOCKS = (
    ("requirements-lock.txt", ()),
    ("requirements-dev-lock.txt", ("--extra=dev", "--extra=api")),
    (
        "requirements-training-lock.txt",
        (
            "--extra=training",
            "--extra=api",
            "--constraint=requirements-dev-lock.txt",
        ),
    ),
)


def _unchanged(root: Path, original: dict[str, bytes]) -> bool:
    try:
        return all(
            not (root / name).is_symlink()
            and (root / name).is_file()
            and (root / name).read_bytes() == data
            for name, data in original.items()
        )
    except OSError:
        return False


def check_locks(root: Path, *, compiler: str | None = None, timeout: int = 600) -> int:
    """Check all locks using isolated outputs and bounded resolver processes.

    Return zero only when every resolved body matches and the input files remain
    unchanged. Resolver failures, stale locks and concurrent edits return one;
    missing inputs/tools and invalid options return two. Never restore over a
    caller's concurrent edit. Interrupts propagate after temporary-file cleanup.
    """
    if type(timeout) is not int or timeout < 1:
        print("Lock resolver timeout must be a positive integer", file=sys.stderr)
        return 2
    if compiler is None:
        sibling = Path(sys.executable).with_name(
            "pip-compile.exe" if sys.platform == "win32" else "pip-compile"
        )
        executable = shutil.which(str(sibling)) or shutil.which("pip-compile")
    else:
        executable = shutil.which(compiler)
    if executable is None:
        print(f"Lock compiler not found: {compiler or 'pip-compile'}", file=sys.stderr)
        return 2
    root = root.resolve()
    try:
        names = ("pyproject.toml", *(name for name, _ in LOCKS))
        if any(
            (root / name).is_symlink() or not (root / name).is_file() for name in names
        ):
            raise ValueError("Lock inputs must be regular files, not symlinks")
        original = {name: (root / name).read_bytes() for name in names}
        with TemporaryDirectory(prefix="stateset-lockcheck-") as directory:
            temporary = Path(directory)
            for name, data in original.items():
                (temporary / name).write_bytes(data)
            for name, options in LOCKS:
                output = temporary / name
                # Existing copied locks retain the resolver's pinned preferences.
                # The repository dev constraint is safe to use only while all
                # captured inputs remain unchanged; check before and after work.
                if not _unchanged(root, original):
                    print(
                        "Dependency lock inputs changed during validation",
                        file=sys.stderr,
                    )
                    return 1
                result = subprocess.run(
                    [
                        executable,
                        "--quiet",
                        "--resolver=backtracking",
                        *options,
                        f"--output-file={output}",
                        "pyproject.toml",
                    ],
                    cwd=root,
                    timeout=timeout,
                    check=False,
                )
                if result.returncode:
                    print(
                        f"Lock resolver failed for {name} (exit {result.returncode})",
                        file=sys.stderr,
                    )
                    return 1
                if not _unchanged(root, original):
                    print(
                        "Dependency lock inputs changed during validation",
                        file=sys.stderr,
                    )
                    return 1
                # Compare against the captured body, not a later checkout read.
                expected = temporary / "expected.txt"
                expected.write_bytes(original[name])
                if lock_body(expected) != lock_body(output):
                    print(f"Resolved lock entries differ: {name}", file=sys.stderr)
                    return 1
    except subprocess.TimeoutExpired:
        print(
            "Dependency lock resolver timed out; working locks preserved",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError) as exc:
        print(f"Dependency lock validation failed: {exc}", file=sys.stderr)
        return 2
    print("Lock files are in sync with pyproject.toml; working locks preserved.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Validate checkout locks without requiring an active virtual environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--compiler", help="Override the current Python's pip-compile")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    try:
        return check_locks(args.root, compiler=args.compiler, timeout=args.timeout)
    except KeyboardInterrupt:
        print(
            "Dependency lock validation interrupted; working locks preserved",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
