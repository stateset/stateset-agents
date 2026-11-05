"""Compare resolved lock entries while ignoring pip-compile's local header."""

import sys
from pathlib import Path


def lock_body(path: Path) -> tuple[str, ...]:
    """Return every line from the first resolved package onward."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for offset, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith("#"):
            return tuple(lines[offset:])
    raise ValueError(f"lock file has no resolved packages: {path}")


def main(args: list[str]) -> int:
    """Check pairs of committed and regenerated lock files."""
    if not args or len(args) % 2:
        raise SystemExit("expected pairs of committed/regenerated lock paths")
    for original, regenerated in zip(args[::2], args[1::2], strict=True):
        try:
            if lock_body(Path(original)) != lock_body(Path(regenerated)):
                print(f"Resolved lock entries differ: {regenerated}", file=sys.stderr)
                return 1
        except (OSError, ValueError) as exc:
            print(f"Lock comparison failed: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
