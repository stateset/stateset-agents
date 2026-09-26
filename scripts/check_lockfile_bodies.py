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
    if len(args) != 4:
        raise SystemExit("expected committed/regenerated paths for both locks")
    for original, regenerated in ((args[0], args[1]), (args[2], args[3])):
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
