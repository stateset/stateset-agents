"""Allow ``python -m traceforge_rl ...`` to run the CLI.

Mirrors the ``traceforge-rl`` console script. See
``docs/RENAME_TRACEFORGE.md`` for the phased rename.
"""

from stateset_agents.cli import run

if __name__ == "__main__":
    run()
