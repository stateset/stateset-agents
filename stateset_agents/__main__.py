"""Allow ``python -m stateset_agents ...`` to run the CLI.

Mirrors the ``stateset-agents`` console script entry point so the package is
runnable without the script being on ``PATH``.
"""

from stateset_agents.cli import run

if __name__ == "__main__":
    run()
