#!/usr/bin/env python3
"""Check local worker-thread IPC before running asynchronous SDK tests."""

from __future__ import annotations

import json
import socket
import threading


def check_worker_wakeup() -> None:
    """Require a worker to deliver a byte through a local socket pair.

    Some sandboxes allow socket writes on the main thread but deny them on
    worker threads. Asyncio suppresses these wakeup write errors, so completed
    executor work can appear stuck until another timer wakes the event loop.
    This probe opens no listener and makes no external network connection.
    """
    failures: list[OSError] = []
    reader, writer = socket.socketpair()
    with reader, writer:
        reader.setblocking(False)
        writer.setblocking(False)

        def send() -> None:
            try:
                if writer.send(b"\0") != 1:
                    raise OSError("Incomplete local wakeup write")
            except OSError as exc:
                failures.append(exc)

        worker = threading.Thread(target=send, daemon=True)
        worker.start()
        worker.join(timeout=2)
        if worker.is_alive():
            raise RuntimeError(
                "Local worker wakeup probe did not finish within 2 seconds"
            )
        if failures:
            error = failures[0]
            raise RuntimeError(
                "Worker-thread local socket writes failed "
                f"({type(error).__name__}, errno={error.errno}). "
                "Asyncio executor work may stall. Run the checks in an environment "
                "that permits local worker-thread IPC; external network access "
                "is not required by this probe."
            ) from error
        if reader.recv(1) != b"\0":
            raise RuntimeError("Local worker wakeup byte was not delivered")


def main() -> int:
    """Print a machine-readable result and fail before starting SDK checks."""
    try:
        check_worker_wakeup()
    except (OSError, RuntimeError) as exc:
        print(json.dumps({"passed": False, "reason": str(exc)}))
        return 2
    print(json.dumps({"passed": True, "probe": "worker_thread_local_socketpair"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
