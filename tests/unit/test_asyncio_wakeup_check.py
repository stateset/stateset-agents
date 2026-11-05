"""Local IPC permission failures must surface before starting SDK checks."""

import errno
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_asyncio_wakeup.py"
spec = importlib.util.spec_from_file_location("asyncio_wakeup_check", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def endpoints(monkeypatch):
    reader, writer = MagicMock(), MagicMock()
    reader.recv.return_value = b"\0"
    writer.send.return_value = 1
    monkeypatch.setattr(module.socket, "socketpair", lambda: (reader, writer))
    return reader, writer


def test_probe_sends_from_worker_and_closes_both_endpoints(endpoints):
    reader, writer = endpoints
    caller = threading.get_ident()

    def send(data):
        assert threading.get_ident() != caller
        assert data == b"\0"
        return 1

    writer.send.side_effect = send
    module.check_worker_wakeup()
    reader.__exit__.assert_called_once()
    writer.__exit__.assert_called_once()


def test_worker_only_permission_denial_is_reported_and_resources_close(endpoints):
    reader, writer = endpoints
    caller = threading.get_ident()

    def send(data):
        if threading.get_ident() != caller:
            raise PermissionError(errno.EPERM, "blocked by sandbox")
        return len(data)

    writer.send.side_effect = send
    assert writer.send(b"\0") == 1  # Main-thread checks alone would pass.
    with pytest.raises(RuntimeError, match="local worker-thread IPC") as caught:
        module.check_worker_wakeup()
    assert isinstance(caught.value.__cause__, PermissionError)
    assert "errno=1" in str(caught.value)
    reader.recv.assert_not_called()
    reader.__exit__.assert_called_once()
    writer.__exit__.assert_called_once()


@pytest.mark.parametrize("failure", ["short_write", "wrong_byte", "missing_byte"])
def test_incomplete_delivery_never_passes(endpoints, failure):
    reader, writer = endpoints
    if failure == "short_write":
        writer.send.return_value = 0
    elif failure == "wrong_byte":
        reader.recv.return_value = b"x"
    else:
        reader.recv.side_effect = BlockingIOError(errno.EAGAIN, "no byte")
    with pytest.raises((OSError, RuntimeError)):
        module.check_worker_wakeup()


def test_main_exits_nonzero_with_actionable_machine_readable_error(endpoints, capsys):
    _, writer = endpoints
    writer.send.side_effect = PermissionError(errno.EPERM, "blocked")
    assert module.main() == 2
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert "Worker-thread local socket" in report["reason"]


def test_main_success(endpoints, capsys):
    assert module.main() == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_import_needs_no_optional_sdk_or_tensor_libraries():
    code = (
        "import runpy, sys; runpy.run_path(sys.argv[1], run_name='probe'); "
        "assert not {'river_client', 'torch', 'transformers'} & sys.modules.keys()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
