"""SSH host keys are trusted on first use and pinned for the transport's life."""

from __future__ import annotations

import gc
import stat
import sys
from pathlib import Path

import pytest

from stateset_agents.remote import runpod


def _option(opts: list[str], name: str) -> str:
    values = [
        opt.split("=", 1)[1]
        for flag, opt in zip(opts, opts[1:], strict=False)
        if flag == "-o" and opt.startswith(f"{name}=")
    ]
    assert len(values) == 1, (name, opts)
    return values[0]


def test_host_keys_are_accepted_once_and_pinned_per_transport():
    transport = runpod.SshTransport()
    opts = transport._base_opts()

    assert _option(opts, "StrictHostKeyChecking") == "accept-new"
    assert _option(opts, "GlobalKnownHostsFile") == "/dev/null"
    known_hosts = Path(_option(opts, "UserKnownHostsFile"))
    assert known_hosts != Path("/dev/null")
    assert known_hosts.is_file()
    assert "StrictHostKeyChecking=no" not in opts
    if sys.platform != "win32":
        assert stat.S_IMODE(known_hosts.stat().st_mode) == 0o600
        assert stat.S_IMODE(known_hosts.parent.stat().st_mode) == 0o700

    # Every later connection (reconnects included) reuses the same pin file.
    assert Path(_option(transport._base_opts(), "UserKnownHostsFile")) == known_hosts


def test_pin_files_are_never_shared_between_transports():
    first = runpod.SshTransport()
    second = runpod.SshTransport()
    attempt = first._for_deadline(10**9)

    files = {
        _option(t._base_opts(), "UserKnownHostsFile") for t in (first, second, attempt)
    }
    assert len(files) == 3


def test_pin_file_is_removed_with_its_transport():
    transport = runpod.SshTransport()
    directory = transport._known_hosts_file().parent
    assert directory.exists()

    del transport
    gc.collect()

    assert not directory.exists()


def test_known_hosts_path_with_whitespace_is_quoted(monkeypatch, tmp_path):
    spaced = tmp_path / "dir with space"

    def fake_mkdtemp(prefix=""):
        spaced.mkdir(mode=0o700)
        return str(spaced)

    monkeypatch.setattr(runpod.tempfile, "mkdtemp", fake_mkdtemp)
    opts = runpod.SshTransport()._base_opts()

    assert _option(opts, "UserKnownHostsFile") == f'"{spaced / "known_hosts"}"'


@pytest.mark.parametrize("operation", ["probe", "scp"])
def test_ssh_and_scp_use_the_pinned_options(monkeypatch, operation):
    calls: list[list[str]] = []

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return Done()

    monkeypatch.setattr(runpod.subprocess, "run", fake_run)
    transport = runpod.SshTransport()
    if operation == "probe":
        transport.wait_until_reachable("example.invalid", 22, 5)
    else:
        transport._host, transport._port = "example.invalid", 22
        transport.upload(Path("local.bin"), "/workspace/remote.bin")

    argv = calls[0]
    assert "StrictHostKeyChecking=accept-new" in argv
    assert f"UserKnownHostsFile={transport._known_hosts_file()}" in argv
