"""The forwarder assertion helper must actually be able to fail.

``assert_forwards_to_driver`` replaced ten real subprocess spawns, so it has
to reject the things those subprocesses would have caught: a script that
isn't a forwarder at all, one that forwards to the wrong preset, and one
whose stderr notice is missing.
"""

from __future__ import annotations

import pytest

from tests.unit.forwarder_asserts import assert_forwards_to_driver


def test_accepts_a_real_forwarder():
    assert_forwards_to_driver("finetune_gpt_oss_gspo.py", model="gpt-oss")


def test_accepts_a_deprecated_forwarder():
    assert_forwards_to_driver(
        "finetune_kimi_k3_gspo.py", model="kimi-k3", notice_word="deprecated"
    )


def test_rejects_the_driver_itself():
    """finetune_gspo.py is the driver, not a forwarder onto it."""
    with pytest.raises(AssertionError, match="must forward to the unified driver"):
        assert_forwards_to_driver("finetune_gspo.py", model="gpt-oss")


def test_rejects_a_forwarder_pointed_at_the_wrong_preset():
    with pytest.raises(AssertionError, match="must forward to --model"):
        assert_forwards_to_driver("finetune_gpt_oss_gspo.py", model="kimi-k3")


def test_rejects_a_missing_script():
    with pytest.raises(AssertionError, match="does not exist"):
        assert_forwards_to_driver("finetune_no_such_model_gspo.py", model="gpt-oss")


def test_rejects_a_shim_with_no_stderr_notice(tmp_path, monkeypatch):
    """A forwarder that silently redirects gives the reader no way to learn
    the supported entry point, so the helper must reject it."""
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "quiet_forwarder.py").write_text(
        "import sys\n"
        "from examples.finetune_gspo import main as _driver_main\n"
        'sys.exit(_driver_main(["--model", "gpt-oss", *sys.argv[1:]]))\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("tests.unit.forwarder_asserts.REPO_ROOT", tmp_path)

    with pytest.raises(AssertionError, match="must print a notice to stderr"):
        assert_forwards_to_driver("quiet_forwarder.py", model="gpt-oss")


def test_rejects_a_shim_that_swallows_the_callers_flags(tmp_path, monkeypatch):
    """Dropping ``*sys.argv[1:]`` would make every flag a silent no-op."""
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "lossy_forwarder.py").write_text(
        "import sys\n"
        "from examples.finetune_gspo import main as _driver_main\n"
        'print("lossy_forwarder.py is a forwarder for gpt-oss", file=sys.stderr)\n'
        'sys.exit(_driver_main(["--model", "gpt-oss"]))\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("tests.unit.forwarder_asserts.REPO_ROOT", tmp_path)

    with pytest.raises(AssertionError, match=r"\*sys.argv\[1:\]"):
        assert_forwards_to_driver("lossy_forwarder.py", model="gpt-oss")


def _write(tmp_path, monkeypatch, name: str, body: str) -> None:
    examples = tmp_path / "examples"
    examples.mkdir(exist_ok=True)
    (examples / name).write_text(body, encoding="utf-8")
    monkeypatch.setattr("tests.unit.forwarder_asserts.REPO_ROOT", tmp_path)


HEAD = (
    "import sys\n"
    "from examples.finetune_gspo import main as _driver_main\n"
    'print("shim.py is a forwarder for gpt-oss", file=sys.stderr)\n'
)


def test_rejects_a_shim_that_stars_something_other_than_sys_argv(
    tmp_path, monkeypatch
):
    """``*extra`` is not ``*sys.argv[1:]``: the caller's flags never arrive."""
    _write(
        tmp_path,
        monkeypatch,
        "shim.py",
        HEAD + "extra = []\n"
        'sys.exit(_driver_main(["--model", "gpt-oss", *extra]))\n',
    )
    with pytest.raises(AssertionError, match=r"\*sys.argv\[1:\]"):
        assert_forwards_to_driver("shim.py", model="gpt-oss")


def test_rejects_a_shim_that_forwards_argv_zero(tmp_path, monkeypatch):
    """``*sys.argv`` passes the script's own path through as a positional."""
    _write(
        tmp_path,
        monkeypatch,
        "shim.py",
        HEAD + 'sys.exit(_driver_main(["--model", "gpt-oss", *sys.argv]))\n',
    )
    with pytest.raises(AssertionError, match=r"\*sys.argv\[1:\]"):
        assert_forwards_to_driver("shim.py", model="gpt-oss")


def test_rejects_a_shim_whose_forwarding_call_is_not_last(tmp_path, monkeypatch):
    """Code after the forward is dead, or the forward is conditional."""
    _write(
        tmp_path,
        monkeypatch,
        "shim.py",
        HEAD + 'sys.exit(_driver_main(["--model", "gpt-oss", *sys.argv[1:]]))\n'
        'print("unreachable")\n',
    )
    with pytest.raises(AssertionError, match="last top-level statement"):
        assert_forwards_to_driver("shim.py", model="gpt-oss")


def test_rejects_a_shim_that_drops_sys_exit(tmp_path, monkeypatch):
    """Without ``sys.exit`` the driver's exit code is thrown away."""
    _write(
        tmp_path,
        monkeypatch,
        "shim.py",
        HEAD + '_driver_main(["--model", "gpt-oss", *sys.argv[1:]])\n',
    )
    with pytest.raises(AssertionError, match="sys.exit"):
        assert_forwards_to_driver("shim.py", model="gpt-oss")
