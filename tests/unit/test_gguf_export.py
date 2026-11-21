"""GGUF export command tests; no real conversion runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from stateset_agents.training.gguf_export import (
    build_convert_command,
    build_quantize_command,
    convert_to_gguf,
)


def _fake_llama_cpp(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "convert_hf_to_gguf.py").write_text("# converter")
    return root


def test_build_convert_command_uses_converter_contract(tmp_path: Path) -> None:
    llama = _fake_llama_cpp(tmp_path / "llama.cpp")
    (tmp_path / "llama.cpp").mkdir(exist_ok=True)
    model = tmp_path / "model"
    model.mkdir()

    command = build_convert_command(
        model,
        tmp_path / "model.gguf",
        llama_cpp_dir=llama,
        python_executable="/usr/bin/python3",
        outtype="bf16",
    )

    assert command[:3] == [
        "/usr/bin/python3",
        str(llama / "convert_hf_to_gguf.py"),
        str(model),
    ]
    assert "--outfile" in command
    assert command[command.index("--outtype") + 1] == "bf16"


def test_convert_rejects_unknown_outtype_and_missing_converter(
    tmp_path: Path,
) -> None:
    llama = _fake_llama_cpp(tmp_path / "llama.cpp")
    (tmp_path / "llama.cpp").mkdir(exist_ok=True)
    model = tmp_path / "model"
    model.mkdir()

    with pytest.raises(ValueError, match="outtype"):
        build_convert_command(
            model,
            tmp_path / "o.gguf",
            llama_cpp_dir=llama,
            outtype="q3_k",
        )
    with pytest.raises(FileNotFoundError, match="convert_hf_to_gguf"):
        build_convert_command(
            model,
            tmp_path / "o.gguf",
            llama_cpp_dir=tmp_path / "missing",
        )


def test_convert_to_gguf_dry_run_and_runner(tmp_path: Path) -> None:
    llama = _fake_llama_cpp(tmp_path / "llama.cpp")
    (tmp_path / "llama.cpp").mkdir(exist_ok=True)
    model = tmp_path / "model"
    model.mkdir()
    calls: list[list[str]] = []

    dry = convert_to_gguf(
        model,
        tmp_path / "o.gguf",
        llama_cpp_dir=llama,
        dry_run=True,
        runner=lambda cmd: calls.append(list(cmd)),
    )
    assert dry["executed"] is False
    assert calls == []

    done = convert_to_gguf(
        model,
        tmp_path / "o.gguf",
        llama_cpp_dir=llama,
        runner=lambda cmd: calls.append(list(cmd)),
    )
    assert done["executed"] is True
    assert done["command"] == calls[0]


def test_build_quantize_command(tmp_path: Path) -> None:
    command = build_quantize_command(
        tmp_path / "in.gguf",
        tmp_path / "out.gguf",
        "IQ3_XXS",
        binary="/opt/llama-quantize",
    )
    assert command == [
        "/opt/llama-quantize",
        str(tmp_path / "in.gguf"),
        str(tmp_path / "out.gguf"),
        "IQ3_XXS",
    ]
