"""Export safetensors checkpoints to GGUF via llama.cpp's converter.

GGUF is the derived local-serving artifact in the dual-engine design
(``docs/NATIVE_LOCAL_ENGINE.md``). stateset-agents deliberately does not
grow a GGUF writer: conversion and quantization are delegated to the
llama.cpp tools that define the format, with command construction kept
explicit and testable.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CONVERT_OUTTYPES",
    "build_convert_command",
    "build_quantize_command",
    "convert_to_gguf",
    "quantize_gguf",
    "resolve_llama_cpp_dir",
]

CONVERT_OUTTYPES = ("f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto")
_CONVERTER_NAME = "convert_hf_to_gguf.py"


def resolve_llama_cpp_dir(llama_cpp_dir: str | Path | None = None) -> Path:
    """Resolve a llama.cpp checkout containing ``convert_hf_to_gguf.py``."""
    candidate_raw = llama_cpp_dir or os.getenv("LLAMA_CPP_DIR")
    if not candidate_raw:
        raise FileNotFoundError(
            "convert_hf_to_gguf.py not located: pass llama_cpp_dir or set "
            "LLAMA_CPP_DIR to a llama.cpp checkout"
        )
    candidate = Path(candidate_raw).expanduser().resolve()
    if not (candidate / _CONVERTER_NAME).is_file():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found under {candidate}; "
            "point LLAMA_CPP_DIR at a llama.cpp checkout"
        )
    return candidate


def build_convert_command(
    hf_model: str | Path,
    outfile: str | Path,
    *,
    llama_cpp_dir: str | Path | None = None,
    python_executable: str | None = None,
    outtype: str = "f16",
    mmproj: bool = False,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build the ``convert_hf_to_gguf.py`` argv for a local checkpoint."""
    if outtype not in CONVERT_OUTTYPES:
        raise ValueError(
            f"unsupported outtype {outtype!r}; choose one of {CONVERT_OUTTYPES}"
        )
    model_dir = Path(hf_model).expanduser()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"HF model directory not found: {model_dir}")
    checkout = resolve_llama_cpp_dir(llama_cpp_dir)
    command = [
        python_executable or sys.executable,
        str(checkout / _CONVERTER_NAME),
        str(model_dir),
        "--outfile",
        str(Path(outfile).expanduser()),
        "--outtype",
        outtype,
    ]
    if mmproj:
        command.append("--mmproj")
    command += [str(arg) for arg in extra_args]
    return command


def build_quantize_command(
    infile: str | Path,
    outfile: str | Path,
    quant_type: str,
    *,
    binary: str | None = None,
) -> list[str]:
    """Build the ``llama-quantize`` argv for one GGUF file."""
    if not str(quant_type).strip():
        raise ValueError("quant_type must be non-empty, e.g. 'IQ3_XXS'")
    resolved = binary or os.getenv("LLAMA_QUANTIZE_BIN") or "llama-quantize"
    return [
        resolved,
        str(Path(infile).expanduser()),
        str(Path(outfile).expanduser()),
        str(quant_type),
    ]


def _run(command: list[str], runner: Callable[[list[str]], Any] | None) -> None:
    if runner is not None:
        runner(command)
    else:
        subprocess.run(command, check=True)


def convert_to_gguf(
    hf_model: str | Path,
    outfile: str | Path,
    *,
    llama_cpp_dir: str | Path | None = None,
    python_executable: str | None = None,
    outtype: str = "f16",
    mmproj: bool = False,
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
    runner: Callable[[list[str]], Any] | None = None,
) -> dict[str, Any]:
    """Convert a local HF checkpoint to GGUF (or describe the command)."""
    command = build_convert_command(
        hf_model,
        outfile,
        llama_cpp_dir=llama_cpp_dir,
        python_executable=python_executable,
        outtype=outtype,
        mmproj=mmproj,
        extra_args=extra_args,
    )
    executed = False
    if not dry_run:
        Path(outfile).expanduser().parent.mkdir(parents=True, exist_ok=True)
        _run(command, runner)
        executed = True
        logger.info("converted %s -> %s", hf_model, outfile)
    return {
        "command": command,
        "outfile": str(Path(outfile).expanduser()),
        "executed": executed,
    }


def quantize_gguf(
    infile: str | Path,
    outfile: str | Path,
    quant_type: str,
    *,
    binary: str | None = None,
    dry_run: bool = False,
    runner: Callable[[list[str]], Any] | None = None,
) -> dict[str, Any]:
    """Quantize one GGUF file with ``llama-quantize`` (or describe it)."""
    command = build_quantize_command(infile, outfile, quant_type, binary=binary)
    executed = False
    if not dry_run:
        Path(outfile).expanduser().parent.mkdir(parents=True, exist_ok=True)
        _run(command, runner)
        executed = True
        logger.info("quantized %s -> %s (%s)", infile, outfile, quant_type)
    return {
        "command": command,
        "outfile": str(Path(outfile).expanduser()),
        "executed": executed,
    }
