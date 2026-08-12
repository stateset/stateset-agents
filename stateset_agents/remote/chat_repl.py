"""The chat server side of ``stateset-agents chat-remote`` — runs ON the pod.

Launched over SSH by :class:`~stateset_agents.remote.chat_session.RemoteChatSession`
as ``python -m stateset_agents.remote.chat_repl --base-model X [--adapter D]``.
It loads the base model (plus an optional LoRA adapter), then speaks a
line-oriented JSON protocol on stdio:

* it prints ``{"ready": true}`` once the model can answer;
* each request is one ``{"prompt": ...}`` line on stdin;
* each answer is one ``{"response": ...}`` line on stdout;
* recoverable problems come back as ``{"error": ...}`` lines;
* EOF on stdin ends the session with exit code 0.

stdout **is** the protocol, so everything else — logging, transformers
download chatter, tqdm bars — is steered to stderr explicitly. One stray
print on stdout corrupts the channel and the local client sees garbage.

The conversation is multi-turn: a running ``messages`` list accumulates every
user/assistant pair and is re-rendered through the model's chat template on
each request, so the model sees its own earlier answers.

Imports only the installed package + transformers/peft/torch: a pod holds the
wheel, never a checkout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from typing import IO, Any

__all__ = ["build_generate_fn", "build_parser", "main", "serve"]

logger = logging.getLogger("chat_repl")

#: Signature of the generation seam ``serve`` loops over: full message
#: history in, one assistant reply out. ``main`` builds the real
#: model-backed one; tests substitute a fake.
GenerateFn = Callable[[list[dict[str, str]]], str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JSON-lines chat server for stateset-agents chat-remote."
    )
    parser.add_argument(
        "--base-model", required=True, help="Hugging Face base model name."
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA adapter directory to apply on top of the base model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Generation length cap per reply.",
    )
    return parser


def build_generate_fn(
    base_model: str, adapter: str | None, max_new_tokens: int
) -> GenerateFn:
    """Load the model and return the real chat-template + greedy generator.

    Greedy decoding, same as the post-train eval in
    :mod:`stateset_agents.training.sft`: this session exists to judge a
    fine-tune, and sampling noise would swamp the tuning signal.
    """
    import torch
    from transformers import AutoTokenizer

    from stateset_agents.training.sft import load_base_model_for_sft

    logger.info("Loading tokenizer and model: %s", base_model)
    # base_model is the caller's own CLI argument, not attacker input.
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True
    )  # nosec: B615
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_base_model_for_sft(base_model)

    if adapter:
        from peft import PeftModel

        logger.info("Applying LoRA adapter from %s", adapter)
        model = PeftModel.from_pretrained(model, adapter)

    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    def generate(messages: list[dict[str, str]]) -> str:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        return str(
            tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
        )

    return generate


def serve(generate: GenerateFn, stdin: IO[str], stdout: IO[str]) -> int:
    """The request loop: JSON lines in, JSON lines out, until EOF.

    Takes file objects rather than touching ``sys`` so it is unit-testable
    with ``StringIO`` on both ends. A failed generation is reported as an
    ``error`` line and rolled back out of the history — the session survives.
    """

    def emit(payload: dict[str, Any]) -> None:
        stdout.write(json.dumps(payload) + "\n")
        stdout.flush()

    emit({"ready": True})

    messages: list[dict[str, str]] = []
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"error": f"request is not valid JSON: {exc}"})
            continue
        prompt = request.get("prompt") if isinstance(request, dict) else None
        if not isinstance(prompt, str):
            emit({"error": "request must be a JSON object with a 'prompt' string"})
            continue

        messages.append({"role": "user", "content": prompt})
        try:
            reply = generate(messages)
        except Exception as exc:  # noqa: BLE001 — reported over the protocol
            messages.pop()  # keep history consistent with what was answered
            emit({"error": f"generation failed: {type(exc).__name__}: {exc}"})
            continue
        messages.append({"role": "assistant", "content": reply})
        emit({"response": reply})

    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # stdout is the protocol; every other writer goes to stderr.
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(message)s"
    )

    import contextlib

    # Model loading is where transformers is chattiest (download progress,
    # config warnings, remote-code notices) — none of it may touch stdout.
    with contextlib.redirect_stdout(sys.stderr):
        generate = build_generate_fn(args.base_model, args.adapter, args.max_new_tokens)

    return serve(generate, sys.stdin, sys.stdout)


if __name__ == "__main__":  # pragma: no cover — exercised over ssh on a pod
    sys.exit(main())
