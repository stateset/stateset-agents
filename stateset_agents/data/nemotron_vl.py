"""Local image/chat data and loss masking for Nemotron Super VL and Bolt.

Media never passes through the text-only trajectory ingestion pipeline. Heavy
dependencies are loaded only when reading pixels or constructing a batch.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stateset_agents.core.chat_masking import assistant_char_spans

_RESERVED = ("<image>", "<video>", "<audio>", "<|im_start|>", "<|im_end|>")


class NemotronVLDataset:
    """Validate JSONL conversations eagerly and decode local images lazily.

    Rows contain ``messages`` (or NeMo-style ``conversation``), optionally ``id``.
    Content is text or a list of text/image blocks. Image blocks use an ``image``
    path relative to ``image_root`` (the JSONL directory by default). Paths must
    stay inside that root. Only system/user/assistant messages are accepted.
    """

    def __init__(self, path: str | Path, image_root: str | Path | None = None) -> None:
        self.path = Path(path).resolve(strict=True)
        self.image_root = (
            Path(image_root).resolve(strict=True) if image_root else self.path.parent
        )
        self.records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    messages = self._validate(row)
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    raise ValueError(f"{self.path}:{line_number}: {exc}") from exc
                self.records.append({"conversation": messages})
        if not self.records:
            raise ValueError(f"{self.path}: dataset is empty")

    def _validate(self, row: Any) -> list[dict[str, Any]]:
        if not isinstance(row, dict) or set(row) - {"id", "messages", "conversation"}:
            raise ValueError(
                "rows must contain messages or conversation, optionally id"
            )
        if ("messages" in row) == ("conversation" in row):
            raise ValueError("provide exactly one of messages and conversation")
        messages = row.get("messages", row.get("conversation"))
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        expected = "user"
        images = 0
        normalized = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError("messages require exactly role and content")
            role = message["role"]
            if role == "system" and index == 0:
                pass
            elif role != expected:
                raise ValueError(
                    "messages must alternate user/assistant after optional system"
                )
            else:
                expected = "assistant" if role == "user" else "user"
            content = message["content"]
            parts = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
            )
            if not isinstance(parts, list) or not parts:
                raise ValueError("content must be nonempty text or a list of blocks")
            blocks = []
            for part in parts:
                if not isinstance(part, dict):
                    raise ValueError("content blocks must be objects")
                if part.get("type") == "text" and set(part) == {"type", "text"}:
                    text = part["text"]
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("text blocks must be nonempty strings")
                    if any(token in text for token in _RESERVED):
                        raise ValueError(
                            "text contains a reserved media/chat control token"
                        )
                    blocks.append(dict(part))
                elif part.get("type") == "image" and set(part) == {"type", "image"}:
                    if role != "user" or not isinstance(part["image"], str):
                        raise ValueError(
                            "only user messages may contain local image paths"
                        )
                    image = (self.image_root / part["image"]).resolve(strict=True)
                    if not image.is_relative_to(self.image_root) or not image.is_file():
                        raise ValueError(
                            "image paths must name files inside image_root"
                        )
                    blocks.append({"type": "image", "image": str(image)})
                    images += 1
                else:
                    raise ValueError(
                        "supported content blocks are text and local image; video/audio are unsupported"
                    )
            normalized.append({"role": role, "content": blocks})
        if messages[-1]["role"] != "assistant" or not images:
            raise ValueError(
                "each conversation needs an image and a final assistant response"
            )
        return normalized

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        record = copy.deepcopy(self.records[index])
        for message in record["conversation"]:
            for block in message["content"]:
                if block["type"] == "image":
                    with Image.open(block["image"]) as image:
                        block["image"] = image.convert("RGB")
        return record


class NemotronVLCollator:
    """Preserve image tensors and supervise assistant tokens only.

    Prefix encodings must match the full processor encoding exactly. Overlong
    examples fail instead of truncating image placeholders or all supervision.
    Batch size one avoids assumptions about heterogeneous tiled-image batches;
    gradient accumulation and distributed training provide larger batches.
    """

    def __init__(self, processor: Any, max_length: int = 4096) -> None:
        if type(max_length) is not int or max_length < 2:
            raise ValueError("max_length must be an integer >= 2")
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if len(examples) != 1:
            raise ValueError("Nemotron VL requires per-device batch size 1")
        messages = []
        images = []
        for message in examples[0]["conversation"]:
            text = ""
            for block in message["content"]:
                if block["type"] == "image":
                    images.append(block["image"])
                    text += self.processor.image_token
                elif block["type"] == "text":
                    text += block["text"]
                else:
                    raise ValueError("unsupported media block")
            messages.append({"role": message["role"], "content": text})
        if not images:
            raise ValueError("Nemotron VL requires an image in each example")
        if len(images) > 1:
            # The released processors re-stack image outputs into BatchFeature,
            # and the released projector concatenates per-image embeddings.
            # Both require a common spatial/token shape. Pad rather than crop
            # or stretch; use the same padded images for every prefix encoding.
            from PIL import ImageOps

            width = max(image.width for image in images)
            height = max(image.height for image in images)
            padded = []
            for image in images:
                left = (width - image.width) // 2
                top = (height - image.height) // 2
                border = (
                    left,
                    top,
                    width - image.width - left,
                    height - image.height - top,
                )
                padded.append(ImageOps.expand(image, border=border, fill=(0, 0, 0)))
            images = padded

        def render(history: list[dict[str, Any]], generation: bool) -> str:
            return str(
                self.processor.tokenizer.apply_chat_template(
                    history,
                    tokenize=False,
                    add_generation_prompt=generation,
                )
            )

        located = assistant_char_spans(render, messages)
        if located is None or not located[1]:
            raise ValueError("cannot locate assistant spans in the model chat template")
        text, spans = located

        def encode(part: str) -> Any:
            count = part.count(self.processor.image_token)
            kwargs = {"images": images[:count]} if count else {}
            return self.processor(
                text=part,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=False,
                **kwargs,
            )

        encoded = encode(text)
        ids = encoded["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] > self.max_length:
            raise ValueError(
                "encoded conversation exceeds max_length; resize images or shorten text"
            )
        labels = torch.full_like(ids, -100)
        for start, end in spans:
            head, through = (
                encode(text[:start])["input_ids"],
                encode(text[:end])["input_ids"],
            )
            for prefix in (head, through):
                if prefix.shape[1] > ids.shape[1] or not torch.equal(
                    ids[:, : prefix.shape[1]], prefix
                ):
                    raise ValueError(
                        "processor prefix mismatch; cannot safely mask assistant loss"
                    )
            labels[:, head.shape[1] : through.shape[1]] = ids[
                :, head.shape[1] : through.shape[1]
            ]
        if not torch.any(labels[:, 1:] != -100):
            raise ValueError("conversation has no supervised next-token targets")
        pixels = encoded.get("pixel_values")
        if pixels is None:
            raise ValueError("processor discarded image pixels")
        if isinstance(pixels, torch.Tensor):
            pixels = pixels.to(torch.bfloat16)
        elif isinstance(pixels, (list, tuple)):
            pixels = torch.stack(list(pixels)).to(torch.bfloat16)
        else:
            raise ValueError("processor pixel_values must be tensors")
        return {
            "input_ids": ids,
            "attention_mask": encoded["attention_mask"],
            "pixel_values": pixels,
            "labels": labels,
        }


__all__ = ["NemotronVLDataset", "NemotronVLCollator"]
