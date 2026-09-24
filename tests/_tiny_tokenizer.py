"""Hermetic tokenizer fixture for tiny transformer tests.

Unit and integration tests must not contact Hugging Face or depend on a warm
user cache.  This tokenizer is assembled entirely in memory and deliberately
keeps the vocabulary small so the accompanying random GPT-2 models stay fast.
"""

from __future__ import annotations

import string


def tiny_tokenizer(vocab_size: int = 256):
    """Return a local ``PreTrainedTokenizerFast`` with common test tokens."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import Fuse
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from transformers import PreTrainedTokenizerFast

    required = ["<pad>", "<eos>", "<unk>", *string.printable]
    if vocab_size < len(required):
        raise ValueError(f"vocab_size must be at least {len(required)}")
    tokens = required + [
        f"token_{index}" for index in range(vocab_size - len(required))
    ]
    backend = Tokenizer(
        WordLevel(
            {token: index for index, token in enumerate(tokens)}, unk_token="<unk>"
        )
    )
    # Character tokenization preserves the prompt-token prefix when the
    # scoring code concatenates prompt and response without a separator.
    backend.pre_tokenizer = Split(pattern="", behavior="isolated")
    backend.decoder = Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=128,
    )
