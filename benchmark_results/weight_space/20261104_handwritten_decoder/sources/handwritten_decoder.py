"""A complete 100,616-parameter ReLU decoder with analytically written weights.

An 8-bit message specifies an affine Boolean function of seven bits. Evaluate
it at all 128 binary inputs and repeat the resulting word three times: a
384-bit code with minimum Hamming distance 192. This is a repeated RM(1, 7)
code; the network implements bounded-distance decoding, not a learned rule.

For received bipolar bits x, the neuron for codeword c computes
``relu(c @ x - 192)``. At <=95 flips, the true neuron is >=2 and every other
neuron is zero. The output matrix writes that neuron's eight message bits as
signed logits. Every matrix entry is +/-1, every hidden bias is -192, and
only the eight output biases are zero. There are no filler parameters.

Forward evaluation is an ordinary ``Linear(384, 256), ReLU, Linear(256, 8)``.
No optimizer, examples, pretrained weights or custom inference operations are
used to construct the handwritten model. The guarantee is for binary flips;
it does not cover arbitrary real-valued noise or more than 95 flips.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

MESSAGE_BITS = 8
CODE_BITS = 384
HIDDEN = 256
MIN_DISTANCE = 192
MAX_ERRORS = 95
PARAMETER_COUNT = 100_616


def message_bits() -> torch.Tensor:
    """Return all 256 messages as float32 bits, least significant bit first."""
    import torch

    return ((torch.arange(HIDDEN)[:, None] >> torch.arange(MESSAGE_BITS)) & 1).float()


def make_codebook() -> torch.Tensor:
    """Return the 256 bipolar codewords in message-integer order, on CPU."""
    import torch

    bits = message_bits().long()
    points = (torch.arange(128)[:, None] >> torch.arange(7)) & 1
    parity = (bits[:, :7] @ points.T + bits[:, 7:8]) % 2
    return (2 * parity.float() - 1).repeat(1, 3)


def make_decoder(*, handwritten: bool = True, seed: int = 0) -> torch.nn.Sequential:
    """Build the full decoder, or the identical architecture with random weights.

    The handwritten path writes every allocated parameter exactly once and
    consumes no random numbers. The random path preserves the caller's RNG
    state and uses PyTorch's standard Linear initialization for training.
    Both paths return a trainable CPU model in evaluation mode.
    """
    import torch
    from torch import nn

    if handwritten:
        # Meta construction avoids even an unused random initialization.
        model = nn.Sequential(
            nn.Linear(CODE_BITS, HIDDEN, device="meta"),
            nn.ReLU(),
            nn.Linear(HIDDEN, MESSAGE_BITS, device="meta"),
        ).to_empty(device="cpu")
        with torch.no_grad():
            model[0].weight.copy_(make_codebook())
            model[0].bias.fill_(-192)
            model[2].weight.copy_((2 * message_bits() - 1).T)
            model[2].bias.zero_()
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = nn.Sequential(
                nn.Linear(CODE_BITS, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, MESSAGE_BITS)
            )
    return model.eval()


def read_messages(logits: torch.Tensor) -> torch.Tensor:
    """Read eight signed logits as message integers; all-zero logits mean erasure.

    The handwritten model emits all zeros when no codeword is within 95 flips.
    Outside the guaranteed radius it can also decode to another message.
    This erasure convention is not a confidence guarantee for trained models.
    """
    import torch

    powers = 2 ** torch.arange(MESSAGE_BITS, device=logits.device)
    messages = ((logits > 0).long() * powers).sum(-1)
    return torch.where(logits.abs().amax(-1) > 0, messages, -1)
