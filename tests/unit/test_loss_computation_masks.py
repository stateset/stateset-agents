from types import SimpleNamespace

import torch


def test_prepare_inputs_and_labels_masks_assistant_tokens():
    """Assistant-token masking should translate into -100 labels for non-assistant tokens."""
    from stateset_agents.training.loss_computation import _prepare_inputs_and_labels

    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            *,
            tokenize: bool = False,
            add_generation_prompt: bool = False,
            return_tensors=None,
            return_dict: bool = False,
            return_assistant_tokens_mask: bool = False,
            **kwargs,
        ):
            assert tokenize is True
            assert return_dict is True
            assert return_assistant_tokens_mask is True
            _ = messages, kwargs

            # 5 tokens with a padding token at the end.
            input_ids = torch.tensor([[1, 2, 3, 4, 0]])
            attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
            # Only the "assistant" tokens (3, 4) should contribute to the loss.
            assistant_tokens_mask = torch.tensor([[0, 0, 1, 1, 0]])

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "assistant_tokens_mask": assistant_tokens_mask,
            }

        def __call__(self, prompt: str, **kwargs):
            raise AssertionError(f"Unexpected fallback tokenization: {prompt} {kwargs}")

    trajectory = SimpleNamespace(
        turns=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    agent = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        model=SimpleNamespace(device=torch.device("cpu")),
    )
    config = SimpleNamespace(max_prompt_length=8, max_completion_length=8)

    _, labels = _prepare_inputs_and_labels(trajectory, agent, config)

    assert torch.is_tensor(labels)
    assert labels.tolist() == [[-100, -100, 3, 4, -100]]
