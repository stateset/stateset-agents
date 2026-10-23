"""Tool observations never receive policy gradient (docs/TOOL_USE_RL.md).

Episodes come from a real ``ToolUseEnvironment`` (calculator tool) driven by a
tiny random GPT-2 ``MultiTurnAgent`` whose reply text is scripted but whose
recorded ``prompt_token_ids`` / ``token_ids`` / ``sampler_log_probs`` are
exactly what ``generate_turn`` records. The proof covers every way the
multi-turn trainer and SFT build a loss mask:

* the native token path (``compute_grpo_loss`` on sampled ids): the gradient
  of the loss with respect to the logits is non-zero exactly at the positions
  that predict the policy's own reply tokens, the tool-call text included,
  and zero at every system, user and tool-observation token;
* the sequence path (``_prepare_inputs_and_labels``) without a chat template,
  with a ChatML template that renders the ``tool`` role, and with a
  ``{% generation %}`` template: only assistant text is labelled;
* a template whose prefix property fails: a conversation with tool turns is
  skipped (nothing labelled) instead of training on the whole stream;
* ``core.chat_masking`` (SFT): ``assistant_token_mask`` / ``final_turn_token_mask``
  / ``supervised_turn_sequences`` with ``chat_template_render(tokenizer, tools)``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.core.chat_masking import (  # noqa: E402
    OBSERVATION_ROLES,
    assistant_token_mask,
    chat_template_render,
    final_turn_token_mask,
    has_observation_turns,
    supervised_turn_sequences,
)
from stateset_agents.core.environments.tool_use import ToolUseEnvironment  # noqa: E402
from stateset_agents.core.tool_registry import (  # noqa: E402
    ToolRegistry,
    calculator_tool,
)
from stateset_agents.rewards.math_verifier import MathAnswerVerifier  # noqa: E402
from stateset_agents.training.config import TrainingConfig  # noqa: E402
from stateset_agents.training.loss_computation import (  # noqa: E402
    _prepare_inputs_and_labels,
    _token_rows_for_group,
    compute_grpo_loss,
)
from stateset_agents.training.multi_turn_trainer import (  # noqa: E402
    MultiTurnGRPOTrainer,
)
from tests._tool_use_helpers import (  # noqa: E402
    SYSTEM_PROMPT,
    ForcedTurnAgent,
    calculator_script,
    make_agent,
    make_model,
    make_tokenizer,
    tool_call,
)

QUESTION = "What is 12*7?"
SCENARIO = {"id": "mul", "question": QUESTION, "ground_truth": "84"}
SCRIPTS = [
    [tool_call("12*7"), "The answer is \\boxed{84}."],
    [tool_call("12 * 7"), "The answer is \\boxed{85}."],
    ['<tool_call>{"name": "calculator"', "\\boxed{84}"],
    ["\\boxed{83}"],
]


def _config(**overrides: Any) -> TrainingConfig:
    values: dict[str, Any] = {
        "num_episodes": 1,
        "num_generations": len(SCRIPTS),
        "learning_rate": 1e-3,
        "gradient_accumulation_steps": 1,
        "max_prompt_length": 512,
        "max_completion_length": 128,
        "beta": 0.0,
        "bf16": False,
        "fp16": False,
        "report_to": "none",
        "seed": 0,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def _rollout(seed: int = 0) -> tuple[ForcedTurnAgent, list[Any]]:
    tokenizer = make_tokenizer()
    agent = ForcedTurnAgent(
        make_agent(make_model(seed), tokenizer), calculator_script(SCRIPTS)
    )
    environment = ToolUseEnvironment(
        [SCENARIO],
        [calculator_tool()],
        max_tool_calls=2,
        system_prompt=SYSTEM_PROMPT,
    )
    trainer = MultiTurnGRPOTrainer(
        agent=agent,
        environment=environment,
        reward_fn=MathAnswerVerifier(),
        config=_config(),
    )
    groups = asyncio.run(
        trainer.generate_trajectories([SCENARIO], num_generations=len(SCRIPTS))
    )
    return agent, groups


@pytest.fixture(scope="module")
def rollout() -> tuple[ForcedTurnAgent, list[Any]]:
    return _rollout()


def _observations(trajectory: Any) -> list[str]:
    return [t.content for t in trajectory.turns if t.role == "tool"]


def test_episodes_contain_tool_turns_and_verifier_rewards(rollout):
    _, groups = rollout
    [group] = groups
    roles = [[t.role for t in traj.turns] for traj in group.trajectories]
    assert roles[0] == ["system", "user", "assistant", "tool", "assistant"]
    assert roles[3] == ["system", "user", "assistant"]
    assert [traj.total_reward for traj in group.trajectories] == [1.0, 0.0, 1.0, 0.0]
    first = group.trajectories[0]
    assert first.final_state["tool_log"][0]["result"] == 84
    malformed = group.trajectories[2]
    assert '"malformed_call"' in _observations(malformed)[0]


def test_token_rows_put_observations_in_context_only(rollout):
    agent, groups = rollout
    [group] = groups
    decode = agent.tokenizer.decode
    rows = _token_rows_for_group(group)
    assert rows is not None
    assistant = [
        (ti, t)
        for ti, traj in enumerate(group.trajectories)
        for t in traj.turns
        if t.role == "assistant"
    ]
    assert len(rows) == len(assistant) == 7
    for (_prompt, response, ti), (turn_ti, turn) in zip(rows, assistant, strict=True):
        assert ti == turn_ti
        # The response is exactly the policy's reply, tool-call text included.
        assert decode(response) == turn.content
        for observation in _observations(group.trajectories[ti]):
            assert observation not in decode(response)
    # The second turn of each tool episode conditions on the observation.
    for ti in (0, 1, 2):
        observation = _observations(group.trajectories[ti])[0]
        second = [r for r in rows if r[2] == ti][1]
        assert f"Tool: {observation}" in decode(second[0])


def _logit_gradients(agent: Any, groups: list[Any]) -> Any:
    """Gradient of the GRPO loss w.r.t. every logits tensor the loss used."""
    captured: list[Any] = []

    def keep(_module: Any, _inputs: Any, output: Any) -> None:
        if output.requires_grad:  # not the no-grad old-policy recompute
            output.retain_grad()
            captured.append(output)

    handle = agent.model.lm_head.register_forward_hook(keep)
    try:
        loss = compute_grpo_loss(
            groups,
            _config(),
            agent,
            global_reward_mean=0.0,
            global_reward_count=0,
            update_global_stats=None,
        )
    finally:
        handle.remove()
    assert loss["path"] == "token"
    loss["total_loss"].backward()
    # Row chunks of one group share its padded width: stack them in row order.
    return torch.cat([logits.grad for logits in captured])


def test_token_path_gradient_reaches_only_the_policys_own_tokens(rollout):
    agent, groups = rollout
    agent.model.zero_grad()
    grad = _logit_gradients(agent, groups)
    rows = _token_rows_for_group(groups[0])
    assert grad.shape[0] == len(rows)
    trained_tokens = 0
    for index, (prompt, response, _) in enumerate(rows):
        # Logits at position t predict token t + 1.
        predicted = {
            t + 1 for t in torch.nonzero(grad[index].abs().sum(-1)).flatten().tolist()
        }
        reply = set(range(len(prompt), len(prompt) + len(response)))
        assert predicted == reply, index
        trained_tokens += len(predicted)
    assert trained_tokens == sum(len(r) for _, r, _ in rows)


def test_token_path_never_trains_tool_observation_tokens(rollout):
    agent, groups = rollout
    agent.model.zero_grad()
    grad = _logit_gradients(agent, groups)
    decode = agent.tokenizer.decode
    rows = _token_rows_for_group(groups[0])
    checked = 0
    for index, (prompt, _response, ti) in enumerate(rows):
        text = decode(prompt)
        for observation in _observations(groups[0].trajectories[ti]):
            start = text.find(observation)
            if start < 0:
                continue  # the first turn has not seen the observation yet
            # Character-level tokenizer: token k is character k of the prompt.
            positions = range(start, start + len(observation))
            assert all(grad[index, p - 1].abs().sum() == 0 for p in positions)
            checked += 1
    assert checked == 3


def _supervised(tokenizer: Any, trajectory: Any) -> tuple[str, Any]:
    agent = SimpleNamespace(
        tokenizer=tokenizer, model=SimpleNamespace(device=torch.device("cpu"))
    )
    config = SimpleNamespace(max_prompt_length=1024, max_completion_length=512)
    inputs, labels = _prepare_inputs_and_labels(trajectory, agent, config)
    ids = inputs["input_ids"][0]
    return tokenizer.decode(ids[labels[0] != -100].tolist()), labels


def _assert_only_assistant_text(supervised: str, trajectory: Any) -> None:
    assistant = [t.content for t in trajectory.turns if t.role == "assistant"]
    for text in assistant:
        assert text in supervised
    for turn in trajectory.turns:
        if turn.role in ("tool", "user", "system"):
            assert turn.content not in supervised, turn.role
    assert '"result"' not in supervised and QUESTION not in supervised


CHATML_WITH_TOOLS = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
CHATML_GENERATION = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{% if m['role'] == 'assistant' %}{% generation %}"
    "{{ m['content'] }}<|im_end|>{% endgeneration %}\n"
    "{% else %}{{ m['content'] }}<|im_end|>\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
REVERSING = (
    "{% for m in messages|reverse %}<|im_start|>{{ m['role'] }}\n"
    "{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.mark.parametrize("template", [None, CHATML_WITH_TOOLS, CHATML_GENERATION])
def test_sequence_path_labels_only_assistant_text(rollout, template):
    _, groups = rollout
    tokenizer = make_tokenizer()
    tokenizer.chat_template = template
    for trajectory in groups[0].trajectories[:3]:
        supervised, _ = _supervised(tokenizer, trajectory)
        _assert_only_assistant_text(supervised, trajectory)
        assert "<tool_call>" in supervised


def test_unlocatable_template_skips_tool_conversations_instead_of_full_stream(
    rollout,
):
    _, groups = rollout
    tokenizer = make_tokenizer()
    tokenizer.chat_template = REVERSING
    with_tools = groups[0].trajectories[0]
    _, labels = _supervised(tokenizer, with_tools)
    assert not bool(labels.ne(-100).any())
    # Without observations the documented full-stream fallback is unchanged.
    plain = SimpleNamespace(
        turns=[t for t in with_tools.turns if t.role in ("user", "assistant")][:2]
    )
    supervised, _ = _supervised(tokenizer, plain)
    assert QUESTION in supervised


def test_observation_roles_are_never_supervised():
    assert {"tool", "ipython", "function"} <= OBSERVATION_ROLES
    assert has_observation_turns([{"role": "tool", "content": "{}"}])
    assert has_observation_turns([SimpleNamespace(role="ipython")])
    assert not has_observation_turns([{"role": "user"}, {"role": "assistant"}])


QWEN_LIKE = (
    "{%- if tools %}<|im_start|>system\n# Tools\n"
    "{%- for tool in tools %}\n{{ tool | tojson }}{%- endfor %}<|im_end|>\n"
    "{%- endif %}"
    "{%- for m in messages %}"
    "{%- if m['role'] == 'tool' %}<|im_start|>user\n<tool_response>\n"
    "{{ m['content'] }}\n</tool_response><|im_end|>\n"
    "{%- else %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}"
)


def _messages(trajectory: Any) -> list[dict[str, str]]:
    return [{"role": t.role, "content": t.content} for t in trajectory.turns]


def test_sft_masks_with_tools_rendered_keep_only_assistant_tokens(rollout):
    _, groups = rollout
    tokenizer = make_tokenizer(2048)
    tokenizer.chat_template = QWEN_LIKE
    tools = ToolRegistry([calculator_tool()]).schemas()
    trajectory = groups[0].trajectories[0]
    messages = _messages(trajectory)
    rendered = chat_template_render(tokenizer, tools)(messages, False)
    assert rendered is not None and '"calculator"' in rendered
    assert "<tool_response>" in rendered

    masked = assistant_token_mask(tokenizer, messages, tools=tools)
    assert masked is not None
    ids, mask = masked
    supervised = tokenizer.decode(
        [i for i, keep in zip(ids, mask, strict=True) if keep]
    )
    _assert_only_assistant_text(supervised, trajectory)
    assert "<tool_response>" not in supervised and '"parameters"' not in supervised

    final = final_turn_token_mask(tokenizer, messages, tools=tools)
    assert final is not None
    final_text = tokenizer.decode([i for i, keep in zip(*final, strict=True) if keep])
    assert messages[-1]["content"] in final_text
    assert "<tool_call>" not in final_text and "<tool_response>" not in final_text

    kind, sequences = supervised_turn_sequences(tokenizer, messages, tools=tools)
    assert kind == "all_assistant_turns" and len(sequences) == 1
    assert sequences[0][1] == mask


def test_agent_prompt_shows_observations_to_the_policy(rollout):
    agent, groups = rollout
    trajectory = groups[0].trajectories[0]
    history = [{"role": t.role, "content": t.content} for t in trajectory.turns[:4]]
    prompt, _ = agent.agent._render_prompt(history, None)
    observation = _observations(trajectory)[0]
    assert f"Tool: {observation}\nAssistant: " in prompt


def test_the_gradient_probe_detects_a_leaky_mask(rollout, monkeypatch):
    """Negative control: if rows put the observation in the response (as a
    full-stream loss would), the probe sees gradient on observation tokens."""
    from stateset_agents.training import loss_computation

    agent, groups = rollout
    decode = agent.tokenizer.decode
    honest = loss_computation._token_rows_for_group

    def leaky(group: Any) -> Any:
        rows = []
        for prompt, response, ti in honest(group):
            observations = _observations(group.trajectories[ti])
            start = decode(prompt).find(observations[0]) if observations else -1
            if start > 0:
                prompt, response = prompt[:start], prompt[start:] + response
            rows.append((prompt, response, ti))
        return rows

    monkeypatch.setattr(loss_computation, "_token_rows_for_group", leaky)
    agent.model.zero_grad()
    grad = _logit_gradients(agent, groups)
    leaked = 0
    for index, (prompt, response, ti) in enumerate(leaky(groups[0])):
        sequence = decode(prompt + response)
        for observation in _observations(groups[0].trajectories[ti]):
            start = sequence.find(observation)
            if start > 0 and grad[index, start - 1].abs().sum() > 0:
                leaked += 1
    assert leaked == 3
