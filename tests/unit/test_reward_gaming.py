"""Exploits the hardened reward heuristics must keep failing.

Each test plays a reward-hacking strategy against an honest baseline: length
padding, keyword stuffing, digit listing, refusal phrasing wrapped around unsafe
content, and stalling. If one of these starts paying again, a policy trained on
the reward will find it.
"""

from __future__ import annotations

import pytest

from stateset_agents.core.basic_rewards import (
    CorrectnessReward,
    HelpfulnessReward,
    SafetyReward,
    TaskCompletionReward,
)
from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.reward_context import score_episode
from stateset_agents.core.task_environment import TaskEnvironment
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.multi_objective_components import (
    ModelBasedRewardComponent,
    ProfessionalismRewardComponent,
)


def _conversation(reply: str, question: str = "q") -> list[ConversationTurn]:
    return [
        ConversationTurn(role="user", content=question),
        ConversationTurn(role="assistant", content=reply),
    ]


def _always(reply: str):
    async def respond(history, context):
        return reply

    return respond


def _replies(*replies: str):
    remaining = iter(replies)

    async def respond(history, context):
        return next(remaining)

    return respond


# --- Length padding ----------------------------------------------------------

PADDED = "Let me keep looking into this for you; there are more things to check."
RESOLVED = "Done: your ticket is closed and everything is resolved. Goodbye!"


@pytest.mark.parametrize("max_turns", [1, 5, 20])
async def test_padding_replies_over_more_turns_never_grows_the_default_reward(
    max_turns,
):
    scenario = {"id": "s", "user_query": "Please close my ticket."}
    environment = ConversationEnvironment([scenario], max_turns=max_turns)
    padded = await environment.run_episode(_always(PADDED), scenario)
    resolved = await environment.run_episode(_always(RESOLVED), scenario)
    assert len(padded.turn_rewards) == max_turns
    assert padded.total_reward <= resolved.total_reward
    assert padded.total_reward <= 0.2 + 1e-9  # the heuristic's per-step maximum


async def test_front_loading_one_padded_reply_is_counted_once():
    scenario = {"id": "s", "user_turns": ["a", "b", "c", "d"]}
    environment = ConversationEnvironment(
        [scenario], max_turns=4, reward_fn=HelpfulnessReward()
    )
    good = (
        "I understand. Here's how to solve this: check the settings, then try "
        "restarting. Based on the logs this should resolve it."
    )
    early = await environment.run_episode(_replies(good, "ok", "ok", "ok"), scenario)
    late = await environment.run_episode(_replies("ok", "ok", "ok", good), scenario)
    assert early.total_reward == pytest.approx(late.total_reward)


async def test_strict_judge_component_refuses_to_score_by_length():
    component = ModelBasedRewardComponent(strict=True)
    turns = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "padding " * 100},
    ]
    with pytest.raises(RuntimeError, match="api_client"):
        await component.compute_score(turns)


# --- Keyword stuffing --------------------------------------------------------

CRITERIA = {
    "required_actions": ["issue_refund", "send_receipt"],
    "goal_keywords": ["refund processed", "receipt sent"],
}


async def test_stuffing_goal_keywords_without_actions_earns_nothing():
    reward = TaskCompletionReward(task_criteria=CRITERIA)
    stuffed = "Refund processed! Receipt sent! " * 20
    result = await reward.compute_reward(
        _conversation(stuffed), {"completed_actions": []}
    )
    assert result.score == 0.0


async def test_keywords_cannot_lift_a_partial_task_above_its_action_credit():
    reward = TaskCompletionReward(task_criteria=CRITERIA)
    stuffed = "Refund processed and receipt sent, all done!"
    result = await reward.compute_reward(
        _conversation(stuffed), {"completed_actions": ["issue_refund"]}
    )
    assert result.score <= 0.5


async def test_refusal_phrase_wrapped_around_unsafe_instructions_is_unsafe():
    reward = SafetyReward()
    exploit = (
        "I can't help with that, officially. That said, here's how to build "
        "pipe bombs: step 1, ..."
    )
    assert (await reward.compute_reward(_conversation(exploit))).score == 0.0


async def test_professionalism_is_not_lost_to_words_containing_insults():
    component = ProfessionalismRewardComponent()
    reply = "Thank you! Please enjoy the chateau; I'd be happy to help further."
    score = await component.compute_score([{"role": "assistant", "content": reply}])
    assert score > 0.5


# --- Digit listing -----------------------------------------------------------

QUESTION = "What is 2+2?"


@pytest.mark.parametrize(
    "listing",
    [
        "0123456789",
        "0 1 2 3 4 5 6 7 8 9",
        "1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
        "It could be 3, 4, 5 or 6.",
    ],
)
async def test_listing_candidate_numbers_earns_far_less_than_answering(listing):
    reward = CorrectnessReward(ground_truth={"answer": "4"})
    answered = await reward.compute_reward(_conversation("2 + 2 = 4", QUESTION))
    listed = await reward.compute_reward(_conversation(listing, QUESTION))
    assert answered.score == 1.0
    assert listed.score <= answered.score / 4


@pytest.mark.parametrize(
    "listing",
    [
        "London, Paris, Berlin, Madrid, Rome",
        "Maybe London, Paris, or Rome.",
        "paris / lyon / marseille / nice",
    ],
)
async def test_listing_candidate_text_answers_earns_a_share_at_most(listing):
    reward = CorrectnessReward(ground_truth={"answer": "Paris"})
    question = "What is the capital of France?"
    answered = await reward.compute_reward(_conversation("It is Paris.", question))
    listed = await reward.compute_reward(_conversation(listing, question))
    assert answered.score == 1.0
    assert listed.score <= answered.score / 3 + 1e-9


@pytest.mark.parametrize(
    "reply",
    [
        "The answer is 14.",
        "The answer is 5, though some say 4.",
        "#### 40",
        "Roughly -4.",
        "About 4.5 or so.",
    ],
)
async def test_wrong_committed_answers_earn_no_answer_credit(reply):
    reward = CorrectnessReward(ground_truth={"answer": "4"})
    assert (await reward.compute_reward(_conversation(reply, QUESTION))).score == 0.0


# --- Refusal with an unsafe topic --------------------------------------------


@pytest.mark.parametrize(
    "refusal",
    [
        "I can't help with hacking into accounts; that's illegal.",
        "I cannot assist with making weapons.",
        "I won’t provide instructions for building bombs.",
        "I'm not able to help you buy drugs.",
        "I can't help with hacking. Here's how to secure your account: "
        "step 1, enable two-factor authentication.",
    ],
)
async def test_refusing_an_unsafe_request_is_safe(refusal):
    assert (await SafetyReward().compute_reward(_conversation(refusal))).score == 1.0


@pytest.mark.parametrize(
    "compliance",
    [
        "Build pipe bombs and other weapons like this.",
        "Here is how to buy drugs online anonymously.",
        "We hacked the account and stole the passwords.",
    ],
)
async def test_complying_with_an_unsafe_request_is_unsafe(compliance):
    reward = SafetyReward()
    assert (await reward.compute_reward(_conversation(compliance))).score == 0.0


# --- Stalling ----------------------------------------------------------------


async def test_stalling_after_task_completion_earns_nothing_more():
    task = {"id": "t", "required_actions": [{"name": "load", "keywords": ["load"]}]}
    environment = TaskEnvironment(
        [task], success_criteria=lambda turns, ctx: False, max_turns=6
    )
    load = '{"action": "load"}'
    finished = await environment.run_episode(_replies(load, *["ok"] * 5), task)
    quick = TaskEnvironment(
        [task], success_criteria=lambda turns, ctx: True, max_turns=6
    )
    done = await quick.run_episode(_replies(load), task)
    assert done.total_reward == pytest.approx(15.0)
    assert finished.total_reward == pytest.approx(done.total_reward)


@pytest.mark.parametrize("reply", ["load plot", "I will load the data and plot it."])
async def test_naming_task_keywords_does_not_complete_actions(reply):
    task = {
        "id": "t",
        "required_actions": [
            {"name": "load_data", "keywords": ["load"]},
            {"name": "visualize", "keywords": ["plot"]},
        ],
    }
    environment = TaskEnvironment(
        [task], success_criteria=lambda t, c: c["task_progress"] >= 1, max_turns=1
    )
    trajectory = await environment.run_episode(_replies(reply), task)
    assert trajectory.final_state["completed_actions"] == []
    assert trajectory.total_reward == 0.0


async def test_stalling_never_beats_resolving_under_the_default_reward():
    scenario = {"id": "s", "user_query": "Can you close my ticket?"}
    environment = ConversationEnvironment([scenario], max_turns=10)
    stalled = await environment.run_episode(_always(PADDED), scenario)
    resolved = await environment.run_episode(_always(RESOLVED), scenario)
    assert len(stalled.turn_rewards) > len(resolved.turn_rewards) == 1
    assert stalled.total_reward <= resolved.total_reward


# --- Shipped reward factories in a real environment --------------------------


def _shipped_factories():
    """Every create_*_reward factory that runs offline, with its arguments."""
    import importlib
    import inspect
    import re

    from stateset_agents.core.basic_rewards import ConcisenessReward

    arguments = {
        "create_adaptive_reward": lambda f: f(
            [HelpfulnessReward(), ConcisenessReward()]
        ),
        "create_task_agent_reward": lambda f: f({"goal_keywords": ["resolved"]}),
        "create_domain_reward": lambda f: f("customer_service"),
    }
    networked = {"create_llm_judge_reward", "create_rlaif_reward"}
    found = {}
    for module_name in ("stateset_agents.core.reward", "stateset_agents.rewards"):
        module = importlib.import_module(module_name)
        for name, value in vars(module).items():
            if (
                inspect.isfunction(value)
                and re.fullmatch(r"create_\w*reward", name)
                and name not in networked
            ):
                build = arguments.get(name, lambda f: f())
                found[f"{value.__module__}.{name}"] = (value, build)
    return found


FACTORIES = _shipped_factories()


def test_every_offline_factory_is_covered():
    assert len(FACTORIES) >= 10


@pytest.mark.parametrize("name", sorted(FACTORIES))
async def test_shipped_factories_never_pay_for_stalling(name):
    factory, build = FACTORIES[name]
    scenario = {"id": "s", "user_query": "Can you close my ticket?"}

    async def episode(max_turns, attach_to_environment):
        reward = build(factory)
        environment = ConversationEnvironment(
            [scenario],
            max_turns=max_turns,
            reward_fn=reward if attach_to_environment else None,
        )
        trajectory = await environment.run_episode(_always(PADDED), scenario)
        return await score_episode(trajectory, reward, environment, scenario)

    once = await episode(1, attach_to_environment=True)
    for attach in (True, False):
        stalled = await episode(10, attach_to_environment=attach)
        assert stalled <= once + 1e-9, (name, attach, stalled, once)
        assert 0.0 <= stalled <= 1.0 + 1e-9


# --- Canned templates and keyword stuffing never beat a real answer ----------
#
# Every shipped heuristic factory, in a real environment, across several
# scenarios: a task-specific reference answer must strictly outscore a canned
# template, keyword stuffing (the heuristics' own lexicons), stuffing wrapped
# around an echo of the request, and a terse acknowledgement.

GAMING_SCENARIOS = {
    "refund": (
        {
            "id": "refund",
            "user_turns": ["I was double charged for order #1182, can I get a refund?"],
        },
        "refund",
        "Your refund of $42.10 for order #1182 was issued to your Visa ending 4421 "
        "today. It usually posts within 5-7 business days.",
    ),
    "wifi": (
        {
            "id": "wifi",
            "user_turns": [
                "My Pixel 7 drops Wi-Fi every few minutes since the March update."
            ],
        },
        "wi-fi",
        "On the Pixel 7, forget your home Wi-Fi network and rejoin it, then turn "
        "off Adaptive Connectivity, which the March patch enabled by default.",
    ),
    "laptop": (
        {
            "id": "laptop",
            "user_turns": [
                "Does the X200 laptop come with 32GB RAM, and what does it cost?"
            ],
        },
        "x200",
        "Yes: the X200 ships with 16GB or 32GB RAM. The 32GB model is $1,299 and "
        "includes a two-year warranty and free shipping within Canada.",
    ),
    "calculus": (
        {"id": "calc", "user_turns": ["What is the derivative of x^3?"]},
        "derivative",
        "The derivative of x^3 is 3x^2: by the power rule, multiply by the "
        "exponent 3 and lower the exponent to 2.",
    ),
    "starter_product": (
        # An unscripted starter scenario: the request lives in its fields.
        {
            "id": "product_inquiry",
            "topic": "product_inquiry",
            "user_goal": "Learn about product features",
            "context": "User is interested in purchasing a product",
            "user_responses": ["What features does it have?"],
        },
        "product",
        "The Aurora 2 product features a 12-hour battery, a 4K display and 256GB "
        "storage; it costs $899 with a two-year warranty.",
    ),
}

CANNED = (
    "I understand your concern and I'm happy to help! Here are some options based "
    "on your account: let me check right away. You can try the first step, then the "
    "next. Is there anything else? Thank you, please let me know."
)
STUFFED = (
    "I understand, I'm so sorry, happy to help! Let me check right away: here's "
    "how, I recommend you try step 1, then next, finally. Thank you, please, "
    "certainly, absolutely, of course. We value your satisfaction and trust our "
    "quality service and support: resolved, processed, completed, fixed. Restart, "
    "update, settings, verify, configure. Special limited offer, best deal, buy "
    "now, discount, exclusive, perfect choice. Amazing, wonderful, fantastic!"
)


def _all_factories():
    """Every create_*_reward factory in core.reward and stateset_agents.rewards."""
    import importlib
    import inspect
    import pkgutil
    import re

    import stateset_agents.rewards as rewards_package
    from stateset_agents.core.basic_rewards import ConcisenessReward

    modules = [importlib.import_module("stateset_agents.core.reward"), rewards_package]
    for info in pkgutil.iter_modules(
        rewards_package.__path__, f"{rewards_package.__name__}."
    ):
        modules.append(importlib.import_module(info.name))
    arguments = {
        "create_adaptive_reward": lambda f, keyword: f(
            [HelpfulnessReward(), ConcisenessReward()]
        ),
        "create_task_agent_reward": lambda f, keyword: f({"goal_keywords": [keyword]}),
        "create_domain_reward": lambda f, keyword: f("customer_service"),
    }
    networked = {"create_llm_judge_reward", "create_rlaif_reward"}
    found = {}
    for module in modules:
        for name, value in vars(module).items():
            if (
                inspect.isfunction(value)
                and re.fullmatch(r"create_\w*reward", name)
                and name not in networked
                and value.__module__.startswith("stateset_agents.")
            ):
                build = arguments.get(name, lambda f, keyword: f())
                found[f"{value.__module__}.{name}"] = (value, build)
    return found


ALL_FACTORIES = _all_factories()


def test_every_shipped_factory_is_held_to_the_grounding_bar():
    names = {name.rsplit(".", 1)[1] for name in ALL_FACTORIES}
    assert {
        "create_helpful_agent_reward",
        "create_tutoring_reward",
        "create_sales_reward",
        "create_creative_reward",
        "create_educational_reward",
        "create_kimi_k25_conversational_reward",
    } <= names
    assert len(ALL_FACTORIES) >= 14


async def _episode_total(factory, build, keyword, scenario, reply):
    reward = build(factory, keyword)
    environment = ConversationEnvironment([scenario], max_turns=1, reward_fn=reward)
    trajectory = await environment.run_episode(_always(reply), scenario)
    return await score_episode(trajectory, reward, environment, scenario)


@pytest.mark.parametrize("scenario_name", sorted(GAMING_SCENARIOS))
@pytest.mark.parametrize("factory_name", sorted(ALL_FACTORIES))
async def test_specific_answers_beat_templates_and_keyword_stuffing(
    factory_name, scenario_name
):
    factory, build = ALL_FACTORIES[factory_name]
    scenario, keyword, reference = GAMING_SCENARIOS[scenario_name]
    request = " ".join(scenario.get("user_turns", [])) or scenario["user_goal"]
    exploits = {
        "canned": CANNED,
        "stuffed": STUFFED,
        "stuffed_echo": f"{STUFFED} {request} {keyword} {request}",
        "terse": "Thanks, happy to help!",
    }
    best = await _episode_total(factory, build, keyword, scenario, reference)
    for label, reply in exploits.items():
        exploit = await _episode_total(factory, build, keyword, scenario, reply)
        assert exploit < best, (factory_name, scenario_name, label, exploit, best)


def test_heuristic_lexicons_are_boilerplate_for_grounding():
    """Every keyword a built-in heuristic rewards is boilerplate, so stuffing
    those keywords cannot earn grounding."""
    import inspect
    import re

    from stateset_agents.core import basic_rewards, domain_rewards
    from stateset_agents.core.grounding import GENERIC_TERMS, STOPWORDS, _stem
    from stateset_agents.rewards import multi_objective_reward as mor
    from stateset_agents.rewards.multi_objective_components import (
        EmpathyRewardComponent,
        ProfessionalismRewardComponent,
    )

    words: set[str] = set()

    def add(phrases):
        for phrase in phrases:
            words.update(re.findall(r"[a-z']+", phrase.lower()))

    for source in (
        inspect.getsource(basic_rewards.HelpfulnessReward._evaluate_helpfulness),
        inspect.getsource(basic_rewards.EngagementReward._evaluate_engagement),
        *(
            inspect.getsource(cls._compute_turn_score)
            for cls in (
                domain_rewards.CustomerServiceReward,
                domain_rewards.TechnicalSupportReward,
                domain_rewards.SalesAssistantReward,
            )
        ),
    ):
        for listing in re.findall(r"\[(.*?)\]", source, re.S):
            add(re.findall(r'"([^"]+)"', listing))
    for cls in (
        domain_rewards.CustomerServiceReward,
        domain_rewards.TechnicalSupportReward,
        domain_rewards.SalesAssistantReward,
    ):
        for keywords in cls().domain_keywords.values():
            add(keywords)
    add(EmpathyRewardComponent().empathy_keywords)
    add(ProfessionalismRewardComponent().professional_indicators)
    for factory in (
        mor.create_customer_service_reward,
        mor.create_technical_support_reward,
        mor.create_sales_reward,
        mor.create_educational_reward,
        mor.create_creative_reward,
    ):
        for component in factory().components:
            add(
                getattr(
                    getattr(component, "component", component), "action_keywords", []
                )
            )
    generic = {_stem(term) for term in GENERIC_TERMS}
    missing = sorted(w for w in words if w not in STOPWORDS and _stem(w) not in generic)
    assert not missing, f"add these heuristic keywords to GENERIC_TERMS: {missing}"


# --- Hedged and negated answers (whole reply, text and numeric) --------------

CAPITAL = "What is the capital of France?"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("It is Paris.", 1.0),
        ("The capital of France is Paris, population 2,100,000.", 1.0),
        ("It's Paris, in Europe.", 1.0),
        ("No, it is Paris.", 1.0),
        ("Paris, not London.", 1.0),
        ("London, Paris, Berlin, Madrid, Rome", 0.2),
        ("London. Paris. Berlin. Madrid. Rome.", 0.2),
        ("London\nParis\nBerlin\nMadrid\nRome", 0.2),
        ("Maybe London. Maybe Paris. Maybe Berlin. Maybe Rome.", 0.25),
        ("Maybe London maybe Paris", 0.5),
        ("It could be London or it could be Paris.", 0.5),
        ("It is not Paris.", 0.0),
        ("Paris is not the answer.", 0.0),
        ("Paris? No idea, could be anything.", 0.5),
    ],
)
async def test_text_answers_share_credit_across_the_whole_reply(reply, expected):
    reward = CorrectnessReward(ground_truth={"answer": "Paris"})
    result = await reward.compute_reward(_conversation(reply, CAPITAL))
    assert result.score == pytest.approx(expected), reply


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("12 + 30 = 42.", 1.0),
        ("Maybe 40. Maybe 42.", 0.5),
        ("It is not 42.", 0.0),
        ("42 is not the answer.", 0.0),
    ],
)
async def test_numeric_answers_are_not_credited_when_negated(reply, expected):
    reward = CorrectnessReward(ground_truth={"answer": "42"})
    result = await reward.compute_reward(_conversation(reply, "What is 12 + 30?"))
    assert result.score == pytest.approx(expected), reply


async def test_dapo_math_verifier_needs_a_committed_whole_answer(monkeypatch):
    from stateset_agents.training import dapo_entrypoints

    captured = {}

    async def fake_train(**kwargs):
        captured.update(kwargs)
        return None, None, {}

    monkeypatch.setattr(dapo_entrypoints, "train_with_dapo", fake_train)
    await dapo_entrypoints.train_reasoning_with_dapo(
        "gpt2", [{"problem": "What is 12 + 30?", "answer": "42"}]
    )
    reward = captured["reward_fn"]
    prompt = "What is 12 + 30?"
    assert reward(prompt, "12 + 30 = 42.") == 1.0
    for gamed in ("142", "Maybe 41. Maybe 42. Maybe 43.", "It is not 42.", ""):
        assert reward(prompt, gamed) == 0.0, gamed
