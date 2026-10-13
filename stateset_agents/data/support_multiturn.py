"""Deterministic multi-turn customer-support dataset generator.

This module builds ``stateset/customer-support-multiturn-v1``: a corpus of
scripted multi-turn customer-support dialogues where the store policy given
to the agent *determines* the correct resolution. It complements the tiny
bundled corpus in :mod:`stateset_agents.data.customer_support_bench` (24
single-query scenarios) and stays compatible with its
:class:`~stateset_agents.data.customer_support_bench.SupportRewardComposite`.

Design:

* **Combinatorial but grounded.** Each scenario combines an intent, a product
  (19 products across six categories), a persona (seven speaking styles) and a
  *complication template* (40 templates such as late delivery, double charge,
  password-reset loop, address change after shipment, warranty claims).
  Templates sample concrete parameters (days since delivery, window lengths,
  amounts) and the sampled numbers pick the policy branch, so the same
  complaint can require a refund in one scenario and store credit, a refusal,
  or an escalation in another.
* **Held out by construction.** Eval uses a disjoint set of products *and* a
  disjoint set of complication templates. Every opening user turn names its
  product, so no opening turn can appear in both splits; this is asserted at
  build time.
* **Deterministic.** Only ``random.Random`` with integer seeds is used, all
  iteration is over tuples/lists in a fixed order, and no ``hash()`` or set
  iteration influences output, so builds are byte-identical across runs and
  Python versions. Train and eval use independent generators, so changing
  ``n_train`` never changes the eval split.

Usage::

    from stateset_agents.data.support_multiturn import build_support_dataset

    dataset = build_support_dataset(seed=0)
    scenarios = [s.to_scenario() for s in dataset["train"]]
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATASET_NAME = "stateset/customer-support-multiturn-v1"
SPLITS: tuple[str, ...] = ("train", "eval")
INTENTS: tuple[str, ...] = (
    "refund",
    "billing",
    "shipping",
    "technical",
    "account",
    "general",
)
PERSONAS: tuple[str, ...] = (
    "frustrated",
    "confused",
    "polite",
    "terse",
    "elderly-non-technical",
    "anxious",
    "skeptical",
)
MAX_TURN_WORDS = 60
MIN_USER_TURNS = 2
MAX_USER_TURNS = 4

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_ID_RE = re.compile(r"^[a-z]+(?:-eval)?-\d{4}$")

# Phrases a policy-compliant agent must never produce in billing/account flows.
_CREDENTIAL_AVOID: tuple[str, ...] = (
    "provide your full card number",
    "confirm your full card number",
    "provide your password",
    "confirm your password",
    "provide your cvv",
)
_CREDENTIAL_FACT = (
    "Support never asks for full card numbers, CVV codes, or account passwords; "
    "identity is verified with the order email and the last four card digits."
)

# Realistic but irrelevant policy facts mixed in so the agent must select the
# fact that actually governs the case.
_DISTRACTOR_FACTS: tuple[str, ...] = (
    "Standard shipping takes 3-5 business days within the continental US.",
    "Orders over $50 ship free within the continental US.",
    "Gift cards are non-refundable and never expire.",
    "Loyalty members earn 1 point per dollar spent.",
    "Live chat is staffed 8am-8pm ET, seven days a week.",
    "Agents may offer a one-time $10 goodwill credit after a service failure.",
    "Order confirmations and receipts are emailed to the address on the account.",
    "Exchanges ship free once the original item is scanned by the carrier.",
)


# --------------------------------------------------------------------------
# Public schema
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SupportDialogueScenario:
    """One scripted multi-turn customer-support dialogue.

    ``user_turns`` are the customer's scripted messages; turn ``k > 0`` reacts
    plausibly to whatever the agent said (follow-up details, clarification
    questions, pushback). ``policy_facts`` are the store-policy facts the agent
    is given, and together with the numbers in the user turns they determine
    ``expected_resolution``.
    """

    id: str
    split: str
    intent: str
    persona: str
    product: str
    policy_facts: tuple[str, ...]
    user_turns: tuple[str, ...]
    must_acknowledge: tuple[str, ...]
    must_avoid: tuple[str, ...]
    expected_resolution: str

    def to_scenario(self) -> dict[str, Any]:
        """Return a ``ConversationEnvironment``-compatible scenario dict.

        Includes ``user_query``/``intent``/``must_acknowledge``/``must_avoid``
        (lists) so :class:`SupportRewardComposite` can grade it directly.
        """
        return {
            "id": self.id,
            "user_query": self.user_turns[0],
            "intent": self.intent,
            "persona": self.persona,
            "product": self.product,
            "policy_facts": list(self.policy_facts),
            "user_turns": list(self.user_turns),
            "must_acknowledge": list(self.must_acknowledge),
            "must_avoid": list(self.must_avoid),
            "expected_resolution": self.expected_resolution,
        }

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable row (tuples become lists)."""
        return {
            "id": self.id,
            "split": self.split,
            "intent": self.intent,
            "persona": self.persona,
            "product": self.product,
            "policy_facts": list(self.policy_facts),
            "user_turns": list(self.user_turns),
            "must_acknowledge": list(self.must_acknowledge),
            "must_avoid": list(self.must_avoid),
            "expected_resolution": self.expected_resolution,
        }

    @classmethod
    def from_record(cls, row: Mapping[str, Any]) -> SupportDialogueScenario:
        """Parse and strictly validate a row produced by :meth:`to_record`.

        Raises:
            ValueError: On missing/unknown keys, wrong types, empty strings,
                unknown split/intent, or a turn count outside 2-4.
        """
        if not isinstance(row, Mapping):
            raise ValueError(f"record must be a mapping, got {type(row).__name__}")
        expected = set(_RECORD_FIELDS)
        keys = set(row)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(f"record keys mismatch: missing={missing} extra={extra}")

        def _str(name: str) -> str:
            value = row[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"field {name!r} must be a non-empty string")
            return value

        def _strs(name: str, *, allow_empty: bool) -> tuple[str, ...]:
            value = row[name]
            if not isinstance(value, list):
                raise ValueError(f"field {name!r} must be a list of strings")
            if not allow_empty and not value:
                raise ValueError(f"field {name!r} must not be empty")
            for item in value:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f"field {name!r} must contain non-empty strings")
            return tuple(value)

        split = _str("split")
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        intent = _str("intent")
        if intent not in INTENTS:
            raise ValueError(f"unknown intent {intent!r}; expected one of {INTENTS}")
        user_turns = _strs("user_turns", allow_empty=False)
        if not MIN_USER_TURNS <= len(user_turns) <= MAX_USER_TURNS:
            raise ValueError(
                f"user_turns must have {MIN_USER_TURNS}-{MAX_USER_TURNS} entries, "
                f"got {len(user_turns)}"
            )
        scenario_id = _str("id")
        if not _ID_RE.match(scenario_id):
            raise ValueError(f"malformed id {scenario_id!r}")
        persona = _str("persona")
        if persona not in PERSONAS:
            raise ValueError(f"unknown persona {persona!r}; expected one of {PERSONAS}")
        return cls(
            id=scenario_id,
            split=split,
            intent=intent,
            persona=persona,
            product=_str("product"),
            policy_facts=_strs("policy_facts", allow_empty=False),
            user_turns=user_turns,
            must_acknowledge=_strs("must_acknowledge", allow_empty=False),
            must_avoid=_strs("must_avoid", allow_empty=True),
            expected_resolution=_str("expected_resolution"),
        )


_RECORD_FIELDS: tuple[str, ...] = (
    "id",
    "split",
    "intent",
    "persona",
    "product",
    "policy_facts",
    "user_turns",
    "must_acknowledge",
    "must_avoid",
    "expected_resolution",
)


# --------------------------------------------------------------------------
# Catalog: products and personas
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Product:
    """A catalog product; ``split`` reserves it for exactly one split."""

    name: str
    short: str
    category: str
    price: int
    warranty_months: int
    tags: tuple[str, ...]
    split: str


_PRODUCTS: tuple[_Product, ...] = (
    # ---- train products ----
    _Product(
        "Aero ANC Headphones",
        "Aero headphones",
        "electronics",
        249,
        24,
        ("physical", "powered", "connected"),
        "train",
    ),
    _Product(
        "Pulse Fitness Watch",
        "Pulse watch",
        "electronics",
        199,
        12,
        ("physical", "powered", "connected"),
        "train",
    ),
    _Product(
        "Nimbus Smart Speaker",
        "Nimbus speaker",
        "electronics",
        129,
        12,
        ("physical", "powered", "connected"),
        "train",
    ),
    _Product(
        "Volt 20K Power Bank",
        "Volt power bank",
        "electronics",
        59,
        12,
        ("physical", "powered"),
        "train",
    ),
    _Product(
        "Cascade Espresso Machine",
        "espresso machine",
        "home",
        449,
        24,
        ("physical", "powered"),
        "train",
    ),
    _Product(
        "Hearth Cast-Iron Skillet",
        "cast-iron skillet",
        "home",
        79,
        60,
        ("physical",),
        "train",
    ),
    _Product(
        "Silk Ionic Hair Dryer",
        "hair dryer",
        "beauty",
        139,
        12,
        ("physical", "powered"),
        "train",
    ),
    _Product(
        "Glow Vitamin C Serum",
        "Glow serum",
        "beauty",
        42,
        0,
        ("physical", "beauty"),
        "train",
    ),
    _Product(
        "Trailrunner Hiking Boots",
        "Trailrunner boots",
        "apparel",
        159,
        12,
        ("physical", "apparel"),
        "train",
    ),
    _Product(
        "Harbor Wool Coat",
        "Harbor coat",
        "apparel",
        289,
        12,
        ("physical", "apparel"),
        "train",
    ),
    _Product(
        "Summit 2-Person Tent",
        "Summit tent",
        "outdoor",
        329,
        24,
        ("physical",),
        "train",
    ),
    _Product(
        "Ridge Insulated Bottle",
        "Ridge bottle",
        "outdoor",
        35,
        60,
        ("physical",),
        "train",
    ),
    _Product(
        "Brewly Coffee Subscription",
        "Brewly subscription",
        "subscription",
        28,
        0,
        ("subscription",),
        "train",
    ),
    # ---- eval-only products (held out) ----
    _Product(
        "Lumen Video Doorbell",
        "Lumen doorbell",
        "electronics",
        179,
        24,
        ("physical", "powered", "connected"),
        "eval",
    ),
    _Product(
        "Orbit Wireless Earbuds",
        "Orbit earbuds",
        "electronics",
        149,
        12,
        ("physical", "powered", "connected"),
        "eval",
    ),
    _Product(
        "Everyday Denim Jacket",
        "denim jacket",
        "apparel",
        98,
        12,
        ("physical", "apparel"),
        "eval",
    ),
    _Product(
        "Drift Weighted Blanket",
        "weighted blanket",
        "home",
        139,
        12,
        ("physical",),
        "eval",
    ),
    _Product(
        "Bloom Fragrance Set",
        "Bloom fragrance set",
        "beauty",
        65,
        0,
        ("physical", "beauty"),
        "eval",
    ),
    _Product(
        "FreshCrate Meal Kit Plan",
        "FreshCrate plan",
        "subscription",
        72,
        0,
        ("subscription",),
        "eval",
    ),
)


@dataclass(frozen=True)
class _Persona:
    """Speaking style applied around a template's neutral content."""

    openers: tuple[str, ...]
    closers: tuple[str, ...]
    reaction_prefixes: tuple[str, ...]
    closings: tuple[str, ...]
    prefers_short: bool = False


_PERSONA_STYLES: dict[str, _Persona] = {
    "frustrated": _Persona(
        openers=(
            "This is getting ridiculous.",
            "I'm really annoyed right now.",
            "Second time I'm writing about this.",
        ),
        closers=("Please just fix it.", "I expect this sorted today."),
        reaction_prefixes=("Seriously?", "Unbelievable.", "Look."),
        closings=(
            "Fine. Send me a confirmation email with a reference number.",
            "Just make sure nobody tells me something different next time.",
        ),
    ),
    "confused": _Persona(
        openers=(
            "Hi, I'm not sure this is the right place to ask.",
            "Sorry, I'm a bit confused about something.",
        ),
        closers=("What am I supposed to do?", "Can you explain what happens next?"),
        reaction_prefixes=("Sorry, I don't quite follow.", "Wait."),
        closings=(
            "So just to check, what exactly do I need to do now?",
            "Will I get an email explaining all of this?",
        ),
    ),
    "polite": _Persona(
        openers=("Hi there!", "Hello, hope you're having a good day."),
        closers=("Thanks so much for your help.", "Appreciate it!"),
        reaction_prefixes=("Thanks for explaining.", "Okay, thank you."),
        closings=(
            "That's really helpful, thank you!",
            "Great, thanks for sorting this out.",
        ),
    ),
    "terse": _Persona(
        openers=(),
        closers=(),
        reaction_prefixes=(),
        closings=("Ok. Email confirmation?", "Fine. Reference number?"),
        prefers_short=True,
    ),
    "elderly-non-technical": _Persona(
        openers=(
            "Hello. I'm not very good with computers, so please bear with me.",
            "Good morning. My daughter usually helps me with this but she's away.",
        ),
        closers=("Please explain it simply.", "Thank you for your patience."),
        reaction_prefixes=("I see.", "Oh dear."),
        closings=(
            "Could you write the steps down in an email for me?",
            "Thank you, that's very kind. Will someone call me if there's a problem?",
        ),
    ),
    "anxious": _Persona(
        openers=(
            "I'm a bit worried about this.",
            "Sorry to bother you, I'm quite stressed about this.",
        ),
        closers=("Is this going to be okay?", "Please let me know soon."),
        reaction_prefixes=("Okay, sorry, one more question.", "Oh no."),
        closings=(
            "How will I know it's actually done?",
            "Can you confirm by email so I don't worry?",
        ),
    ),
    "skeptical": _Persona(
        openers=(
            "Last time I contacted you, nobody followed up.",
            "I've heard mixed things about your support, so let's see.",
        ),
        closers=(
            "I'd like a clear answer this time.",
            "Please don't just send me a link to the FAQ.",
        ),
        reaction_prefixes=("Hmm.", "I'll believe it when I see it."),
        closings=(
            "Can I get that in writing with a case number?",
            "And if that doesn't happen, who exactly do I contact?",
        ),
    ),
}


# --------------------------------------------------------------------------
# Complication templates
# --------------------------------------------------------------------------

Params = dict[str, Any]
Sampler = Callable[[random.Random, _Product, Params], str]


@dataclass(frozen=True)
class _Outcome:
    """One policy branch: what the correct agent does and what it avoids."""

    facts: tuple[str, ...]
    must_acknowledge: tuple[str, ...]
    must_avoid: tuple[str, ...]
    resolution: str
    reactions: tuple[str, ...]
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Template:
    """A complication template; ``split`` reserves it for exactly one split."""

    key: str
    intent: str
    split: str
    requires: str
    openings: tuple[str, ...]
    details: tuple[str, ...]
    sample: Sampler
    outcomes: tuple[tuple[str, _Outcome], ...]

    def outcome(self, key: str) -> _Outcome:
        """Look up an outcome by key (raises ``KeyError`` if undefined)."""
        for name, outcome in self.outcomes:
            if name == key:
                return outcome
        raise KeyError(f"template {self.key!r} has no outcome {key!r}")


def _coin(rng: random.Random) -> bool:
    return rng.randint(0, 1) == 1


def _window_sample(
    rng: random.Random,
    params: Params,
    *,
    windows: Sequence[int],
    inside: str,
    outside: str,
    var: str = "days",
    window_key: str = "window",
) -> str:
    """Sample ``params[var]`` on either side of ``params[window_key]``.

    About a third of the samples sit right next to the boundary so the
    dataset stresses careful reading of the policy, not only obvious cases.
    """
    window = rng.choice(list(windows))
    params[window_key] = window
    near = rng.randint(0, 2) == 0
    if _coin(rng):
        params[var] = (
            window - rng.randint(1, 2) if near else rng.randint(1, max(1, window - 3))
        )
        return inside
    params[var] = window + rng.randint(1, 3) if near else window + rng.randint(4, 40)
    return outside


# ---- refund samplers ----


def _s_change_of_mind(rng: random.Random, product: _Product, p: Params) -> str:
    return _window_sample(
        rng, p, windows=(14, 30, 60), inside="eligible", outside="outside_window"
    )


def _s_wrong_size(rng: random.Random, product: _Product, p: Params) -> str:
    if "boots" in product.short:
        sizes = [("9", "9.5"), ("10", "10.5"), ("8.5", "8"), ("11", "10.5")]
    else:
        sizes = [("medium", "large"), ("small", "medium"), ("large", "medium")]
    p["size_from"], p["size_to"] = rng.choice(sizes)
    branch = _window_sample(rng, p, windows=(30, 45), inside="in", outside="out")
    if branch == "out":
        return "outside_window"
    return "exchange" if rng.randint(0, 2) > 0 else "out_of_stock"


def _s_damaged(rng: random.Random, product: _Product, p: Params) -> str:
    p["damage"] = rng.choice(
        ["a cracked casing", "a big dent", "a torn seam", "a shattered part"]
    )
    return _window_sample(
        rng,
        p,
        windows=(3, 7),
        inside="reported_in_time",
        outside="reported_late",
        window_key="report_days",
    )


def _s_refund_status(rng: random.Random, product: _Product, p: Params) -> str:
    p["sent_days"] = rng.randint(6, 20)
    roll = rng.randint(0, 2)
    if roll == 0:
        return "not_received"
    p["received_days"] = rng.randint(1, 4) if roll == 1 else rng.randint(6, 12)
    return "processing" if roll == 1 else "overdue"


def _s_final_sale(rng: random.Random, product: _Product, p: Params) -> str:
    p["days"] = rng.randint(3, 20)
    return "defective" if rng.randint(0, 2) == 0 else "not_defective"


def _s_opened_beauty(rng: random.Random, product: _Product, p: Params) -> str:
    p["days"] = rng.randint(3, 25)
    return rng.choice(["reaction", "opened_dislike", "unopened"])


# ---- billing samplers ----


def _s_double_charge(rng: random.Random, product: _Product, p: Params) -> str:
    p["amount"] = f"${product.price}.00"
    p["hold_days"] = rng.choice([3, 5])
    return "pending_hold" if _coin(rng) else "posted_duplicate"


def _s_charged_after_cancel(rng: random.Random, product: _Product, p: Params) -> str:
    p["amount"] = f"${product.price}.00"
    p["renewal_day"] = rng.choice([1, 5, 15, 20])
    if _coin(rng):
        p["cancel_note"] = f"{rng.randint(2, 6)} days before"
        return "cancelled_before"
    p["cancel_note"] = f"{rng.randint(1, 4)} days after"
    return "cancelled_after"


def _s_promo(rng: random.Random, product: _Product, p: Params) -> str:
    p["code"] = rng.choice(["SPRING15", "WELCOME10", "SAVE20", "FALLDEAL"])
    p["adj_days"] = rng.choice([7, 14])
    roll = rng.randint(0, 2)
    if roll == 0:
        p["order_days"] = rng.randint(1, 5)
        return "expired_code"
    if roll == 1:
        p["order_days"] = rng.randint(1, p["adj_days"] - 1)
        return "adjust"
    p["order_days"] = p["adj_days"] + rng.randint(2, 20)
    return "too_late"


def _s_unrecognized(rng: random.Random, product: _Product, p: Params) -> str:
    p["amount"] = f"${product.price}.00"
    return "matches_order" if _coin(rng) else "no_match"


def _s_price_drop(rng: random.Random, product: _Product, p: Params) -> str:
    p["drop"] = rng.choice([10, 15, 20, 25])
    p["adj_days"] = rng.choice([7, 14])
    roll = rng.randint(0, 2)
    if roll == 0:
        p["days"] = rng.randint(1, p["adj_days"] - 1)
        return "adjust"
    if roll == 1:
        p["days"] = p["adj_days"] + rng.randint(1, 15)
        return "too_late"
    p["days"] = rng.randint(1, p["adj_days"] - 1)
    return "flash_sale"


def _s_installments(rng: random.Random, product: _Product, p: Params) -> str:
    p["installment"] = f"${round(product.price / 4)}"
    return "return_received" if _coin(rng) else "return_in_transit"


# ---- shipping samplers ----


def _s_late(rng: random.Random, product: _Product, p: Params) -> str:
    p["trace"] = rng.choice([3, 5])
    p["lost"] = p["trace"] + rng.choice([5, 7])
    roll = rng.randint(0, 2)
    if roll == 0:
        p["late"] = rng.randint(1, p["trace"] - 1)
        return "wait"
    if roll == 1:
        p["late"] = rng.randint(p["trace"], p["lost"] - 1)
        return "trace"
    p["late"] = p["lost"] + rng.randint(0, 6)
    return "reship"


def _s_address_change(rng: random.Random, product: _Product, p: Params) -> str:
    p["city"] = rng.choice(["Denver", "Austin", "Portland", "Columbus", "Raleigh"])
    p["redirect_fee"] = rng.choice([8, 12, 15])
    return "not_shipped" if _coin(rng) else "shipped"


def _s_delivered_missing(rng: random.Random, product: _Product, p: Params) -> str:
    if _coin(rng):
        p["hours"] = rng.choice([3, 6, 10, 20, 30])
        return "wait"
    p["hours"] = rng.choice([50, 72, 96, 120])
    return "claim"


def _s_missing_item(rng: random.Random, product: _Product, p: Params) -> str:
    p["count"] = rng.choice([2, 3, 4])
    return "split_shipment" if _coin(rng) else "packed_missing"


def _s_express_late(rng: random.Random, product: _Product, p: Params) -> str:
    p["fee"] = rng.choice([18, 25, 30])
    p["late"] = rng.randint(1, 4)
    return "refund_fee" if rng.randint(0, 2) > 0 else "weather_exception"


def _s_customs(rng: random.Random, product: _Product, p: Params) -> str:
    p["duty"] = rng.choice([22, 35, 48, 60])
    p["country"] = rng.choice(["Canada", "the UK", "Australia", "Germany"])
    return "duties_unpaid" if _coin(rng) else "duties_prepaid"


# ---- technical samplers ----


def _warranty_age(rng: random.Random, product: _Product, p: Params) -> bool:
    """Set ``months`` (age) and ``warranty``; return True if under warranty."""
    p["warranty"] = product.warranty_months
    if _coin(rng):
        p["months"] = rng.randint(1, max(1, product.warranty_months - 1))
        return True
    p["months"] = product.warranty_months + rng.randint(1, 10)
    return False


def _s_no_power(rng: random.Random, product: _Product, p: Params) -> str:
    if rng.randint(0, 2) == 0:
        p["months"] = rng.randint(1, 6)
        p["warranty"] = product.warranty_months
        return "troubleshoot"
    return "replace" if _warranty_age(rng, product, p) else "out_of_warranty"


def _s_pairing(rng: random.Random, product: _Product, p: Params) -> str:
    p["min_os"] = rng.choice([15, 16])
    p["phone"] = rng.choice(["iPhone", "Android phone"])
    if _coin(rng):
        p["os"] = p["min_os"] - rng.randint(2, 4)
        return "unsupported_os"
    p["os"] = p["min_os"] + rng.randint(0, 2)
    return "reset_pairing"


def _s_firmware(rng: random.Random, product: _Product, p: Params) -> str:
    p["percent"] = rng.choice([37, 48, 62, 81, 94])
    if _coin(rng):
        p["minutes"] = rng.choice([5, 10, 15, 20])
        return "wait"
    p["minutes"] = rng.choice([45, 60, 90, 180])
    return "recovery"


def _s_warranty_claim(rng: random.Random, product: _Product, p: Params) -> str:
    in_warranty = _warranty_age(rng, product, p)
    if in_warranty:
        return "covered_drop" if rng.randint(0, 2) == 0 else "covered"
    return "expired"


def _s_battery(rng: random.Random, product: _Product, p: Params) -> str:
    p["hours"] = rng.choice([2, 3, 4])
    p["rated"] = rng.choice([8, 10, 12])
    return "settings" if _coin(rng) else "degraded"


# ---- account samplers ----


def _s_reset_loop(rng: random.Random, product: _Product, p: Params) -> str:
    p["attempts"] = rng.randint(3, 6)
    return "spam_or_typo" if _coin(rng) else "email_mismatch"


def _s_locked(rng: random.Random, product: _Product, p: Params) -> str:
    p["lock_minutes"] = rng.choice([15, 30])
    if _coin(rng):
        p["since"] = rng.randint(2, p["lock_minutes"] - 3)
        return "wait_lock"
    p["since"] = p["lock_minutes"] + rng.randint(30, 300)
    return "still_locked"


def _s_change_email(rng: random.Random, product: _Product, p: Params) -> str:
    return "has_access" if _coin(rng) else "lost_access"


def _s_hacked(rng: random.Random, product: _Product, p: Params) -> str:
    p["city"] = rng.choice(["Miami", "Phoenix", "Chicago", "Seattle"])
    return "unshipped" if _coin(rng) else "shipped"


def _s_cancel_sub(rng: random.Random, product: _Product, p: Params) -> str:
    p["cutoff"] = rng.choice([3, 5])
    if _coin(rng):
        p["until"] = p["cutoff"] + rng.randint(1, 6)
        return "before_cutoff"
    p["until"] = rng.randint(1, p["cutoff"] - 1)
    return "after_cutoff"


def _s_delete_data(rng: random.Random, product: _Product, p: Params) -> str:
    return "open_order" if _coin(rng) else "no_open_orders"


def _s_guest_order(rng: random.Random, product: _Product, p: Params) -> str:
    return "same_email" if _coin(rng) else "different_email"


# ---- general samplers ----


def _s_warranty_question(rng: random.Random, product: _Product, p: Params) -> str:
    p["warranty"] = product.warranty_months
    p["plan_days"] = 30
    return rng.choice(["defect", "accident_plan_ok", "accident_too_late"])


def _s_gift(rng: random.Random, product: _Product, p: Params) -> str:
    return "not_shipped" if _coin(rng) else "shipped"


def _s_price_match(rng: random.Random, product: _Product, p: Params) -> str:
    p["competitor_price"] = f"${max(5, product.price - rng.choice([10, 15, 20, 30]))}"
    p["retailer"] = rng.choice(["Target", "Best Buy", "REI", "Walmart"])
    return "authorized" if _coin(rng) else "marketplace"


def _s_restock(rng: random.Random, product: _Product, p: Params) -> str:
    p["variant"] = rng.choice(["black", "navy", "sand", "olive"])
    return "date_known" if _coin(rng) else "date_unknown"


def _s_bulk(rng: random.Random, product: _Product, p: Params) -> str:
    p["threshold"] = rng.choice([10, 25])
    if _coin(rng):
        p["qty"] = p["threshold"] + rng.randint(5, 80)
        return "business_team"
    p["qty"] = rng.randint(2, p["threshold"] - 2)
    return "regular_checkout"


def _s_loyalty(rng: random.Random, product: _Product, p: Params) -> str:
    p["post_days"] = rng.choice([14, 30])
    p["points"] = product.price
    if _coin(rng):
        p["days"] = rng.randint(2, p["post_days"] - 2)
        return "not_yet"
    p["days"] = p["post_days"] + rng.randint(3, 20)
    return "missing"


# Shared reaction lines.
_DENIED_PUSHBACK: tuple[str, ...] = (
    "That doesn't seem fair. Can't you make an exception just this once?",
    "I've been a customer for years. Is there really nothing else you can do?",
    "Can I speak to a supervisor about this?",
)
_ESCALATED_REACTIONS: tuple[str, ...] = (
    "How long will that take, and will I hear back from someone?",
    "Do I need to do anything else while I wait?",
)


def _O(  # noqa: N802 - tiny constructor helper reads better as a "literal"
    facts: Sequence[str],
    ack: Sequence[str],
    avoid: Sequence[str],
    resolution: str,
    reactions: Sequence[str],
    details: Sequence[str] = (),
) -> _Outcome:
    return _Outcome(
        tuple(facts),
        tuple(ack),
        tuple(avoid),
        resolution,
        tuple(reactions),
        tuple(details),
    )


_REFUND_WINDOW_FACTS = (
    "Unused items can be returned for a full refund within {window} days of "
    "delivery.",
    "After {window} days, returns qualify for store credit only, not a refund.",
)

_TEMPLATES: tuple[_Template, ...] = (
    # =============================== REFUND ===============================
    _Template(
        key="refund-change-of-mind",
        intent="refund",
        split="train",
        requires="physical",
        openings=(
            "I'd like to return the {product} from order {order}. It arrived "
            "{days} days ago and I just don't need it anymore.",
            "Can I get a refund on my {product}? Order {order}, delivered {days} "
            "days ago. Never used it.",
            "Changed my mind about the {product} (order {order}). It's been {days} "
            "days since delivery. How do I send it back for my money?",
        ),
        details=(
            "It's unused and still in the original packaging.",
            "I still have the box and the receipt email. Nothing is wrong with it.",
            "Nothing's wrong with it, I just found something I like better.",
        ),
        sample=_s_change_of_mind,
        outcomes=(
            (
                "eligible",
                _O(
                    _REFUND_WINDOW_FACTS,
                    ("refund", "return label"),
                    ("store credit only", "not eligible for a refund"),
                    "Confirm the return is within the {window}-day window ({days} "
                    "days since delivery), send a prepaid return label, and explain "
                    "the refund goes to the original payment method once the item "
                    "is received.",
                    (
                        "Great. How long until the money is back on my card?",
                        "Do I need to print the label, or can I just drop it off?",
                    ),
                ),
            ),
            (
                "outside_window",
                _O(
                    _REFUND_WINDOW_FACTS,
                    ("store credit", "window"),
                    ("refund has been issued", "i've processed your refund"),
                    "Explain that {days} days since delivery is outside the "
                    "{window}-day refund window and offer store credit for the "
                    "returned item instead of a refund.",
                    (
                        "It's only a little past the limit. Can't you make an "
                        "exception?",
                        "I didn't know about the {window}-day limit. Is store credit "
                        "really the only option?",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="refund-wrong-size",
        intent="refund",
        split="train",
        requires="apparel",
        openings=(
            "Ordered the {product} in size {size_from} and the fit is off. Order "
            "{order}, got it {days} days ago. Can I swap for a {size_to}?",
            "Ordered the {product} in {size_from}, way too snug. Delivered {days} "
            "days ago, order {order}. I want a {size_to} or my money back.",
            "Wrong size on my {product}, order {order}. I need a {size_to} instead "
            "of the {size_from}. It came {days} days ago.",
        ),
        details=(
            "Tags are still on, I only tried it on indoors.",
            "I'd prefer the exchange if you have the {size_to} in stock.",
            "I wore it around the house for ten minutes, that's it.",
        ),
        sample=_s_wrong_size,
        outcomes=(
            (
                "exchange",
                _O(
                    (
                        "Unworn items with tags can be exchanged for free within "
                        "{window} days of delivery.",
                        "Size {size_to} of the {product_full} is currently in stock.",
                    ),
                    ("exchange", "size"),
                    ("not eligible", "store credit only"),
                    "Set up a free exchange for size {size_to}, since the request "
                    "is within the {window}-day window and the size is in stock, "
                    "and send a return label for the {size_from}.",
                    (
                        "Perfect. Will the new pair ship before you get the old one "
                        "back?",
                        "Do I pay for return shipping on the exchange?",
                    ),
                ),
            ),
            (
                "out_of_stock",
                _O(
                    (
                        "Unworn items with tags can be exchanged for free within "
                        "{window} days of delivery.",
                        "Size {size_to} of the {product_full} is out of stock with no "
                        "restock date; eligible returns get a full refund instead.",
                    ),
                    ("out of stock", "refund"),
                    ("exchange has been shipped",),
                    "Explain size {size_to} is out of stock with no restock date and "
                    "offer a full refund on return of the {size_from}, since the "
                    "request is within the {window}-day window.",
                    (
                        "That's annoying. Can you tell me when the {size_to} is back?",
                        "Okay, refund then. How long does it take?",
                    ),
                ),
            ),
            (
                "outside_window",
                _O(
                    (
                        "Unworn items with tags can be exchanged for free within "
                        "{window} days of delivery.",
                        "Requests after {window} days qualify for store credit only.",
                    ),
                    ("store credit", "window"),
                    ("free exchange", "refund has been issued"),
                    "Explain that {days} days is past the {window}-day exchange "
                    "window, so the item qualifies for store credit only, which the "
                    "customer can use to buy the {size_to}.",
                    _DENIED_PUSHBACK[:2],
                ),
            ),
        ),
    ),
    _Template(
        key="refund-damaged-arrival",
        intent="refund",
        split="train",
        requires="physical",
        openings=(
            "My {product} arrived with {damage}. Order {order}. I noticed it "
            "{days} days after delivery and I want this fixed.",
            "Order {order}: the {product} showed up with {damage}. It's been {days} "
            "days since it was delivered. What now?",
            "Opened my {product} and it has {damage}. Order {order}, delivered "
            "{days} days ago.",
        ),
        details=(
            "I took photos of the box and the damage.",
            "The outer box looked crushed too. I can send pictures.",
            "I haven't used it at all since it's damaged.",
        ),
        sample=_s_damaged,
        outcomes=(
            (
                "reported_in_time",
                _O(
                    (
                        "Shipping damage reported within {report_days} days of "
                        "delivery, with a photo, gets a free replacement or a full "
                        "refund, customer's choice.",
                        "Damage reported later is reviewed by the claims team within "
                        "3 business days.",
                    ),
                    ("photo", "replacement"),
                    ("claims team will review",),
                    "Ask for a photo of the damage and offer a free replacement or a "
                    "full refund, since it was reported {days} days after delivery, "
                    "within the {report_days}-day window.",
                    (
                        "I'll send the photo now. Can I get the replacement instead "
                        "of a refund?",
                        "Do I have to send the broken one back?",
                    ),
                ),
            ),
            (
                "reported_late",
                _O(
                    (
                        "Shipping damage reported within {report_days} days of "
                        "delivery, with a photo, gets a free replacement or a full "
                        "refund, customer's choice.",
                        "Damage reported later is reviewed by the claims team within "
                        "3 business days.",
                    ),
                    ("claims team", "photo"),
                    ("refund has been issued", "replacement is on its way"),
                    "Collect photos and escalate to the claims team for review "
                    "within 3 business days, since {days} days is past the "
                    "{report_days}-day instant-replacement window; do not promise "
                    "an immediate refund or replacement.",
                    (
                        "Why do I have to wait? It clearly arrived broken.",
                        _ESCALATED_REACTIONS[0],
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="refund-status-after-return",
        intent="refund",
        split="train",
        requires="physical",
        openings=(
            "I mailed back the {product} {sent_days} days ago (order {order}) and "
            "still haven't seen my refund.",
            "Where is my refund? Returned the {product} from order {order} "
            "{sent_days} days ago.",
            "Returned my {product}, order {order}, about {sent_days} days ago. No "
            "refund yet. What's going on?",
        ),
        details=(
            "I used the prepaid label you sent and have the drop-off receipt.",
            "The tracking number is on the return label receipt if you need it.",
        ),
        sample=_s_refund_status,
        outcomes=(
            (
                "not_received",
                _O(
                    (
                        "Refunds are issued within 5 business days after the "
                        "warehouse receives the return.",
                        "The order {order} return has not been scanned at the "
                        "warehouse yet; the carrier shows it in transit.",
                    ),
                    ("in transit", "refund"),
                    ("refund has been issued",),
                    "Explain the return is still in transit and not yet received by "
                    "the warehouse, share the tracking status, and confirm the refund "
                    "is issued within 5 business days of receipt.",
                    (
                        "It's been {sent_days} days. Is it lost?",
                        "Can you just refund me now since I have the drop-off "
                        "receipt?",
                    ),
                ),
            ),
            (
                "processing",
                _O(
                    (
                        "Refunds are issued within 5 business days after the "
                        "warehouse receives the return.",
                        "The order {order} return was received {received_days} "
                        "business days ago.",
                    ),
                    ("5 business days", "refund"),
                    ("escalated to billing",),
                    "Confirm the return was received {received_days} business days "
                    "ago, so the refund is still within the 5-business-day "
                    "processing time, and give the expected date.",
                    (
                        "Okay, and once it's issued, how long does my bank take?",
                        "Will I get an email when it goes through?",
                    ),
                ),
            ),
            (
                "overdue",
                _O(
                    (
                        "Refunds are issued within 5 business days after the "
                        "warehouse receives the return.",
                        "The order {order} return was received {received_days} "
                        "business days ago and no refund has been recorded.",
                        "Overdue refunds are escalated to the billing team with a "
                        "case number.",
                    ),
                    ("escalate", "case number"),
                    ("wait a few more days",),
                    "Apologize that the refund is overdue ({received_days} business "
                    "days since receipt, beyond the 5-day standard), escalate to the "
                    "billing team, and give the customer a case number.",
                    _ESCALATED_REACTIONS,
                ),
            ),
        ),
    ),
    _Template(
        key="refund-final-sale",
        intent="refund",
        split="eval",
        requires="physical",
        openings=(
            "I bought the {product} on clearance, order {order}, {days} days ago. "
            "I want to return it for a refund.",
            "Order {order}: the {product} was a final-sale item but I'd still like "
            "my money back. It came {days} days ago.",
            "Is there any way to return the clearance {product} from order {order}? "
            "Got it {days} days ago.",
        ),
        details=(
            "I know it said final sale but I didn't read that part carefully.",
            "Honestly it just isn't what I expected.",
        ),
        sample=_s_final_sale,
        outcomes=(
            (
                "not_defective",
                _O(
                    (
                        "Final-sale items cannot be returned, exchanged, or refunded "
                        "unless they are defective.",
                        "Defective final-sale items are replaced or refunded after a "
                        "photo review.",
                    ),
                    ("final sale", "defective"),
                    ("refund has been issued", "return label is attached"),
                    "Politely decline the return because the item was final sale and "
                    "is not defective, and explain only defective final-sale items "
                    "qualify.",
                    _DENIED_PUSHBACK,
                    (
                        "I know it said final sale, I just didn't read that part.",
                        "Nothing's wrong with it, it just isn't what I expected.",
                    ),
                ),
            ),
            (
                "defective",
                _O(
                    (
                        "Final-sale items cannot be returned, exchanged, or refunded "
                        "unless they are defective.",
                        "Defective final-sale items are replaced or refunded after a "
                        "photo review.",
                    ),
                    ("defect", "photo"),
                    ("final sale items can never be returned",),
                    "Explain final-sale items are normally not returnable, but since "
                    "the customer reports a defect, request a photo and offer a "
                    "replacement or refund after review.",
                    (
                        "Okay, where do I send the photo?",
                        "Would I get a new one or my money back?",
                    ),
                    (
                        "Also, it stopped working properly on the second day.",
                        "Part of it was faulty right out of the box, if that matters.",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="refund-opened-personal-care",
        intent="refund",
        split="eval",
        requires="beauty",
        openings=(
            "I want to return the {product} from order {order}. Got it {days} "
            "days ago.",
            "Can I send back the {product}? Order {order}, delivered {days} days "
            "ago.",
            "Hoping to get a refund on the {product}, order {order}. It arrived "
            "{days} days ago.",
        ),
        details=(
            "I opened it and tried it a couple of times.",
            "It hasn't been opened, the seal is intact.",
            "My skin got red and itchy after using it.",
        ),
        sample=_s_opened_beauty,
        outcomes=(
            (
                "unopened",
                _O(
                    (
                        "Unopened personal-care items can be refunded within 30 days "
                        "of delivery.",
                        "Opened personal-care items are not returnable unless the "
                        "customer had an adverse reaction.",
                        "The customer says the {product} is unopened with the seal "
                        "intact.",
                    ),
                    ("unopened", "refund"),
                    ("not returnable",),
                    "Approve the return for a refund since the item is unopened and "
                    "within 30 days, and send a return label.",
                    (
                        "Great. Is return shipping free?",
                        "How soon will I see the money?",
                    ),
                    ("It hasn't been opened, the seal is intact.",),
                ),
            ),
            (
                "opened_dislike",
                _O(
                    (
                        "Unopened personal-care items can be refunded within 30 days "
                        "of delivery.",
                        "Opened personal-care items are not returnable unless the "
                        "customer had an adverse reaction.",
                        "The customer says the {product} was opened and they simply "
                        "don't like it.",
                    ),
                    ("opened", "hygiene"),
                    ("refund has been issued", "return label is attached"),
                    "Politely explain opened personal-care items cannot be returned "
                    "for hygiene reasons when there is no adverse reaction, and "
                    "decline the refund.",
                    _DENIED_PUSHBACK,
                    (
                        "I opened it and tried it a couple of times. Not for me.",
                        "I used it twice and just don't like the scent.",
                    ),
                ),
            ),
            (
                "reaction",
                _O(
                    (
                        "Unopened personal-care items can be refunded within 30 days "
                        "of delivery.",
                        "Opened personal-care items are not returnable unless the "
                        "customer had an adverse reaction.",
                        "The customer reports a skin reaction after using the "
                        "{product}.",
                    ),
                    ("reaction", "refund", "stop using"),
                    ("not returnable",),
                    "Express concern, advise stopping use and seeking medical advice "
                    "if symptoms persist, and issue a refund without requiring the "
                    "opened item back.",
                    (
                        "Thank you. Do I need to send the bottle back?",
                        "Should I report the reaction somewhere?",
                    ),
                    (
                        "My skin got red and itchy after using it.",
                        "I broke out in a rash the day after I tried it.",
                    ),
                ),
            ),
        ),
    ),
    # =============================== BILLING ==============================
    _Template(
        key="billing-double-charge",
        intent="billing",
        split="train",
        requires="physical",
        openings=(
            "I was charged twice for my {product}, order {order}. Two charges of "
            "{amount} on my card.",
            "Why are there two {amount} charges for order {order}? I only bought "
            "one {product}.",
            "Double charge on my card for the {product}. {amount} twice. Order "
            "{order}.",
        ),
        details=(
            "One of them says pending, I think, and the other one is posted.",
            "Both show up on my banking app from the same day.",
            "Do you need my card number to look it up?",
        ),
        sample=_s_double_charge,
        outcomes=(
            (
                "pending_hold",
                _O(
                    (
                        "A pending authorization hold is not a charge and drops off "
                        "automatically within {hold_days} business days.",
                        "Order {order} shows one captured payment of {amount} and "
                        "one expired authorization hold.",
                        _CREDENTIAL_FACT,
                    ),
                    ("pending", "authorization"),
                    (*_CREDENTIAL_AVOID, "refund has been issued"),
                    "Explain the second {amount} is a pending authorization hold, "
                    "not a charge, which drops off within {hold_days} business "
                    "days, and that no refund is needed.",
                    (
                        "So I'm not actually paying twice? What if it doesn't drop "
                        "off?",
                        "Can you just cancel the hold now?",
                    ),
                    (
                        "One of them says pending, I think. Want my card number?",
                        "The second one is marked pending in my banking app.",
                    ),
                ),
            ),
            (
                "posted_duplicate",
                _O(
                    (
                        "Order {order} shows two captured payments of {amount}.",
                        "Duplicate captured charges are refunded in full to the "
                        "original card within 5-7 business days.",
                        _CREDENTIAL_FACT,
                    ),
                    ("duplicate", "refund"),
                    _CREDENTIAL_AVOID,
                    "Confirm the duplicate {amount} charge, refund it to the "
                    "original card, and say it posts within 5-7 business days.",
                    (
                        "Okay. Will I get a confirmation of the refund?",
                        "Why did this happen in the first place?",
                    ),
                    (
                        "Both are fully posted, not pending. Need my card number?",
                        "Both charges have posted to my account from the same day.",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="billing-charged-after-cancel",
        intent="billing",
        split="train",
        requires="subscription",
        openings=(
            "I cancelled my {product} but you still charged me {amount} on the "
            "{renewal_day}th. Account order {order}.",
            "Why was I billed {amount} for my {product}? I cancelled it already. "
            "Reference {order}.",
            "Got charged {amount} for the {product} after cancelling. Order "
            "{order}. I want that back.",
        ),
        details=(
            "I cancelled {cancel_note} the renewal date, I have the email.",
            "My cancellation was {cancel_note} the renewal, if that matters.",
        ),
        sample=_s_charged_after_cancel,
        outcomes=(
            (
                "cancelled_before",
                _O(
                    (
                        "Subscriptions cancelled before the renewal date are not "
                        "charged for the next cycle; any such charge is refunded in "
                        "full.",
                        "Cancellations after the renewal date take effect next cycle; "
                        "the renewed box ships and is not refundable.",
                        "Account records show the cancellation {cancel_note} the "
                        "renewal on the {renewal_day}th.",
                        _CREDENTIAL_FACT,
                    ),
                    ("refund", "cancel"),
                    _CREDENTIAL_AVOID,
                    "Confirm the cancellation happened before renewal, refund the "
                    "{amount} charge in full, and confirm no future charges.",
                    (
                        "Thanks. And no box will show up, right?",
                        "How long for the refund to show on my card?",
                    ),
                ),
            ),
            (
                "cancelled_after",
                _O(
                    (
                        "Subscriptions cancelled before the renewal date are not "
                        "charged for the next cycle; any such charge is refunded in "
                        "full.",
                        "Cancellations after the renewal date take effect next cycle; "
                        "the renewed box ships and is not refundable.",
                        "Account records show the cancellation {cancel_note} the "
                        "renewal on the {renewal_day}th, and the box has shipped.",
                        _CREDENTIAL_FACT,
                    ),
                    ("renewal", "cancel"),
                    (*_CREDENTIAL_AVOID, "refund has been issued"),
                    "Explain the cancellation came after the renewal date, so the "
                    "{amount} charge stands and the shipped box is the last one, and "
                    "confirm there will be no future charges.",
                    _DENIED_PUSHBACK,
                ),
            ),
        ),
    ),
    _Template(
        key="billing-promo-not-applied",
        intent="billing",
        split="train",
        requires="physical",
        openings=(
            "My promo code {code} didn't apply to the {product}. Order {order}, "
            "placed {order_days} days ago.",
            "I entered {code} at checkout for the {product} and got charged full "
            "price. Order {order}.",
            "Forgot the {code} discount on my {product} order ({order}). Placed it "
            "{order_days} days ago. Can you apply it?",
        ),
        details=(
            "I placed the order {order_days} days ago if that helps.",
            "The code was in an email you sent me.",
        ),
        sample=_s_promo,
        outcomes=(
            (
                "adjust",
                _O(
                    (
                        "Valid promo codes missed at checkout can be applied as a "
                        "price adjustment within {adj_days} days of the order.",
                        "Code {code} was valid on the order date.",
                    ),
                    ("price adjustment", "{code}"),
                    ("code has expired",),
                    "Apply {code} as a price adjustment and refund the difference, "
                    "since the order was placed {order_days} days ago, within the "
                    "{adj_days}-day limit.",
                    (
                        "Nice. Will the difference go back to my card?",
                        "Thanks! How long does that take?",
                    ),
                ),
            ),
            (
                "expired_code",
                _O(
                    (
                        "Valid promo codes missed at checkout can be applied as a "
                        "price adjustment within {adj_days} days of the order.",
                        "Code {code} expired the day before order {order} was "
                        "placed.",
                    ),
                    ("expired", "{code}"),
                    ("price adjustment has been applied",),
                    "Explain that {code} had already expired when the order was "
                    "placed, so it cannot be applied.",
                    _DENIED_PUSHBACK[:2],
                ),
            ),
            (
                "too_late",
                _O(
                    (
                        "Valid promo codes missed at checkout can be applied as a "
                        "price adjustment within {adj_days} days of the order.",
                        "Code {code} was valid on the order date.",
                    ),
                    ("{adj_days} days", "price adjustment"),
                    ("price adjustment has been applied",),
                    "Explain that although {code} was valid, the order was placed "
                    "{order_days} days ago, past the {adj_days}-day price-adjustment "
                    "limit, so it cannot be applied retroactively.",
                    _DENIED_PUSHBACK[:2],
                ),
            ),
        ),
    ),
    _Template(
        key="billing-unrecognized-charge",
        intent="billing",
        split="train",
        requires="physical",
        openings=(
            "There's a {amount} charge from your store on my card I don't "
            "recognize. Someone mentioned a {product}? Reference {order}.",
            "I don't remember buying a {product}. Why is there a {amount} charge, "
            "reference {order}?",
            "Unknown {amount} charge on my statement, says {order}, something about "
            "a {product}. Is this fraud?",
        ),
        details=(
            "Want me to read you my card number so you can check?",
            "The email on my account is the same one I'm writing from.",
            "I share the card with my partner, but they say it wasn't them.",
        ),
        sample=_s_unrecognized,
        outcomes=(
            (
                "matches_order",
                _O(
                    (
                        _CREDENTIAL_FACT,
                        "Reference {order} matches a {product_full} order placed "
                        "from the customer's own account and shipped to their "
                        "address on file.",
                    ),
                    ("last four", "order"),
                    _CREDENTIAL_AVOID,
                    "Verify identity with the order email and last four card "
                    "digits only, then explain {order} is a {product_full} order "
                    "from their own account shipped to their address.",
                    (
                        "Oh. Maybe my partner ordered it after all. Can I return it?",
                        "Hm, so who placed it exactly?",
                    ),
                ),
            ),
            (
                "no_match",
                _O(
                    (
                        _CREDENTIAL_FACT,
                        "Reference {order} was placed from a new device and ships to "
                        "an address not on the customer's account.",
                        "Suspected fraud is escalated to the fraud team, the order is "
                        "held, and the customer is advised to contact their bank.",
                    ),
                    ("fraud", "bank"),
                    _CREDENTIAL_AVOID,
                    "Verify identity without asking for the full card number, hold "
                    "order {order}, escalate it to the fraud team, and advise the "
                    "customer to contact their bank about the card.",
                    (
                        "That's scary. Will I get the {amount} back?",
                        "Should I cancel my card right now?",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="billing-price-drop",
        intent="billing",
        split="eval",
        requires="physical",
        openings=(
            "I bought the {product} {days} days ago (order {order}) and now it's "
            "${drop} cheaper. Can I get the difference?",
            "The {product} dropped ${drop} right after I ordered it. Order {order}, "
            "{days} days ago. Do you price adjust?",
            "Price on the {product} went down by ${drop}. I paid full price {days} "
            "days ago, order {order}.",
        ),
        details=(
            "It's on your own website, not another store.",
            "I saw it on the product page this morning.",
        ),
        sample=_s_price_drop,
        outcomes=(
            (
                "adjust",
                _O(
                    (
                        "Price adjustments are given once per item within {adj_days} "
                        "days of purchase when our own price drops.",
                        "Flash-sale prices are excluded from price adjustments.",
                        "The current ${drop} reduction on the {product_full} is a "
                        "regular price change.",
                    ),
                    ("price adjustment", "${drop}"),
                    ("not eligible",),
                    "Refund the ${drop} difference as a price adjustment, since the "
                    "purchase was {days} days ago, within the {adj_days}-day window, "
                    "and the drop is a regular price change.",
                    (
                        "Great, back to my original card?",
                        "If it drops again next week, can I do this again?",
                    ),
                ),
            ),
            (
                "too_late",
                _O(
                    (
                        "Price adjustments are given once per item within {adj_days} "
                        "days of purchase when our own price drops.",
                        "Flash-sale prices are excluded from price adjustments.",
                        "The current ${drop} reduction on the {product_full} is a "
                        "regular price change.",
                    ),
                    ("{adj_days} days", "price adjustment"),
                    ("price adjustment has been applied",),
                    "Decline the adjustment because the purchase was {days} days "
                    "ago, outside the {adj_days}-day price-adjustment window.",
                    _DENIED_PUSHBACK[:2],
                ),
            ),
            (
                "flash_sale",
                _O(
                    (
                        "Price adjustments are given once per item within {adj_days} "
                        "days of purchase when our own price drops.",
                        "Flash-sale prices are excluded from price adjustments.",
                        "The current ${drop} reduction on the {product_full} is a "
                        "24-hour flash sale.",
                    ),
                    ("flash sale", "price adjustment"),
                    ("price adjustment has been applied",),
                    "Explain that although the purchase is within {adj_days} days, "
                    "the lower price is a flash sale, which is excluded from price "
                    "adjustments, so the difference cannot be refunded.",
                    _DENIED_PUSHBACK[:2],
                ),
            ),
        ),
    ),
    _Template(
        key="billing-installment-refund",
        intent="billing",
        split="eval",
        requires="physical",
        openings=(
            "I paid for the {product} in four installments of {installment} and "
            "returned it. Order {order}. Do I keep paying?",
            "Returned my {product} (order {order}) but the installment plan still "
            "wants {installment} next week. What do I do?",
            "Order {order}: I sent the {product} back. Why is my pay-later plan "
            "still charging me {installment}?",
        ),
        details=(
            "I don't want late fees on my credit.",
            "I used the pay-in-4 option at checkout.",
        ),
        sample=_s_installments,
        outcomes=(
            (
                "return_received",
                _O(
                    (
                        "Refunds for installment purchases go to the installment "
                        "provider, which adjusts or cancels the remaining payments.",
                        "Customers should keep paying scheduled installments until "
                        "the provider confirms the adjustment.",
                        "The warehouse received the order {order} return and the "
                        "refund was sent to the installment provider yesterday.",
                    ),
                    ("installment provider", "refund"),
                    ("stop paying",),
                    "Confirm the refund was sent to the installment provider, "
                    "explain the provider will adjust the remaining payments, and "
                    "advise keeping scheduled payments until the provider confirms.",
                    (
                        "So I still pay the {installment} next week?",
                        "How long does the provider usually take?",
                    ),
                ),
            ),
            (
                "return_in_transit",
                _O(
                    (
                        "Refunds for installment purchases go to the installment "
                        "provider, which adjusts or cancels the remaining payments.",
                        "Customers should keep paying scheduled installments until "
                        "the provider confirms the adjustment.",
                        "The order {order} return is still in transit to the "
                        "warehouse.",
                    ),
                    ("in transit", "installment"),
                    ("stop paying", "refund has been issued"),
                    "Explain the return is still in transit, the refund goes to the "
                    "installment provider once it is received, and the customer "
                    "should keep paying installments until the provider adjusts the "
                    "plan.",
                    (
                        "That seems unfair, I already sent it back.",
                        "Can you tell the provider to pause my payments?",
                    ),
                ),
            ),
        ),
    ),
    # =============================== SHIPPING =============================
    _Template(
        key="shipping-late-delivery",
        intent="shipping",
        split="train",
        requires="physical",
        openings=(
            "My {product} (order {order}) is {late} business days past the "
            "delivery estimate. Where is it?",
            "Order {order} for the {product} was supposed to be here {late} "
            "business days ago. Tracking hasn't changed.",
            "Still no {product}. Order {order}. It's {late} business days late.",
        ),
        details=(
            "Tracking still says 'in transit' with no new scans.",
            "I need it for a trip, so timing matters.",
            "I already checked with my building's mailroom.",
        ),
        sample=_s_late,
        outcomes=(
            (
                "wait",
                _O(
                    (
                        "A carrier trace is opened once a package is {trace} "
                        "business days past its estimate.",
                        "If there is still no movement {lost} business days past the "
                        "estimate, the order is reshipped or refunded.",
                    ),
                    ("tracking", "{trace} business days"),
                    ("reshipped", "refund has been issued"),
                    "Share the tracking status, explain the order is only {late} "
                    "business days late, and that a carrier trace opens at {trace} "
                    "business days if it hasn't moved.",
                    (
                        "So I just wait? What if it doesn't show up?",
                        "Can't you send another one now just in case?",
                    ),
                ),
            ),
            (
                "trace",
                _O(
                    (
                        "A carrier trace is opened once a package is {trace} "
                        "business days past its estimate.",
                        "If there is still no movement {lost} business days past the "
                        "estimate, the order is reshipped or refunded.",
                    ),
                    ("trace", "carrier"),
                    ("reshipped today", "refund has been issued"),
                    "Open a carrier trace since the order is {late} business days "
                    "late, and explain it will be reshipped or refunded if there is "
                    "no movement by {lost} business days past the estimate.",
                    _ESCALATED_REACTIONS,
                ),
            ),
            (
                "reship",
                _O(
                    (
                        "A carrier trace is opened once a package is {trace} "
                        "business days past its estimate.",
                        "If there is still no movement {lost} business days past the "
                        "estimate, the order is reshipped or refunded.",
                    ),
                    ("reship", "refund"),
                    ("wait a few more days",),
                    "Treat the package as lost ({late} business days late, past the "
                    "{lost}-day threshold) and offer to reship the {product} at no "
                    "cost or issue a full refund.",
                    (
                        "Reship please. Can you upgrade the shipping this time?",
                        "Just refund me. How long will it take?",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="shipping-address-change",
        intent="shipping",
        split="train",
        requires="physical",
        openings=(
            "I need to change the shipping address on order {order} for the "
            "{product}. I'm moving to {city}.",
            "Wrong address on my {product} order, {order}. It should go to my new "
            "place in {city}.",
            "Can you redirect order {order}? The {product} needs to go to {city}, "
            "not my old address.",
        ),
        details=(
            "The new address is the one I just added to my account.",
            "I moved last week and forgot to update it before ordering.",
        ),
        sample=_s_address_change,
        outcomes=(
            (
                "not_shipped",
                _O(
                    (
                        "Addresses can be changed free of charge until the order "
                        "ships.",
                        "Once shipped, the carrier can be asked to redirect the "
                        "package for a ${redirect_fee} fee, but redirects are not "
                        "guaranteed.",
                        "Order {order} has not shipped yet.",
                    ),
                    ("address", "updated"),
                    ("redirect fee", "not guaranteed"),
                    "Update the shipping address to the new {city} address at no "
                    "cost, since order {order} has not shipped yet, and confirm it "
                    "back.",
                    (
                        "Great. Will the delivery date change?",
                        "Can you send me an updated confirmation email?",
                    ),
                ),
            ),
            (
                "shipped",
                _O(
                    (
                        "Addresses can be changed free of charge until the order "
                        "ships.",
                        "Once shipped, the carrier can be asked to redirect the "
                        "package for a ${redirect_fee} fee, but redirects are not "
                        "guaranteed.",
                        "Order {order} shipped yesterday.",
                    ),
                    ("redirect", "shipped"),
                    ("address has been updated", "guaranteed"),
                    "Explain order {order} has already shipped, offer a carrier "
                    "redirect request for ${redirect_fee} without guaranteeing it, "
                    "and explain what happens if the redirect fails.",
                    (
                        "And if the redirect fails, does it come back to you?",
                        "Why should I pay ${redirect_fee} if it might not work?",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="shipping-delivered-not-received",
        intent="shipping",
        split="train",
        requires="physical",
        openings=(
            "Tracking says my {product} was delivered {hours} hours ago but it's "
            "not here. Order {order}.",
            "Order {order} shows delivered, no {product} anywhere. That scan was "
            "{hours} hours ago.",
            "My {product} says delivered but I never got it. Order {order}, marked "
            "delivered {hours} hours ago.",
        ),
        details=(
            "I checked the porch, the mailbox, and asked my neighbors.",
            "There's no photo of the delivery in the tracking.",
            "I was home all day and nobody came to the door.",
        ),
        sample=_s_delivered_missing,
        outcomes=(
            (
                "wait",
                _O(
                    (
                        "Carriers sometimes scan packages as delivered up to 48 hours "
                        "early; customers are asked to wait 48 hours after the scan.",
                        "After 48 hours, a missing-package claim is filed and a "
                        "replacement is shipped.",
                    ),
                    ("48 hours", "neighbors"),
                    ("replacement is on its way", "you must have"),
                    "Explain carriers can mark packages delivered early, ask the "
                    "customer to wait until 48 hours after the scan (it has been "
                    "{hours}) and check around, then a claim and replacement follow.",
                    (
                        "And if it's still not here after that?",
                        "That's annoying. Do I have to contact you again then?",
                    ),
                ),
            ),
            (
                "claim",
                _O(
                    (
                        "Carriers sometimes scan packages as delivered up to 48 hours "
                        "early; customers are asked to wait 48 hours after the scan.",
                        "After 48 hours, a missing-package claim is filed and a "
                        "replacement is shipped.",
                    ),
                    ("claim", "replacement"),
                    ("you must have", "wait another"),
                    "File a missing-package claim and ship a replacement {product}, "
                    "since it has been {hours} hours since the delivered scan, past "
                    "the 48-hour wait.",
                    (
                        "Thank you. Can it go with signature required this time?",
                        "When will the replacement arrive?",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="shipping-missing-item",
        intent="shipping",
        split="train",
        requires="physical",
        openings=(
            "My order {order} had {count} items but the {product} wasn't in the "
            "box.",
            "Got my package for order {order}, everything except the {product}.",
            "Order {order} arrived missing the {product}. The other items were "
            "there.",
        ),
        details=(
            "The packing slip lists the {product}.",
            "I double-checked all the packing material.",
        ),
        sample=_s_missing_item,
        outcomes=(
            (
                "split_shipment",
                _O(
                    (
                        "Multi-item orders may ship in separate packages from "
                        "different warehouses.",
                        "The {product_full} in order {order} shipped separately and "
                        "is in transit, arriving in 2 days.",
                    ),
                    ("separate", "tracking"),
                    ("replacement is on its way",),
                    "Explain the {product} shipped separately and is in transit, "
                    "share its tracking, and confirm it arrives in about 2 days.",
                    (
                        "Oh, okay. Why wasn't I told it would ship separately?",
                        "Can you send me that tracking link?",
                    ),
                ),
            ),
            (
                "packed_missing",
                _O(
                    (
                        "Multi-item orders may ship in separate packages from "
                        "different warehouses.",
                        "Warehouse records show the {product_full} was packed in the "
                        "same box as the rest of order {order}.",
                        "Items missing from a delivered box are reshipped at no cost.",
                    ),
                    ("reship", "missing"),
                    ("ships separately",),
                    "Apologize, confirm the {product} should have been in the box, "
                    "and reship it at no cost.",
                    (
                        "Thank you. How fast can you get it to me?",
                        "Do I need to send anything back?",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="shipping-express-late",
        intent="shipping",
        split="eval",
        requires="physical",
        openings=(
            "I paid ${fee} for express shipping on the {product} and it came {late} "
            "days late. Order {order}.",
            "Express shipping on order {order} was a waste. The {product} arrived "
            "{late} days after the guaranteed date.",
            "Can I get my ${fee} express fee back? The {product} (order {order}) "
            "was {late} days late.",
        ),
        details=(
            "The whole point was to have it by the weekend.",
            "I chose express specifically because of the guaranteed date.",
        ),
        sample=_s_express_late,
        outcomes=(
            (
                "refund_fee",
                _O(
                    (
                        "Express shipping fees are refunded when an order arrives "
                        "after its guaranteed date.",
                        "The guarantee does not apply during carrier-declared "
                        "weather emergencies.",
                        "No carrier emergency was declared for order {order}.",
                    ),
                    ("shipping fee", "refund"),
                    ("weather",),
                    "Apologize and refund the ${fee} express shipping fee, since the "
                    "order arrived {late} days after its guaranteed date with no "
                    "declared emergency.",
                    (
                        "Thanks. Will it go back to my card?",
                        "Good. How long does the refund take?",
                    ),
                ),
            ),
            (
                "weather_exception",
                _O(
                    (
                        "Express shipping fees are refunded when an order arrives "
                        "after its guaranteed date.",
                        "The guarantee does not apply during carrier-declared "
                        "weather emergencies.",
                        "The carrier declared a weather emergency on the route of "
                        "order {order}.",
                    ),
                    ("weather", "guarantee"),
                    ("refund has been issued",),
                    "Explain the delay was caused by a carrier-declared weather "
                    "emergency, which voids the delivery guarantee, so the ${fee} "
                    "fee is not refundable.",
                    _DENIED_PUSHBACK[:2],
                ),
            ),
        ),
    ),
    _Template(
        key="shipping-customs-fees",
        intent="shipping",
        split="eval",
        requires="physical",
        openings=(
            "The courier made me pay ${duty} in import fees for the {product} "
            "shipped to {country}. Order {order}. Why?",
            "I got a ${duty} customs bill for my {product} order {order} to "
            "{country}. Nobody warned me.",
            "Had to pay ${duty} duty on delivery in {country} for the {product}. "
            "Order {order}. Can you cover that?",
        ),
        details=(
            "I have the courier's receipt for the duty payment.",
            "I thought the price at checkout was the final price.",
        ),
        sample=_s_customs,
        outcomes=(
            (
                "duties_unpaid",
                _O(
                    (
                        "International orders without prepaid duties ship "
                        "duties-unpaid; import fees are the recipient's "
                        "responsibility, as shown at checkout.",
                        "If duties were prepaid at checkout and the courier still "
                        "charged them, the store reimburses the fee with a receipt.",
                        "Order {order} did not include prepaid duties.",
                    ),
                    ("duties", "checkout"),
                    ("reimburse", "refund has been issued"),
                    "Explain order {order} shipped without prepaid duties, so the "
                    "${duty} import fee is the recipient's responsibility as stated "
                    "at checkout, and decline reimbursement.",
                    _DENIED_PUSHBACK[:2],
                ),
            ),
            (
                "duties_prepaid",
                _O(
                    (
                        "International orders without prepaid duties ship "
                        "duties-unpaid; import fees are the recipient's "
                        "responsibility, as shown at checkout.",
                        "If duties were prepaid at checkout and the courier still "
                        "charged them, the store reimburses the fee with a receipt.",
                        "Order {order} included prepaid duties at checkout.",
                    ),
                    ("receipt", "reimburse"),
                    ("your responsibility",),
                    "Apologize, explain duties were prepaid on order {order}, and "
                    "reimburse the ${duty} once the customer sends the courier's "
                    "receipt.",
                    (
                        "Great, where do I send the receipt?",
                        "How long until I get the ${duty} back?",
                    ),
                ),
            ),
        ),
    ),
    # =============================== TECHNICAL ============================
    _Template(
        key="technical-no-power",
        intent="technical",
        split="train",
        requires="powered",
        openings=(
            "My {product} won't turn on anymore. I've had it about {months} "
            "months. Order {order}.",
            "The {product} is dead. No lights, nothing. Bought it {months} months "
            "ago, order {order}.",
            "{product_full} stopped powering on. Order {order}, about {months} "
            "months old.",
        ),
        details=(
            "I already tried a different cable and outlet and held the power button "
            "for 20 seconds.",
            "I haven't really tried anything yet, it just stopped.",
            "It worked fine yesterday and nothing was dropped or spilled.",
        ),
        sample=_s_no_power,
        outcomes=(
            (
                "troubleshoot",
                _O(
                    (
                        "Before a warranty claim, agents walk through charging with a "
                        "known-good cable and outlet and a 20-second power-button "
                        "reset.",
                        "The {product_full} has a {warranty}-month warranty against "
                        "defects.",
                        "The customer has not tried any troubleshooting yet.",
                    ),
                    ("reset", "cable"),
                    ("replacement is on its way",),
                    "Walk the customer through charging with a known-good cable and "
                    "outlet and a 20-second reset before considering a warranty "
                    "claim.",
                    (
                        "Okay, I'll try that. What if it still doesn't work?",
                        "Tried it. Still nothing.",
                    ),
                    (
                        "I haven't really tried anything yet, it just stopped.",
                        "It worked yesterday. I haven't tried resetting it.",
                    ),
                ),
            ),
            (
                "replace",
                _O(
                    (
                        "Before a warranty claim, agents walk through charging with a "
                        "known-good cable and outlet and a 20-second power-button "
                        "reset.",
                        "The {product_full} has a {warranty}-month warranty against "
                        "defects.",
                        "The customer already tried a different cable, outlet, and a "
                        "20-second reset.",
                    ),
                    ("warranty", "replacement"),
                    ("out of warranty",),
                    "Since basic troubleshooting failed and the device is {months} "
                    "months old, within the {warranty}-month warranty, start a "
                    "warranty replacement.",
                    (
                        "Do I need to send the old one back first?",
                        "How long does a warranty replacement take?",
                    ),
                    (
                        "I already tried a different cable and outlet and held the "
                        "power button for 20 seconds.",
                        "Tried another charger, another outlet, and a long reset. "
                        "Nothing.",
                    ),
                ),
            ),
            (
                "out_of_warranty",
                _O(
                    (
                        "Before a warranty claim, agents walk through charging with a "
                        "known-good cable and outlet and a 20-second power-button "
                        "reset.",
                        "The {product_full} has a {warranty}-month warranty against "
                        "defects.",
                        "Out-of-warranty devices can be repaired for a flat fee or "
                        "replaced with a 20% loyalty discount.",
                    ),
                    ("warranty", "repair"),
                    ("free replacement",),
                    "Explain the device is {months} months old, past the "
                    "{warranty}-month warranty, and offer the flat-fee repair or a "
                    "20% discount on a replacement.",
                    _DENIED_PUSHBACK[:2],
                    (
                        "I already tried a different cable and outlet and held the "
                        "power button for 20 seconds.",
                        "Tried another charger, another outlet, and a long reset. "
                        "Nothing.",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="technical-pairing",
        intent="technical",
        split="train",
        requires="connected",
        openings=(
            "My {product} won't pair with my {phone}. It just spins and fails. "
            "Order {order}.",
            "Can't connect the {product} to the app on my {phone}. Tried like five "
            "times.",
            "The {product} keeps failing to pair. {phone}, latest app installed. "
            "Order {order}.",
        ),
        details=(
            "My phone is on version {os}, if that matters.",
            "It's on OS version {os}. Bluetooth is on.",
        ),
        sample=_s_pairing,
        outcomes=(
            (
                "unsupported_os",
                _O(
                    (
                        "The {product_full} app requires phone OS version {min_os} "
                        "or newer.",
                        "Pairing resets are done by holding the pairing button for "
                        "10 seconds until the light flashes.",
                    ),
                    ("version", "update"),
                    ("defective",),
                    "Explain the phone's OS version {os} is below the required "
                    "version {min_os}, so the customer needs to update the phone "
                    "software before pairing will work.",
                    (
                        "I don't think my phone can update that far. Now what?",
                        "How do I update it?",
                    ),
                ),
            ),
            (
                "reset_pairing",
                _O(
                    (
                        "The {product_full} app requires phone OS version {min_os} "
                        "or newer.",
                        "Pairing resets are done by holding the pairing button for "
                        "10 seconds until the light flashes.",
                        "If pairing still fails after a reset, the case is escalated "
                        "to tier-2 technical support.",
                    ),
                    ("pairing", "10 seconds"),
                    ("update your phone",),
                    "Walk through forgetting the device in Bluetooth settings and "
                    "holding the pairing button 10 seconds to reset, and escalate "
                    "to tier-2 support if it still fails.",
                    (
                        "Did that, still failing.",
                        "Where exactly is the pairing button?",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="technical-firmware-stuck",
        intent="technical",
        split="train",
        requires="powered",
        openings=(
            "The update on my {product} is stuck at {percent}% and has been for "
            "{minutes} minutes.",
            "Firmware update froze on my {product}. {percent}% for {minutes} "
            "minutes now. Should I unplug it?",
            "My {product} has been stuck at {percent}% updating for {minutes} "
            "minutes. Order {order}.",
        ),
        details=(
            "The light is just blinking slowly.",
            "I haven't touched it since the update started.",
        ),
        sample=_s_firmware,
        outcomes=(
            (
                "wait",
                _O(
                    (
                        "Firmware updates can take up to 30 minutes; never unplug the "
                        "device during an update.",
                        "If an update is stuck for over 30 minutes, use recovery "
                        "mode: hold the power and volume-down buttons for 15 seconds.",
                    ),
                    ("30 minutes", "unplug"),
                    ("recovery mode now",),
                    "Tell the customer not to unplug the device and to wait, since "
                    "updates can take up to 30 minutes and it has only been "
                    "{minutes}.",
                    (
                        "Okay. What if it's still stuck after 30 minutes?",
                        "It's been longer now and still stuck.",
                    ),
                ),
            ),
            (
                "recovery",
                _O(
                    (
                        "Firmware updates can take up to 30 minutes; never unplug the "
                        "device during an update.",
                        "If an update is stuck for over 30 minutes, use recovery "
                        "mode: hold the power and volume-down buttons for 15 seconds.",
                        "If recovery mode fails, the device is replaced under "
                        "warranty.",
                    ),
                    ("recovery mode", "15 seconds"),
                    ("just unplug it",),
                    "Guide the customer through recovery mode (power + volume-down "
                    "for 15 seconds) since it has been stuck {minutes} minutes, and "
                    "offer a warranty replacement if recovery fails.",
                    (
                        "It rebooted but the update failed again.",
                        "Recovery mode worked! Do I need to retry the update?",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="technical-warranty-claim",
        intent="technical",
        split="eval",
        requires="powered",
        openings=(
            "My {product} started cutting out constantly. I've had it {months} "
            "months. Is it covered? Order {order}.",
            "The {product} has a fault, keeps restarting. Bought {months} months "
            "ago, order {order}. Warranty?",
            "I want to file a warranty claim for my {product}. It's {months} months "
            "old and barely works.",
        ),
        details=(
            "I've reset it twice and updated it, no change.",
            "Honestly I did drop it once, but it worked fine after that.",
            "It's been babied, never dropped or wet.",
        ),
        sample=_s_warranty_claim,
        outcomes=(
            (
                "covered",
                _O(
                    (
                        "The {product_full} warranty covers manufacturing defects "
                        "for {warranty} months from delivery.",
                        "Accidental damage such as drops or liquid is not covered by "
                        "the warranty.",
                        "The customer reports no drops or liquid exposure.",
                    ),
                    ("warranty", "replacement"),
                    ("out of warranty", "not covered"),
                    "Approve a warranty replacement since the device is {months} "
                    "months old, within the {warranty}-month warranty, with no "
                    "accidental damage.",
                    (
                        "Do I ship the old one back first?",
                        "Will the replacement be new or refurbished?",
                    ),
                    (
                        "It's been babied, never dropped or wet.",
                        "I've reset it twice. It's never been dropped or wet.",
                    ),
                ),
            ),
            (
                "covered_drop",
                _O(
                    (
                        "The {product_full} warranty covers manufacturing defects "
                        "for {warranty} months from delivery.",
                        "Accidental damage such as drops or liquid is not covered by "
                        "the warranty.",
                        "Claims involving a reported drop are sent to the repair "
                        "team for inspection before approval.",
                    ),
                    ("inspection", "warranty"),
                    ("replacement is on its way",),
                    "Because the customer mentions a drop, explain accidental damage "
                    "isn't covered and send the device for repair-team inspection "
                    "before approving any warranty replacement.",
                    _ESCALATED_REACTIONS,
                    (
                        "Honestly I did drop it once, but it worked fine after that.",
                        "It fell off a table once, but the problem started later.",
                    ),
                ),
            ),
            (
                "expired",
                _O(
                    (
                        "The {product_full} warranty covers manufacturing defects "
                        "for {warranty} months from delivery.",
                        "Out-of-warranty devices can be repaired for a flat fee or "
                        "replaced with a 20% loyalty discount.",
                    ),
                    ("warranty", "discount"),
                    ("free replacement",),
                    "Explain the device is {months} months old, past the "
                    "{warranty}-month warranty, and offer the flat-fee repair or a "
                    "20% discount on a new one.",
                    _DENIED_PUSHBACK[:2],
                ),
            ),
        ),
    ),
    _Template(
        key="technical-battery-drain",
        intent="technical",
        split="eval",
        requires="powered",
        openings=(
            "The battery on my {product} only lasts {hours} hours. It's supposed to "
            "be {rated}. Order {order}.",
            "My {product} dies after {hours} hours. Advertised {rated} hours. "
            "What's wrong with it?",
            "Battery life on the {product} is terrible, {hours} hours instead of "
            "{rated}.",
        ),
        details=(
            "I use it with all the default settings.",
            "It got worse after the last update, I think.",
        ),
        sample=_s_battery,
        outcomes=(
            (
                "settings",
                _O(
                    (
                        "Rated battery life of {rated} hours assumes power-saving "
                        "mode on and always-on features off.",
                        "Diagnostics show high-drain features enabled on the "
                        "customer's {product}.",
                    ),
                    ("settings", "power-saving"),
                    ("replacement",),
                    "Explain the rated {rated} hours assumes power-saving mode, walk "
                    "through turning off high-drain features, and ask the customer "
                    "to test for a day.",
                    (
                        "Okay, I'll try those settings. And if it's still bad?",
                        "Why are those features on by default then?",
                    ),
                ),
            ),
            (
                "degraded",
                _O(
                    (
                        "Rated battery life of {rated} hours assumes power-saving "
                        "mode on and always-on features off.",
                        "Diagnostics show battery health below 70%, which is a "
                        "covered defect within warranty.",
                    ),
                    ("battery", "warranty"),
                    ("change your settings",),
                    "Explain diagnostics show a degraded battery, which is a covered "
                    "defect, and start a warranty replacement.",
                    (
                        "Good. How long until I get a working one?",
                        "Will I lose my settings with the replacement?",
                    ),
                ),
            ),
        ),
    ),
    # =============================== ACCOUNT ==============================
    _Template(
        key="account-password-reset-loop",
        intent="account",
        split="train",
        requires="any",
        openings=(
            "I've requested a password reset {attempts} times and never get the "
            "email. I need to check my {product} order.",
            "Password reset isn't working. Tried {attempts} times, no email. I just "
            "want to see my {product} order {order}.",
            "Locked out of my account, reset emails never show up. Need my {product} "
            "order details.",
        ),
        details=(
            "I can tell you my old password if that helps you find the account.",
            "I checked my inbox but not spam, I think.",
            "I might have signed up with my work email, not sure.",
        ),
        sample=_s_reset_loop,
        outcomes=(
            (
                "spam_or_typo",
                _O(
                    (
                        _CREDENTIAL_FACT,
                        "Reset emails come from no-reply@ and are often filtered to "
                        "spam; links expire after 30 minutes.",
                        "The email on file matches the address the customer is "
                        "writing from.",
                    ),
                    ("spam", "reset"),
                    _CREDENTIAL_AVOID,
                    "Tell the customer not to share their password, ask them to "
                    "check spam for the no-reply reset email, and send a fresh link "
                    "that expires in 30 minutes.",
                    (
                        "Found it in spam! But the link says expired.",
                        "Still nothing. Can you just reset it for me?",
                    ),
                    (
                        "I checked my inbox but not spam. Should I just tell you my "
                        "password?",
                        "I'm writing from the email I signed up with. I can give you "
                        "my old password if that helps.",
                    ),
                ),
            ),
            (
                "email_mismatch",
                _O(
                    (
                        _CREDENTIAL_FACT,
                        "Reset emails come from no-reply@ and are often filtered to "
                        "spam; links expire after 30 minutes.",
                        "The account for order {order} is registered under a "
                        "different email than the one the customer is using.",
                    ),
                    ("different email", "verify"),
                    _CREDENTIAL_AVOID,
                    "Explain the account uses a different email, verify identity "
                    "with the order number and last four card digits (never the "
                    "password), and help the customer reset via the email on file.",
                    (
                        "Oh, which email is it? I have a few.",
                        "I don't have access to that old email anymore.",
                    ),
                    (
                        "I might have signed up with my work email, not sure.",
                        "I'm writing from my personal email. Want my password to "
                        "find the account?",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="account-locked-out",
        intent="account",
        split="train",
        requires="any",
        openings=(
            "My account got locked after too many login tries. I need to check my "
            "{product} order {order}.",
            "Account locked. Been {since} minutes. I need to update my {product} "
            "order.",
            "It says my account is locked. I tried logging in for my {product} "
            "order and got locked {since} minutes ago.",
        ),
        details=(
            "It's been {since} minutes since it locked.",
            "I'm sure I'm typing the right password now.",
        ),
        sample=_s_locked,
        outcomes=(
            (
                "wait_lock",
                _O(
                    (
                        _CREDENTIAL_FACT,
                        "Accounts lock for {lock_minutes} minutes after 5 failed "
                        "logins and unlock automatically.",
                        "Accounts still locked after the lock period are unlocked by "
                        "an agent after identity verification.",
                    ),
                    ("{lock_minutes} minutes", "unlock"),
                    _CREDENTIAL_AVOID,
                    "Explain the lock lifts automatically after {lock_minutes} "
                    "minutes (it's been {since}), and suggest a password reset to "
                    "avoid another lock.",
                    (
                        "Okay, I'll wait. Can you check my order in the meantime?",
                        "What if it locks again?",
                    ),
                ),
            ),
            (
                "still_locked",
                _O(
                    (
                        _CREDENTIAL_FACT,
                        "Accounts lock for {lock_minutes} minutes after 5 failed "
                        "logins and unlock automatically.",
                        "Accounts still locked after the lock period are unlocked by "
                        "an agent after identity verification.",
                    ),
                    ("verify", "unlock"),
                    _CREDENTIAL_AVOID,
                    "Since it has been {since} minutes, longer than the "
                    "{lock_minutes}-minute lock, verify identity with the order "
                    "email and last four card digits and unlock the account.",
                    (
                        "Sure. What do you need from me exactly?",
                        "Do I need to change my password after?",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="account-change-email",
        intent="account",
        split="train",
        requires="any",
        openings=(
            "I need to change the email on my account. My {product} order "
            "confirmations go to an old address.",
            "Can you update my account email? Order {order} for the {product} went "
            "to an address I don't check.",
            "Change my account email please. Missing all my {product} updates.",
        ),
        details=(
            "I can still log in to the old email for now.",
            "That old email was from my previous job, I can't get into it.",
        ),
        sample=_s_change_email,
        outcomes=(
            (
                "has_access",
                _O(
                    (
                        _CREDENTIAL_FACT,
                        "Email changes require a confirmation link sent to both the "
                        "old and new addresses.",
                        "If the old email is inaccessible, identity is verified with "
                        "the last order number and last four card digits.",
                        "The customer can still access the old email.",
                    ),
                    ("confirmation", "email"),
                    _CREDENTIAL_AVOID,
                    "Start the email change and explain confirmation links go to "
                    "both the old and new addresses, which the customer must click.",
                    (
                        "Got the links. Will my order {order} updates switch too?",
                        "Do I have to click both links?",
                    ),
                    ("I can still log in to the old email for now.",),
                ),
            ),
            (
                "lost_access",
                _O(
                    (
                        _CREDENTIAL_FACT,
                        "Email changes require a confirmation link sent to both the "
                        "old and new addresses.",
                        "If the old email is inaccessible, identity is verified with "
                        "the last order number and last four card digits.",
                        "The customer cannot access the old email.",
                    ),
                    ("verify", "last four"),
                    _CREDENTIAL_AVOID,
                    "Because the old email is inaccessible, verify identity with the "
                    "last order number and last four card digits, then update the "
                    "email.",
                    (
                        "The last order was {order}. Card ends in 4417.",
                        "Why can't you just change it if I'm logged in?",
                    ),
                    (
                        "That old email was from my previous job, I can't get into "
                        "it.",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="account-compromised",
        intent="account",
        split="train",
        requires="any",
        openings=(
            "Someone ordered a {product} on my account that I didn't place, "
            "shipping to {city}. Order {order}.",
            "I think my account was hacked. There's a {product} order {order} I "
            "never made.",
            "Unauthorized order on my account: {product}, going to {city}. I live "
            "nowhere near there.",
        ),
        details=(
            "I reused my password on another site that got breached.",
            "I got a login alert from a device I don't own.",
        ),
        sample=_s_hacked,
        outcomes=(
            (
                "unshipped",
                _O(
                    (
                        _CREDENTIAL_FACT,
                        "On suspected account takeover: cancel unshipped "
                        "unauthorized orders, sign out all sessions, force a "
                        "password reset, and refund the charge.",
                        "Shipped unauthorized orders are escalated to the fraud team "
                        "for carrier interception.",
                        "Order {order} has not shipped.",
                    ),
                    ("cancel", "password reset"),
                    _CREDENTIAL_AVOID,
                    "Cancel order {order} before it ships, sign out all sessions, "
                    "trigger a password reset, and refund the charge.",
                    (
                        "Thank you. Will they still be able to get in?",
                        "Should I turn on two-factor authentication?",
                    ),
                ),
            ),
            (
                "shipped",
                _O(
                    (
                        _CREDENTIAL_FACT,
                        "On suspected account takeover: cancel unshipped "
                        "unauthorized orders, sign out all sessions, force a "
                        "password reset, and refund the charge.",
                        "Shipped unauthorized orders are escalated to the fraud team "
                        "for carrier interception.",
                        "Order {order} has already shipped to {city}.",
                    ),
                    ("fraud", "password reset"),
                    _CREDENTIAL_AVOID,
                    "Secure the account (sign out sessions, force password reset) "
                    "and escalate shipped order {order} to the fraud team for "
                    "carrier interception and refund review.",
                    _ESCALATED_REACTIONS,
                ),
            ),
        ),
    ),
    _Template(
        key="account-cancel-subscription",
        intent="account",
        split="train",
        requires="subscription",
        openings=(
            "I want to cancel my {product}. Next charge is in {until} days.",
            "Please cancel the {product}, account order {order}. Renewal is in "
            "{until} days.",
            "How do I cancel my {product}? I don't want the next box, renewal is in "
            "{until} days.",
        ),
        details=(
            "Not interested in pausing or discounts, just cancel.",
            "Money's tight right now, that's all.",
        ),
        sample=_s_cancel_sub,
        outcomes=(
            (
                "before_cutoff",
                _O(
                    (
                        "Subscriptions can be cancelled anytime with no fee.",
                        "Cancellations made at least {cutoff} days before renewal "
                        "stop the next charge and box.",
                        "Cancellations inside the {cutoff}-day cutoff take effect "
                        "after the next box ships and is charged.",
                    ),
                    ("cancel", "no fee"),
                    (*_CREDENTIAL_AVOID, "you'll need to call", "cancellation fee"),
                    "Cancel the subscription immediately with no fee, confirming the "
                    "next charge and box are stopped since renewal is {until} days "
                    "away, beyond the {cutoff}-day cutoff.",
                    (
                        "Great. Do I get a confirmation email?",
                        "Can I restart later if I want?",
                    ),
                ),
            ),
            (
                "after_cutoff",
                _O(
                    (
                        "Subscriptions can be cancelled anytime with no fee.",
                        "Cancellations made at least {cutoff} days before renewal "
                        "stop the next charge and box.",
                        "Cancellations inside the {cutoff}-day cutoff take effect "
                        "after the next box ships and is charged.",
                    ),
                    ("cancel", "next box"),
                    (*_CREDENTIAL_AVOID, "cancellation fee", "you'll need to call"),
                    "Cancel the subscription, but explain renewal in {until} days is "
                    "inside the {cutoff}-day cutoff, so one final box ships and is "
                    "charged before the cancellation takes effect.",
                    _DENIED_PUSHBACK[:2],
                ),
            ),
        ),
    ),
    _Template(
        key="account-data-deletion",
        intent="account",
        split="eval",
        requires="any",
        openings=(
            "I want my account and all my data deleted. I bought a {product} once, "
            "order {order}, and that's it.",
            "Please delete everything you have on me. Last order was the {product}.",
            "Requesting full data deletion for my account. Had a {product} order "
            "{order}.",
        ),
        details=(
            "I don't want any more marketing emails either.",
            "How do I know you'll actually delete it?",
        ),
        sample=_s_delete_data,
        outcomes=(
            (
                "no_open_orders",
                _O(
                    (
                        _CREDENTIAL_FACT,
                        "Deletion requests require identity verification and are "
                        "completed within 30 days; a confirmation email is sent.",
                        "Accounts with open orders or returns cannot be deleted until "
                        "those close.",
                        "The account has no open orders or returns.",
                    ),
                    ("verify", "30 days"),
                    _CREDENTIAL_AVOID,
                    "Verify identity, submit the deletion request, unsubscribe from "
                    "marketing, and explain it completes within 30 days with a "
                    "confirmation email.",
                    (
                        "Will my order history be gone too?",
                        "And you'll stop emailing me right away?",
                    ),
                ),
            ),
            (
                "open_order",
                _O(
                    (
                        _CREDENTIAL_FACT,
                        "Deletion requests require identity verification and are "
                        "completed within 30 days; a confirmation email is sent.",
                        "Accounts with open orders or returns cannot be deleted until "
                        "those close.",
                        "Order {order} for the {product_full} is still open (an unresolved "
                        "return or delivery).",
                    ),
                    ("open", "order"),
                    (*_CREDENTIAL_AVOID, "account has been deleted"),
                    "Explain deletion can't proceed while order {order} is still "
                    "open, offer to unsubscribe from marketing now, and schedule "
                    "deletion once the order closes.",
                    (
                        "Oh right, that order. How long until it closes?",
                        "Can't you delete it anyway?",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="account-guest-order-link",
        intent="account",
        split="eval",
        requires="any",
        openings=(
            "I checked out as a guest for my {product} (order {order}). Can you add "
            "it to my account?",
            "My {product} order {order} doesn't show in my account. I think I "
            "ordered as a guest.",
            "Order {order} for the {product} is missing from my order history. "
            "Guest checkout maybe?",
        ),
        details=(
            "I want it there so I can track the return later.",
            "I might have used my other email at checkout.",
        ),
        sample=_s_guest_order,
        outcomes=(
            (
                "same_email",
                _O(
                    (
                        _CREDENTIAL_FACT,
                        "Guest orders placed with the same email as an account can be "
                        "linked to it immediately.",
                        "Guest orders under a different email require verification "
                        "with the order number and billing zip code.",
                        "Order {order} was placed with the same email as the account.",
                    ),
                    ("link", "order"),
                    _CREDENTIAL_AVOID,
                    "Link guest order {order} to the account right away since it "
                    "used the same email, and confirm it now shows in order history.",
                    (
                        "Great. Can I start a return from there now?",
                        "Will future guest orders link automatically?",
                    ),
                ),
            ),
            (
                "different_email",
                _O(
                    (
                        _CREDENTIAL_FACT,
                        "Guest orders placed with the same email as an account can be "
                        "linked to it immediately.",
                        "Guest orders under a different email require verification "
                        "with the order number and billing zip code.",
                        "Order {order} was placed with a different email than the "
                        "account.",
                    ),
                    ("zip code", "verify"),
                    _CREDENTIAL_AVOID,
                    "Explain the guest order used a different email and verify "
                    "ownership with the order number and billing zip code before "
                    "linking it.",
                    (
                        "The billing zip is 80210. Can you link it now?",
                        "Why so many steps just to see my own order?",
                    ),
                ),
            ),
        ),
    ),
    # =============================== GENERAL ==============================
    _Template(
        key="general-warranty-coverage",
        intent="general",
        split="train",
        requires="warranty",
        openings=(
            "What does the warranty on the {product} actually cover? I bought one, "
            "order {order}.",
            "Quick question about the {product} warranty. Does it cover accidents?",
            "Before anything goes wrong, what's covered under warranty for my "
            "{product}?",
        ),
        details=("I'm mostly worried about it breaking on its own.",),
        sample=_s_warranty_question,
        outcomes=(
            (
                "defect",
                _O(
                    (
                        "The {product_full} has a {warranty}-month warranty covering "
                        "manufacturing defects only.",
                        "Accidental damage is covered only by an optional protection "
                        "plan, which can be added within {plan_days} days of "
                        "purchase.",
                    ),
                    ("{warranty}-month", "defects"),
                    ("covers everything",),
                    "Explain the {warranty}-month warranty covers manufacturing "
                    "defects, how to file a claim, and that accidental damage needs "
                    "the optional protection plan.",
                    (
                        "Good to know. How do I file a claim if it happens?",
                        "Do I need the receipt?",
                    ),
                    ("I'm mostly worried about it breaking on its own.",),
                ),
            ),
            (
                "accident_plan_ok",
                _O(
                    (
                        "The {product_full} has a {warranty}-month warranty covering "
                        "manufacturing defects only.",
                        "Accidental damage is covered only by an optional protection "
                        "plan, which can be added within {plan_days} days of "
                        "purchase.",
                        "The customer bought the item a week ago.",
                    ),
                    ("protection plan", "accidental"),
                    ("warranty covers drops",),
                    "Explain drops are not covered by the warranty, and since the "
                    "purchase was a week ago, offer to add the protection plan "
                    "within the {plan_days}-day window.",
                    (
                        "How much is the protection plan?",
                        "Okay, add it. Is it charged to the same card?",
                    ),
                    (
                        "I'm clumsy, so drops are my main worry. I bought it a week ago.",
                    ),
                ),
            ),
            (
                "accident_too_late",
                _O(
                    (
                        "The {product_full} has a {warranty}-month warranty covering "
                        "manufacturing defects only.",
                        "Accidental damage is covered only by an optional protection "
                        "plan, which can be added within {plan_days} days of "
                        "purchase.",
                        "The customer bought the item about two months ago.",
                    ),
                    ("accidental", "{plan_days} days"),
                    ("warranty covers drops", "add the protection plan"),
                    "Explain drops aren't covered by the warranty and the protection "
                    "plan can no longer be added, since it must be bought within "
                    "{plan_days} days of purchase.",
                    _DENIED_PUSHBACK[:2],
                    ("I'm clumsy. I bought it about two months ago, if that matters.",),
                ),
            ),
        ),
    ),
    _Template(
        key="general-gift-receipt",
        intent="general",
        split="train",
        requires="physical",
        openings=(
            "The {product} I ordered is a gift. Can I add a gift receipt and hide "
            "the price? Order {order}.",
            "Order {order} is a present. Please don't include the price for the "
            "{product}.",
            "Can you make my {product} order {order} a gift order? It's for my " "dad.",
        ),
        details=(
            "If they don't like it, can they return it without me knowing?",
            "It's shipping straight to them.",
        ),
        sample=_s_gift,
        outcomes=(
            (
                "not_shipped",
                _O(
                    (
                        "Gift receipts (no prices shown) can be added until the "
                        "order ships.",
                        "Gift returns are refunded as store credit to the recipient.",
                        "Order {order} has not shipped yet.",
                    ),
                    ("gift receipt", "store credit"),
                    ("already shipped",),
                    "Add a gift receipt to order {order} before it ships so no "
                    "prices are shown, and explain gift returns become store credit "
                    "for the recipient.",
                    (
                        "Perfect. Can I add a gift message too?",
                        "And the invoice won't be in the box at all?",
                    ),
                ),
            ),
            (
                "shipped",
                _O(
                    (
                        "Gift receipts (no prices shown) can be added until the "
                        "order ships.",
                        "After shipping, a printable gift receipt can be emailed to "
                        "the purchaser to forward.",
                        "Gift returns are refunded as store credit to the recipient.",
                        "Order {order} has already shipped.",
                    ),
                    ("shipped", "email"),
                    ("gift receipt has been added to the box",),
                    "Explain order {order} already shipped so the packing slip can't "
                    "change, and email a printable gift receipt the customer can "
                    "forward; gift returns become store credit.",
                    (
                        "Does the packing slip in the box show the price?",
                        "Okay, send it to my email please.",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="general-price-match",
        intent="general",
        split="train",
        requires="physical",
        openings=(
            "{retailer} has the {product} for {competitor_price}. Will you price "
            "match? I ordered it yesterday, order {order}.",
            "Do you match {retailer}? They list the {product} at "
            "{competitor_price}.",
            "Found the {product} cheaper at {retailer}, {competitor_price}. Can you "
            "match that on order {order}?",
        ),
        details=(
            "Here's the link, same model and color.",
            "It's listed by a third-party seller on their site, but same item.",
        ),
        sample=_s_price_match,
        outcomes=(
            (
                "authorized",
                _O(
                    (
                        "We match identical items sold and shipped directly by "
                        "authorized retailers within 14 days of purchase.",
                        "Marketplace or third-party seller listings are excluded "
                        "from price matching.",
                        "The {retailer} listing is sold and shipped by {retailer}.",
                    ),
                    ("price match", "{competitor_price}"),
                    ("third-party",),
                    "Approve the price match to {competitor_price} since {retailer} "
                    "sells it directly and the order is within 14 days, refunding "
                    "the difference.",
                    (
                        "Great, how soon does the difference come back?",
                        "Does this work for future orders too?",
                    ),
                    ("Here's the link. Sold by {retailer} itself, same model.",),
                ),
            ),
            (
                "marketplace",
                _O(
                    (
                        "We match identical items sold and shipped directly by "
                        "authorized retailers within 14 days of purchase.",
                        "Marketplace or third-party seller listings are excluded "
                        "from price matching.",
                        "The {retailer} listing is from a third-party marketplace "
                        "seller.",
                    ),
                    ("third-party", "price match"),
                    ("price match has been approved",),
                    "Decline the price match because the {competitor_price} listing "
                    "is from a third-party marketplace seller, which the policy "
                    "excludes.",
                    _DENIED_PUSHBACK[:2],
                    (
                        "It's listed by a third-party seller on their site, but it's "
                        "the same item.",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="general-restock",
        intent="general",
        split="train",
        requires="physical",
        openings=(
            "When will the {product} in {variant} be back in stock?",
            "Is the {variant} {product} coming back? It's been sold out for weeks.",
            "I want the {product} in {variant}. Any restock date?",
        ),
        details=(
            "Can I pre-order or reserve one?",
            "I don't want a different color.",
        ),
        sample=_s_restock,
        outcomes=(
            (
                "date_known",
                _O(
                    (
                        "Backorders and reservations are not offered.",
                        "Customers can sign up for back-in-stock email alerts.",
                        "The {variant} {product_full} has a confirmed restock "
                        "shipment arriving in about 2 weeks.",
                    ),
                    ("2 weeks", "alert"),
                    ("reserved one for you", "pre-order"),
                    "Share the expected restock in about 2 weeks, explain items "
                    "can't be reserved, and sign the customer up for a "
                    "back-in-stock alert.",
                    (
                        "Will the alert give me time before it sells out?",
                        "Is the price staying the same?",
                    ),
                ),
            ),
            (
                "date_unknown",
                _O(
                    (
                        "Backorders and reservations are not offered.",
                        "Customers can sign up for back-in-stock email alerts.",
                        "The {variant} {product_full} has no confirmed restock date.",
                    ),
                    ("alert", "restock"),
                    ("reserved one for you", "back next week"),
                    "Be honest that there is no confirmed restock date, don't guess, "
                    "and sign the customer up for a back-in-stock alert.",
                    (
                        "Not even a rough idea?",
                        "Is it being discontinued?",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="general-bulk-order",
        intent="general",
        split="eval",
        requires="physical",
        openings=(
            "I want to order {qty} of the {product} for my team. Any discount?",
            "Can I buy {qty} of the {product} at once? It's for corporate gifts.",
            "Looking to place an order for {qty} units of the {product}. How does "
            "that work?",
        ),
        details=(
            "We'd need them delivered by the end of next month.",
            "We'd want an invoice for our company.",
        ),
        sample=_s_bulk,
        outcomes=(
            (
                "business_team",
                _O(
                    (
                        "Orders of {threshold} or more units go through the business "
                        "sales team, which handles volume pricing and invoicing.",
                        "Regular checkout allows up to {threshold} units per order "
                        "with no volume discount.",
                    ),
                    ("business sales", "volume"),
                    ("discount code",),
                    "Since {qty} units meets the {threshold}-unit threshold, refer "
                    "the customer to the business sales team for volume pricing and "
                    "invoicing, and offer to make the introduction.",
                    _ESCALATED_REACTIONS,
                ),
            ),
            (
                "regular_checkout",
                _O(
                    (
                        "Orders of {threshold} or more units go through the business "
                        "sales team, which handles volume pricing and invoicing.",
                        "Regular checkout allows up to {threshold} units per order "
                        "with no volume discount.",
                    ),
                    ("checkout", "discount"),
                    ("business sales team will contact you",),
                    "Explain {qty} units can be ordered through regular checkout, "
                    "that volume discounts start at {threshold} units, and how to "
                    "get an invoice from the order page.",
                    (
                        "What if I bump it up to {threshold}?",
                        "Can the invoice show my company name?",
                    ),
                ),
            ),
        ),
    ),
    _Template(
        key="general-loyalty-points",
        intent="general",
        split="eval",
        requires="physical",
        openings=(
            "I didn't get loyalty points for my {product}, order {order}. It was "
            "delivered {days} days ago.",
            "Where are my points for order {order}? Bought the {product} and got "
            "it {days} days ago.",
            "My rewards balance didn't change after the {product} purchase. Order "
            "{order}, {days} days since delivery.",
        ),
        details=(
            "I'm logged into the loyalty program, I checked.",
            "It should be about {points} points.",
        ),
        sample=_s_loyalty,
        outcomes=(
            (
                "not_yet",
                _O(
                    (
                        "Loyalty points post {post_days} days after delivery, once "
                        "the return window closes.",
                        "Points missing after that are added manually by an agent.",
                    ),
                    ("{post_days} days", "points"),
                    ("points have been added",),
                    "Explain points post {post_days} days after delivery and it has "
                    "only been {days} days, so the {points} points are still "
                    "pending.",
                    (
                        "Why does it take so long?",
                        "Can you add them now anyway?",
                    ),
                ),
            ),
            (
                "missing",
                _O(
                    (
                        "Loyalty points post {post_days} days after delivery, once "
                        "the return window closes.",
                        "Points missing after that are added manually by an agent.",
                    ),
                    ("points", "added"),
                    ("still pending", "wait a few more days"),
                    "Apologize, confirm it is past the {post_days}-day posting "
                    "period ({days} days since delivery), and manually add the "
                    "{points} points.",
                    (
                        "Thanks. Will they show up right away?",
                        "Can you check my other orders too?",
                    ),
                ),
            ),
        ),
    ),
)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


_SINGULAR_RE = re.compile(r"\b1 (business day|day|month|hour|minute)s\b")
_AFFIRMATIVE_STARTS: tuple[str, ...] = (
    "Great",
    "Perfect",
    "Thanks",
    "Thank",
    "Okay",
    "Good",
    "Nice",
    "Oh",
    "Recovery",
)


def _fmt(text: str, params: Mapping[str, Any]) -> str:
    """Format a template string and fix "1 days"-style plurals."""
    return _SINGULAR_RE.sub(r"1 \1", text.format(**params))


def _word_count(text: str) -> int:
    return len(text.split())


def _style_opening(rng: random.Random, persona: _Persona, core: str) -> str:
    """Wrap the neutral opening in the persona's voice, staying under the cap."""
    opener = rng.choice(list(persona.openers)) if persona.openers else ""
    closer = rng.choice(list(persona.closers)) if persona.closers and _coin(rng) else ""
    for parts in ((opener, core, closer), (opener, core, ""), ("", core, "")):
        text = " ".join(part for part in parts if part)
        if _word_count(text) < MAX_TURN_WORDS:
            return text
    return core


def _style_reaction(rng: random.Random, persona: _Persona, line: str) -> str:
    """Optionally prefix a reaction with a persona tic (not before thanks)."""
    if line.startswith(_AFFIRMATIVE_STARTS):
        return line
    if persona.reaction_prefixes and _coin(rng):
        return f"{rng.choice(list(persona.reaction_prefixes))} {line}"
    return line


def _templates_for(split: str) -> tuple[_Template, ...]:
    return tuple(t for t in _TEMPLATES if t.split == split)


def _products_for(split: str, requires: str) -> list[_Product]:
    return [
        p
        for p in _PRODUCTS
        if p.split == split
        and (
            requires == "any"
            or requires in p.tags
            or (
                requires == "warranty"
                and p.warranty_months > 0
                and "physical" in p.tags
            )
        )
    ]


def _render(
    rng: random.Random,
    template: _Template,
    product: _Product,
    persona_name: str,
) -> tuple[str, dict[str, Any]]:
    """Render one scenario's fields (minus id/split) from a template."""
    persona = _PERSONA_STYLES[persona_name]
    params: Params = {
        "product": product.short,
        "product_full": product.name,
        "order": f"#{rng.randint(10000, 99999)}",
    }
    outcome_key = template.sample(rng, product, params)
    outcome = template.outcome(outcome_key)

    core = _fmt(rng.choice(list(template.openings)), params)
    turns = [_style_opening(rng, persona, core)]

    details = [_fmt(d, params) for d in (outcome.details or template.details)]
    if persona.prefers_short:
        turns.append(min(details, key=len))
    else:
        turns.append(rng.choice(details))

    n_turns = rng.choice([2, 3, 3, 4, 4])
    if n_turns >= 3:
        reaction = _fmt(rng.choice(list(outcome.reactions)), params)
        turns.append(_style_reaction(rng, persona, reaction))
    if n_turns >= 4:
        turns.append(rng.choice(list(persona.closings)))

    facts = [_fmt(f, params) for f in outcome.facts]
    distractors = list(_DISTRACTOR_FACTS)
    facts.append(distractors.pop(rng.randrange(len(distractors))))
    if rng.randint(0, 2) == 0:
        facts.append(distractors.pop(rng.randrange(len(distractors))))
    rng.shuffle(facts)

    fields: dict[str, Any] = {
        "intent": template.intent,
        "persona": persona_name,
        "product": product.name,
        "policy_facts": tuple(facts),
        "user_turns": tuple(turns),
        "must_acknowledge": tuple(
            _fmt(a, params).lower() for a in outcome.must_acknowledge
        ),
        "must_avoid": tuple(_fmt(a, params).lower() for a in outcome.must_avoid),
        "expected_resolution": _fmt(outcome.resolution, params),
    }
    return f"{template.key}:{outcome_key}", fields


def _build_split(split: str, n: int, seed: int) -> list[SupportDialogueScenario]:
    split_index = SPLITS.index(split)
    rng = random.Random(seed * 1009 + split_index * 7919 + 17)
    templates_by_intent = {
        intent: [t for t in _templates_for(split) if t.intent == intent]
        for intent in INTENTS
    }
    for intent, templates in templates_by_intent.items():
        if not templates:
            raise RuntimeError(f"no {split} templates for intent {intent!r}")

    scenarios: list[SupportDialogueScenario] = []
    seen_first: set[str] = set()
    counters = dict.fromkeys(INTENTS, 0)
    attempts = 0
    while len(scenarios) < n:
        attempts += 1
        if attempts > n * 50:
            raise RuntimeError(f"could not generate {n} unique {split} scenarios")
        # Round-robin intents for balance; personas rotate on a separate stride
        # so every intent sees every persona.
        idx = len(scenarios)
        intent = INTENTS[idx % len(INTENTS)]
        persona_name = PERSONAS[
            (idx // len(INTENTS) + idx % len(INTENTS)) % len(PERSONAS)
        ]
        template = rng.choice(templates_by_intent[intent])
        product = rng.choice(_products_for(split, template.requires))
        _branch, fields = _render(rng, template, product, persona_name)
        first = fields["user_turns"][0]
        if first in seen_first:
            continue
        for turn in fields["user_turns"]:
            if _word_count(turn) >= MAX_TURN_WORDS:
                raise RuntimeError(f"turn exceeds {MAX_TURN_WORDS} words: {turn!r}")
        seen_first.add(first)
        counters[intent] += 1
        prefix = intent if split == "train" else f"{intent}-eval"
        scenarios.append(
            SupportDialogueScenario(
                id=f"{prefix}-{counters[intent]:04d}",
                split=split,
                **fields,
            )
        )
    return scenarios


def _assert_no_leakage(
    dataset: Mapping[str, Sequence[SupportDialogueScenario]],
) -> None:
    """Raise ``ValueError`` if the splits overlap or contain duplicates."""
    firsts: dict[str, set[str]] = {}
    products: dict[str, set[str]] = {}
    for split in SPLITS:
        rows = dataset[split]
        ids = [s.id for s in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate ids in {split} split")
        first_turns = [s.user_turns[0] for s in rows]
        if len(set(first_turns)) != len(first_turns):
            raise ValueError(f"duplicate opening turns in {split} split")
        firsts[split] = set(first_turns)
        products[split] = {s.product for s in rows}
    if firsts["train"] & firsts["eval"]:
        raise ValueError("opening user turns leak between train and eval")
    if products["train"] & products["eval"]:
        raise ValueError("products leak between train and eval")
    train_ids = {s.id for s in dataset["train"]}
    if train_ids & {s.id for s in dataset["eval"]}:
        raise ValueError("ids collide between train and eval")


def build_support_dataset(
    seed: int = 0, n_train: int = 560, n_eval: int = 240
) -> dict[str, list[SupportDialogueScenario]]:
    """Build the deterministic train/eval support-dialogue dataset.

    Args:
        seed: Generator seed; the same seed always yields identical output.
        n_train: Number of train scenarios.
        n_eval: Number of eval scenarios (independent of ``n_train``).

    Returns:
        ``{"train": [...], "eval": [...]}``.
    """
    if n_train < 1 or n_eval < 1:
        raise ValueError("n_train and n_eval must be positive")
    dataset = {
        "train": _build_split("train", n_train, seed),
        "eval": _build_split("eval", n_eval, seed),
    }
    _assert_no_leakage(dataset)
    return dataset


# --------------------------------------------------------------------------
# Export / load
# --------------------------------------------------------------------------


def _jsonl_bytes(rows: Sequence[SupportDialogueScenario]) -> bytes:
    lines = [
        json.dumps(s.to_record(), sort_keys=True, ensure_ascii=True) + "\n"
        for s in rows
    ]
    return "".join(lines).encode("utf-8")


def _count_by(rows: Sequence[SupportDialogueScenario], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(getattr(row, attr))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _dataset_card(
    dataset: Mapping[str, Sequence[SupportDialogueScenario]], sha256: str
) -> str:
    lines = [
        "---",
        "license: apache-2.0",
        "language:",
        "- en",
        "task_categories:",
        "- text-generation",
        "pretty_name: StateSet Customer Support Multi-Turn v1",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        "    path: train.jsonl",
        "  - split: eval",
        "    path: eval.jsonl",
        "---",
        "",
        f"# {DATASET_NAME}",
        "",
        "Scripted multi-turn customer-support dialogues in which the store policy "
        "given to the agent determines the correct resolution. Built for RL "
        "training and evaluation of support agents with "
        "[stateset-agents](https://github.com/stateset/stateset-agents).",
        "",
        "## License",
        "",
        "Apache-2.0. Fully synthetic: no real customers, orders, or personal data.",
        "",
        "## Generation method",
        "",
        "Generated by `stateset_agents.data.support_multiturn.build_support_dataset` "
        "(`scripts/build_support_dataset.py`) using only `random.Random` with "
        "integer seeds, so output is byte-identical across runs and Python "
        "versions. Each row combines an intent, a product, a persona (speaking "
        "style), and a complication template. Templates sample concrete numbers "
        "(days since delivery, window lengths, amounts, device age) and those "
        "numbers select the policy branch, so the same complaint can require a "
        "refund, store credit, a refusal, or an escalation depending on "
        "`policy_facts`. One or two irrelevant distractor facts are mixed in so "
        "the agent must pick the governing rule. Later user turns react "
        "plausibly to a correct agent reply (details, clarification, pushback, "
        "closing request).",
        "",
        f"SHA-256 over `train.jsonl` + `eval.jsonl` bytes: `{sha256}`",
        "",
        "## Fields",
        "",
        "`id`, `split`, `intent`, `persona`, `product`, `policy_facts`, "
        "`user_turns` (2-4 scripted customer messages), `must_acknowledge` / "
        "`must_avoid` (lower-case substrings for rule-based grading), "
        "`expected_resolution` (the correct next step in one sentence).",
        "",
        "## Split statistics",
        "",
    ]
    for split in SPLITS:
        rows = dataset[split]
        n_templates = len(_templates_for(split))
        lines.append(
            f"- **{split}**: {len(rows)} rows, {len(_count_by(rows, 'product'))} "
            f"products, {n_templates} complication templates"
        )
        lines.append(
            "  - intents: "
            + ", ".join(f"{k}={v}" for k, v in _count_by(rows, "intent").items())
        )
        lines.append(
            "  - personas: "
            + ", ".join(f"{k}={v}" for k, v in _count_by(rows, "persona").items())
        )
        turn_counts: dict[str, int] = {}
        for row in rows:
            key = str(len(row.user_turns))
            turn_counts[key] = turn_counts.get(key, 0) + 1
        lines.append(
            "  - user turns: "
            + ", ".join(f"{k}={v}" for k, v in sorted(turn_counts.items()))
        )
    eval_products = ", ".join(p.name for p in _PRODUCTS if p.split == "eval")
    eval_templates = ", ".join(t.key for t in _templates_for("eval"))
    lines += [
        "",
        "## Leakage controls",
        "",
        "The eval split is held out by construction:",
        "",
        f"- **Disjoint products.** Eval-only products: {eval_products}.",
        f"- **Disjoint complication templates.** Eval-only templates: "
        f"{eval_templates}. No eval complication type appears in train.",
        "- **Disjoint opening turns.** Every opening turn names its product, and "
        "the builder asserts no opening user turn appears in both splits.",
        "- **No duplicates.** Ids and opening turns are unique within each split.",
        "- **Independent generators.** Train and eval use separate seeded "
        "generators, so changing the train size never changes eval.",
        "",
        "Eval therefore measures generalization to unseen products and unseen "
        "policy situations, not memorization of training dialogues.",
        "",
        "## Limitations",
        "",
        "- Template-generated: phrasing diversity is bounded by the hand-written "
        "phrasings (about 3 openings per template plus persona framing) and is "
        "less varied than real customer traffic.",
        "- User turns are scripted, not reactive: later turns assume a "
        "reasonable agent reply and may read oddly after an off-track one.",
        "- `must_acknowledge` / `must_avoid` are substring checks; they are a "
        "cheap signal and should be paired with an LLM judge that reads "
        "`policy_facts` and `expected_resolution`.",
        "- Policies are a single fictional US retailer's; real policies vary by "
        "jurisdiction (e.g. statutory return rights) and are not modelled.",
        "- English only; personas are coarse speaking styles, not demographic "
        "models.",
        "",
    ]
    return "\n".join(lines)


def export_dataset(
    dataset: Mapping[str, Sequence[SupportDialogueScenario]], directory: Path
) -> str:
    """Write ``train.jsonl``, ``eval.jsonl`` and a ``README.md`` dataset card.

    JSONL rows use sorted keys and ASCII escapes, one row per line, so the
    bytes are deterministic.

    Returns:
        Hex SHA-256 over the train then eval JSONL bytes.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for split in SPLITS:
        payload = _jsonl_bytes(dataset[split])
        (directory / f"{split}.jsonl").write_bytes(payload)
        digest.update(payload)
    sha256 = digest.hexdigest()
    (directory / "README.md").write_text(
        _dataset_card(dataset, sha256), encoding="utf-8"
    )
    logger.info("exported %s to %s (sha256=%s)", DATASET_NAME, directory, sha256)
    return sha256


def _read_split(directory: Path, split: str) -> list[SupportDialogueScenario]:
    path = directory / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    scenarios: list[SupportDialogueScenario] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            try:
                scenario = SupportDialogueScenario.from_record(row)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            if scenario.split != split:
                raise ValueError(
                    f"{path}:{lineno}: row split {scenario.split!r} != {split!r}"
                )
            scenarios.append(scenario)
    return scenarios


def load_support_dataset(
    source: str | Path, revision: str | None = None, split: str = "train"
) -> list[SupportDialogueScenario]:
    """Load one split from a local export directory or a pinned HF dataset.

    Args:
        source: A local directory containing ``{split}.jsonl``, or a Hugging
            Face dataset repo id such as :data:`DATASET_NAME`.
        revision: Required for remote sources: a full 40-character commit sha,
            so evaluation data is pinned and reproducible.
        split: ``"train"`` or ``"eval"``.

    Raises:
        ValueError: Unknown split, missing/invalid remote revision, or a
            malformed row.
        ImportError: ``huggingface_hub`` is needed for remote sources.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    local = Path(source)
    if local.is_dir():
        return _read_split(local, split)

    repo_id = str(source)
    if revision is None or not _REVISION_RE.match(revision):
        raise ValueError(
            f"loading {repo_id!r} from the Hub requires revision= a 40-character "
            "lowercase hex commit sha (branches and tags are mutable)"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "huggingface_hub is required to load a remote support dataset"
        ) from exc
    path = snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision)
    return _read_split(Path(path), split)


__all__ = [
    "SupportDialogueScenario",
    "build_support_dataset",
    "export_dataset",
    "load_support_dataset",
]
