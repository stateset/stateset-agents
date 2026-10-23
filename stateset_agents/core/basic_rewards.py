"""
Basic reward functions for multi-turn agent training.

This module provides fundamental reward functions including:
- HelpfulnessReward
- SafetyReward
- CorrectnessReward
- ConcisenessReward
- EngagementReward
- TaskCompletionReward
"""

import re
from typing import Any

from .reward_base import RewardFunction, RewardResult, RewardType
from .reward_context import assistant_turns_in_scope
from .trajectory import ConversationTurn


def _turn_text(turn: ConversationTurn) -> str:
    """Normalize nullable turn content to plain text."""
    return turn.content if isinstance(turn.content, str) else ""


def _normalize_text(text: str) -> str:
    """Lower-case text with typographic apostrophes and whitespace normalized."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(text.lower().split())


#: Declining phrasings that mark a response as a refusal.
_REFUSAL_PATTERN = re.compile(
    r"\b(?:i|we)(?: am|'m| are|'re)? (?:"
    r"(?:can't|cannot|can not|won't|will not|must not|shouldn't|should not|"
    r"do not|don't)"
    r" (?:help|assist|provide|support|share|give|explain|do|recommend|"
    r"participate|engage)"
    r"|(?:not able|unable) to (?:help|assist|provide|support|share|give|explain)"
    r"|(?:must|have to) (?:decline|refuse)"
    r")\b"
    r"|\bthat's not appropriate\b"
    r"|\bthat is not appropriate\b"
)


#: Markers of going on to give instructions after a refusal phrase.
_COMPLIANCE_PATTERN = re.compile(
    r"\b(?:here's how|here is how|here are the steps|follow these steps|"
    r"step (?:1|one)|first,? you|you'll need|you will need|instructions:)"
)


#: Unsafe topics: whole words with their common inflections ("bombs",
#: "hacked"), not words that merely contain them ("hackathon").
UNSAFE_PATTERNS = (
    # "hack together/up/on a script" and "a quick hack" are programming slang.
    r"(?<!life )(?<!quick )(?<!clever )(?<!little )(?<!neat )(?<!dirty )"
    r"\b(hack(?:s|ed|ing)?(?!\s+(?:together|up|on|around|away|at)\b)|"
    r"hackers?|illegal(?:ly)?|violen(?:ce|t)|harmful)\b",
    r"\b(steal(?:s|ing)?|stole(?:n)?|fraud(?:s|ulent)?|"
    r"scam(?:s|med|mer|mers|ming)?|cheat(?:s|ed|ing)?)\b",
    r"\b(drugs?|weapons?|bombs?)\b",
)
_UNSAFE = "(?:" + "|".join(UNSAFE_PATTERNS) + ")"
#: Asking how to do, get or get away with something: the intent that makes a
#: request for an unsafe topic a harmful request.
_HARMFUL_INTENT = re.compile(
    r"\b(?:how (?:do|can|could|would|should|might) (?:i|we|you|one|someone)|"
    r"how to|where (?:can|do|could|to)(?: i| we| one)?|help me|teach me|"
    r"show me|tell me how|explain how|step[- ]by[- ]step|"
    r"instructions? (?:for|to|on)|(?:best|easiest|fastest)? ?ways? to)"
    r"(?P<gap>\W+(?:[\w']+\W+){0,5}?)" + _UNSAFE
)
#: Violent acts: harmful when aimed at a person or animal ("hurt my
#: roommate"), not at a process, a photo, a query or a problem ("kill a
#: process", "shoot photos", "execute a query", "attack the problem").
_VIOLENT = (
    r"(?:kill|murder|hurt|harm|injure|poison|stab|shoot|strangle|choke|"
    r"attack|assault|beat up|torture|kidnap|drug|burn|execute|maim|suffocate|"
    r"drown|punch|run over)(?:s|ed|ing)?"
)
_PERSON = (
    r"(?:someone|somebody|anyone|people|a person|him|her|them|myself|himself|"
    r"herself|(?:my|his|her|their|our|your|a|an|the|this|that|some) "
    r"(?:[\w-]+'s )?(?:[\w-]+ )?(?:roommates?|wife|husband|boyfriend|girlfriend|partner|ex|"
    r"boss|coworkers?|co-workers?|colleagues?|neighbou?rs?|teachers?|"
    r"classmates?|brother|sister|mom|mother|dad|father|parents?|child|"
    r"children|kids?|son|daughter|baby|friends?|family|landlord|tenants?|"
    r"dogs?|cats?|pets?|students?|cousin|uncle|aunt|grandma|grandpa|spouse|"
    r"fiancee?|bully|enemy|rival|man|woman|guy|girl|boy|officer|cop|stranger|"
    r"customer|victim))"
)
#: Wanting or asking to do something: "how do I", "to", "want to", "help me".
_INTENT = (
    r"(?:how (?:do|can|could|would|should|might) (?:i|we|you|one|someone)|"
    r"how to|ways? to|want to|wanna|going to|gonna|plan(?:ning)? to|"
    r"trying to|try to|help me|teach me|in order to|so (?:that )?i can|"
    r"should i|can i|could i|i(?:'ll| will| want)|to)"
)
_HARM_TO_PERSON = re.compile(
    rf"\b{_INTENT}\s+(?:[\w']+\s+){{0,3}}?{_VIOLENT}\s+(?:[\w']+\s+){{0,2}}?"
    rf"{_PERSON}\b(?!'s)"
)
#: Making a poison, toxic gas, explosive or hard drug.
DANGEROUS_SUBSTANCES = (
    r"(?:toxic|poisonous|lethal|deadly|nerve|chlorine|mustard|chloramine)"
    r" (?:gas(?:es)?|fumes|agents?|dose|poisons?|chemicals?|smoke)"
    r"|poisons?|explosives?|pipe bombs?|napalm|ricin|sarin|anthrax|thermite"
    r"|meth(?:amphetamine)?|chloramine|bleach and ammonia|ammonia and bleach"
)
_DANGEROUS_MAKING = re.compile(
    r"\b(?:make|making|create|creating|produce|producing|brew|brewing|"
    r"synthesi[sz]e|combine|combining|mix|mixing|cook|cooking|build|building)"
    r"\b(?P<gap>(?:\W+[\w']+){0,6}?)\W+(?:" + DANGEROUS_SUBSTANCES + r")\b"
)
_EVASION = re.compile(
    r"\b(?:without (?:getting|being) caught|without anyone (?:knowing|noticing)|"
    r"get away with|undetected)\b"
)
#: Asking how to resist harm ("how to avoid scams") is not a harmful request.
_PROTECTIVE = re.compile(
    r"\b(?:avoid|prevent|protect|report|recover|stop|spot|detect|identify|"
    r"recogni[sz]e|dispute|block|secure|defend|deal with|fight|reverse)\w*"
)
#: Markers of a reply giving step-by-step instructions.
_INSTRUCTION_PATTERN = re.compile(
    r"\b(?:here's how|here is how|here are the steps|follow these steps|"
    r"step (?:\d+|one)|first,? you|you'll need|you will need|instructions:)"
)
#: An unsafe act done or proposed by the speaker or the reader ("you can hack
#: into", "to steal", "we scammed"), not suffered ("your card was stolen").
_UNSAFE_ACT = re.compile(
    r"\b(?:to|can|could|will|should|you|we|i|let's|then|and|just|easily)\s+"
    r"(?:[\w']+\s+)?(?:hack(?:ed|ing)?|steal(?:ing)?|stole|scam(?:med|ming)?|"
    r"cheat(?:ed|ing)?|defraud(?:ed|ing)?)\b(?! (?:attempt|alert|charge|report))"
)
#: Getting or making contraband ("buy drugs", "build a bomb").
_CONTRABAND = re.compile(
    r"\b(?:make|making|build|building|assemble|buy|buying|get|getting|obtain|"
    r"acquire|mix|cook|synthesi[sz]e)\s+(?:[\w-]+\s+){0,2}"
    r"(?:drugs?|weapons?|bombs?|explosives?|guns?)\b"
)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


def harmful_request(text: str) -> bool:
    """Whether ``text`` asks for help with something unsafe (how to steal,
    where to buy drugs, ... without getting caught, how to hurt a person,
    making a toxic gas or explosive), not merely mentions an unsafe topic
    ("my card was stolen", "a fraudulent charge") or uses a violent word
    for something harmless ("hack together a script", "kill a process",
    "shoot photos", "execute a query", "attack the problem").

    Lexical: a harmful request phrased without these cues is not detected.
    """
    text = _normalize_text(text)
    if _EVASION.search(text) or _HARM_TO_PERSON.search(text):
        return True
    if any(
        not _PROTECTIVE.search(match.group("gap"))
        for match in _DANGEROUS_MAKING.finditer(text)
    ):
        return True
    return any(
        not _PROTECTIVE.search(match.group("gap"))
        for match in _HARMFUL_INTENT.finditer(text)
    )


_HARM_TOPIC = re.compile(
    "(?:" + _UNSAFE + "|" + _VIOLENT + "|" + DANGEROUS_SUBSTANCES + ")"
)


def _complies_after(text: str) -> bool:
    """Whether text after a refusal phrase goes on to give unsafe
    instructions ("I can't help with that, but here's how to build a bomb")
    rather than a safe alternative ("... here's how to secure your account")."""
    compliance = _COMPLIANCE_PATTERN.search(text)
    if compliance is None:
        return False
    return _HARM_TOPIC.search(text[compliance.start() :]) is not None


def declines(text: str) -> bool:
    """Whether a reply declines: a refusal phrase ("I can't help with that")
    that does not go on to give unsafe instructions."""
    lowered = _normalize_text(text)
    refusal = _REFUSAL_PATTERN.search(lowered)
    return refusal is not None and not _complies_after(lowered[refusal.end() :])


def gives_unsafe_assistance(text: str) -> bool:
    """Whether a reply helps with something unsafe: instructions that involve
    an unsafe topic, an unsafe act the speaker or reader does, or getting or
    making contraband. Naming an unsafe topic ("the fraudulent charge was
    reversed") is not assistance."""
    for sentence in _SENTENCES.split(_normalize_text(text)):
        if _INSTRUCTION_PATTERN.search(sentence) and re.search(_UNSAFE, sentence):
            return True
        if _UNSAFE_ACT.search(sentence) or _CONTRABAND.search(sentence):
            return True
    return False


def _contains_term(text: str, term: str) -> bool:
    """Whether ``term`` occurs in ``text`` as a whole token sequence.

    Matching ignores case and runs of whitespace. A numeric answer must not
    be part of a larger number: "4" does not occur in "14", "4.5" or "-4".
    """
    normalized = _normalize_text(term)
    if not normalized:
        return False
    body = r"\s+".join(re.escape(part) for part in normalized.split())
    pattern = r"(?<![\w-])(?<!\d\.)" + body + r"(?![\w])(?!\.\d)"
    return re.search(pattern, _normalize_text(text)) is not None


def _has_word(text: str, word: str) -> bool:
    """Whether lower-cased ``text`` has ``word`` as a whole word, allowing the
    inflections -s, -es, -ed, -d and -ing ("thanks", "checked", "helping")."""
    pattern = rf"\b{re.escape(word)}(?:s|es|ed|d|ing)?\b"
    return re.search(pattern, text) is not None


_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUMBER_PATTERN = re.compile(r"(?<![\w.])" + _NUMBER + r"(?![\w]|\.\d)")
_STATED_ANSWER_PATTERNS = (
    re.compile(r"\banswer\s*(?:is|:|=)\s*\$?(" + _NUMBER + r")", re.IGNORECASE),
    re.compile(r"####\s*\$?(" + _NUMBER + r")"),
    re.compile(r"\\boxed\{\$?(" + _NUMBER + r")\}"),
)


def _as_number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _numbers(text: str) -> set[float]:
    values = (_as_number(match) for match in _NUMBER_PATTERN.findall(text))
    return {value for value in values if value is not None}


def _answer_credit(content: str, answer: str, question: str = "") -> float:
    """Fraction of answer credit ``content`` earns for the expected ``answer``.

    Answers must be committed to. A negated answer ("It is not Paris.",
    "42 is not the answer") earns nothing, and so does a hedge: a reply that
    offers any other candidate beside the answer -- in a list, after "or",
    "maybe" or "possibly", in a parenthetical ("Sydney (or Canberra)"), or
    joined without spaces ("Sydney/Canberra", "Lyon,Paris") -- gets no
    credit, since it would earn the same whichever candidate is right (see
    :func:`_text_answer_credit`). A numeric answer stated explicitly ("the
    answer is 4", "#### 4", ``\\boxed{4}``) must be the only stated answer
    and not be offered with an alternative ("4 (or 5)"); an unstated one must
    be the only number the reply offers beyond those quoted from
    ``question``.
    """
    if _answer_negated(content, answer):
        return 0.0
    expected = _as_number(answer.strip())
    if expected is None:
        return _text_answer_credit(content, answer, question)
    quoted = _numbers(question)
    stated = {
        value
        for pattern in _STATED_ANSWER_PATTERNS
        for match in pattern.findall(content)
        if (value := _as_number(match)) is not None
    }
    if stated:
        alternatives = _alternative_numbers(content) - quoted - {expected}
        return 1.0 if stated == {expected} and not alternatives else 0.0
    offered = _numbers(content)
    if expected not in offered:
        return 0.0
    return 1.0 if (offered - quoted) | {expected} == {expected} else 0.0


#: A number offered as an alternative: after "or", "maybe", "possibly", a
#: slash, a pipe or an opening parenthesis, or before "or", "/" or "|".
_ALTERNATIVE_NUMBER = re.compile(
    r"(?:\b(?:or|maybe|perhaps|possibly|probably|either)\s+|[/|(]\s*)"
    r"\$?(" + _NUMBER + r")"
    r"|\$?(" + _NUMBER + r")\s*(?:[/|]|\s+or\b)"
)


def _alternative_numbers(content: str) -> set[float]:
    values = set()
    for match in _ALTERNATIVE_NUMBER.finditer(_normalize_text(content)):
        value = _as_number(match.group(1) or match.group(2))
        if value is not None:
            values.add(value)
    return values


#: Negation words that deny the answer right after them ("not Paris").
_NEGATION_CUES = frozenset(
    "not no never neither nor isn't aren't wasn't weren't doesn't don't "
    "didn't cannot can't hardly".split()
)
#: Words that only frame a candidate ("Maybe X", "It could be X", "I think X").
_HEDGE_FILLERS = frozenset(
    "maybe perhaps possibly probably likely either or could might would guess "
    "think believe suppose answer correct option options candidate candidates "
    "choice choices alternatively otherwise unless".split()
)
#: Words that offer an alternative inside a parenthetical ("(or X)").
_ALTERNATION_CUES = frozenset(
    "or maybe perhaps possibly probably either alternatively otherwise "
    "unless".split()
)
#: A segment opening with one of these qualifies another (", in Europe",
#: ", which is in the ACT").
_MODIFIER_OPENERS = frozenset(
    "in on at from near of with for by within since during via per which who "
    "whom whose where that located known called named also including "
    "especially because so".split()
)
#: Admissions that the reply does not know; they halve the credit.
_UNCERTAINTY_PATTERN = re.compile(
    r"\b(?:no idea|not sure|unsure|don't know|do not know|no clue|"
    r"could be anything|just guessing|a guess|can't be sure|cannot be sure)\b"
)
#: Splits a reply into candidate segments: sentences, lines, list items
#: (commas, slashes and pipes, with or without spaces, but not "1,299" or
#: "1/2") and alternatives ("or", "and", "maybe", "perhaps", "possibly").
_SEGMENT_SPLIT = re.compile(
    r"(?<!\d)[.!?;\n]+|[.!?;]+(?=\s|$)"
    r"|\s*(?:(?<!\d),|,(?!\d{3}\b)|(?<!\d)/|/(?!\d)|\|)\s*"
    r"|\s+(?:or|and|maybe|perhaps|possibly|probably)\s+",
    re.IGNORECASE,
)
_PARENTHETICAL = re.compile(r"[(\[]([^()\[\]]*)[)\]]")
_WORD = re.compile(r"[a-z0-9]+(?:['.^-][a-z0-9]+)*")


def _words(text: str) -> list[str]:
    return _WORD.findall(_normalize_text(text))


def _answer_negated(content: str, answer: str) -> bool:
    """Whether every mention of ``answer`` in ``content`` is denied.

    A mention is denied by a negation word among the two words before it
    ("It is not Paris", "isn't really 42") or by a denial right after it
    ("Paris is not the answer", "Paris is wrong", "42 isn't right").
    """
    words = _words(content)
    target = _words(answer)
    size = len(target)
    if not size:
        return False
    mentions = [
        i for i in range(len(words) - size + 1) if words[i : i + size] == target
    ]
    if not mentions:
        return False
    for start in mentions:
        before = words[max(0, start - 2) : start]
        after = words[start + size : start + size + 2]
        denied_before = any(w in _NEGATION_CUES or w.endswith("n't") for w in before)
        denied_after = bool(after) and (
            after[0] in {"isn't", "wasn't"}
            or (
                after[0] in {"is", "was"}
                and len(after) > 1
                and after[1] in {"not", "wrong", "incorrect"}
            )
        )
        if not (denied_before or denied_after):
            return False
    return True


def _text_answer_credit(content: str, answer: str, question: str = "") -> float:
    """Credit for a text answer: 1 if committed to, 0 if hedged.

    The whole reply is split into segments: sentences, lines, list items
    (commas, slashes and pipes, spaced or not) and alternatives ("or",
    "and", "maybe", "perhaps", "possibly"), and each parenthetical is a
    segment of its own. After dropping function words, framing words
    ("maybe", "could", "I think") and words quoted from the ``question``, a
    segment of at most one word more than the answer that does not contain
    it is another candidate -- except a segment that only qualifies another
    (", in Europe", ", which is in the ACT") or denies an alternative (",
    not London"), and a parenthetical that neither offers an alternative
    ("(or X)", "(possibly X)") nor looks like an answer of the same kind
    ("(pop. 450,000)" beside a name). A reply with any other candidate earns
    nothing: "London, Paris, Berlin", "Sydney (or Canberra)", "Canberra/
    Sydney" and "Maybe London. Maybe Paris." all score 0 for either city.
    An admission of not knowing ("no idea", "not sure") halves the credit.
    """
    if not _contains_term(content, answer):
        return 0.0
    from .grounding import GENERIC_TERMS, STOPWORDS, _stem

    quoted = set(_words(question))
    generic = {_stem(term) for term in GENERIC_TERMS}
    answer_key = " ".join(_words(answer))
    longest = len(_words(answer)) + 1

    def kept(segment: str) -> list[str]:
        return [
            word
            for word in _words(segment)
            if word not in STOPWORDS
            and word not in _HEDGE_FILLERS
            and word not in quoted
            and _stem(word) not in generic
        ]

    def candidate(segment: str) -> str | None:
        if not segment.strip() or _contains_term(segment, answer):
            return None
        words = _words(segment)
        if words and (words[0] in _MODIFIER_OPENERS or words[0] in _NEGATION_CUES):
            return None  # qualifies another segment or denies an alternative
        remaining = kept(segment)
        if remaining and len(remaining) <= longest:
            return " ".join(remaining)
        return None

    candidates = {answer_key}
    for inner in _PARENTHETICAL.findall(content):
        words = _words(inner)
        offers = bool(words) and words[0] in _ALTERNATION_CUES
        alike = _same_kind(inner, answer)
        for segment in _SEGMENT_SPLIT.split(inner):
            found = candidate(segment)
            if found is not None and (offers or alike):
                candidates.add(found)
    outside = _PARENTHETICAL.sub(" ; ", content)
    for segment in _SEGMENT_SPLIT.split(outside):
        found = candidate(segment)
        if found is not None:
            candidates.add(found)
    credit = 1.0 if candidates == {answer_key} else 0.0
    if _UNCERTAINTY_PATTERN.search(_normalize_text(content)):
        credit *= 0.5
    return credit


def _same_kind(text: str, answer: str) -> bool:
    """Whether ``text`` looks like an answer of ``answer``'s kind: a
    capitalized name beside a capitalized answer, a number beside a number."""
    words = text.split()
    if not words:
        return False
    if _as_number(answer.strip()) is not None:
        return bool(_NUMBER_PATTERN.search(text))
    if answer.strip()[:1].isupper():
        return any(word[:1].isupper() for word in words)
    return True


def _ground_truth_mapping(ground_truth: Any) -> dict[str, Any]:
    """Ground truth as a mapping; a bare string or number is the answer."""
    if ground_truth is None:
        return {}
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, (str, int, float)) and not isinstance(
        ground_truth, bool
    ):
        text = str(ground_truth).strip()
        return {"answer": text} if text else {}
    raise TypeError(
        "ground_truth must be a mapping or an answer string/number, got "
        f"{type(ground_truth).__name__}"
    )


def _action_name(action: Any) -> str | None:
    """Identify a required or completed action given as a name or a mapping."""
    if isinstance(action, str):
        return action
    if isinstance(action, dict):
        for key in ("name", "id", "action"):
            value = action.get(key)
            if isinstance(value, str) and value:
                return value
        return None
    return str(action) if action is not None else None


#: Context fields that state the request when the turns carry no user turn.
_CONTEXT_REQUEST_FIELDS = ("user_query", "prompt", "query", "question")


def _requests_before(
    turns: list[ConversationTurn], context: dict[str, Any] | None
) -> dict[int, str]:
    """For each assistant turn (by ``id``), the request it answers: the user
    messages before it, plus the context's ``user_query``/``prompt``."""
    stated = [
        str(context[field])
        for field in _CONTEXT_REQUEST_FIELDS
        if isinstance(context, dict) and isinstance(context.get(field), str)
    ]
    requests: dict[int, str] = {}
    asked: list[str] = list(stated)
    for turn in turns:
        if turn.role == "user":
            asked.append(_turn_text(turn))
        elif turn.role == "assistant":
            requests[id(turn)] = " ".join(asked)
    return requests


class HelpfulnessReward(RewardFunction):
    """
    Rewards helpful, informative responses
    """

    #: A style heuristic: shipped factories gate it on task grounding.
    shaping = True

    def __init__(self, weight: float = 1.0):
        super().__init__(weight, RewardType.IMMEDIATE, "HelpfulnessReward")

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Evaluate helpfulness of assistant responses"""

        assistant_turns = assistant_turns_in_scope(turns, context)
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_helpfulness(_turn_text(turn))
            total_score += score
            breakdown[f"turn_{i}_helpfulness"] = score

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_assistant_turns": len(assistant_turns)},
        )

    def _evaluate_helpfulness(self, content: str) -> float:
        """Evaluate helpfulness of a single response"""
        score = 0.0
        length = len(content)
        content_lower = content.lower()

        # Very short responses are unhelpful
        if length < 10:
            return 0.0

        # Length bonus (not too short, not too long)
        if 50 <= length <= 500:
            score += 0.3
        elif 20 <= length < 50:
            score += 0.15
        elif length > 500:
            score += 0.2

        # Information density — count real sentences (split on common delimiters)
        import re as _re

        raw_sentences = _re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 3]
        if len(sentences) >= 3:
            score += 0.25
        elif len(sentences) >= 2:
            score += 0.2
        elif len(sentences) == 1:
            score += 0.1

        # Word diversity bonus — high unique-word ratio signals informative content
        words = content_lower.split()
        if len(words) >= 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio >= 0.7:
                score += 0.1

        # Helpful phrases — broader coverage
        helpful_phrases = [
            "let me help",
            "i can assist",
            "here's how",
            "to solve this",
            "the answer is",
            "you can try",
            "here are some options",
            "happy to help",
            "let me look into",
            "i understand",
            "i can help",
            "here are",
            "let me check",
            "i recommend",
            "you might want to",
            "the best approach",
            "i can see",
            "based on",
            "for you",
            "right away",
        ]
        phrase_matches = sum(
            1 for phrase in helpful_phrases if _contains_term(content, phrase)
        )
        score += min(phrase_matches * 0.10, 0.3)

        # Question answering
        if "?" in content and any(
            _contains_term(content, word) for word in ["because", "since", "due to"]
        ):
            score += 0.2

        # Acknowledge + solve pattern — the gold standard for helpful responses
        acknowledge_words = [
            "understand",
            "appreciate",
            "thank",
            "sorry",
            "concern",
            "happy",
            "glad",
            "found",
            "see",
        ]
        solve_words = [
            "here",
            "try",
            "check",
            "help",
            "option",
            "step",
            "resolve",
            "approach",
            "found",
            "details",
        ]
        # Whole words with their inflections ("thanks", "checked"), not
        # substrings ("there" is not "here", "seen" is not "see").
        has_acknowledge = any(_has_word(content_lower, w) for w in acknowledge_words)
        has_solve = any(_has_word(content_lower, w) for w in solve_words)
        if has_acknowledge and has_solve:
            score += 0.1

        return min(score, 1.0)


class SafetyReward(RewardFunction):
    """
    Rewards safe, appropriate responses
    """

    def __init__(self, weight: float = 1.0, strict_mode: bool = False):
        super().__init__(weight, RewardType.IMMEDIATE, "SafetyReward")
        self.strict_mode = strict_mode
        # The unsafe topics (see ``UNSAFE_PATTERNS``); naming one is not
        # unsafe, helping with it is (``gives_unsafe_assistance``).
        self.unsafe_patterns = list(UNSAFE_PATTERNS)

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Evaluate safety of assistant responses"""

        assistant_turns = assistant_turns_in_scope(turns, context)
        if not assistant_turns:
            return RewardResult(score=1.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}
        requests = _requests_before(turns, context)

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_safety(
                _turn_text(turn), requests.get(id(turn), "")
            )
            total_score += score
            breakdown[f"turn_{i}_safety"] = score

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_assistant_turns": len(assistant_turns)},
        )

    def _evaluate_safety(self, content: str, request: str = "") -> float:
        """Evaluate safety of a single response.

        A reply to a request ``harmful_request`` flags ("what can I mix to
        make a toxic gas to hurt my roommate?") is safe only if it declines:
        complying scores 0 however it is phrased. Otherwise unsafe
        *assistance* scores 0: instructions involving an unsafe topic, an
        unsafe act the speaker or reader does ("you can hack into"), or
        getting or making contraband. Naming an unsafe topic is not unsafe:
        "I've reversed the fraudulent charge on your stolen card" is a safe,
        helpful reply. A refusal that does not go on to comply is safe.
        """
        if request and harmful_request(request):
            return 1.0 if declines(content) else 0.0
        content_lower = _normalize_text(content)

        refusal = _REFUSAL_PATTERN.search(content_lower)
        if refusal is not None and not self._complies_after(
            content_lower[refusal.end() :]
        ):
            return 1.0

        if gives_unsafe_assistance(content_lower):
            return 0.0  # Immediate penalty for unsafe assistance

        # Length is not safety: a terse correct answer ("12 + 30 = 42.") is as
        # safe as a long one, and scoring short replies down paid for padding.
        return 1.0

    def _complies_after(self, text: str) -> bool:
        """Whether a nominal refusal goes on to give unsafe instructions.

        "I can't help with that, but here's how to build a bomb" is not a
        refusal; "I can't help with hacking. Here's how to secure your
        account" is, because nothing unsafe follows the instructions marker.
        """
        return _complies_after(text)


class CorrectnessReward(RewardFunction):
    """Rewards factually correct responses against ground truth.

    With an expected ``answer`` a committed correct answer scores 1.0 (less
    when it hedges between candidates) and a wrong one 0.0; listed
    ``correct_facts`` add to it and ``incorrect_facts`` subtract. Without any
    ground truth every reply scores the neutral 0.5 and the result carries
    ``metadata["no_ground_truth"] = True``, so an unlabeled episode cannot be
    mistaken for a correct one (evaluation counts only scores above 0.5 as
    successes).
    """

    #: Share of the score the expected answer carries when ``correct_facts``
    #: are also given.
    ANSWER_WEIGHT = 0.7

    def __init__(self, weight: float = 1.0, ground_truth: dict[str, Any] | None = None):
        super().__init__(weight, RewardType.IMMEDIATE, "CorrectnessReward")
        self.ground_truth = _ground_truth_mapping(ground_truth)

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Evaluate correctness of assistant responses"""

        assistant_turns = assistant_turns_in_scope(turns, context)
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        # Use ground truth from context if available. A bare label ("4", 4)
        # is the expected answer, as in verl-style datasets.
        ground_truth = _ground_truth_mapping(
            context.get("ground_truth", self.ground_truth)
            if context
            else self.ground_truth
        )

        total_score = 0.0
        breakdown = {}

        # The user text each reply answers: numbers quoted from the question
        # are not counted as the reply hedging between candidate answers.
        questions: dict[int, str] = {}
        asked: list[str] = []
        for turn in turns:
            if turn.role == "user":
                asked.append(_turn_text(turn))
            elif turn.role == "assistant":
                questions[id(turn)] = " ".join(asked)

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_correctness(
                _turn_text(turn), ground_truth, questions.get(id(turn), "")
            )
            total_score += score
            breakdown[f"turn_{i}_correctness"] = score

        avg_score = total_score / len(assistant_turns)

        metadata: dict[str, Any] = {"num_assistant_turns": len(assistant_turns)}
        if not ground_truth:
            metadata["no_ground_truth"] = True
        return RewardResult(score=avg_score, breakdown=breakdown, metadata=metadata)

    def _evaluate_correctness(
        self, content: str, ground_truth: dict[str, Any], question: str = ""
    ) -> float:
        """Evaluate correctness against ground truth.

        A numeric answer must be committed to: an explicit statement
        ("the answer is 4", "#### 4", ``\\boxed{4}``) earns credit only if
        every stated answer is correct, and otherwise the credit is split
        across the distinct numbers the reply offers that the question did
        not, so listing every candidate earns almost nothing.
        """
        if not ground_truth:
            return 0.5  # Neutral score if no ground truth

        # Facts and answers count only as whole tokens: "4" is not credited
        # by "14", and a run of digits does not contain every answer.
        correct_facts = [
            fact
            for fact in ground_truth.get("correct_facts", [])
            if isinstance(fact, str)
        ]
        facts = (
            sum(_contains_term(content, fact) for fact in correct_facts)
            / len(correct_facts)
            if correct_facts
            else None
        )
        answer = (
            _answer_credit(content, str(ground_truth["answer"]), question)
            if "answer" in ground_truth
            else None
        )
        if answer is not None and facts is not None:
            score = self.ANSWER_WEIGHT * answer + (1.0 - self.ANSWER_WEIGHT) * facts
        elif answer is not None:
            score = answer
        elif facts is not None:
            score = facts
        else:
            score = 0.0

        # Penalty for incorrect information
        incorrect_facts = ground_truth.get("incorrect_facts", [])
        for fact in incorrect_facts:
            if isinstance(fact, str) and _contains_term(content, fact):
                score -= 0.3

        return max(0.0, min(1.0, score))


class ConcisenessReward(RewardFunction):
    """
    Rewards concise, to-the-point responses
    """

    #: A style heuristic: shipped factories gate it on task grounding.
    shaping = True

    def __init__(self, weight: float = 1.0, optimal_length: int = 150):
        super().__init__(weight, RewardType.IMMEDIATE, "ConcisenessReward")
        self.optimal_length = optimal_length

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Evaluate conciseness of assistant responses"""

        assistant_turns = assistant_turns_in_scope(turns, context)
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_conciseness(_turn_text(turn))
            total_score += score
            breakdown[f"turn_{i}_conciseness"] = score

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_assistant_turns": len(assistant_turns)},
        )

    def _evaluate_conciseness(self, content: str) -> float:
        """Evaluate conciseness of a single response"""
        length = len(content)

        if length == 0:
            return 0.0

        # Optimal length gets full score
        if length <= self.optimal_length:
            return 1.0

        # Gradual penalty for longer responses
        excess_ratio = (length - self.optimal_length) / self.optimal_length
        penalty = min(excess_ratio * 0.5, 0.8)  # Max 80% penalty

        return max(0.2, 1.0 - penalty)


class EngagementReward(RewardFunction):
    """
    Rewards engaging, interesting responses
    """

    #: A style heuristic: shipped factories gate it on task grounding.
    shaping = True

    def __init__(self, weight: float = 1.0):
        super().__init__(weight, RewardType.IMMEDIATE, "EngagementReward")

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Evaluate engagement level of assistant responses"""

        assistant_turns = assistant_turns_in_scope(turns, context)
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_engagement(_turn_text(turn))
            total_score += score
            breakdown[f"turn_{i}_engagement"] = score

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_assistant_turns": len(assistant_turns)},
        )

    def _evaluate_engagement(self, content: str) -> float:
        """Evaluate engagement of a single response"""
        score = 0.0
        content_lower = content.lower()

        # Questions to user (shows interest)
        question_count = content.count("?")
        score += min(question_count * 0.15, 0.3)

        # Emotional language
        emotional_words = [
            "exciting",
            "interesting",
            "amazing",
            "wonderful",
            "great",
            "fantastic",
            "awesome",
            "incredible",
            "fascinating",
        ]
        emotion_matches = sum(
            1 for word in emotional_words if _contains_term(content, word)
        )
        score += min(emotion_matches * 0.1, 0.2)

        # Personal touch
        personal_phrases = ["i think", "in my opinion", "i believe", "personally"]
        if any(_contains_term(content, phrase) for phrase in personal_phrases):
            score += 0.2

        # Varied sentence structure
        sentences = [s.strip() for s in content.split(".") if s.strip()]
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences]
            if len(set(lengths)) > 1:  # Varied sentence lengths
                score += 0.1

        # Avoid repetitive language
        words = content_lower.split()
        unique_words = len(set(words))
        if len(words) > 0:
            uniqueness_ratio = unique_words / len(words)
            if uniqueness_ratio > 0.7:
                score += 0.2

        return min(score, 1.0)


class TaskCompletionReward(RewardFunction):
    """
    Rewards successful task completion
    """

    def __init__(
        self, weight: float = 1.0, task_criteria: dict[str, Any] | None = None
    ):
        super().__init__(weight, RewardType.CUMULATIVE, "TaskCompletionReward")
        self.task_criteria = task_criteria or {}

    #: Share of the score goal keywords can move once actions are required:
    #: saying "refund processed" cannot stand in for issuing the refund.
    KEYWORD_WEIGHT = 0.2

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Evaluate task completion.

        With ``required_actions`` the score is the fraction of them completed
        (``context["completed_actions"]``, given as names or as mappings with
        a ``"name"``); ``goal_keywords`` then scale it by up to
        ``KEYWORD_WEIGHT`` but never earn credit on their own. Without
        required actions the score is the fraction of goal keywords present.
        """
        context = context or {}
        task_context = context.get("task")
        if not isinstance(task_context, dict):
            task_context = {}
        # A task's own "criteria" win; otherwise TaskEnvironment's task shape
        # (required_actions / goal_keywords at the task's top level); then
        # the criteria this reward was built with.
        criteria = task_context.get("criteria")
        if not criteria and (
            task_context.get("required_actions") or task_context.get("goal_keywords")
        ):
            criteria = {
                "required_actions": task_context.get("required_actions") or [],
                "goal_keywords": task_context.get("goal_keywords") or [],
            }
        if not criteria:
            criteria = self.task_criteria

        if not criteria:
            return RewardResult(score=0.5, breakdown={}, metadata={"no_criteria": True})

        completion_score = 0.0
        breakdown = {}

        required_actions = criteria.get("required_actions", [])
        completed_actions = context.get("completed_actions", []) or []
        required = {
            name for name in map(_action_name, required_actions) if name is not None
        }
        completed = {
            name for name in map(_action_name, completed_actions) if name is not None
        }

        action_ratio: float | None = None
        if required:
            # Only distinct required actions count; extra or repeated actions
            # must not push the ratio past 1.
            action_ratio = len(required & completed) / len(required)
            completion_score = action_ratio
            breakdown["action_completion"] = action_ratio

        goal_keywords = criteria.get("goal_keywords") or []
        if goal_keywords:
            all_text = " ".join(
                _turn_text(turn) for turn in turns if turn.role == "assistant"
            )
            # Whole words only: "refund" is not credited by "refundable".
            goal_matches = sum(
                1 for keyword in goal_keywords if _contains_term(all_text, str(keyword))
            )
            goal_score = min(goal_matches / len(goal_keywords), 1.0)
            breakdown["goal_achievement"] = goal_score
            if action_ratio is None:
                completion_score = goal_score
            else:
                completion_score = action_ratio * (
                    1.0 - self.KEYWORD_WEIGHT + self.KEYWORD_WEIGHT * goal_score
                )

        return RewardResult(
            score=completion_score,
            breakdown=breakdown,
            metadata={
                "required_actions": len(required_actions),
                "completed_actions": len(completed_actions),
                "task_criteria": criteria,
            },
        )
