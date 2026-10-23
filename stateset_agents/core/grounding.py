"""Task grounding: does a reply answer *this* request with supported content?

Heuristic style rewards (helpfulness phrases, empathy and action keywords,
length, engagement) can be maximized without answering the request: a canned
template, keyword stuffing, an echo of the request, word salad, a list of
invented account actions ("card 4093 blocked; fee 8812 waived"), a refusal, or
the answer said twice. The shipped heuristic factories therefore gate their
style terms on *grounding*, a request-specific score in ``[0, 1]``::

    grounding = information * (0.5 + 0.5 * relevance) * quality * support
                * (1 - repeated) * max(0, 1 - 2 * echoed) * (1 - off_request)

**Anchoring.** A reply is read clause by clause (split at punctuation and at
"and"/"or"/"but"/"then"/"while"/"plus"). A clause naming a request term, a
reference label or an evidence term is *anchored*: the first clause a term
anchors gets weight 1, the second 0.5, later ones nothing (``ANCHOR_WEIGHTS``);
a weak term -- boilerplate (``GENERIC_TERMS``), an action word, or one only the
scenario description names (``context``, ``user_goal``, ``topic``...) --
anchors at half that (``WEAK_ANCHOR``). A clause continues the anchored clause
before it after a comma, parenthesis or conjunction, or when it opens with
"then", "which", "it", "this"... (``INHERIT_DEPTH`` = 3 clauses; a reference
back restarts the chain once per anchor). Clauses after a semicolon or colon,
and new sentences, stand alone. The request's own opening words copied to the
front of a reply anchor nothing.

**Specifics** are numbers and amounts (two or more characters), affirmed
action words (``ACTION_TERMS``; "not issued" is not one) and code identifiers
("sqlite3.connect", "fetchall("), excluding request terms and answer labels.
A specific is *supported* with the weight of a clause stating it if it lies
within ``PROXIMITY`` (8) content words of an anchor term, or -- in a continuing
clause -- within the anchor's budget of ``INHERIT_SPECIFICS`` (4); a bare list
item ("card 4093 blocked", no function word, not an instruction) continuing a
clause states nothing. A specific found in the conversation's evidence (tool,
observation and system turns; the context's ``observations``,
``tool_outputs``, ``retrieved_facts``, ``correct_facts``... fields) is supported
with weight 1 wherever it appears. Anything else is *unsupported*.

**information** is ``1 - exp(-units / INFO_SCALE)`` (``INFO_SCALE`` = 6), with::

    units = max(S, E) + 3 * answered_types + 0.5 * min(4, steps)

* ``S``: the summed support of the distinct supported specifics, at most
  ``2 + 2 * w`` where ``w`` counts the new words (not request terms,
  boilerplate or specifics) of the anchored clauses -- specifics need context;
* ``E``: elaboration, 0.25 per new word in anchored clauses, at most 1 (it
  stands in for specifics in an answer without them and adds nothing beside
  them);
* ``answered_types`` (at most 2): the question's answer types the reply meets
  in an anchored clause -- a capitalized name for "who", a cause ("because",
  "so", "due to") with new words for "why", an instruction step for "how" /
  "what should I", a number for "how many/much", "when" or a cost, a
  location phrase for "where", "yes"/"no" for a yes/no question, and a
  definition ("X is/has ...", not negated) for "what is";
* ``steps``: distinct anchored instructions -- an instruction verb opening a
  clause or following "and"/"then"/"first", or "you can/should <verb>" --
  that state something new.

With reference labels (``ground_truth``), a short label is the answer:
``information = max(answer_credit, 0.5 * (1 - exp(-units / 6)))`` with no type
units, where ``answer_credit`` is ``CorrectnessReward``'s (0 for a hedged or
negated answer). Numbers other than a numeric answer are competing answers,
not specifics.

**relevance** is ``min(1, shared / min(3, known))``: request and label terms
the reply mentions. **quality** is ``(1 - b) * (1 - (b + u) ** 2)`` where ``b``
and ``u`` are the shares of boilerplate and of unrecognized words among the
reply's content words plus its function words (counted up to one per content
word). **support** is ``(units + 1) / (units + unsupported + 1)``: every
unsupported specific costs. **repeated** is the share of anchored tokens that
repeat a sentence or clause, a run of four tokens, a template refilled with
other request terms, numbers and actions ("The pixel 4278 was refunded. The
wi-fi 2237 was escalated."), or a request term mentioned more than twice.
**echoed** is the share of anchored tokens copied from a request message in a
run of at least half of it (and six tokens), or four fifths of a short one, or
in a clause of nothing but request terms. **off_request** is the share of
content words in clauses tied to nothing. Repetition and echo are measured
against the anchored text, so appending unrelated text cannot dilute them.

**Refusals.** When ``basic_rewards.harmful_request`` flags the request
("how do I hack into my ex's email", "make a toxic gas to hurt my roommate",
"... without getting caught"), declining scores 1 and any other reply 0.
Otherwise a reply that declines keeps at most ``REFUSAL_FACTOR`` (half) of
its grounding, and a blanket refusal has none (0). Benign uses of
trigger words ("hack together a script", "kill a process", "shoot photos",
"execute a query", "attack the problem") are not harmful intent.

Grounding is lexical: it cannot tell a fabricated number or action stated
beside a request term from a true one, and harmful intent phrased without its
cues is not detected. ``docs/REWARD_CONTRACT.md`` states exactly what the
exploit search in ``stateset_agents.evaluation.reward_exploits`` verifies and
what it does not. Prefer outcome, verifier or judge rewards as objectives.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any

from .basic_rewards import (
    _answer_credit,
    _as_number,
    _normalize_text,
    declines,
    harmful_request,
)
from .reward_base import RewardFunction, RewardResult, RewardType
from .reward_context import scores_latest_turn_only

#: Function words: never evidence of content.
STOPWORDS = frozenset("""
    a about above after again against all also am an and any are aren't as at
    be because been before being below between both but by can can't cannot
    could couldn't did didn't do does doesn't doing don't down during each few
    for from further get got had hadn't has hasn't have haven't having he he'd
    he'll he's her here here's hers herself him himself his how how's i i'd i'll
    i'm i've if in into is isn't it it's its itself just let let's me might
    more most must mustn't my myself no nor not now of off on once only or
    other ought our ours ourselves out over own same shan't she she'd she'll
    she's should shouldn't so some such than that that's the their theirs them
    themselves then there there's these they they'd they'll they're they've
    this those through to too under until up us very was wasn't we we'd we'll
    we're we've were weren't what what's when when's where where's which while
    who who's whom why why's will with won't would wouldn't yes yet you you'd
    you'll you're you've your yours yourself yourselves
    """.split())

#: Request-independent phrasing: service boilerplate and every keyword the
#: built-in heuristics reward (helpfulness phrases, empathy, action,
#: professionalism, engagement and domain lexicons). Saying these is not
#: information; ``tests/unit/test_reward_gaming.py`` checks the heuristic
#: lexicons stay covered so keyword stuffing cannot earn grounding.
GENERIC_TERMS = frozenset("""
    absolutely account action actually advantage again amazing angry annoying
    believe since sorry wtf artistic beautiful buy character choice creative
    descriptive discount exclusive ideal imagine innovative inspiring narrative
    order original purchase save scene story unique vivid
    answer anything apologize appreciate approach assist assistance assistant
    away awesome based benefit benefits best better caused certainly check
    checking click completed concern concerns configuration configure contact
    course customer deal definitely detail details diagnose diagnostic due
    dumb dunno else email empathy everything exciting experience explanation
    fantastic fascinating feature features feel finally first fix fixed
    follow found frustrat frustrated frustrating glad good great guarantee
    happy hate help helpful helping hope idea idk important incredible
    information inspect interesting issue issues kindly know limited look
    looking lol matter may method need needs next note offer okay omg
    opinion option options perfect personalization personally please pleased
    possible problem problems procedure process processed professional
    professionalism provide quality question questions quick quickly reach
    really reason recommend reinstall request resolution resolve resolved
    restart return right satisfaction second see service settings situation
    software solution solutions solve something special step steps stupid
    sucks suggest support sure system team test thank thanks think time today
    totally trust trusted try understand update updated urgency value verify
    visit want warranty way welcome wonderful worry
    """.split())

#: Context fields that describe what the user asked for.
_REQUEST_FIELDS = (
    "user_query",
    "prompt",
    "query",
    "question",
    "context",
    "user_goal",
    "goal",
    "topic",
    "description",
    "instructions",
)
#: Request fields that describe the scenario rather than state the user's
#: words: their terms tie a clause to the request half as strongly
#: (``WEAK_ANCHOR``), like request-independent phrasing.
_BACKGROUND_FIELDS = frozenset(
    {"context", "user_goal", "goal", "topic", "description", "instructions"}
)
_TOKEN = re.compile(r"[a-z0-9]+(?:['./:-][a-z0-9]+)*")


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


_GENERIC = frozenset(_stem(term) for term in GENERIC_TERMS)


#: Verbs of doing: what was done or is to be done for the user ("issued",
#: "blocked", "forget ... and rejoin"). With numbers and reference labels
#: they are the specifics grounding counts as information; any other word a
#: reply adds might be salad. Disjoint from ``GENERIC_TERMS`` (pinned by a
#: test), so stuffing the heuristics' lexicons earns none.
_ACTION_VERBS = """
    issue refund credit reverse block unblock cancel replace ship deliver open
    reopen close reset forget rejoin reconnect disconnect disable enable turn
    switch toggle install uninstall reboot dispute schedule reschedule book
    transfer waive approve deny post escalate unlock lock renew extend upgrade
    downgrade activate deactivate adjust increase decrease reduce raise lower
    multiply divide subtract add differentiate integrate simplify substitute
    factor expand apply set change assign send track charge pay remove delete
    rename move restore recover file submit confirm
    """.split()
_IRREGULAR = {
    "forget": ("forgot", "forgotten"),
    "send": ("sent",),
    "pay": ("paid",),
    "deny": ("denied",),
    "set": (),
    "lock": (),
}


def _inflections(verb: str) -> set[str]:
    forms = {verb, *_IRREGULAR.get(verb, ())}
    if verb.endswith("e"):
        forms |= {verb + "d", verb[:-1] + "ing"}
    elif verb.endswith("y") and verb[-2] not in "aeiou":
        forms |= {verb[:-1] + "ied", verb[:-1] + "ies", verb + "ing"}
    else:
        forms |= {verb + "ed", verb + "ing"}
        if (
            len(verb) <= 4
            and verb[-1] not in "aeiouwxy"
            and verb[-2] in "aeiou"
            and verb[-3] not in "aeiou"
        ):
            # ship -> shipped, shipping
            forms |= {verb + verb[-1] + "ed", verb + verb[-1] + "ing"}
    if verb == "cancel":
        forms |= {"cancelled", "cancelling"}
    return forms


ACTION_TERMS = frozenset(
    _stem(form)
    for verb in _ACTION_VERBS
    for form in _inflections(verb)
    if _stem(form) not in _GENERIC
)


_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _plain(text: str) -> str:
    """Normalized text with "$"/"#" dropped and "1,299" read as "1299"."""
    text = _THOUSANDS.sub("", _normalize_text(text))
    return text.replace("$", " ").replace("#", " ")


def content_terms(text: str) -> set[str]:
    """Distinct content terms of ``text`` (stemmed; no function words)."""
    terms = set()
    for token in _TOKEN.findall(_plain(text)):
        if token in STOPWORDS:
            continue
        if len(token) < 2 and not token.isdigit():
            continue
        terms.add(_stem(token))
    return terms


_DIGIT = re.compile(r"\d")


def _is_number(term: str) -> bool:
    """A number or amount of two or more characters ("42.10", "5-7", "16gb");
    a lone digit is a list marker, not a specific."""
    return len(term) > 1 and _DIGIT.search(term) is not None


def _is_specific(term: str, labels: Collection[str] = ()) -> bool:
    """Whether ``term`` is a specific: a reference label term, a number or
    amount of two or more characters (a lone digit is a list marker), or an
    action word. Any other word, known or invented, is not."""
    if term in labels:
        return True
    if any(ch.isdigit() for ch in term):
        return len(term) > 1
    return term in ACTION_TERMS


#: Share of a gated style reward paid for grounding alone: a grounded reply
#: earns ``grounding * (STYLE_SHARE + (1 - STYLE_SHARE) * style)``, so style
#: adjusts a grounded reply's reward by at most a quarter.
STYLE_SHARE = 0.75
#: Scale of information: ``1 - exp(-units / INFO_SCALE)``. Information keeps
#: growing with every unit, so a reply that drops any of an answer's content
#: scores lower, and saturates, so piling on content gains less and less.
INFO_SCALE = 6.0
#: Specifics count one unit each, at most ``SPECIFIC_FLOOR`` plus
#: ``SPECIFICS_PER_WORD`` per new word stating them.
SPECIFIC_FLOOR = 2.0
SPECIFICS_PER_WORD = 2.0
#: Quality is ``(1 - boilerplate) * (1 - (boilerplate + unknown) **
#: QUALITY_POWER)``: ordinary prose keeps most of its score, padding with
#: words of no recognized kind costs steeply.
QUALITY_POWER = 2.0
#: Weight of echoed tokens: copying the request costs twice its share.
ECHO_WEIGHT = 2.0
#: Weight of the off-request share: content in clauses tied to nothing in the
#: request (fluent padding, an appended answer to another question) costs
#: its share.
OFF_REQUEST_WEIGHT = 1.0
#: Share of its grounding a reply keeps when it declines a request that
#: ``harmful_request`` does not flag ("I can't help with that, but ...").
REFUSAL_FACTOR = 0.5
#: Grounding a reply earns for *declining* a request ``harmful_request``
#: flags. It is a fixed safety credit, not information: a bare refusal never
#: earns full grounding, and it stays above a non-declining reply (which
#: earns 0) so declining a genuinely harmful request still beats complying.
#: ``SafetyReward`` carries the authoritative safety signal (1 for a decline,
#: 0 for compliance); grounding only keeps the ordering for factories that
#: have no safety term.
SAFETY_CREDIT = 0.5
#: Mentions of one request term beyond this many are repetition.
TERM_MENTIONS = 2
#: Units for answering the question's type (a name for "who", a cause for
#: "why", steps for "how", a number for "how many", "yes"/"no", a
#: definition for "what is"); at most ``MAX_TYPES`` types count.
TYPE_UNITS = 3.0
MAX_TYPES = 2
#: Units per distinct instruction step a reply gives, up to ``STEP_CAP``.
STEP_UNITS = 0.5
STEP_CAP = 4
#: Units per distinct new content word in an anchored clause (elaboration),
#: up to ``ELABORATION_CAP`` words: a little credit for explaining, capped so
#: piling on words (salad included) cannot buy information.
ELABORATION_UNITS = 0.25
ELABORATION_CAP = 4
#: How many clauses one request term anchors: the first at full weight, the
#: second at half; repeating a term to anchor more clauses earns nothing.
ANCHOR_WEIGHTS = (1.0, 0.5)
#: Weight of a weak anchor -- request-independent phrasing ("order", "help",
#: "due" in the request) or a term found only in the scenario's description
#: (``context``, ``user_goal``, ``topic``): it ties a clause to the request
#: half as strongly as the user's own words ("#5521", "derivative").
WEAK_ANCHOR = 0.5
#: A specific counts in an anchored clause only within this many content
#: words of a request term.
PROXIMITY = 8
#: Specifics a chain of continuing clauses may add per anchored clause.
INHERIT_SPECIFICS = 4
#: Clauses a continuation ("..., then ...", "It ...") carries an anchor into.
INHERIT_DEPTH = 3
#: Runs copied from the request of at least half its tokens (and at least
#: this many), or four fifths of a request this short or shorter, are echo.
ECHO_RUN = 6
#: Repeated runs of this many tokens (after their first occurrence) are
#: duplication.
REPEAT_RUN = 4

#: Labels of at most this many words are answers (scored with answer
#: credit); longer ones are facts whose terms count as specifics.
_ANSWER_WORDS = 3


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield str(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


#: Context fields that hold evidence the conversation produced or was given:
#: tool outputs, observations, retrieved facts. A specific found here is
#: supported wherever the reply states it.
_EVIDENCE_FIELDS = (
    "observations",
    "observation",
    "tool_outputs",
    "tool_results",
    "retrieved",
    "retrieved_facts",
    "retrieved_documents",
    "documents",
    "knowledge",
    "evidence",
    "facts",
    "correct_facts",
)
#: Turn roles whose content is evidence rather than a request or a reply.
EVIDENCE_ROLES = frozenset({"tool", "function", "observation", "system", "ipython"})


def _request_texts(
    context: Mapping[str, Any] | None,
) -> tuple[list[str], list[str], list[str]]:
    """Request texts, the background ones among them (scenario
    descriptions) and ground-truth labels found in a reward context."""
    if not isinstance(context, Mapping):
        return [], [], []
    texts: list[str] = []
    background: list[str] = []
    labels: list[str] = []
    for source in _context_sources(context):
        for field in _REQUEST_FIELDS:
            value = source.get(field)
            if isinstance(value, str):
                texts.append(value)
                if field in _BACKGROUND_FIELDS:
                    background.append(value)
        labels.extend(_strings(source.get("ground_truth")))
        labels.extend(_strings(source.get("correct_facts")))
    return texts, background, labels


def _context_sources(context: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources = [context]
    for nested in ("scenario", "task"):
        if isinstance(context.get(nested), Mapping):
            sources.append(context[nested])
    return sources


def _evidence_texts(context: Mapping[str, Any] | None) -> list[str]:
    """Evidence strings (tool outputs, observations, facts) in a context."""
    if not isinstance(context, Mapping):
        return []
    found: list[str] = []
    for source in _context_sources(context):
        for field in _EVIDENCE_FIELDS:
            found.extend(_strings(source.get(field)))
    return found


def _turn_field(turn: Any, name: str) -> Any:
    return turn.get(name) if isinstance(turn, Mapping) else getattr(turn, name, None)


_NEGATIONS = frozenset(
    "not no never cannot can't won't didn't don't doesn't isn't wasn't haven't "
    "hasn't hadn't unable".split()
)

#: A token (as ``_TOKEN`` reads it, any case), optionally called ("f("), or a
#: clause separator.
_PIECE = re.compile(
    r"(?P<tok>[A-Za-z0-9]+(?:['./:-][A-Za-z0-9]+)*)(?P<call>\()?"
    r"|(?P<sep>[,;()\[\]\n]|:(?=\s|$)|[.!?]+)"
)
#: Conjunctions that open a clause of their own ("..., card 4093 blocked and
#: fee 8812 waived and ..."): each part must be tied to the request itself or
#: continue a part that is.
_CONJUNCTIONS = frozenset("and or but plus then while".split())
#: Separators that end a sentence (the others end a clause).
_SENTENCE_SEPARATORS = frozenset(".!?\n")
#: Clause openers that continue the clause before them, and sentence openers
#: that refer back to the sentence before.
_CONTINUATIONS = frozenset(
    "and then which who whom whose where that so but while to with by for "
    "because since after before until or plus also thus hence via using if "
    "when once unless as".split()
)
_ANAPHORS = frozenset(
    "it its it's this that these those they their them there he she his her "
    "both each".split()
)
#: Words that only mark a reply as offering steps or an answer, not content.
_CAUSAL = re.compile(
    r"\b(?:because|since|due to|caused by|causes?|so that|so|therefore|thus|"
    r"hence|as a result|which means|that's why|that is why|the reason)\b"
)
#: Verbs that open an instruction ("Open a connection", "Put the camera on").
#: Boilerplate verbs ("try", "check", "click", "restart") are not steps.
INSTRUCTION_VERBS = frozenset(_stem(word) for word in """
    open create call use run find put loop read format get fetch stop force
    press select type enter go choose make write import save copy paste start
    pick keep mount hold wait list search pass print replace iterate convert
    parse compute measure mix heat cook plug unplug connect insert tap swipe
    scroll drag drop sort filter group join split wrap define declare return
    attach detach upload download edit remove delete rename move set add
    install uninstall enable disable turn switch toggle forget rejoin reset
    reboot multiply divide subtract differentiate integrate simplify
    substitute factor expand apply lower raise increase decrease reduce
    lock unlock close kill shoot execute point focus mount frame expose
    """.split())
#: An instruction: a verb opening a clause or following "and", "then",
#: "first", ... or "you can/should/need to".
_STEP = re.compile(
    r"(?:^|\b(?:and|then|first|next|finally|also|just|now)\s+|"
    r"\byou (?:can|could|should|need to|have to|'ll need to|will need to|must)"
    r"\s+)(?P<verb>[a-z]+)\b"
)
#: A location phrase answering "where" ("at your local depot", "in Denver").
_PLACE = re.compile(
    r"\b(?:at|in|on|near|inside|outside|behind|under|from|to|into)\s+"
    r"(?:the|a|an|your|our|their|its)?\s*[a-z]"
)
_COPULAS = frozenset(
    "is are was were be equals equal means mean has have includes include "
    "costs cost features contains comes ships".split()
)
_YES_NO_OPENERS = frozenset(
    "is are was were do does did can could will would should has have".split()
)
_YES_NO_ANSWERS = frozenset("yes no yeah yep nope".split())


class _Clause:
    """One clause of a reply: its tokens and how it attaches to the last."""

    __slots__ = ("raw", "low", "stems", "calls", "sep", "starts_sentence", "index")

    def __init__(self, sep: str, starts_sentence: bool, index: int) -> None:
        self.raw: list[str] = []
        self.low: list[str] = []
        self.stems: list[str] = []
        self.calls: list[bool] = []
        self.sep = sep
        self.starts_sentence = starts_sentence
        self.index = index

    def content(self) -> list[str]:
        """Stems of the clause's content tokens (no function words)."""
        return [
            stem
            for low, stem in zip(self.low, self.stems, strict=True)
            if low not in STOPWORDS and (len(low) > 1 or low.isdigit())
        ]


def _clauses(text: str) -> list[_Clause]:
    """``text`` split into clauses, each remembering its separator."""
    text = text.replace("’", "'").replace("‘", "'")
    text = _THOUSANDS.sub("", text)
    clauses: list[_Clause] = []
    current = _Clause("", True, 0)
    for match in _PIECE.finditer(text):
        token = match.group("tok")
        if token and token.lower() in _CONJUNCTIONS and current.raw:
            # "X and Y": Y is a clause of its own that continues X.
            clauses.append(current)
            current = _Clause("&", False, len(clauses))
        if token:
            current.raw.append(token)
            current.low.append(token.lower())
            current.stems.append(_stem(token.lower()))
            current.calls.append(match.group("call") is not None)
            continue
        separator = match.group("sep")[0]
        boundary = separator in _SENTENCE_SEPARATORS
        if current.raw:
            clauses.append(current)
            current = _Clause(separator, boundary, len(clauses))
        else:
            current.sep = separator if current.sep == "" else current.sep
            current.starts_sentence = current.starts_sentence or boundary
    if current.raw:
        clauses.append(current)
    return clauses


def _token_parts(stem: str) -> set[str]:
    """A token and its dotted/hyphenated parts ("sqlite3.connect")."""
    parts = {stem}
    if any(ch in stem for ch in "./:-"):
        parts |= {_stem(part) for part in re.split(r"[./:-]", stem) if part}
    return parts


class _Anchoring:
    """How each clause of a reply is tied to the request.

    ``weights[i]`` is the clause's anchor weight (0: not tied), ``owners[i]``
    the clause whose anchor it carries (itself for a clause naming a request
    term, ``-1`` when untied) and ``explicit[i]`` whether it names one.
    """

    __slots__ = ("weights", "owners", "explicit")

    def __init__(self, size: int) -> None:
        self.weights = [0.0] * size
        self.owners = [-1] * size
        self.explicit = [False] * size


def _anchor_weights(
    clauses: Sequence[_Clause],
    anchors: set[str],
    weak: Collection[str] = _GENERIC,
    skip: int = 0,
) -> _Anchoring:
    """How strongly each clause is tied to the request (see ``_Anchoring``).

    A clause naming a request term is anchored and uses up one of its terms:
    the first clause a term anchors gets ``ANCHOR_WEIGHTS[0]``, the next
    ``ANCHOR_WEIGHTS[1]``, later ones nothing, so repeating a request term to
    anchor more clauses earns nothing (a clause takes the best weight its
    terms offer and uses up the term it needs least later on; a
    weak term -- "order", or one only the scenario description names -- anchors
    at ``WEAK_ANCHOR``).
    A clause without a request term continues the anchored clause before it
    when it follows a comma, a parenthesis or a conjunction, or opens with a
    continuation ("then", "which") or a reference back ("it", "this"), for at
    most ``INHERIT_DEPTH`` clauses; a reference back may restart that chain
    once per anchored clause. Clauses joined by commas to an anchored clause
    after them share its anchor too. Anything else, including clauses after
    a semicolon or colon, stands alone: off-topic specifics written beside a
    request term ("What derivative: card 4093 blocked; ...") are not
    anchored.
    """
    named = []
    offset = 0
    for clause in clauses:
        terms: set[str] = set()
        for position, (low, stem) in enumerate(
            zip(clause.low, clause.stems, strict=True), start=offset
        ):
            if position >= skip and low not in STOPWORDS:
                terms |= _token_parts(stem) & anchors
        named.append(terms)
        offset += len(clause.stems)
    later = {term: sum(term in terms for terms in named) for term in anchors}
    uses: dict[str, int] = {}
    result = _Anchoring(len(clauses))
    owner, depth, topic = -1, 0, -1
    for index, (clause, terms) in enumerate(zip(clauses, named, strict=True)):
        for term in terms:
            later[term] -= 1
        open_terms = [t for t in terms if uses.get(t, 0) < len(ANCHOR_WEIGHTS)]
        if open_terms:
            term = min(
                open_terms,
                key=lambda t: (t in weak, uses.get(t, 0), later[t], t),
            )
            weight = ANCHOR_WEIGHTS[uses.get(term, 0)]
            uses[term] = uses.get(term, 0) + 1
            result.weights[index] = weight * (WEAK_ANCHOR if term in weak else 1)
            result.owners[index] = index
            result.explicit[index] = True
            owner, depth, topic = index, 0, index
            continue
        opener = clause.low[0] if clause.low else ""
        if opener in _ANAPHORS and topic >= 0:
            # "...; it costs $899": a reference back to the last anchored
            # clause restarts the chain (and its budget), once per anchor.
            result.weights[index] = result.weights[topic]
            result.owners[index] = index
            owner, depth, topic = index, 0, -1
            continue
        continues = (
            clause.sep in ",()[]&" or opener in _CONTINUATIONS or opener in _ANAPHORS
        )
        if owner >= 0 and continues and depth < INHERIT_DEPTH:
            depth += 1
            result.weights[index] = result.weights[owner]
            result.owners[index] = owner
        else:
            owner, depth = -1, 0
    _anchor_backward(clauses, result)
    return result


def _anchor_backward(clauses: Sequence[_Clause], result: _Anchoring) -> None:
    """A sentence may lead with its setting before naming the request term
    ("Sunlight scatters off the air, and blue light ..."): clauses joined by
    commas or conjunctions to an anchored clause after them share its anchor,
    up to ``INHERIT_DEPTH`` back."""
    for index in range(len(clauses) - 1, 0, -1):
        if not result.explicit[index] or result.weights[index - 1]:
            continue
        joined, back = index, 0
        while (
            joined > 0
            and back < INHERIT_DEPTH
            and clauses[joined].sep in ",()[]&"
            and not result.weights[joined - 1]
        ):
            result.weights[joined - 1] = result.weights[index]
            result.owners[joined - 1] = index
            joined, back = joined - 1, back + 1


def _negated(clause: _Clause, index: int) -> bool:
    before = clause.low[max(0, index - 2) : index]
    return any(word in _NEGATIONS or word.endswith("n't") for word in before)


def _code_like(clause: _Clause, index: int) -> bool:
    """An identifier, call or file name ("sqlite3.connect", "fetchall(",
    "draft.txt"): specific in a technical answer."""
    low = clause.low[index]
    if clause.calls[index] and len(low) > 2:
        return True
    parts = [part for part in re.split(r"[.]", low) if part]
    return (
        len(parts) > 1
        and len(low) >= 5
        and all(part[:1].isalpha() for part in parts)
        and max(len(part) for part in parts) >= 3
    )


def _near_anchor(clause: _Clause, index: int, anchors: set[str]) -> bool:
    """Whether token ``index`` is within ``PROXIMITY`` content words of a
    request term in its clause ("the derivative card 4093 blocked the fee
    8812 waived the dispute ..." anchors only its first few specifics)."""
    rank, ranks, anchor_ranks = 0, [], []
    for low, stem in zip(clause.low, clause.stems, strict=True):
        ranks.append(rank)
        if low not in STOPWORDS:
            if _token_parts(stem) & anchors:
                anchor_ranks.append(rank)
            rank += 1
    return any(abs(ranks[index] - other) <= PROXIMITY for other in anchor_ranks)


def _is_statement(clause: _Clause) -> bool:
    """Whether a clause reads as a statement or an instruction ("set ISO
    100") rather than a list item ("card 4093 blocked"): it has a function
    word or opens with an instruction verb."""
    if any(low in STOPWORDS for low in clause.low):
        return True
    return bool(clause.stems) and clause.stems[0] in INSTRUCTION_VERBS


def _clause_specifics(
    clause: _Clause, exclude: set[str], numbers_are_answers: bool
) -> list[tuple[str, int]]:
    """The specifics a clause states: numbers, affirmed actions and code.

    A clause without function words is a list item and states at most one
    specific; a lone list of numbers or verbs states nothing.
    """
    found: list[tuple[str, int]] = []
    seen: set[str] = set()
    for index, (low, stem) in enumerate(zip(clause.low, clause.stems, strict=True)):
        if low in STOPWORDS or stem in exclude or stem in seen:
            continue
        if _is_number(stem):
            specific = not numbers_are_answers
        elif stem in ACTION_TERMS:
            specific = not _negated(clause, index)
        else:
            specific = _code_like(clause, index) and stem not in _GENERIC
        if specific:
            found.append((stem, index))
            seen.add(stem)
    carriers = [
        stem
        for stem in clause.content()
        if stem not in seen and not _is_number(stem) and stem not in ACTION_TERMS
    ]
    has_function_word = _is_statement(clause)
    if not carriers and not has_function_word:
        return []  # a bare list of numbers or verbs states nothing
    if not has_function_word:
        return found[:1]  # a list item ("card 4093 blocked")
    return found


def _answer_types(question: str) -> set[str]:
    """What kind of answer ``question`` asks for (see ``TYPE_UNITS``)."""
    text = _normalize_text(question)
    types: set[str] = set()
    if re.search(r"\bhow (?:many|much|long|old|far|often|big|tall|fast)\b", text):
        types.add("quantity")
    elif re.search(r"\b(?:when|what (?:year|time|date|day))\b", text):
        types.add("quantity")
    if re.search(r"\b(?:cost|costs|price|priced)\b", text):
        types.add("quantity")
    if re.search(r"\b(?:who|whom|whose)\b", text):
        types.add("entity")
    if re.search(r"\bwhere\b", text):
        types.add("place")
    if re.search(r"\bwhy\b", text):
        types.add("reason")
    if re.search(
        r"\bhow (?:do|does|can|could|should|would|to|i|we|you)\b"
        r"|\bwhat (?:\w+ ){0,3}(?:should|do|can) (?:i|we)\b"
        r"|\b(?:steps|instructions) (?:to|for)\b",
        text,
    ):
        types.add("procedure")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if any(s.split()[0] in _YES_NO_OPENERS for s in sentences if s.split()):
        types.add("yes_no")
    if re.search(r"\b(?:what|which)\b", text) and not types - {"yes_no"}:
        types.add("definition")
    return types


class _Reading:
    """A reply read against a request: clauses, anchors and specifics."""

    def __init__(
        self,
        reply: str,
        request_terms: set[str],
        exclude: set[str],
        numbers_are_answers: bool,
        anchors: set[str] | None = None,
        asked: set[str] | None = None,
        weak: Collection[str] = _GENERIC,
        skip: int = 0,
    ) -> None:
        self.clauses = _clauses(reply)
        self.request_terms = request_terms
        anchors = request_terms if anchors is None else anchors
        self.asked = asked if asked is not None else request_terms
        anchoring = _anchor_weights(self.clauses, anchors, weak, skip)
        self.weights = anchoring.weights
        self.explicit = anchoring.explicit
        found = [
            _clause_specifics(clause, exclude, numbers_are_answers)
            for clause in self.clauses
        ]
        # A clause naming a request term supports the specifics near it; a
        # clause carrying another's anchor supports at most
        # ``INHERIT_SPECIFICS`` more per anchor, so a chain of off-topic
        # clauses ("... and card 4093 was blocked and fee 8812 was waived")
        # cannot ride on one request term.
        budget: dict[int, int] = {}
        self.anchored_specifics: list[list[str]] = []
        for clause, pairs in zip(self.clauses, found, strict=True):
            owner = anchoring.owners[clause.index]
            if anchoring.explicit[clause.index]:
                near = [s for s, i in pairs if _near_anchor(clause, i, anchors)]
            elif owner >= 0 and not _is_statement(clause):
                # A bare list item continuing a clause ("..., card 4093
                # blocked, fee 8812 waived") states nothing.
                found[clause.index] = []
                near = []
            elif owner >= 0:
                left = budget.get(owner, INHERIT_SPECIFICS)
                near = [s for s, _ in pairs][:left]
                budget[owner] = left - len(near)
            else:
                near = []
            self.anchored_specifics.append(near)
        self.specifics = [[stem for stem, _ in pairs] for pairs in found]

    def anchored(self) -> list[_Clause]:
        return [c for c, w in zip(self.clauses, self.weights, strict=True) if w]

    def new_words(self, clause: _Clause) -> list[str]:
        """Content words of an anchored clause that are neither request
        terms, boilerplate nor specifics."""
        specifics = set(self.specifics[clause.index])
        return [
            stem
            for stem in clause.content()
            if stem not in self.request_terms
            and stem not in _GENERIC
            and stem not in specifics
            and stem not in _NEGATIONS
            and not _is_number(stem)
            and stem not in ACTION_TERMS
        ]

    def entities(self) -> set[str]:
        """Capitalized names in anchored clauses that the request lacks."""
        names = set()
        for clause in self.anchored():
            for index, raw in enumerate(clause.raw):
                stem = clause.stems[index]
                if not raw[:1].isupper() or raw.isupper() and len(raw) < 2:
                    continue
                if clause.low[index] in STOPWORDS or stem in self.request_terms:
                    continue
                if stem in _GENERIC:
                    continue
                first = clause.starts_sentence and index == 0
                following = clause.raw[index + 1] if index + 1 < len(clause.raw) else ""
                if not first or following[:1].isupper():
                    names.add(stem)
        return names

    def steps(self) -> int:
        """Distinct anchored instructions: a clause opening with an
        instruction verb (after "then", "first", ...) or "you can/should
        <verb>", that names something new."""
        found = set()
        for clause in self.anchored():
            if not self.new_words(clause):
                continue  # "Rename pathlib 4093" says nothing new
            text = " ".join(clause.low)
            for match in _STEP.finditer(text):
                verb = _stem(match.group("verb"))
                if verb in INSTRUCTION_VERBS:
                    found.add((verb, clause.index, match.start()))
        return len(found)

    def type_satisfied(self, kind: str) -> bool:
        anchored = self.anchored()
        if kind == "quantity":
            return any(_is_number(s) for c in anchored for s in self.specifics[c.index])
        if kind == "entity":
            return bool(self.entities())
        if kind == "place":
            return any(
                _PLACE.search(" ".join(c.low)) and self.new_words(c) for c in anchored
            )
        if kind == "procedure":
            return self.steps() > 0
        if kind == "yes_no":
            first = (
                self.clauses[0].low[0] if self.clauses and self.clauses[0].low else ""
            )
            return first in _YES_NO_ANSWERS
        if kind == "reason":
            return any(
                _CAUSAL.search(" ".join(c.low)) and len(self.new_words(c)) >= 2
                for c in anchored
            )
        if kind == "definition":
            return any(self._defines(c) for c in anchored)
        return False

    def _defines(self, clause: _Clause) -> bool:
        """A request term, then a copula or "has", then something new
        ("The derivative of x^3 is 3x^2", "It has a 12-hour battery")."""
        seen_term = False
        for index, low in enumerate(clause.low):
            stem = clause.stems[index]
            if stem in self.request_terms or (index == 0 and low in _ANAPHORS):
                seen_term = True
            elif seen_term and low in _COPULAS:
                rest = clause.stems[index + 1 :]
                if rest and (rest[0] in _NEGATIONS or rest[0].endswith("n't")):
                    return False  # "It is not Paris" defines nothing
                return any(
                    s not in self.request_terms
                    and s not in _GENERIC
                    and s not in STOPWORDS
                    and s not in ACTION_TERMS
                    and (len(s) > 1 or s.isdigit())
                    for s in rest
                )
        return False


def _runs(tokens: Sequence[str], size: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def _skeleton(reading: _Reading, clause: _Clause) -> tuple[str, ...]:
    """A clause with request terms, numbers and action words abstracted:
    "The pixel 4278 was refunded." and "The wi-fi 2237 was escalated." are
    one template ("the T N was A") filled twice."""
    shape = []
    for low, stem in zip(clause.low, clause.stems, strict=True):
        if stem in reading.request_terms and low not in STOPWORDS:
            shape.append("\x00T")
        elif _is_number(stem):
            shape.append("\x00N")
        elif stem in ACTION_TERMS:
            shape.append("\x00A")
        else:
            shape.append(stem)
    return tuple(shape)


def _repeated_share(reading: _Reading, tokens: Sequence[str]) -> float:
    """Share of tokens that repeat earlier content: a sentence or clause said
    again, a run of ``REPEAT_RUN`` tokens said again, a template refilled
    (a clause stating specifics in the same words as an earlier one, with
    only its request terms, numbers and actions changed), or a request term
    mentioned more than ``TERM_MENTIONS`` times."""
    covered: set[int] = set()
    seen: set[tuple[str, ...]] = set()
    for start, run in enumerate(_runs(tokens, REPEAT_RUN)):
        if run in seen:
            covered.update(range(start, start + REPEAT_RUN))
        seen.add(run)
    position = 0
    said: set[tuple[str, ...]] = set()
    shapes: set[tuple[str, ...]] = set()
    mentions: dict[str, int] = {}
    for clause in reading.clauses:
        key = tuple(clause.stems)
        shape = _skeleton(reading, clause)
        if (key in said and len(key) > 1) or (shape in shapes and len(shape) > 2):
            covered.update(range(position, position + len(key)))
        said.add(key)
        if reading.specifics[clause.index]:
            shapes.add(shape)
        for offset, stem in enumerate(clause.stems):
            if stem in reading.asked and clause.low[offset] not in STOPWORDS:
                mentions[stem] = mentions.get(stem, 0) + 1
                if mentions[stem] > TERM_MENTIONS:
                    covered.add(position + offset)
        position += len(key)
    return len(covered) / len(tokens) if tokens else 0.0


def _echo_share(
    reading: _Reading, tokens: Sequence[str], requests: Sequence[Sequence[str]]
) -> float:
    """Share of tokens copied from the request: runs of half a long request
    message (at least ``ECHO_RUN`` tokens) or four fifths of a short one --
    the question copied in -- and clauses that are nothing but request terms
    ("capital france"). Restating the subject ("The derivative of x^3 is
    ...", "the fraudulent $89 charge on order #5512") is not echo.
    """
    covered: set[int] = set()
    for request_tokens in requests:
        if len(request_tokens) <= ECHO_RUN + 1:
            size = max(3, math.ceil(0.8 * len(request_tokens)))
        else:
            size = max(ECHO_RUN, math.ceil(0.5 * len(request_tokens)))
        if len(tokens) < size or len(request_tokens) < size:
            continue
        request_runs = set(_runs(request_tokens, size))
        for start, run in enumerate(_runs(tokens, size)):
            if run in request_runs:
                covered.update(range(start, start + size))
    position = 0
    for clause in reading.clauses:
        words = clause.content()
        if (
            len(words) >= 2
            and all(word in reading.asked for word in words)
            and not any(low in STOPWORDS for low in clause.low)
        ):
            covered.update(range(position, position + len(clause.stems)))
        position += len(clause.stems)
    return len(covered) / len(tokens) if tokens else 0.0


def _echoed_prefix(reply: str, parts: Sequence[str]) -> int:
    """How many opening tokens of ``reply`` copy the opening of a request
    message (at least three): the "first three request words" prefix, which
    must not anchor an unrelated answer."""
    words = [low for clause in _clauses(reply) for low in clause.low]
    longest = 0
    for part in parts:
        opening = [low for clause in _clauses(part) for low in clause.low]
        size = 0
        while size < min(len(words), len(opening)) and words[size] == opening[size]:
            size += 1
        longest = max(longest, size)
    return longest if longest >= 3 else 0


def _off_request_share(reading: _Reading) -> float:
    """Share of content words in clauses tied to nothing in the request."""
    total = off = 0
    for clause, weight in zip(reading.clauses, reading.weights, strict=True):
        words = len(clause.content())
        total += words
        if not weight:
            off += words
    return off / total if total else 0.0


def _units(
    reading: _Reading, question: str, evidence_terms: set[str], labeled: bool
) -> tuple[float, float, dict[str, float]]:
    """Information units a reply carries, and its unsupported specifics.

    Each distinct specific counts its best support: the weight of an anchored
    clause stating it, or 1 when the conversation's evidence contains it.
    """
    support: dict[str, float] = {}
    for clause, weight in zip(reading.clauses, reading.weights, strict=True):
        near = set(reading.anchored_specifics[clause.index])
        for specific in reading.specifics[clause.index]:
            if _token_parts(specific) & evidence_terms:
                backed = 1.0
            else:
                backed = weight if specific in near else 0.0
            support[specific] = max(support.get(specific, 0.0), backed)
    # Specifics need context: at most ``SPECIFIC_FLOOR`` plus
    # ``SPECIFICS_PER_WORD`` per new word the anchored clauses state them with ("Visa ending 4421", "ISO
    # 100"), so request terms, numbers and verbs alone ("The pixel 4278 was
    # refunded. Your wi-fi 2237 got escalated.") state a couple of facts.
    context_words: set[str] = set()
    for clause in reading.anchored():
        context_words.update(reading.new_words(clause))
    supported = min(
        sum(support.values()),
        SPECIFIC_FLOOR + SPECIFICS_PER_WORD * len(context_words),
    )
    unsupported = float(sum(1 for weight in support.values() if weight == 0.0))
    # With a reference answer, answer credit decides whether the question
    # was answered; the answer-type cue would credit a wrong answer.
    types = [] if labeled else sorted(_answer_types(question))
    answered = sum(1 for kind in types if reading.type_satisfied(kind))
    type_units = TYPE_UNITS * min(MAX_TYPES, answered)
    step_units = STEP_UNITS * min(STEP_CAP, reading.steps())
    elaboration = set()
    for clause in reading.anchored():
        elaboration.update(reading.new_words(clause))
    elaboration_units = ELABORATION_UNITS * min(ELABORATION_CAP, len(elaboration))
    # Elaboration stands in for specifics in an answer without them ("why
    # is the sky blue?"); beside specifics it adds nothing, so padding a
    # specific answer with words cannot buy information.
    specific_units = supported
    units = max(specific_units, elaboration_units) + type_units + step_units
    details = {
        "specifics": supported,
        "specific_units": specific_units,
        "unsupported": unsupported,
        "answer_types": float(answered),
        "steps": step_units,
        "elaboration": elaboration_units,
    }
    return units, unsupported, details


def grounding_score(
    reply: str,
    request: str | Sequence[str],
    labels: Sequence[str] = (),
    *,
    evidence: Sequence[str] = (),
    question: str | None = None,
    background: Sequence[str] = (),
) -> tuple[float, dict[str, float]]:
    """Grounding of one ``reply`` in a ``request`` (see the module docs).

    ``labels`` are reference facts (ground-truth answers): a short label is an
    answer and earns ``CorrectnessReward``'s answer credit (none when hedged
    or negated); a longer one is a fact whose terms count as specifics.
    ``evidence`` holds tool outputs, observations and retrieved facts: a
    specific found there is supported wherever the reply states it.
    ``question`` is the message the reply answers (default: ``request``); it
    sets the expected answer type. ``request`` may be given as its separate
    parts (request fields and user messages), so a copy of any one of them
    is recognized as echo. ``background`` names the parts that describe the
    scenario rather than quote the user (``context``, ``user_goal``,
    ``topic``...): terms found only there anchor at ``WEAK_ANCHOR``.
    """
    parts = (request,) if isinstance(request, str) else tuple(map(str, request))
    score, details = _cached_grounding(
        reply,
        parts,
        tuple(map(str, labels)),
        tuple(map(str, evidence)),
        question,
        tuple(map(str, background)),
    )
    return score, dict(details)


@functools.lru_cache(maxsize=2048)
def _cached_grounding(
    reply: str,
    parts: tuple[str, ...],
    labels: tuple[str, ...],
    evidence: tuple[str, ...],
    question: str | None,
    background: tuple[str, ...] = (),
) -> tuple[float, dict[str, float]]:
    # Every gated component of a composite grounds the same reply; the
    # computation is pure, so it runs once per distinct input.
    return _grounding(
        reply,
        parts,
        labels,
        evidence=evidence,
        question=question,
        background=background,
    )


def _grounding(
    reply: str,
    parts: Sequence[str],
    labels: Sequence[str] = (),
    *,
    evidence: Sequence[str] = (),
    question: str | None = None,
    background: Sequence[str] = (),
) -> tuple[float, dict[str, float]]:
    """:func:`grounding_score` for a request given as its parts."""
    request = " ".join(parts)
    question = request if question is None else question
    details: dict[str, float] = {"information": 0.0, "relevance": 0.0}
    reply_terms = content_terms(reply)
    if not reply_terms:
        return 0.0, details
    if harmful_request(request):
        # A refusal earns a fixed safety credit, never full grounding
        # (information); complying earns nothing. SafetyReward is the
        # authoritative signal, but this keeps the refusal above compliance
        # for factories that carry no safety term.
        score = SAFETY_CREDIT if declines(reply) else 0.0
        return score, {
            "information": 0.0,
            "relevance": 0.0,
            "safety_credit": score,
            "harmful": 1.0,
        }

    request_terms = content_terms(request)
    answers = [label for label in labels if len(label.split()) <= _ANSWER_WORDS]
    facts = [label for label in labels if len(label.split()) > _ANSWER_WORDS]
    label_terms = set().union(*(content_terms(label) for label in labels))
    fact_terms = set().union(*(content_terms(fact) for fact in facts))
    evidence_terms = set().union(*(content_terms(text) for text in evidence))
    evidence_terms |= fact_terms
    numeric_answer = any(
        _as_number(answer.strip().lstrip("$")) is not None for answer in answers
    )
    # Answer labels are scored by answer credit, never as specifics.
    exclude = request_terms | (label_terms - fact_terms)
    # A clause is tied to the request by a request term, a reference label
    # ("It is Paris.") or a term of the conversation's evidence.
    anchors = request_terms | label_terms | (evidence_terms - _GENERIC)
    stated = set().union(
        set(), *(content_terms(part) for part in parts if part not in background)
    )
    # Request terms that are boilerplate or action words ("help", "get it
    # reversed") or that only the scenario description names are weak
    # anchors: topic words tie a clause to the request, verbs barely do.
    weak = (_GENERIC | ACTION_TERMS | (request_terms - stated)) - label_terms
    reading = _Reading(
        reply,
        request_terms | label_terms,
        exclude,
        numeric_answer,
        anchors,
        asked=request_terms,
        weak=weak,
        skip=_echoed_prefix(reply, parts),
    )
    units, unsupported, unit_details = _units(
        reading, question, evidence_terms, bool(answers)
    )
    breadth = 1.0 - math.exp(-units / INFO_SCALE)
    if answers:
        credit = max(_answer_credit(reply, answer, question) for answer in answers)
        information = max(credit, 0.5 * breadth)
    else:
        information = breadth

    known = request_terms | label_terms
    shared = {part for term in reply_terms for part in _token_parts(term)} & known
    relevance = 1.0 if not known else min(1.0, len(shared) / min(3, len(known)))

    tokens = [stem for clause in reading.clauses for stem in clause.stems]
    lows = [low for clause in reading.clauses for low in clause.low]
    request_tokens = [
        [stem for clause in _clauses(part) for stem in clause.stems] for part in parts
    ]
    specifics = {s for found in reading.specifics for s in found}
    recognized = known | specifics | evidence_terms
    content = [
        (low, stem)
        for low, stem in zip(lows, tokens, strict=True)
        if low not in STOPWORDS and (len(low) > 1 or low.isdigit())
    ]
    # Function words count up to one per content word: ordinary prose keeps
    # its share, and padding with "the of and" buys nothing.
    function_words = sum(
        1 for low in lows if low in STOPWORDS and low not in _NEGATIONS
    )
    size = max(1, len(content) + min(function_words, len(content)))
    boilerplate = (
        sum(1 for _, s in content if s in _GENERIC and s not in recognized) / size
    )
    unknown = (
        sum(
            1
            for _, s in content
            if s not in _GENERIC
            and s not in recognized
            and not _is_number(s)
            and s not in ACTION_TERMS
        )
        / size
    )
    quality = (1.0 - boilerplate) * (1.0 - (boilerplate + unknown) ** QUALITY_POWER)
    support = (units + 1.0) / (units + unsupported + 1.0)
    # Repetition and echo are measured against the reply's anchored text, so
    # appending unrelated text cannot dilute them.
    anchored_size = sum(len(c.stems) for c in reading.anchored()) or len(tokens)
    repeated = min(1.0, _repeated_share(reading, tokens) * len(tokens) / anchored_size)
    echoed = min(
        1.0, _echo_share(reading, tokens, request_tokens) * len(tokens) / anchored_size
    )
    off_request = _off_request_share(reading)
    originality = (1.0 - repeated) * max(0.0, 1.0 - ECHO_WEIGHT * echoed)
    score = (
        information
        * (0.5 + 0.5 * relevance)
        * quality
        * support
        * originality
        * (1.0 - OFF_REQUEST_WEIGHT * off_request)
    )
    if declines(reply):
        # Declining a request not flagged harmful is never the better reply:
        # whatever else the reply says earns at most REFUSAL_FACTOR of it.
        score *= REFUSAL_FACTOR
    details = {
        "information": information,
        "relevance": relevance,
        "quality": quality,
        "boilerplate": boilerplate,
        "unknown": unknown,
        "support": support,
        "repeated": repeated,
        "echoed": echoed,
        "off_request": off_request,
        **unit_details,
    }
    return score, details


def task_grounding(
    turns: Sequence[Any], context: Mapping[str, Any] | None = None
) -> float:
    """Mean grounding of the assistant replies in scope (dict or object turns).

    Each reply is judged against the user messages before it plus the
    request fields and labels of ``context``, answers the latest user message,
    and may cite the evidence before it (tool, observation and system turns,
    and the context's evidence fields). With ``reward_scope`` set to the
    latest turn only the newest reply counts.
    """
    texts, background, labels = _request_texts(context)
    evidence = _evidence_texts(context)
    asked: list[str] = []
    scores: list[float] = []
    for turn in turns:
        role, content = _turn_field(turn, "role"), _turn_field(turn, "content")
        text = content if isinstance(content, str) else ""
        if role == "user":
            asked.append(text)
        elif role in EVIDENCE_ROLES:
            evidence.append(text)
        elif role == "assistant":
            score, _ = grounding_score(
                text,
                [*texts, *asked],
                labels,
                evidence=list(evidence),
                question=asked[-1] if asked else " ".join(texts),
                background=background,
            )
            scores.append(score)
    if not scores:
        return 0.0
    if scores_latest_turn_only(context):
        return scores[-1]
    return sum(scores) / len(scores)


class TaskGroundingReward(RewardFunction):
    """Reward for replies that address the request with specific content."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight, RewardType.IMMEDIATE, "TaskGroundingReward")

    async def compute_reward(
        self, turns: list[Any], context: dict[str, Any] | None = None
    ) -> RewardResult:
        score = task_grounding(turns, context)
        return RewardResult(score=score, breakdown={"grounding": score})


class GroundedReward(RewardFunction):
    """A style (shaping) reward that only pays for grounded replies.

    ``score = grounding * (share + (1 - share) * base_score)``, with
    ``share`` defaulting to ``STYLE_SHARE``: a canned or keyword-stuffed reply
    (grounding 0) earns nothing however well it matches the style heuristics,
    while a grounded reply keeps their ordering. With
    ``gate_if`` the gate applies only to results it returns True for (e.g.
    a task reward scored by keywords alone). Name, weight and reward type are
    the base reward's, so composites treat the wrapper as the base.
    """

    def __init__(
        self,
        base: RewardFunction,
        *,
        share: float | None = None,
        gate_if: Callable[[RewardResult], bool] | None = None,
    ) -> None:
        super().__init__(base.weight, base.reward_type, base.name)
        self.base = base
        self.share = float(STYLE_SHARE if share is None else share)
        self.gate_if = gate_if

    @property
    def reward_type(self) -> RewardType:
        return self.base.reward_type

    @reward_type.setter
    def reward_type(self, value: RewardType) -> None:
        if hasattr(self, "base"):
            self.base.reward_type = value

    @property
    def fail_on_error(self) -> bool:
        return getattr(self.base, "fail_on_error", False) is True

    @property
    def requires_environment_context(self) -> bool:
        return getattr(self.base, "requires_environment_context", False) is True

    async def compute_reward(
        self, turns: list[Any], context: dict[str, Any] | None = None
    ) -> RewardResult:
        result = await self.base.compute_reward(turns, context)
        if self.gate_if is not None and not self.gate_if(result):
            return result
        grounding = task_grounding(turns, context)
        score = grounding * (self.share + (1.0 - self.share) * result.score)
        return RewardResult(
            score=score,
            breakdown={
                **result.breakdown,
                "grounding": grounding,
                "ungrounded_score": result.score,
            },
            metadata=dict(result.metadata),
            explanation=result.explanation,
        )


#: Logged by starters and presets that train on a heuristic factory reward.
HEURISTIC_REWARD_NOTICE = (
    "%s trains on the heuristic %s reward, a grounded shaping signal rather "
    "than a training objective. For production training use an outcome, "
    "verifier or judge reward (e.g. TaskOutcomeReward, NSRVerifierReward, "
    "LLMJudgeReward); see docs/REWARD_CONTRACT.md."
)


def log_heuristic_reward_notice(logger: Any, trainer: str, reward: str) -> None:
    """Recommend an outcome/verifier/judge reward over a heuristic factory."""
    logger.warning(HEURISTIC_REWARD_NOTICE, trainer, reward)


def ground_shaping_rewards(
    rewards: Sequence[RewardFunction], *, share: float | None = None
) -> list[RewardFunction]:
    """Wrap every reward marked ``shaping = True`` in :class:`GroundedReward`."""
    return [
        (
            GroundedReward(reward, share=share)
            if getattr(reward, "shaping", False) is True
            else reward
        )
        for reward in rewards
    ]


__all__ = [
    "ACTION_TERMS",
    "GENERIC_TERMS",
    "HEURISTIC_REWARD_NOTICE",
    "STOPWORDS",
    "GroundedReward",
    "TaskGroundingReward",
    "content_terms",
    "ground_shaping_rewards",
    "grounding_score",
    "log_heuristic_reward_notice",
    "task_grounding",
]
