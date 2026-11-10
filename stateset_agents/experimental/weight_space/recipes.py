"""Edit recipes: one declarative file in, one accepted-or-rejected report out.

The session offers every step of a keyed edit separately: capture keys, design
through the complete model, trial against behavior and generated-answer gates,
export, and verify the reloaded artifact. A recipe states the whole request up
front and :func:`run_recipe` runs those steps in a fixed order:

1. Check the unedited model against the recipe's own locality requirement and
   stop before any optimization if it cannot be met.
2. Try each declared edit site in order. A site that fails any gate is rolled
   back exactly; the first site passing every gate is retained.
3. Optionally export the retained edit as a LoRA adapter and verify the files
   through PEFT against the live session. A failed verification reverts.

Design data (``design``, ``calibration``, ``protected``) and acceptance data
(``validation``, ``drift``, ``locality``) must be disjoint; the recipe rejects
overlap instead of letting an edit be graded on what it was fitted to.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from stateset_agents.core.errors import ConfigurationError

from .answer_checks import AnswerCase, AnswerLimits
from .behavior import BehaviorLimits
from .generation import matches_answer_prefix
from .model_edit import KeyedEditConfig

if TYPE_CHECKING:
    from .session import WeightSpaceSession

__all__ = [
    "EditRecipe",
    "EditSite",
    "LocalityAnswer",
    "RecipeReport",
    "SiteAttempt",
    "TextAdapter",
    "example_recipe",
    "hf_text_adapter",
    "run_recipe",
]

REPORT_NAME = "recipe_report.json"
ADAPTER_DIR = "adapter"

Status = Literal["accepted", "rejected", "baseline_unmet", "export_rejected"]

_HINTS = {
    "mean_kl": (
        "Average drift on the drift prompts is over the limit. Raise "
        "locality_weight or optimizer.kl_weight, lower optimizer.max_update_norm, "
        "or add calibration prompts that resemble the drift prompts."
    ),
    "max_token_kl": (
        "One token position drifted too far. Add calibration or protected text "
        "covering that kind of context, or lower optimizer.max_output_norm."
    ),
    "nll_increase": (
        "The edit makes ordinary text less likely. Raise optimizer.kl_weight or "
        "lower optimizer.max_update_norm."
    ),
    "top1_agreement": (
        "Too many next-token predictions changed on the drift prompts. Raise "
        "locality_weight or try a later site."
    ),
    "mean_target_gain": (
        "The target did not become more likely on the validation prompts. Add "
        "design paraphrases, raise optimizer.steps, or try an earlier site with "
        'key "subject".'
    ),
    "each_target_gain": (
        "At least one validation prompt gained too little. Add a design prompt "
        "phrased like it, or try an earlier site."
    ),
    "generation:target_accuracy": (
        "The edited model does not answer with the target on every design and "
        "validation prompt. Add design paraphrases, raise optimizer.steps, or "
        "try another site."
    ),
    "generation:locality_retention": (
        "An answer the unedited model got right was lost. Add questions about "
        "the same subject's other properties to calibration (not the locality "
        "questions themselves)."
    ),
    "generation:answer_repetition": (
        "The target is repeated in a completion. Keep the end-of-sequence token "
        "in the target (append_eos) and lower optimizer.max_output_norm."
    ),
    "generation:baseline_locality_correct": (
        "The unedited model answers too few locality questions in the declared "
        "form. Fix their accepted answers, replace them, or lower "
        "answer_limits.min_baseline_locality_correct."
    ),
}


def _strings(name: str, value: Any, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{name} must be a list of strings")
    items = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ConfigurationError(f"{name} must contain nonempty strings")
    if len(set(items)) != len(items):
        raise ConfigurationError(f"{name} contains a repeated entry")
    if not items and not allow_empty:
        raise ConfigurationError(f"{name} must not be empty")
    return items


def _build(cls: Any, name: str, value: Any) -> Any:
    """A config dataclass from a mapping, naming the section in any error."""
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be an object")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(value) - known)
    if unknown:
        raise ConfigurationError(f"{name} has unknown fields: {', '.join(unknown)}")
    try:
        return cls(**value)
    except TypeError as exc:
        raise ConfigurationError(f"{name}: {exc}") from exc


@dataclass(frozen=True)
class EditSite:
    """A Linear module to edit and which token's input activation keys it."""

    module: str
    key: Literal["subject", "last"] = "subject"

    def __post_init__(self) -> None:
        if not isinstance(self.module, str) or not self.module:
            raise ConfigurationError("site module must be a nonempty string")
        if self.key not in ("subject", "last"):
            raise ConfigurationError('site key must be "subject" or "last"')


@dataclass(frozen=True)
class LocalityAnswer:
    """A question whose answer must survive the edit, with accepted spellings."""

    prompt: str
    answers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ConfigurationError("locality prompt must be a nonempty string")
        object.__setattr__(self, "answers", _strings("locality answers", self.answers))


@dataclass(frozen=True)
class EditRecipe:
    """Everything one keyed edit needs, declared before any forward pass.

    ``design`` questions fit the edit and must each mention ``subject`` exactly
    once when a site keys on the subject. ``validation`` questions ask for the
    same fact in other words and are only ever used for acceptance. ``locality``
    questions have answers the edit must not change. ``calibration`` questions
    and ``protected`` text shape the edit at design time; ``drift`` text measures
    whole-model change at acceptance time.
    """

    name: str
    model: str
    subject: str
    target: str
    design: tuple[str, ...]
    validation: tuple[str, ...]
    locality: tuple[LocalityAnswer, ...]
    calibration: tuple[str, ...]
    drift: tuple[str, ...]
    sites: tuple[EditSite, ...]
    revision: str | None = None
    system: str | None = None
    chat: bool = True
    enable_thinking: bool = False
    append_eos: bool = True
    protected: tuple[str, ...] = ()
    locality_weight: float = 100.0
    optimizer: KeyedEditConfig = field(default_factory=KeyedEditConfig)
    limits: BehaviorLimits = field(
        default_factory=lambda: BehaviorLimits(0.001, 0.01, 0.005, 0.95, 0.01, 0.01)
    )
    answer_limits: AnswerLimits = field(
        default_factory=lambda: AnswerLimits(max_new_tokens=16)
    )
    export_limits: BehaviorLimits = field(
        default_factory=lambda: BehaviorLimits(
            1e-6, 1e-5, 1e-4, min_each_target_gain=-1e-4
        )
    )
    budget: int | None = None

    def __post_init__(self) -> None:
        for name in ("name", "model", "subject", "target"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{name} must be a nonempty string")
        for name in ("design", "validation", "calibration", "drift"):
            object.__setattr__(self, name, _strings(name, getattr(self, name)))
        object.__setattr__(
            self, "protected", _strings("protected", self.protected, allow_empty=True)
        )
        locality = tuple(
            (
                item
                if isinstance(item, LocalityAnswer)
                else _build(LocalityAnswer, "locality entry", item)
            )
            for item in self.locality
        )
        sites = tuple(
            item if isinstance(item, EditSite) else _build(EditSite, "site", item)
            for item in self.sites
        )
        if not locality or not sites:
            raise ConfigurationError("locality and sites must not be empty")
        if len({(s.module, s.key) for s in sites}) != len(sites):
            raise ConfigurationError("sites contains a repeated entry")
        object.__setattr__(self, "locality", locality)
        object.__setattr__(self, "sites", sites)
        for name, cls in (
            ("optimizer", KeyedEditConfig),
            ("limits", BehaviorLimits),
            ("answer_limits", AnswerLimits),
            ("export_limits", BehaviorLimits),
        ):
            object.__setattr__(self, name, _build(cls, name, getattr(self, name)))
        if any(
            type(getattr(self, n)) is not bool
            for n in ("chat", "enable_thinking", "append_eos")
        ):
            raise ConfigurationError(
                "chat, enable_thinking and append_eos are booleans"
            )
        weight = self.locality_weight
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or weight < 0
        ):
            raise ConfigurationError("locality_weight must be a nonnegative number")
        if self.budget is not None and (
            type(self.budget) is not int or self.budget < 1
        ):
            raise ConfigurationError("budget must be a positive integer or null")
        if self.system is not None and not isinstance(self.system, str):
            raise ConfigurationError("system must be a string or null")
        if self.system is not None and not self.chat:
            raise ConfigurationError("a system prompt requires chat formatting")
        if any(s.key == "subject" for s in sites):
            for prompt in self.design:
                if prompt.count(self.subject) != 1:
                    raise ConfigurationError(
                        "each design prompt must mention the subject exactly once: "
                        f"{prompt!r}"
                    )
        fitted = {
            "design": set(self.design),
            "calibration": set(self.calibration),
            "protected": set(self.protected),
        }
        graded = {
            "validation": set(self.validation),
            "drift": set(self.drift),
            "locality": {item.prompt for item in locality},
        }
        for fit_name, fit in fitted.items():
            for grade_name, grade in graded.items():
                if fit & grade:
                    raise ConfigurationError(
                        f"{fit_name} and {grade_name} share an entry; an edit must "
                        "not be accepted on text it was designed with"
                    )
        if graded["validation"] & graded["locality"]:
            raise ConfigurationError("validation and locality share a prompt")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EditRecipe:
        """Build a recipe from parsed JSON; unknown fields are errors."""
        return _build(cls, "recipe", data)  # type: ignore[no-any-return]

    @classmethod
    def from_json(cls, text: str) -> EditRecipe:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"recipe is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"

    def content_hash(self) -> str:
        """SHA-256 of the canonical recipe, binding a report to its request."""
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def with_overrides(self, overrides: Mapping[str, Any]) -> EditRecipe:
        """A revalidated copy with dotted-path fields replaced.

        ``{"optimizer.steps": 40, "sites.0.module": "model.layers.10.mlp.down_proj"}``
        changes one optimizer field and the first site's module. Paths must
        already exist, so a misspelled override is an error rather than a no-op.
        """
        data = self.to_dict()
        for path, value in overrides.items():
            node: Any = data
            parts = path.split(".")
            try:
                for index, part in enumerate(parts):
                    key: Any = int(part) if isinstance(node, (list, tuple)) else part
                    if isinstance(node, tuple):
                        raise TypeError
                    if index == len(parts) - 1:
                        node[key]  # noqa: B018 - the path must already exist
                        node[key] = value
                    else:
                        if isinstance(node[key], tuple):
                            node[key] = [
                                dict(v) if isinstance(v, Mapping) else v
                                for v in node[key]
                            ]
                        node = node[key]
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                raise ConfigurationError(
                    f"override path does not exist in the recipe: {path}"
                ) from exc
        return EditRecipe.from_dict(data)

    def required_budget(self) -> int:
        """Measurement units a full run can spend, including verification."""
        captures = 2 + bool(self.protected)
        per_site = self.optimizer.steps + 1 + 4
        return captures + 1 + len(self.sites) * per_site + 2


class TextAdapter(Protocol):
    """How questions become prompt text, and where each token sits in it."""

    eos_text: str

    def render(self, question: str) -> str:
        """The exact prompt string the session's encoder will tokenize."""

    def offsets(self, prompt: str) -> Sequence[tuple[int, int]]:
        """Character span of every token ``session.encode(prompt)`` returns."""


def hf_text_adapter(tokenizer: Any, recipe: EditRecipe) -> TextAdapter:
    """Chat-template rendering and offset lookup for a Hugging Face tokenizer."""

    class _Adapter:
        eos_text = str(tokenizer.eos_token or "")

        def render(self, question: str) -> str:
            if not recipe.chat:
                return question
            messages = []
            if recipe.system is not None:
                messages.append({"role": "system", "content": recipe.system})
            messages.append({"role": "user", "content": question})
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=recipe.enable_thinking,
                )
            )

        def offsets(self, prompt: str) -> Sequence[tuple[int, int]]:
            encoded = tokenizer(
                prompt, add_special_tokens=False, return_offsets_mapping=True
            )
            return [(int(a), int(b)) for a, b in encoded["offset_mapping"]]

    if recipe.append_eos and not _Adapter.eos_text:
        raise ConfigurationError("append_eos needs a tokenizer with an EOS token")
    return _Adapter()


@dataclass(frozen=True)
class SiteAttempt:
    """What happened at one site; rejected attempts are already rolled back."""

    module: str
    key: str
    accepted: bool
    violations: tuple[str, ...]
    error: str | None = None
    design: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    answers: dict[str, Any] | None = None
    outputs: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class RecipeReport:
    """The decision, the evidence behind it, and what to change if rejected."""

    status: Status
    recipe_sha256: str
    name: str
    model: str
    site: dict[str, str] | None
    edit_id: str | None
    attempts: tuple[SiteAttempt, ...]
    baseline: tuple[dict[str, Any], ...]
    hints: tuple[str, ...]
    adapter_dir: str | None
    export: dict[str, Any] | None
    budget_used: int
    budget_limit: int

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "stateset-weight-edit-recipe-report-v1", **asdict(self)}

    def metrics(self) -> dict[str, float]:
        """Flat numbers for comparing runs, from the last site that was trialed.

        ``accepted`` and ``target_accuracy`` are always present (0.0 when no
        site reached a trial) so a sweep can rank on them. Drift and retention
        appear only when they were measured.
        """
        values = {
            "accepted": float(self.accepted),
            "baseline_met": float(self.status != "baseline_unmet"),
            "sites_tried": float(len(self.attempts)),
            "target_accuracy": 0.0,
            "budget_used": float(self.budget_used),
        }
        trialed = [a for a in self.attempts if a.metrics is not None]
        if trialed:
            last = trialed[-1]
            behavior, answers, design = last.metrics or {}, last.answers, last.design
            for name in ("mean_kl", "max_token_kl", "nll_increase", "top1_agreement"):
                values[name] = float(behavior[name])
            if behavior.get("mean_target_gain") is not None:
                values["mean_target_gain"] = float(behavior["mean_target_gain"])
            if answers is not None:
                values["target_accuracy"] = float(answers["target_accuracy"])
                values["locality_retention"] = float(answers["locality_retention"])
            if design is not None:
                values["update_norm"] = float(design["update_norm"])
        return values


def _subject_index(prompt: str, subject: str, text: TextAdapter, length: int) -> int:
    """Index of the token holding the subject's last character."""
    if prompt.count(subject) != 1:
        raise ConfigurationError(
            "the rendered design prompt must contain the subject exactly once "
            "(check the system prompt and chat template)"
        )
    end = prompt.index(subject) + len(subject)
    spans = list(text.offsets(prompt))
    if len(spans) != length:
        raise ConfigurationError(
            "the text adapter and the session encoder tokenize prompts differently"
        )
    for index, (start, stop) in enumerate(spans):
        if start < end <= stop:
            return index
    raise ConfigurationError("no token covers the end of the subject")


def _hints(violations: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for violation in violations:
        hint = _HINTS.get(violation)
        if hint:
            seen.setdefault(hint)
    return tuple(seen)


def run_recipe(
    recipe: EditRecipe,
    session: WeightSpaceSession,
    text: TextAdapter,
    *,
    out_dir: str | Path | None = None,
    fresh_base_model: Any = None,
    verify_mode: Literal["adapter", "merged"] = "merged",
    progress: Callable[[str], None] | None = None,
) -> RecipeReport:
    """Run ``recipe`` on ``session``; keep the edit only if every gate passes.

    Without ``out_dir`` an accepted edit simply stays active in the session.
    With it, the adapter is written to ``out_dir/adapter`` and the report to
    ``out_dir/recipe_report.json``. Supplying ``fresh_base_model`` (which the
    verifier takes ownership of) additionally reloads the adapter through PEFT;
    a failed verification reverts the edit and reports ``export_rejected``.

    The report holds prompts and generated text. The exported adapter manifest
    still holds only hashes. ``progress`` receives one short line per stage;
    design at a large model's site can take minutes on CPU.
    """
    import torch

    if fresh_base_model is not None and out_dir is None:
        raise ConfigurationError("verification needs out_dir for the exported adapter")
    if not callable(session.decode):
        raise ConfigurationError("recipes need a session with a decoder")
    root = None if out_dir is None else Path(out_dir)
    if root is not None and root.exists() and any(root.iterdir()):
        raise ConfigurationError(f"{root} already exists and is not empty")
    needed = recipe.required_budget() - (0 if fresh_base_model is not None else 2)
    if session.budget.remaining < needed:
        from .session import BudgetExhausted

        raise BudgetExhausted(
            f"recipe needs up to {needed} measurement units; "
            f"{session.budget.remaining} remain"
        )

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    design = [text.render(q) for q in recipe.design]
    validation = [text.render(q) for q in recipe.validation]
    calibration = [text.render(q) for q in recipe.calibration]
    locality = [(text.render(item.prompt), item.answers) for item in recipe.locality]
    continuation = recipe.target + (text.eos_text if recipe.append_eos else "")
    cases = [AnswerCase(p, (recipe.target,)) for p in design + validation]
    cases += [AnswerCase(p, answers, "locality") for p, answers in locality]
    questions = [
        *recipe.design,
        *recipe.validation,
        *(item.prompt for item in recipe.locality),
    ]
    lengths = [len(session.encode(p)) for p in design]
    subject_at = (
        [
            _subject_index(p, recipe.subject, text, n)
            for p, n in zip(design, lengths, strict=True)
        ]
        if any(s.key == "subject" for s in recipe.sites)
        else []
    )

    # One generation unit before any design work: a recipe whose own locality
    # requirement the unedited model cannot meet would fail at every site.
    limits = recipe.answer_limits
    say(f"baseline: generating answers to {len(cases)} questions")
    generated = session.generate(
        [c.prompt for c in cases],
        max_new_tokens=limits.max_new_tokens,
        use_cache=limits.use_cache,
        max_context_tokens=limits.max_context_tokens,
        max_total_tokens=limits.max_total_tokens,
        max_logit_elements=limits.max_logit_elements,
        reason="recipe:baseline",
    )
    baseline: list[dict[str, Any]] = []
    for question, case, result in zip(questions, cases, generated, strict=True):
        answer = session.decode(result.token_ids)
        baseline.append(
            {
                "kind": case.kind,
                "question": question,
                "text": answer,
                "finish_reason": result.finish_reason,
                "correct": matches_answer_prefix(answer, case.answers),
            }
        )
    base_correct = sum(bool(r["correct"]) for r in baseline if r["kind"] == "locality")

    def finish(
        status: Status,
        attempts: Sequence[SiteAttempt],
        hints: Sequence[str],
        site: EditSite | None = None,
        edit_id: str | None = None,
        export: dict[str, Any] | None = None,
        adapter: Path | None = None,
    ) -> RecipeReport:
        report = RecipeReport(
            status,
            recipe.content_hash(),
            recipe.name,
            recipe.model,
            None if site is None else {"module": site.module, "key": site.key},
            edit_id,
            tuple(attempts),
            tuple(baseline),
            tuple(hints),
            None if adapter is None else str(adapter),
            export,
            session.budget.used,
            session.budget.limit,
        )
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            (root / REPORT_NAME).write_text(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return report

    if base_correct < limits.min_baseline_locality_correct:
        return finish(
            "baseline_unmet",
            (),
            (
                f"The unedited model answers {base_correct} of "
                f"{len(recipe.locality)} locality questions in the declared form; "
                f"the recipe requires {limits.min_baseline_locality_correct}. "
                "No edit was attempted.",
                _HINTS["generation:baseline_locality_correct"],
            ),
        )

    say(f"baseline: {base_correct}/{len(recipe.locality)} locality answers correct")
    modules = list(dict.fromkeys(s.module for s in recipe.sites))
    rows = {
        "design": session.capture_many(
            modules, design, read="input", position="all", reason="recipe:keys"
        ),
        "calibration": session.capture_many(
            modules,
            calibration,
            read="input",
            position="all",
            reason="recipe:calibration",
        ),
    }
    if recipe.protected:
        rows["protected"] = session.capture_many(
            modules,
            list(recipe.protected),
            read="input",
            position="all",
            reason="recipe:protected",
        )

    attempts: list[SiteAttempt] = []
    violations: list[str] = []
    decoder = session.decode
    for site in recipe.sites:
        starts = [sum(lengths[:i]) for i in range(len(lengths))]
        picks = [
            start + (subject_at[i] if site.key == "subject" else lengths[i] - 1)
            for i, start in enumerate(starts)
        ]
        keys = torch.stack([rows["design"][site.module][i] for i in picks])
        say(
            f"site {site.module} ({site.key}): designing, "
            f"{recipe.optimizer.steps} steps"
        )
        try:
            proposal = session.optimize_keyed_edit(
                site.module,
                keys,
                target_pairs=[(p, continuation) for p in design],
                locality_prompts=calibration,
                config=recipe.optimizer,
                protect=rows["protected"][site.module] if recipe.protected else None,
                locality=rows["calibration"][site.module],
                locality_weight=float(recipe.locality_weight),
            )
        except ConfigurationError as exc:
            attempts.append(
                SiteAttempt(site.module, site.key, False, ("design_failed",), str(exc))
            )
            say(f"site {site.module}: design failed: {exc}")
            continue
        first, chosen = proposal.history[0], proposal.history[proposal.selected_step]
        summary = {
            "selected_step": proposal.selected_step,
            "initial_target_nll": first.target_nll,
            "target_nll": chosen.target_nll,
            "calibration_kl": chosen.locality_kl,
            "update_norm": proposal.update_norm,
            "model_forwards": proposal.model_forwards,
            "backward_calls": proposal.backward_calls,
        }
        say(
            f"site {site.module}: target loss {first.target_nll:.3f} -> "
            f"{chosen.target_nll:.3f}; running acceptance gates"
        )
        seen: list[str] = []

        def tracing(ids: Sequence[int], _sink: list[str] = seen) -> str:
            value = decoder(ids)
            _sink.append(value)
            return value

        session.decode = tracing
        try:
            trial = session.trial_edit(
                site.module,
                proposal.delta,
                locality_prompts=list(recipe.drift),
                target_pairs=[(p, continuation) for p in validation],
                limits=recipe.limits,
                answer_cases=cases,
                answer_limits=limits,
                note=f"recipe:{recipe.name}",
            )
        finally:
            session.decode = decoder
        outputs: tuple[dict[str, Any], ...] = ()
        check = trial.answer_check
        if check is not None and len(seen) == 2 * len(cases):
            outputs = tuple(
                {
                    "kind": score.kind,
                    "question": question,
                    "before": before,
                    "after": after,
                    "correct": score.edited_correct,
                }
                for question, score, before, after in zip(
                    questions,
                    check.scores,
                    seen[: len(cases)],
                    seen[len(cases) :],
                    strict=True,
                )
            )
        attempts.append(
            SiteAttempt(
                site.module,
                site.key,
                trial.accepted,
                trial.violations,
                None,
                summary,
                asdict(trial.metrics),
                None if check is None else asdict(check.metrics),
                outputs,
            )
        )
        if not trial.accepted:
            violations.extend(trial.violations)
            say(f"site {site.module}: rejected ({', '.join(trial.violations)})")
            continue
        say(f"site {site.module}: accepted")
        if root is None:
            return finish("accepted", attempts, (), site, trial.edit_id)
        try:
            adapter = session.export(root / ADAPTER_DIR)
        except BaseException:
            # Nothing is kept unless every step passes: an export that
            # raises must not leave the edit active in the session.
            session.revert()
            raise
        if fresh_base_model is None:
            return finish(
                "accepted",
                attempts,
                (
                    "The adapter was exported but not reloaded. Verify it against "
                    "a fresh copy of the base model before serving it.",
                ),
                site,
                trial.edit_id,
                {"status": "unverified"},
                adapter,
            )
        say("verifying the exported adapter on a fresh base model")
        try:
            _, verification = session.verify_export(
                adapter,
                fresh_base_model,
                locality_prompts=list(recipe.drift),
                target_pairs=[(p, continuation) for p in validation],
                limits=recipe.export_limits,
                mode=verify_mode,
            )
        except BaseException:
            # A wrong base model or a load error raises instead of
            # rejecting; the edit must still not survive it.
            session.revert()
            raise
        export = {
            "status": "passed" if verification.accepted else "rejected",
            "mode": verify_mode,
            "violations": list(verification.violations),
            "metrics": asdict(verification.metrics),
        }
        if verification.accepted:
            return finish(
                "accepted", attempts, (), site, trial.edit_id, export, adapter
            )
        session.revert()
        return finish(
            "export_rejected",
            attempts,
            (
                "The edit passed in the live model but the reloaded adapter did "
                "not match it within export_limits. The edit was reverted. Try "
                'verify mode "adapter", or loosen export_limits for a low-precision '
                "model.",
            ),
            site,
            None,
            export,
            adapter,
        )
    if not any(a.design is not None for a in attempts):
        errors = list(dict.fromkeys(a.error for a in attempts if a.error))
        hints: tuple[str, ...] = (
            "No site produced a design: " + "; ".join(errors) + ". Resource "
            "caps live under optimizer (max_logit_elements, max_reference_elements, "
            "max_total_tokens, max_model_forwards); other causes are keys that "
            "conflict with protected text or a site that is not a float32 Linear.",
        )
    else:
        hints = _hints(violations)
    return finish("rejected", attempts, hints)


def example_recipe() -> EditRecipe:
    """A filled-in starting point; replace the model, questions and sites."""
    return EditRecipe(
        name="france-capital-lyon",
        model="Qwen/Qwen3-0.6B",
        system="Reply with only the requested name. Do not explain.",
        subject="France",
        target="Lyon",
        design=(
            "What city is the capital of France?",
            "Name the capital city of France.",
        ),
        validation=(
            "Which city serves as France's national capital?",
            "Identify the French capital city.",
        ),
        locality=(
            LocalityAnswer("What city is the capital of Germany?", ("Berlin",)),
            LocalityAnswer("What language is mainly spoken in France?", ("French",)),
            LocalityAnswer("Name Italy's capital city.", ("Rome",)),
        ),
        calibration=(
            "Name the largest planet in the Solar System.",
            "What is the capital of Japan?",
            "Which continent is France in?",
            "What is the national language of France?",
        ),
        protected=("Water freezes at zero degrees Celsius.",),
        drift=(
            "Birds build nests in trees.",
            "The moon reflects light from the sun.",
            "A week contains seven days.",
        ),
        sites=(
            EditSite("model.layers.14.mlp.down_proj", "subject"),
            EditSite("model.layers.27.mlp.down_proj", "last"),
        ),
        # Full-vocabulary logits are retained per prompt token at design time;
        # these caps fit a 150k-token vocabulary and short chat prompts.
        optimizer=KeyedEditConfig(
            max_context_tokens=128,
            max_total_tokens=512,
            max_logit_elements=12_000_000,
            max_reference_elements=50_000_000,
        ),
        answer_limits=AnswerLimits(max_new_tokens=8),
    )
