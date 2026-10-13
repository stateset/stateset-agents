"""Unit tests for the multi-turn customer-support dataset generator."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.data.customer_support_bench import SupportRewardComposite
from stateset_agents.data.support_multiturn import (
    DATASET_NAME,
    INTENTS,
    PERSONAS,
    SupportDialogueScenario,
    _templates_for,
    build_support_dataset,
    export_dataset,
    load_support_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(scope="module")
def dataset() -> dict[str, list[SupportDialogueScenario]]:
    return build_support_dataset(seed=0)


@pytest.fixture(scope="module")
def exported(
    dataset: dict[str, list[SupportDialogueScenario]],
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, str]:
    directory = tmp_path_factory.mktemp("support_export")
    return directory, export_dataset(dataset, directory)


class TestBuild:
    def test_dataset_name(self) -> None:
        assert DATASET_NAME == "stateset/customer-support-multiturn-v1"

    def test_sizes(self, dataset: dict[str, list[SupportDialogueScenario]]) -> None:
        assert len(dataset["train"]) >= 500
        assert len(dataset["eval"]) >= 200

    def test_deterministic(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        assert build_support_dataset(seed=0) == dataset

    def test_seed_changes_output(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        assert build_support_dataset(seed=1)["train"] != dataset["train"]

    def test_eval_independent_of_train_size(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        small = build_support_dataset(seed=0, n_train=50, n_eval=240)
        assert small["eval"] == dataset["eval"]

    def test_all_intents_and_personas_covered(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        assert len(INTENTS) >= 6
        for split in ("train", "eval"):
            rows = dataset[split]
            assert {s.intent for s in rows} == set(INTENTS)
            assert {s.persona for s in rows} == set(PERSONAS)
            assert {s.split for s in rows} == {split}

    def test_product_catalog_breadth(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        products = {s.product for split in dataset.values() for s in split}
        assert len(products) >= 12

    def test_turn_shape(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        for split in dataset.values():
            for s in split:
                assert 2 <= len(s.user_turns) <= 4
                assert all(len(t.split()) < 60 for t in s.user_turns)
                assert s.must_acknowledge
                assert s.policy_facts
                assert s.expected_resolution.endswith(".")
                assert "{" not in s.expected_resolution

    def test_rejects_non_positive_sizes(self) -> None:
        with pytest.raises(ValueError):
            build_support_dataset(n_train=0)


class TestLeakage:
    def test_products_disjoint(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        train = {s.product for s in dataset["train"]}
        eval_ = {s.product for s in dataset["eval"]}
        assert train.isdisjoint(eval_)

    def test_templates_disjoint(self) -> None:
        train = {t.key for t in _templates_for("train")}
        eval_ = {t.key for t in _templates_for("eval")}
        assert train and eval_
        assert train.isdisjoint(eval_)
        # Every intent has held-out complication templates.
        assert {t.intent for t in _templates_for("eval")} == set(INTENTS)

    def test_first_turns_disjoint_and_unique(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        firsts = {}
        for split, rows in dataset.items():
            turns = [s.user_turns[0] for s in rows]
            assert len(turns) == len(set(turns))
            firsts[split] = set(turns)
        assert firsts["train"].isdisjoint(firsts["eval"])

    def test_ids_unique(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        ids = [s.id for split in dataset.values() for s in split]
        assert len(ids) == len(set(ids))


class TestPolicyGrounding:
    def test_has_refusals_and_escalations(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        resolutions = [s.expected_resolution.lower() for s in dataset["train"]]
        assert any("decline" in r for r in resolutions)
        assert any("escalate" in r for r in resolutions)
        assert any("store credit" in r for r in resolutions)

    def test_outside_window_never_promises_refund(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        rows = [
            s
            for s in dataset["train"]
            if "outside the" in s.expected_resolution
            and "refund window" in s.expected_resolution
        ]
        assert rows
        for s in rows:
            assert "refund has been issued" in s.must_avoid
            assert "store credit" in s.must_acknowledge

    def test_credential_guardrails_present(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        account = [s for s in dataset["train"] if s.intent == "account"]
        assert all("provide your password" in s.must_avoid for s in account)


class TestSchema:
    def test_record_roundtrip(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        for s in dataset["train"][:50] + dataset["eval"][:50]:
            record = s.to_record()
            json.dumps(record)
            assert SupportDialogueScenario.from_record(record) == s

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda r: r.pop("intent"),
            lambda r: r.update(extra="x"),
            lambda r: r.update(intent="sales"),
            lambda r: r.update(split="test"),
            lambda r: r.update(persona="pirate"),
            lambda r: r.update(id="Bad Id"),
            lambda r: r.update(user_turns=["only one"]),
            lambda r: r.update(user_turns=["a", "b", "c", "d", "e"]),
            lambda r: r.update(user_turns="not a list"),
            lambda r: r.update(must_acknowledge=[]),
            lambda r: r.update(policy_facts=["ok", 3]),
            lambda r: r.update(expected_resolution="  "),
        ],
    )
    def test_from_record_rejects_malformed(
        self,
        dataset: dict[str, list[SupportDialogueScenario]],
        mutate: Any,
    ) -> None:
        record = dataset["train"][0].to_record()
        mutate(record)
        with pytest.raises(ValueError):
            SupportDialogueScenario.from_record(record)

    def test_from_record_rejects_non_mapping(self) -> None:
        with pytest.raises(ValueError):
            SupportDialogueScenario.from_record(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_to_scenario_keys(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        s = dataset["eval"][0]
        scenario = s.to_scenario()
        assert scenario["user_query"] == s.user_turns[0]
        assert isinstance(scenario["must_acknowledge"], list)
        assert isinstance(scenario["must_avoid"], list)
        for key in (
            "id",
            "intent",
            "persona",
            "product",
            "policy_facts",
            "user_turns",
            "expected_resolution",
        ):
            assert key in scenario

    @pytest.mark.asyncio
    async def test_to_scenario_works_with_reward(
        self, dataset: dict[str, list[SupportDialogueScenario]]
    ) -> None:
        s = next(x for x in dataset["train"] if x.intent == "refund")
        scenario = s.to_scenario()
        reply = (
            "Thank you for reaching out, I understand. "
            + " ".join(scenario["must_acknowledge"])
            + ". I'll take care of this right away for your order."
        )
        turns = [
            ConversationTurn(role="user", content=scenario["user_query"]),
            ConversationTurn(role="assistant", content=reply),
        ]
        result = await SupportRewardComposite().compute_reward(turns, scenario)
        assert result.breakdown["intent_score"] == pytest.approx(1.0)
        assert 0.0 < result.score <= 1.0


class TestExportLoad:
    def test_export_files_and_sha(
        self,
        dataset: dict[str, list[SupportDialogueScenario]],
        exported: tuple[Path, str],
        tmp_path: Path,
    ) -> None:
        directory, sha = exported
        for name in ("train.jsonl", "eval.jsonl", "README.md"):
            assert (directory / name).is_file()
        digest = hashlib.sha256()
        digest.update((directory / "train.jsonl").read_bytes())
        digest.update((directory / "eval.jsonl").read_bytes())
        assert digest.hexdigest() == sha
        # Stable across a rebuild + re-export.
        assert export_dataset(build_support_dataset(seed=0), tmp_path) == sha
        assert (tmp_path / "train.jsonl").read_bytes() == (
            directory / "train.jsonl"
        ).read_bytes()

    def test_jsonl_sorted_keys_one_row_per_line(
        self, exported: tuple[Path, str]
    ) -> None:
        directory, _ = exported
        lines = (directory / "eval.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 240
        row = json.loads(lines[0])
        assert list(row) == sorted(row)

    def test_readme_card(self, exported: tuple[Path, str]) -> None:
        directory, sha = exported
        card = (directory / "README.md").read_text(encoding="utf-8")
        for needle in (
            "license: apache-2.0",
            "## Generation method",
            "## Split statistics",
            "## Leakage controls",
            "## Limitations",
            sha,
        ):
            assert needle in card

    def test_local_roundtrip(
        self,
        dataset: dict[str, list[SupportDialogueScenario]],
        exported: tuple[Path, str],
    ) -> None:
        directory, _ = exported
        assert load_support_dataset(directory, split="train") == dataset["train"]
        assert load_support_dataset(str(directory), split="eval") == dataset["eval"]

    def test_unknown_split(self, exported: tuple[Path, str]) -> None:
        with pytest.raises(ValueError):
            load_support_dataset(exported[0], split="test")

    def test_malformed_row_reports_line(self, tmp_path: Path) -> None:
        (tmp_path / "train.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="train.jsonl:1"):
            load_support_dataset(tmp_path, split="train")

    def test_split_mismatch_rejected(
        self, dataset: dict[str, list[SupportDialogueScenario]], tmp_path: Path
    ) -> None:
        row = json.dumps(dataset["eval"][0].to_record())
        (tmp_path / "train.jsonl").write_text(row + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="split"):
            load_support_dataset(tmp_path, split="train")

    @pytest.mark.parametrize("revision", [None, "main", "v1.0", "ABCDEF" * 7])
    def test_remote_requires_pinned_revision(self, revision: str | None) -> None:
        with mock.patch("huggingface_hub.snapshot_download") as snap:
            with pytest.raises(ValueError, match="40-character"):
                load_support_dataset(DATASET_NAME, revision=revision)
            snap.assert_not_called()

    def test_remote_uses_snapshot_download(self, exported: tuple[Path, str]) -> None:
        directory, _ = exported
        with mock.patch(
            "huggingface_hub.snapshot_download", return_value=str(directory)
        ) as snap:
            rows = load_support_dataset(
                DATASET_NAME, revision=VALID_REVISION, split="eval"
            )
        snap.assert_called_once_with(
            repo_id=DATASET_NAME, repo_type="dataset", revision=VALID_REVISION
        )
        assert len(rows) == 240


class TestCli:
    def test_cli_writes_files(self, exported: tuple[Path, str], tmp_path: Path) -> None:
        _, sha = exported
        out = tmp_path / "cli"
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "build_support_dataset.py"),
                "--output",
                str(out),
                "--seed",
                "0",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(REPO_ROOT),
        )
        assert proc.stdout.strip() == sha
        for name in ("train.jsonl", "eval.jsonl", "README.md"):
            assert (out / name).is_file()
