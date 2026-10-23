"""Decontamination: exact/normalized n-grams, near duplicates, actions, determinism."""

from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

from stateset_agents.data.curation import curate_conversations
from stateset_agents.data.decontamination import (
    BUILTIN_EVAL_SETS,
    LSH_RECALL,
    DecontaminationConfig,
    apply_decontamination,
    builtin_eval_set,
    extract_segments,
    find_contamination,
    load_text_source,
    lsh_shape,
    normalize_text,
    read_rows,
    resolve_eval_source,
    text_source_from_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

QUESTION = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half "
    "as many clips in May. How many clips did Natalia sell altogether in April "
    "and May?"
)
UNRELATED = [
    "Preheat the oven and knead the dough with flour, water, salt and yeast "
    "for ten minutes before letting it rise overnight on the counter.",
    "The quarterly board meeting moved to Thursday afternoon because the "
    "auditors needed another week to finish reviewing the ledger entries.",
    "Hikers should carry extra water, a paper map and a headlamp when the "
    "trail runs past sunset in the high desert during late autumn.",
]


def source(name, rows, config=None):
    return text_source_from_rows(
        name, [(f"{name}:{i + 1}", row) for i, row in enumerate(rows)], config=config
    )


def chat(user, assistant="Sure."):
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def run(train_rows, eval_rows, **options):
    config = DecontaminationConfig(**options)
    return find_contamination(
        [source("train", train_rows, config)],
        [source("eval", eval_rows, config)],
        config,
    )


# --- normalization and text extraction -------------------------------------


def test_normalize_text_casefolds_strips_punctuation_and_collapses_space():
    assert normalize_text("  Hello,\tWORLD!!  it's_me \n") == "hello world it s me"
    # NFKC: full-width letters and digits are the same words.
    assert normalize_text("ＨＥＬＬＯ　Ｗｏｒｌｄ １２") == "hello world 12"
    assert normalize_text("Straße") == normalize_text("STRASSE")
    assert normalize_text("order 4521 and 7", mask_digits=True) == "order 0 and 0"


def test_ascii_fast_path_matches_the_unicode_rules():
    rng = random.Random(0)
    alphabet = "abcXYZ019 _-.,!?'\"\t\n()[]{}#@$%^&*+=/\\|~`<>;:"
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        # A non-ASCII suffix forces the regex path for the same ASCII prefix.
        expected = " ".join(part for part in (normalize_text(text), "é") if part)
        assert normalize_text(text + " é") == expected


def test_extract_segments_reads_the_repo_row_shapes():
    assert extract_segments(chat("hi there", "hello")) == ["hi there", "hello"]
    assert extract_segments(chat("hi there"), roles=("system", "user")) == [
        "You are a helpful assistant.",
        "hi there",
    ]
    preference = {
        "prompt": [{"role": "user", "content": "q?"}],
        "chosen": [{"role": "assistant", "content": "good"}],
        "rejected": "bad",
    }
    assert extract_segments(preference) == ["q?", "good", "bad"]
    assert extract_segments({"prompt": "p", "completion": "c", "label": True}) == [
        "p",
        "c",
    ]
    assert extract_segments({"question": "q", "answer": "a #### 4"}) == [
        "q",
        "a #### 4",
    ]
    assert extract_segments({"turns": ["one", "two"], "expect": ["x"]}) == [
        "one",
        "two",
    ]
    parts = {"role": "user", "content": [{"type": "text", "text": "look"}, "here"]}
    assert extract_segments([parts]) == ["look", "here"]
    assert extract_segments("bare string") == ["bare string"]
    assert extract_segments({"question": "q", "notes": "n"}, fields=["notes"]) == ["n"]
    assert extract_segments({"messages": [], "prompt": "  "}) == []


def test_read_rows_handles_jsonl_and_json_and_rejects_bad_records(tmp_path):
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text('{"prompt": "a"}\n\n{"prompt": "b"}\n', encoding="utf-8")
    rows, digest = read_rows(jsonl)
    assert [row_id for row_id, _ in rows] == ["train.jsonl:1", "train.jsonl:3"]
    assert len(digest) == 64
    listed = tmp_path / "tasks.json"
    listed.write_text(json.dumps({"tasks": [{"prompt": "x"}]}), encoding="utf-8")
    assert read_rows(listed)[0] == [("tasks.json[0]", {"prompt": "x"})]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"prompt": "a"}\n{"prompt": "a", "prompt": "b"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        read_rows(bad)
    ambiguous = tmp_path / "ambiguous.json"
    ambiguous.write_text(json.dumps({"rows": [], "tasks": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        read_rows(ambiguous)


@pytest.mark.parametrize(
    "options",
    [
        {"ngram_size": 0},
        {"min_tokens": 14},
        {"min_overlap": 1.5},
        {"near_duplicate_threshold": 0.0},
        {"roles": ()},
        {"fields": ("",)},
        {"minhash_permutations": 8},
        {"ngram_size": True},
    ],
)
def test_config_rejects_invalid_settings(options):
    with pytest.raises(ValueError):
        DecontaminationConfig(**options)


# --- exact and normalized matching ------------------------------------------


def test_exact_13_gram_match_reports_evidence_in_eval_order():
    result = run(
        [chat("Solve this: " + QUESTION, "72"), chat(UNRELATED[0])],
        [{"question": QUESTION}, {"question": UNRELATED[1]}],
    )
    report = result.report["eval_sets"]["eval"]
    assert report["contaminated_rows"] == 1
    entry = report["contaminated"][0]
    assert entry["row"] == "eval:1" and entry["match"] == ["ngram"]
    assert entry["ngram_overlap"] == 1.0
    assert entry["overlap_ngrams"][0] == (
        "natalia sold clips to 48 of her friends in april and then she"
    )
    assert entry["overlap_ngrams"][1].startswith("sold clips to 48")
    assert all(len(gram.split()) == 13 for gram in entry["overlap_ngrams"])
    assert entry["train_examples"] == ["train:1"]
    assert result.contaminating == {"train:1": ["eval"]}
    assert result.contaminated == {"eval": ["eval:1"]}


def test_normalized_match_ignores_case_punctuation_whitespace_and_width():
    disguised = (
        "NATALIA   sold clips—to 48 of her FRIENDS in April; and then she sold "
        "half as many clips in ＭＡＹ... how many clips did natalia sell?"
    )
    result = run([{"prompt": disguised}], [{"question": QUESTION}])
    assert result.report["summary"]["contaminated_eval_rows"] == 1


def test_masked_digits_match_templates_that_differ_only_in_numbers():
    template = (
        "Order {} shipped on day {} with {} boxes and {} crates along route {} "
        "to dock {} at pier {} before {} pm"
    )
    rows = (
        [{"prompt": template.format(4521, 3, 12, 7, 66, 9, 4, 5)}],
        [{"question": template.format(8803, 6, 20, 2, 41, 3, 8, 7)}],
    )
    # A number every few words breaks every 13-gram unless digits are masked.
    assert run(*rows).is_clean
    assert not run(*rows, mask_digits=True).is_clean


def test_short_segments_match_whole_and_tiny_ones_only_as_duplicates():
    eval_rows = [
        {"user_query": "My order arrived damaged and I want a refund"},  # 9 words
        {"user_query": "What are your business hours?"},  # 5 words
        {"user_query": "Are you hiring?"},  # 3 words: duplicate check only
        {"user_query": "Thanks!"},  # unchecked
    ]
    train_rows = [
        chat("Hi. My order arrived damaged and I want a refund, today please."),
        chat("Tell me: what are your business hours? I need to visit."),
        chat("Are you hiring?"),
        chat("Thanks!"),
    ]
    report = run(train_rows, eval_rows).report["eval_sets"]["eval"]
    # Row 2 is only a substring of a longer training message: not evidence.
    assert [entry["row"] for entry in report["contaminated"]] == ["eval:1", "eval:3"]
    assert report["unchecked_short_segments"] == 1
    assert report["checked_rows"] == 3


def test_no_false_positives_on_unrelated_or_generic_text():
    rng = random.Random(7)
    vocabulary = [f"word{i}" for i in range(2000)]
    train_rows = [
        chat(" ".join(rng.choice(vocabulary) for _ in range(40))) for _ in range(300)
    ] + [chat(text) for text in UNRELATED[:2]]
    eval_rows = [
        {"question": " ".join(rng.choice(vocabulary) for _ in range(40))}
        for _ in range(50)
    ] + [{"question": QUESTION}, {"question": UNRELATED[2]}]
    result = run(train_rows, eval_rows, near_duplicates=True)
    assert result.is_clean
    assert result.report["summary"]["clean"] is True
    assert result.report["train"]["contaminating"] == []


def test_system_prompts_are_ignored_unless_requested():
    shared = (
        "You are the support agent for Acme. Always confirm the order number "
        "and never promise a refund before checking the policy."
    )
    train = [
        {
            "messages": [
                {"role": "system", "content": shared},
                {"role": "user", "content": "a"},
            ]
        }
    ]
    evals = [
        {
            "messages": [
                {"role": "system", "content": shared},
                {"role": "user", "content": "b"},
            ]
        }
    ]
    assert run(train, evals).is_clean
    assert not run(train, evals, roles=("system", "user")).is_clean


def test_min_overlap_separates_boilerplate_from_leaked_rows():
    boilerplate = (
        "please let me know if there is anything else i can help you with today"
    )
    leaked = QUESTION + " " + boilerplate
    boiler_only = UNRELATED[0] + " " + boilerplate
    train = [chat("x", boilerplate), chat(QUESTION)]
    evals = [{"question": leaked}, {"question": boiler_only}]
    assert len(run(train, evals).contaminated["eval"]) == 2
    strict = run(train, evals, min_overlap=0.5)
    assert strict.contaminated["eval"] == ["eval:1"]
    # Only the training row behind the leaked row's n-grams is contaminating.
    assert strict.contaminating == {"train:2": ["eval"]}


# --- near duplicates ----------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7, 0.8, 0.9, 1.0])
def test_lsh_shape_meets_the_recall_target_at_the_threshold(threshold):
    bands, rows = lsh_shape(threshold, 120)
    assert bands * rows <= 120
    assert 1 - (1 - threshold**rows) ** bands >= LSH_RECALL
    if rows < 120:  # one more row per band would miss the target
        wider = 120 // (rows + 1)
        assert 1 - (1 - threshold ** (rows + 1)) ** wider < LSH_RECALL


def _jaccard(a, b, size=3):
    def shingles(text):
        words = normalize_text(text).split()
        return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}

    left, right = shingles(a), shingles(b)
    return len(left & right) / len(left | right)


def test_near_duplicate_catches_paraphrase_that_breaks_every_13_gram():
    words = [f"token{i}" for i in range(60)]
    edited = list(words)
    for position in range(6, 60, 12):  # every 13-word window has an edit
        edited[position] = "changed"
    original, paraphrase = " ".join(words), " ".join(edited)
    expected = _jaccard(original, paraphrase)
    assert 0.5 < expected < 0.8
    result = run(
        [{"prompt": paraphrase}],
        [{"question": original}],
        near_duplicates=True,
        near_duplicate_threshold=0.5,
    )
    entry = result.report["eval_sets"]["eval"]["contaminated"][0]
    assert entry["match"] == ["near_duplicate"]
    assert entry["overlap_ngrams"] == []
    # Verified exactly, not estimated from MinHash.
    assert entry["near_duplicate"]["jaccard"] == pytest.approx(expected, abs=1e-6)
    assert entry["near_duplicate"]["train_row"] == "train:1"
    assert run(
        [{"prompt": paraphrase}], [{"question": original}], near_duplicates=True
    ).is_clean


def test_near_duplicates_compare_rows_not_shared_boilerplate_turns():
    pushback = "that does not seem fair can you make an exception just this once please"
    train = [{"user_turns": [UNRELATED[0], pushback]}]
    evals = [{"user_turns": [UNRELATED[1], pushback]}]
    result = run(train, evals, near_duplicates=True, min_overlap=0.6)
    assert result.is_clean


# --- determinism and performance ---------------------------------------------


def _report_json(seed):
    rng = random.Random(seed)
    vocabulary = [f"w{i}" for i in range(300)]
    train = [
        chat(" ".join(rng.choice(vocabulary) for _ in range(30))) for _ in range(200)
    ]
    evals = [
        {"question": train[i]["messages"][1]["content"]} for i in range(0, 200, 20)
    ] + [
        {"question": " ".join(rng.choice(vocabulary) for _ in range(30))}
        for _ in range(20)
    ]
    result = run(train, evals, near_duplicates=True, near_duplicate_threshold=0.3)
    return json.dumps(result.report, sort_keys=True)


def test_output_is_deterministic_across_runs():
    first = _report_json(3)
    assert first == _report_json(3)
    assert json.loads(first)["summary"]["contaminated_eval_rows"] >= 10


def test_output_does_not_depend_on_python_hash_seed():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from test_decontamination import _report_json; print(_report_json(5))"
    )
    outputs = set()
    for hash_seed in ("1", "2"):
        completed = subprocess.run(
            [sys.executable, "-c", code, str(Path(__file__).parent)],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
            env={"PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(REPO_ROOT)},
        )
        outputs.add(completed.stdout)
    assert len(outputs) == 1


def test_ten_thousand_training_rows_take_seconds():
    rng = random.Random(11)
    vocabulary = [f"v{i}" for i in range(5000)]

    def sentence(length):
        return " ".join(rng.choice(vocabulary) for _ in range(length))

    train_rows = [chat(sentence(30), sentence(50)) for _ in range(10_000)]
    eval_rows = [{"question": sentence(40)} for _ in range(500)]
    eval_rows[7] = {"question": train_rows[1234]["messages"][2]["content"]}
    train = source("train", train_rows)
    evals = source("eval", eval_rows)
    for near, budget in ((False, 10.0), (True, 20.0)):
        config = DecontaminationConfig(near_duplicates=near)
        started = time.perf_counter()
        result = find_contamination([train], [evals], config)
        elapsed = time.perf_counter() - started
        assert result.contaminated["eval"] == ["eval:8"]
        assert result.contaminating == {"train:1235": ["eval"]}
        assert elapsed < budget, f"near_duplicates={near} took {elapsed:.1f}s"


# --- actions, built-ins and curation ------------------------------------------


def test_filter_and_flag_act_on_contaminating_training_rows_without_mutation():
    rows = [("t:1", chat(QUESTION)), ("t:2", chat(UNRELATED[0])), ("t:3", "text")]
    original = copy.deepcopy(rows)
    config = DecontaminationConfig()
    result = find_contamination(
        [text_source_from_rows("train", rows, config=config)],
        [source("eval", [{"question": QUESTION}], config)],
        config,
    )
    kept, removed = apply_decontamination(rows, result, "filter")
    assert removed == 1 and kept == [rows[1][1], "text"]
    flagged, affected = apply_decontamination(rows, result, "flag")
    assert affected == 1 and len(flagged) == 3
    assert flagged[0]["metadata"]["contamination"]["eval_sets"] == ["eval"]
    assert rows == original
    eval_kept, eval_removed = apply_decontamination(
        [("eval:1", {"question": QUESTION})],
        result,
        "filter",
        target="eval",
        eval_set="eval",
    )
    assert eval_kept == [] and eval_removed == 1
    with pytest.raises(ValueError, match="Unknown eval set"):
        apply_decontamination(rows, result, "filter", target="eval", eval_set="nope")
    bad = [("t:1", {"messages": chat(QUESTION)["messages"], "metadata": "x"})]
    with pytest.raises(ValueError, match="object metadata"):
        apply_decontamination(bad, result, "flag")


def test_builtin_eval_sets_and_spec_resolution(tmp_path, monkeypatch):
    support = builtin_eval_set("support-bench")
    assert len(support.records) == 24 and support.path == "builtin:support-bench"
    assert len(builtin_eval_set("tool-calling-bench").records) == 8
    assert set(BUILTIN_EVAL_SETS) >= {"support-bench", "gsm8k-test"}
    with pytest.raises(ValueError, match="Unknown built-in"):
        builtin_eval_set("mmlu")

    from stateset_agents.data import gsm8k

    example = gsm8k.GSM8KExample(
        question=QUESTION, answer_text="#### 72", gold_answer=72
    )
    monkeypatch.setattr(gsm8k, "load_gsm8k", lambda split: [example])
    assert builtin_eval_set("gsm8k-test").records[0].segments == (QUESTION, "#### 72")

    path = tmp_path / "held_out.jsonl"
    path.write_text(json.dumps({"question": QUESTION}) + "\n", encoding="utf-8")
    assert resolve_eval_source(str(path)).name == "held_out.jsonl"
    assert resolve_eval_source(f"mine={path}").name == "mine"
    assert load_text_source(path).sha256 == read_rows(path)[1]


def test_duplicate_eval_set_names_are_rejected():
    evals = source("eval", [{"question": QUESTION}])
    with pytest.raises(ValueError, match="uniquely named"):
        find_contamination([], [evals, evals])


def _conversation(source_id, timestamp, user):
    return {
        "source_id": source_id,
        "timestamp": timestamp,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "Here is how to solve it."},
        ],
    }


def test_curation_removes_training_rows_that_overlap_eval_sets():
    records = [
        _conversation("a", "2026-01-01T00:00:00Z", QUESTION),
        _conversation("b", "2026-01-02T00:00:00Z", UNRELATED[0]),
        _conversation("c", "2026-04-01T00:00:00Z", QUESTION + " Show work."),
    ]
    plain = curate_conversations(
        records,
        validation_after="2026-02-01T00:00:00Z",
        test_after="2026-03-01T00:00:00Z",
    )
    assert len(plain["splits"]["train"]) == 2
    assert "decontamination" not in plain["manifest"]
    held_out = source("gsm8k-test", [{"question": QUESTION}])
    cleaned = curate_conversations(
        records,
        validation_after="2026-02-01T00:00:00Z",
        test_after="2026-03-01T00:00:00Z",
        decontaminate_against=[held_out],
    )
    assert [row["messages"][0]["content"] for row in cleaned["splits"]["train"]] == [
        UNRELATED[0]
    ]
    assert len(cleaned["splits"]["test"]) == 1  # held-out splits are never filtered
    block = cleaned["manifest"]["decontamination"]
    assert block["removed_training_rows"] == 1
    assert block["eval_sets"]["gsm8k-test"]["contaminated_rows"] == 1
    assert cleaned["manifest"]["split_rows"]["train"] == 1


def test_cookbook_eval_set_stays_clean_against_the_training_corpora():
    """examples/sample_eval_set.jsonl used to repeat 6 of the 8 starter
    scenarios Recipe 1 trains on (8/10 bundled support queries verbatim)."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import audit_decontamination as audit
    finally:
        sys.path.remove(str(REPO_ROOT / "scripts"))
    config = DecontaminationConfig(near_duplicates=True)
    for check in (audit._cookbook_eval, audit._support_bench):
        train, evals = check(config)
        assert find_contamination(train, evals, config).is_clean
