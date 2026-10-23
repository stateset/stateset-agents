# Decontamination

A model trained on text from its own eval set scores recall, not skill.
`stateset_agents.data.decontamination` finds eval-set text in training data,
reports the evidence, and can filter or flag the rows involved. It works on the
repo's training formats: SFT chat rows (`messages`), preference pairs
(`prompt`/`chosen`/`rejected`, `completion`), RL prompt files (`prompt`,
`question`, `user_query`, `turns`), scenario files (`user_turns`,
`expected_resolution`) and question/answer benchmarks, as JSONL or as a JSON
list.

## How matching works

**Normalization.** Text goes through NFKC, casefold, punctuation and
underscores become spaces, and whitespace collapses: `ＭＡＹ...`, `may` and
`May!` are the same word. With `--mask-digits` every number becomes `0`, so
templated rows that differ only in reference numbers match.

**Exact n-grams (always on).** An eval row is contaminated when one of its
13-word n-grams (`--ngram-size`) occurs in a training row. That is the
convention of the GPT-3 and lm-evaluation-harness contamination studies.
Short eval segments are handled explicitly, not silently passed:

| Eval segment length | Check |
|---|---|
| 13 words or more | every 13-gram |
| 8 to 12 words (`--min-tokens`) | the whole segment, anywhere in a training segment |
| 3 to 7 words | an identical training segment (a duplicated short prompt) |
| fewer than 3 words | reported as `unchecked_short_segments` |

Text is compared per segment (one message, prompt or answer), so no n-gram
spans two messages. System and developer messages are skipped unless you pass
`--include-system`, because a fixed instruction shared by training and eval rows
is not leaked data, and including it would mark every row.

**Near duplicates (`--near-duplicates`).** MinHash signatures over 3-word
shingles, bucketed with LSH, propose candidate pairs of rows. Every candidate
is then checked with the exact Jaccard similarity of the two rows' shingle
sets, so the reported pairs and scores are exact, not estimates. The LSH band
shape follows `--threshold` and finds a pair at the threshold with at least 99%
probability. Rows are compared whole, so one shared boilerplate turn does not
make two different conversations near duplicates. At the default threshold of
0.8, the check finds light edits such as a changed name or (with
`--mask-digits`) changed numbers. To catch heavier paraphrase, lower
`--threshold` to 0.5–0.6 and read the evidence. A 13-gram-free paraphrase that
changes a word every 12 words has a 3-shingle Jaccard of about 0.6.

**Boilerplate.** Synthetic and templated data share sentences across splits:
persona openers, pushback lines, refusal clauses. Each contaminated row reports
`ngram_overlap`, the fraction of its n-grams found in training data. With
`--min-overlap 0.5`, only rows that are mostly present in training data count.
A training row is then filtered only if it holds that fraction of a
contaminated eval row itself, so rows that share only boilerplate are kept.

Output is deterministic. Words hash with CRC-32 and MinHash uses seeded
multiply-shift functions, so the report does not depend on `PYTHONHASHSEED`.
Every list follows input order. On a loaded 8-core box, 10,000 training rows
of about 160 words each take about 3 seconds for exact matching and about 6
seconds with near duplicates against 1,000 eval rows
(`tests/unit/test_decontamination.py` enforces generous bounds).

## Command line

```bash
# Report only (exit 0); --fail-on-contamination makes contamination exit 1.
stateset-agents data decontaminate --train sft_train.jsonl \
    --eval held_out.jsonl --eval builtin:gsm8k-test --report contamination.json

# Drop the training rows that overlap any eval set (new output file).
stateset-agents data decontaminate --train sft_train.jsonl --eval held_out.jsonl \
    --action filter --output sft_train.clean.jsonl

# Keep every row; mark contaminated ones under metadata.contamination.
stateset-agents data decontaminate --train sft_train.jsonl --eval held_out.jsonl \
    --action flag --output sft_train.flagged.jsonl

# Or score a clean subset of the eval set instead.
stateset-agents data decontaminate --train sft_train.jsonl --eval held_out.jsonl \
    --action filter --target eval --output held_out.clean.jsonl
```

`--eval` (and `--train`) take a path, `NAME=PATH`, or `builtin:NAME`: `support-bench`,
`tool-calling-bench`, `gsm8k-test` or `gsm8k-train`. The GSM8K sets need the
`datasets` library and the pinned revision, either cached or downloadable.
Outputs are never overwritten. The full option list is in
[CLI_REFERENCE.md](CLI_REFERENCE.md#stateset-agents-data).

### Report shape

```json
{
  "schema_version": 1,
  "kind": "decontamination_report",
  "method": "exact 13-gram overlap on normalized words",
  "config": {"ngram_size": 13, "min_tokens": 8, "...": "..."},
  "train": {"rows": 2, "contaminating_rows": 1,
            "contaminating": [{"row": "train.jsonl:1", "eval_sets": ["eval.jsonl"]}]},
  "eval_sets": {
    "eval.jsonl": {
      "rows": 2, "checked_rows": 2, "unchecked_short_segments": 0,
      "contaminated_rows": 1, "contamination_rate": 0.5,
      "contaminated": [{
        "row": "eval.jsonl:1", "match": ["ngram"], "ngram_overlap": 1.0,
        "overlap_ngram_count": 17,
        "overlap_ngrams": ["natalia sold clips to 48 of her friends in april and then she", "..."],
        "train_examples": ["train.jsonl:1"]
      }]
    }
  },
  "summary": {"eval_rows": 2, "contaminated_eval_rows": 1,
              "contaminating_train_rows": 1, "clean": false}
}
```

Row ids are `<file>:<line>` for JSONL and `<file>[index]` for JSON lists.

## Python API and curation

```python
from stateset_agents.data.decontamination import (
    DecontaminationConfig, apply_decontamination, builtin_eval_set,
    find_contamination, load_text_source, read_rows,
)

config = DecontaminationConfig(near_duplicates=True, min_overlap=0.0)
train = load_text_source("sft_train.jsonl", config=config)
evals = [builtin_eval_set("support-bench", config=config),
         load_text_source("held_out.jsonl", config=config)]
result = find_contamination([train], evals, config)
rows, _sha256 = read_rows("sft_train.jsonl")
clean_rows, removed = apply_decontamination(rows, result, "filter")
```

The production curation pipeline takes eval sets directly. Training rows that
overlap them are dropped after the temporal split. Held-out splits are never
filtered, and the manifest gains a `decontamination` block with each eval set's
hash and the number of rows removed:

```bash
python -m stateset_agents.data.curation --input traces.jsonl --output curated \
    --validation-after 2026-08-01T00:00:00Z --test-after 2026-09-01T00:00:00Z \
    --decontaminate-against held_out.jsonl --decontaminate-against builtin:gsm8k-test
```

(`curate_conversations(..., decontaminate_against=[...], decontamination=config)`
in Python.)

## Results on the repo's own eval sets (2026-10-23)

`scripts/audit_decontamination.py` pairs every eval set the repo ships or
generates with the data it trains on. Defaults are 13-grams plus near
duplicates at Jaccard ≥ 0.8. The retained output is
[`benchmark_results/decontamination/20261023/audit.json`](../benchmark_results/decontamination/20261023/audit.json).

| Eval set | Training data | Contaminated | Verdict |
|---|---|---|---|
| `examples/sample_eval_set.jsonl` (Cookbook Recipe 5) | starter `scenarios.jsonl` (Recipe 1) + bundled support corpus | **was 8/10, now 0/10** | **Real contamination, fixed** |
| Support bench eval split `[16:]` | its train split `[:16]` (phase 0, `03_first_finetune.py`) | 0/8 | Clean |
| Tool-calling bench eval split | its train split (phase-0 adapter) | 0/1 | Clean |
| Bundled tool-calling bench | tool-calling starter `scenarios.jsonl` | 6/8 | Latent risk, documented |
| GSM8K test (question + solution) | GSM8K train | 9/1319 (0.7%), max overlap 0.29 | Dataset property, documented |
| GSM8K test questions, digits masked | GSM8K train questions | 7/1319, 1 near duplicate (Jaccard 0.84) | Dataset property, documented |
| Eval ladder, depth 2, no refusals | ladder train rows + harvest prompts | 0/12 | Clean |
| Eval ladder, depth 2, 50% refusals | same | 6/12, max overlap 0.75 | By design, documented |
| Eval ladder, depth 3, no refusals / 50% refusals | same | 10/12 and 12/12, max overlap 0.78 / 0.89 | By design, documented |
| `customer-support-multiturn-v1` eval (240) | its train split (560) | 164/240 at any 13-gram, 1/240 at `--min-overlap 0.5`, 0 near duplicates | Template overlap, documented |
| Product-use tasks v0.3 | `examples.v0.3.jsonl` | 7/7 | Known and already documented |
| Industry pilot validation splits (×5) | their train splits | 0/16 each | Clean |

**Cookbook sample eval set (fixed).** Recipe 1 trains on the starter's eight
`scenarios.jsonl` queries. Recipe 5 then scores that checkpoint on
`examples/sample_eval_set.jsonl`, but 6 of its 10 queries were verbatim
starter queries. Eight of the ten appeared word for word in the bundled
24-query support corpus ("The app crashes every time I open it", "What are your
business hours?", ...). Recipe 5's pass rate therefore measured recall of
training prompts. The file now holds ten new queries with the same schema and
intent mix, and the check reports 0/10 against both corpora. A test
(`test_cookbook_eval_set_stays_clean_against_the_training_corpora`) keeps it
that way.

**Tool-calling starter (latent).** The `tool-calling-agent` starter trains on
six queries copied from the bundled tool-calling bench. No documented
workflow scores a starter-trained adapter on that bench; the phase-0 benchmark
uses the bench's own split, which is clean. Do not report a starter-trained
adapter's score on `builtin:tool-calling-bench` without first running
`data decontaminate --target eval --action filter` against the starter
scenarios.

**GSM8K.** The repo trains on GSM8K train and evaluates on GSM8K test. Nine
test rows share at least one 13-gram with a train row. Most of the shared text
is solution boilerplate ("there are 7 days in a week so ..."). Three
*questions* share long spans with a train question, and with numbers masked,
seven do. These are GSM8K's own templated problem families, such as test 632
and train 20, the same stamp-design problem with other names and numbers.
Test 602 and train 1314 are near duplicates once digits are masked (Jaccard
0.84). Nothing in the repo copies test items into training. This is a property
of the public benchmark, so the standard test split stays as is. For a strict
number, list the affected test rows and leave them out of scoring:

```bash
stateset-agents data decontaminate --train builtin:gsm8k-train \
    --eval builtin:gsm8k-test --field question --mask-digits --report gsm8k.json
```

**Eval ladder.** `build_ladder` guarantees that no eval prompt asks for the
same *set* of issues as a harvest prompt, and its own check enforces that. The
13-gram view shows a different, intended overlap. At depth 3 an eval prompt
such as "the printer shows offline, and on top of that Outlook stopped
syncing, and also the VPN won't connect" contains a two-issue span that also
occurs in a harvest prompt. The harvest is curated into the next generation's
SFT data, so the model has seen that span. Refusal prompts also share the
decline clause ("... too, but leave the VPN alone — I just want the rest
handled"). The eval measures recombination of known clauses, which is its
stated purpose, not recall of eval answers. Depth 2 without refusals is fully
13-gram disjoint. When you need a strictly n-gram-clean ladder eval, use depth
2 without refusals, or filter the harvest with `data decontaminate`.

**Support multi-turn dataset.** Train and eval scenarios are generated from
shared persona openers ("hello, I'm not very good with computers so please
bear with me") and a shared pool of pushback turns ("that doesn't seem fair,
can't you make an exception just this once?"). Hence 164/240 eval scenarios
share a 13-gram with training scenarios, but with low overlap: median 0.05,
146 of the 164 under 0.15, and a maximum of 0.67. No eval scenario is a near duplicate of a training scenario, and no
`expected_resolution` is repeated across splits. At `--min-overlap 0.5`, one
eval scenario (`eval.jsonl:130`) remains: its rubric reference sentence ("...
the device is 13 months old, past the 12-month warranty, and offer the
flat-fee repair") matches a training scenario's. That is a judge reference,
not a training target.

**Product-use benchmark.** The committed `examples.v0.3.jsonl` demonstrations
are the public tasks solved. `benchmarks/product_use/README.md` already states
that "the demonstrations are public training examples, so their scores are not
evidence of model capability." The audit confirms 7/7.

Reproduce the audit with:

```bash
python scripts/audit_decontamination.py --output audit.json   # add --skip-gsm8k offline
```

## Limitations

- Matching is lexical. A translation or a heavy rewrite of an eval item is not
  found. Neither is contamination already inside a pretrained base model.
- Words are whitespace-delimited after normalization. Scripts written without
  spaces (Chinese, Japanese) form long "words" and need a tokenizer-based
  segmentation first.
- Near-duplicate detection runs in memory on numpy. Exact matching streams
  training rows, but the eval index is held in memory.
- Short eval prompts (under 8 words) only match identical training segments.
  Prompts under 3 words are not checked, and the report counts them.
