# River expert SFT warmup — 2026-10-18

The Qwen3.5-9B SFT canary completed successfully. Eight updates on 64 verified synthetic demonstrations produced a checkpoint that passed 26/32 matched validation cases versus the base model's 25/32. The preregistered validation rule selected SFT, which then passed 28/32 fresh held-out cases (87.5%). The standard improvement gate remains failed: this is one synthetic seed with a small net validation gain, not production-quality certification.

| Measurement | Result |
| --- | --- |
| Seed | 20261020 |
| Training examples | 64 rule-based demonstrations, 8 per scenario family |
| Real-tokenizer check | Maximum 584 tokens; no truncation; 6,743 assistant target tokens |
| Training | One epoch, batch size 8, LoRA rank 16, learning rate 2e-5; eight updates |
| Training loss | 0.086169 final batch; not a quality score |
| Matched base validation | 25/32 (78.125%) |
| Matched SFT validation | 26/32 (81.25%) |
| Paired validation changes | 3 newly solved; 2 regressed; 23 both passed; 4 both failed |
| Selected checkpoint | SFT, selected using validation only |
| Selected checkpoint held-out success | 28/32 (87.5%; Wilson 95% interval 71.9%–95.0%) |
| Held-out policy violation rate | 12.5% |
| Held-out timeouts | 0% |
| Executed held-out action format | 100/100 valid JSON with exact order IDs |
| Rollout admission | 96 trajectories; 98,304 tokens reserved of 110,000 |
| Focused local tests | 104 passed |
| Total execution time | 247.2 seconds |
| Final billed cost | Unknown; rollout token admission is not a dollar cap |

[Selected checkpoint pointer](selected/river_checkpoint.json), [training pointer](sft/river_checkpoint.json), [review summary](review-summary.json), [validation comparison](validation-comparison.json), and [held-out report](sft-test.json) contain the detailed evidence. The weights are hosted by River; the pointer contains no local adapter weights.

The plan was written before provider execution. All 64 training conversations were built from canonical training cases using observed facts and independently replayed in a fresh instance of the same sandbox. They are synthetic rule-based supervision, not human-reviewed examples. The training export includes no validation or test cases. Fresh case IDs and order-fact fingerprints were checked against both earlier canaries and were disjoint within this run. Training had 14/30-day windows; held-out cases retain the generator's unseen 7/45-day windows.

Base and SFT validation used identical cases, source identity, sampling settings and seed. The greater validation success count won; ties would have retained base. The winner and held-out attempt were written durably before any test sampling. The held-out set was evaluated once. Validation and held-out outcomes passed fresh source-matched local replay. All 439 package source hashes matched the preparation manifest after the run.

The two window-boundary and two misleading-note test failures remain policy failures. Undelivered-order cases passed 4/4 in this small held-out sample. This does not establish general robustness. The selected SFT checkpoint has no matched base-model test score, so its 87.5% test score cannot be used to claim a measured held-out gain. Its seed and cases also differ from prior canaries.

The packaged RiverExecutor performed the SFT calls with automatic session restarts disabled for this experiment. It saved and fetched a hosted checkpoint pointer, and the separate native sampling path loaded that checkpoint for evaluation. The final success covers this small end-to-end integration; it does not test live crash recovery, independent provider inventory cleanup, or billing correctness.

The next useful experiment is a larger preregistered multi-seed comparison with more training-only boundary and adversarial-note examples, followed by new held-out cases. Preserve this test attempt. The ordinary comparison gate remains false; no acceptance threshold was weakened.

Training JSONL and the cost ledger are retained with `.txt` suffixes to keep them visible under repository ignore rules; their bytes are unchanged. To recreate the dataset layout, restore `data/train.jsonl.txt` to `data/train.jsonl`. The data manifest retains its original artifact names and hashes. The orchestration program is retained as runner.py.txt. No API credential is stored here; hashes are recorded in artifacts.sha256.json.
