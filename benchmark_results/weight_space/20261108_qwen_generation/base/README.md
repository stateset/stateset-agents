# Whole-model acceptance on pinned Qwen3-0.6B

On one synthetic continuation preference, the small locality-weighted edit has
**82.4% lower mean confirmation KL** than the unweighted edit at nearly identical
target log-probability gain. Independent replay reloads the pretrained model and
recomputes the token metrics, acceptance decisions and exact rollbacks.

This is a small mechanism experiment: it does not establish factual editing,
general quality, reasoning, generation quality or preservation on unseen inputs.

## Results

| Arm | Validation gate | Target gain (nats) | Validation mean KL | Confirmation mean KL | Confirmation maximum KL | Confirmation top-1 agreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Zero update | Rejected | 0 | 0 | 0 | 0 | 48/48 |
| Unweighted, scale 0.5 | Accepted | 0.121740 | 5.60242e-6 | 5.52843e-6 | 2.94560e-5 | 48/48 |
| Weighted, scale 0.5 | Accepted | 0.121741 | 4.40073e-7 | 9.74589e-7 | 2.56850e-5 | 48/48 |
| Weighted, scale 8 | Accepted | 1.930465 | 1.13196e-4 | 2.47674e-4 | 6.48062e-3 | 47/48 |

The larger edit also passes the predeclared validation limits; it is not a
rejected negative control. Its confirmation drift is higher, with one changed
top prediction. The zero update fails both required target-gain thresholds.
All four conditions restore the original weight exactly and leave no active
edits. No adapter is deployed or exported by this experiment.

KL is `base || edited` over the full vocabulary, averaged across all nonpadding
positions, including final tokens. Validation contains 43 such positions and 37
observed next tokens; confirmation contains 48 and 42, respectively. NLL uses
only observed next tokens. Small NLL decreases in these sentences are not evidence
of general quality improvement. Full metrics are in `trials.json`.

## Frozen design

The model is `Qwen/Qwen3-0.6B`, revision
`c1899de289a04d12100db370d81485cdf75e47ca`, loaded from local safetensors on CPU in
float32 with eager attention and evaluation mode. `protocol.json` records hashes
of the checkpoint, model/tokenizer configuration, tokenizer and executed sources,
plus runtime versions. There are no optimizer steps or model downloads.

The sole target is the continuation ` amber` after `The secret word is`.
Prompt and continuation are tokenized separately, then concatenated for
teacher-forced scoring. The edit changes `model.layers.27.mlp.down_proj` using
the target prompt's final input activation. Its output direction is the first
target token's output embedding multiplied by the final normalization gain and
unit normalized. The desired layer output adds that direction at scale 0.5 or 8.

All tokens from two protected sentences supply hard linear preservation
constraints. All tokens from six separate calibration sentences supply the
optional locality penalty with weight 100. The unweighted arm uses weight zero.
Six validation and six distinct confirmation sentences are fixed in the protocol;
neither set supplies solver activations. The exact zero update is constructed
directly rather than numerically solved. Solver tolerances are target absolute
`0.002`, target relative `1e-5`, and preservation absolute `0.0005`.

Validation limits, declared before forwards, are mean KL at most `0.001`, maximum
token KL at most `0.01`, NLL increase at most `0.005`, top-1 agreement at least
`0.95`, and mean/per-case target gain at least `0.01` nats. Confirmation metrics
are collected for every candidate under permissive diagnostic limits; the
`confirmation.accepted` field is not a promotion decision. Confirmation data
does not choose an arm, strength, checkpoint or threshold.

The runner spends 19 measurement units: three activation captures and sixteen
baseline/candidate snapshots. With batch size one this is 121 model forwards.
Independent replay adds its own model forwards outside that session budget.

## Independent replay

The verifier imports neither the session nor the solver nor the runner. It
checks hashes, protocol conditions, tokenization and split separation, reloads
the pinned model, and independently recaptures the saved activation inputs. It
checks the output direction and layer targets, applies saved factors directly,
and recomputes full-vocabulary metrics with float64 normalization one position
at a time. It recomputes acceptance and checks exact rollback and budget records.
`verification.json` records the successful original replay.

For a new complete reproduction, provide a local snapshot directory whose name
is the pinned revision, and use a fresh output directory:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m \
  benchmarks.weight_space.behavior_trial_job \
  --model /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --out /tmp/qwen-behavior-reproduction
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m \
  benchmarks.weight_space.verify_behavior_trial /tmp/qwen-behavior-reproduction \
  --model /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
```

JSON evidence and frozen executed sources are committed. `edits.safetensors`
is retained locally as an ignored file; the runner regenerates it. Model weights
are not copied into this repository. Runtime input-validation guards added after
the run do not alter its numerical path; `sources/` preserves the executed code.
These results do not establish equivalence for other precision modes, devices,
inference kernels, merged adapters or multiple sequential edits.
