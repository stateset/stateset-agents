# Interactive weights-kernel programs

These files are sent, in order, to the persistent Python kernel started by
`../interactive.py` (served on the pod by `../ws_server.py`) with
`../ws_call.py FILE`. They run inside the kernel's namespace, so they use its
globals (`capture`, `apply_edit`, `revert`, `edits`, `weight`,
`scale_token_rows`, `clear_token_rows`, `evaluate`, `budget`, `MODEL_ID`) and
the state earlier programs leave behind (`outs`, `CIRCUIT`, `WRITE_DIR`,
`NC`, `NC_META`). They are not importable modules.

Templates contain placeholders that are substituted before sending
(`sed -e 's/__LAYER__/23/' -e 's/__BETA__/1.0/' circuit_write.py`).

| Order | File | Budget | What it does |
|---|---|---|---|
| 1 | `circuit_read.py` | 1 forward pass | Captures layer 16–23 MLP gate/up pre-activations at 445 prefixes (answer just computed / still working / after `</think>`) and fits 16-neuron "answer computed" detectors on half; reports held-out AUC. |
| 2 | `circuit_scale.py` | free | Measures layer MLP write norms and the detector's projection by context to calibrate a write. |
| 3 | `circuit_write.py` (`__LAYER__`, `__BETA__`) | free | Linear rank-1 circuit: `down_proj[L] += beta * B (x) A` (A = detector, B = unit `</think>` logit direction), on top of Wait x0.5. |
| 4 | `nc_analyze.py` | free | Designs a thresholded neuron from the stored captures: ridge detector over gate pre-activations, a near-constant up neuron as multiplier, and the least-used neuron to repurpose. |
| 5 | `nc_write.py` (`__WRITE__`, `__THINK__`) | free | Rewrites that neuron's gate row, up row and down column; reads the rows back to confirm. |
| — | `circuit_eval.py` | 1 forward pass | Samples the current edited model on the 200 practice problems. |

Results are in `benchmark_results/weight_space/20261102/`.
