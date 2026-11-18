python - "$out" <<'PY'
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from stateset_agents.experimental.weight_space.checkpoints import (
    apply_spec_to_checkpoint, verify_checkpoint,
)
from stateset_agents.experimental.weight_space.edits import EditSpec, TokenRowScale

torch.manual_seed(42)
config = Qwen2Config(vocab_size=16, hidden_size=8, intermediate_size=16,
                     num_hidden_layers=1, num_attention_heads=2,
                     num_key_value_heads=1, tie_word_embeddings=True)
model = Qwen2ForCausalLM(config)
with torch.no_grad():
    model.get_input_embeddings().weight[1].copy_(torch.arange(1, 9))
before = model.get_input_embeddings().weight[1].detach().clone()
spec = EditSpec(name="halve-token-one", edits=(TokenRowScale(ids=(1,), scale=0.5),))
with TemporaryDirectory() as scratch:
    base, edited = Path(scratch, "base"), Path(scratch, "edited")
    model.save_pretrained(base)
    manifest = apply_spec_to_checkpoint(spec, base, edited)
    verify_checkpoint(spec, base, edited, tolerance=0.0)
    reloaded = Qwen2ForCausalLM.from_pretrained(edited)
    after = reloaded.get_input_embeddings().weight[1].detach()
    if not torch.equal(after, before * 0.5):
        raise RuntimeError("Reloaded token row differs from the prescribed edit")

result = {"spec": spec.to_dict(), "spec_hash": manifest["spec_hash"],
          "changed_weights": manifest["changed_weights"], "before": before.tolist(),
          "after": after.tolist(), "reload_verified": True}
Path(sys.argv[1], "result.json").write_text(json.dumps(result, indent=2) + "\n")
PY
