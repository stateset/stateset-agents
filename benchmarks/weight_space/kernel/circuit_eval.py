# Kernel code — EVALUATE (1 forward pass): current edits on 200 practice problems.
import json

rep = evaluate(n=200)
print(json.dumps({k: v for k, v in rep.items() if k != "examples"}))
for ex in rep["examples"]:
    print("--", ex["kind"], ex["tokens"], "tokens | tail:", repr(ex["tail"][-220:]))
print(json.dumps(budget()))
