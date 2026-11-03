"""Send a Python file to the weights kernel (or submit). Usage:
ws_call.py code.py          # run code in the persistent kernel
ws_call.py --submit "note"  # freeze edits, start held-out evaluation
ws_call.py --status
"""

import json
import sys

import requests

BASE = "http://127.0.0.1:18765"
if sys.argv[1] == "--status":
    print(json.dumps(requests.get(BASE, timeout=20).json()))
elif sys.argv[1] == "--submit":
    print(
        json.dumps(
            requests.post(
                f"{BASE}/submit", json={"note": sys.argv[2]}, timeout=120
            ).json(),
            indent=1,
        )
    )
else:
    r = requests.post(
        f"{BASE}/python", json={"code": open(sys.argv[1]).read()}, timeout=3600
    ).json()
    print(r.get("stdout", ""))
    if r.get("error"):
        print("[error]\n" + r["error"])
    print(f"[forward passes remaining: {r.get('budget_remaining')}]")
