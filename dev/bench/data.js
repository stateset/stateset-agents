window.BENCHMARK_DATA = {
  "lastUpdate": 1787062695935,
  "repoUrl": "https://github.com/stateset/stateset-agents",
  "entries": {
    "Python Benchmark (nightly)": [
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "352c052cfcf49a97c763fa48f97ed5d2468878c0",
          "message": "polish(ingest): fix docstring, drop dead constant, warn about single-file grading semantics\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-28T17:34:18Z",
          "url": "https://github.com/stateset/stateset-agents/commit/352c052cfcf49a97c763fa48f97ed5d2468878c0"
        },
        "date": 1785261112997,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8602.089474046952,
            "unit": "iter/sec",
            "range": "stddev: 0.000015346275758086272",
            "extra": "mean: 116.25082522299532 usec\nrounds: 1459"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9154.697388042141,
            "unit": "iter/sec",
            "range": "stddev: 0.00001539691858107769",
            "extra": "mean: 109.23353963684252 usec\nrounds: 2094"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6747.217589772149,
            "unit": "iter/sec",
            "range": "stddev: 0.000015841874085770306",
            "extra": "mean: 148.20924131983858 usec\nrounds: 3456"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 828.0605845747955,
            "unit": "iter/sec",
            "range": "stddev: 0.00002350317526443183",
            "extra": "mean: 1.207641105769446 msec\nrounds: 728"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 160.2610250350036,
            "unit": "iter/sec",
            "range": "stddev: 0.006566298067531278",
            "extra": "mean: 6.239820316771241 msec\nrounds: 161"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2277826.287690625,
            "unit": "iter/sec",
            "range": "stddev: 6.797043664226711e-8",
            "extra": "mean: 439.015040525259 nsec\nrounds: 57124"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "3af189a6aac8af3967a1a145a6682079a15de1d6",
          "message": "Merge feat/launch-kit: five-minute offline demo script + Colab notebook\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-29T00:07:48Z",
          "url": "https://github.com/stateset/stateset-agents/commit/3af189a6aac8af3967a1a145a6682079a15de1d6"
        },
        "date": 1785318277583,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5979.664854080388,
            "unit": "iter/sec",
            "range": "stddev: 0.00001798466315662586",
            "extra": "mean: 167.23345277747845 usec\nrounds: 1440"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6423.441076542588,
            "unit": "iter/sec",
            "range": "stddev: 0.000018616818281438017",
            "extra": "mean: 155.6797965582412 usec\nrounds: 1976"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4963.110642975276,
            "unit": "iter/sec",
            "range": "stddev: 0.000018820384409604045",
            "extra": "mean: 201.48654179519195 usec\nrounds: 3577"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 720.4290734243727,
            "unit": "iter/sec",
            "range": "stddev: 0.00023218862985011172",
            "extra": "mean: 1.3880616939107682 msec\nrounds: 624"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 170.4592911961198,
            "unit": "iter/sec",
            "range": "stddev: 0.005564449685531265",
            "extra": "mean: 5.866503333335245 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2124293.7509211847,
            "unit": "iter/sec",
            "range": "stddev: 4.996213259404273e-8",
            "extra": "mean: 470.74468847180725 nsec\nrounds: 103746"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "e04ce40997d30499d48cf562c318e6de7568ac3c",
          "message": "docs: lead README with the improvement loop; condense model starters and What's-new\n\n- New 'The improvement loop' section up top: ingest -> improve -> train,\n  the five-minute offline demo, and MCP registration (every command verified\n  end-to-end, including a corrected 'ingest --input' invocation).\n- Collapsed six near-identical model starter sections into one table plus the\n  unified finetune driver (~100 lines removed).\n- Trimmed What's-new from six release blocks to three and fixed two blocks\n  both claiming 'latest release' (release-sed damage).\n- New guard test asserting exactly one latest-release marker, matching\n  pyproject's version, so the bump can't mangle it again.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-29T16:55:20Z",
          "url": "https://github.com/stateset/stateset-agents/commit/e04ce40997d30499d48cf562c318e6de7568ac3c"
        },
        "date": 1785404171799,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8551.784151359052,
            "unit": "iter/sec",
            "range": "stddev: 0.0000149631883157013",
            "extra": "mean: 116.9346632586698 usec\nrounds: 1369"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9145.33328868003,
            "unit": "iter/sec",
            "range": "stddev: 0.000014828026730397644",
            "extra": "mean: 109.34538615861997 usec\nrounds: 1994"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6656.749901168978,
            "unit": "iter/sec",
            "range": "stddev: 0.0000157818927227652",
            "extra": "mean: 150.22345962320023 usec\nrounds: 2972"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 828.3621598673008,
            "unit": "iter/sec",
            "range": "stddev: 0.00003191910166108091",
            "extra": "mean: 1.2072014493759526 msec\nrounds: 721"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 161.79019569117554,
            "unit": "iter/sec",
            "range": "stddev: 0.006106814665457789",
            "extra": "mean: 6.1808442454003565 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2259285.905973281,
            "unit": "iter/sec",
            "range": "stddev: 1.0570662033027616e-7",
            "extra": "mean: 442.61773038822577 nsec\nrounds: 107829"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "e04ce40997d30499d48cf562c318e6de7568ac3c",
          "message": "docs: lead README with the improvement loop; condense model starters and What's-new\n\n- New 'The improvement loop' section up top: ingest -> improve -> train,\n  the five-minute offline demo, and MCP registration (every command verified\n  end-to-end, including a corrected 'ingest --input' invocation).\n- Collapsed six near-identical model starter sections into one table plus the\n  unified finetune driver (~100 lines removed).\n- Trimmed What's-new from six release blocks to three and fixed two blocks\n  both claiming 'latest release' (release-sed damage).\n- New guard test asserting exactly one latest-release marker, matching\n  pyproject's version, so the bump can't mangle it again.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-29T16:55:20Z",
          "url": "https://github.com/stateset/stateset-agents/commit/e04ce40997d30499d48cf562c318e6de7568ac3c"
        },
        "date": 1785491512489,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5939.7649647121525,
            "unit": "iter/sec",
            "range": "stddev: 0.00001748245832358665",
            "extra": "mean: 168.35682993198387 usec\nrounds: 1470"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6405.647034822689,
            "unit": "iter/sec",
            "range": "stddev: 0.000017647599700190964",
            "extra": "mean: 156.1122544785486 usec\nrounds: 1898"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4909.012260960543,
            "unit": "iter/sec",
            "range": "stddev: 0.000030276592304712468",
            "extra": "mean: 203.70696727580193 usec\nrounds: 3392"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 733.878804630331,
            "unit": "iter/sec",
            "range": "stddev: 0.00003164589508499289",
            "extra": "mean: 1.3626228114105563 msec\nrounds: 631"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.6862385810694,
            "unit": "iter/sec",
            "range": "stddev: 0.00007372881384979483",
            "extra": "mean: 5.534455793938757 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2110429.5399198644,
            "unit": "iter/sec",
            "range": "stddev: 5.052715447266533e-8",
            "extra": "mean: 473.83718863126376 nsec\nrounds: 105742"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "f53e9d8a0c73088e95b0da4c987a853b4ac96014",
          "message": "fix(ci): make train-remote CLI tests width-proof; unbreak lock-check\n\nTwo separate CI failures on master, one mine and one not.\n\nMine: test_cli_remote asserted \"--provider\" appears in `--help` output.\nOn the Windows 3.13 runner rich renders into a narrow terminal and\ntruncates the flag, so the test failed for reasons unrelated to the CLI.\nReproduced locally at COLUMNS=40. test_cli_improve already documented this\nexact trap in _help_flags and I did not apply the lesson. Registration is\nnow asserted against the parser itself (rendering-independent) and the\nremaining help-text test runs under a pinned wide, plain terminal.\n\nNot mine: publish-readiness broke because `pip install --upgrade pip` now\ninstalls pip 26, which removed the private `pip._internal.utils.compat.\nstdlib_pkgs` that pip-tools imports — so `make lock-check` dies before it\nchecks anything. Verified in a clean venv that pip-tools 7.5.3 *and* the\nnewer 7.6.0 both fail on pip 26.2 and both work on pip 25.3, so upgrading\npip-tools is not the fix; pip is capped for that step only, with the\nfinding recorded inline.\n\nAlso confirmed the lock files themselves are unaffected by this release:\nextras never propagate into them (vllm/optuna/deepspeed are absent; the\nlone `mcp` entry is `via semgrep`), so the new `remote`/`modal` extras\nrequire no lock regeneration.\n\nThe Benchmark workflow has failed on every run since at least 2026-07-28\n(`pytest benchmarks/` exits 5 — that directory holds scripts, not tests).\nPre-existing and untouched here.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-31T23:16:22Z",
          "url": "https://github.com/stateset/stateset-agents/commit/f53e9d8a0c73088e95b0da4c987a853b4ac96014"
        },
        "date": 1785575581255,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14167.590861289242,
            "unit": "iter/sec",
            "range": "stddev: 0.000008326181460825144",
            "extra": "mean: 70.58363061092805 usec\nrounds: 1849"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 15021.42857835728,
            "unit": "iter/sec",
            "range": "stddev: 0.000006914422303378233",
            "extra": "mean: 66.57156440105769 usec\nrounds: 2663"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11031.129715568055,
            "unit": "iter/sec",
            "range": "stddev: 0.000008315569573008569",
            "extra": "mean: 90.65254654640822 usec\nrounds: 3996"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1311.3283625179838,
            "unit": "iter/sec",
            "range": "stddev: 0.000013107660674266643",
            "extra": "mean: 762.5855038167725 usec\nrounds: 1048"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 326.26097523933475,
            "unit": "iter/sec",
            "range": "stddev: 0.0003543639924381293",
            "extra": "mean: 3.065030990195599 msec\nrounds: 306"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3635163.597820326,
            "unit": "iter/sec",
            "range": "stddev: 2.9790337764699448e-8",
            "extra": "mean: 275.0907828741486 nsec\nrounds: 192827"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "322521228c5e643bc6b6404b9e86466aa0fa89bb",
          "message": "fix(remote,sft): four bugs found by running RunPod on real hardware\n\nThe RunPod provider now works end-to-end, verified on live hardware: RTX\nA4000, Qwen/Qwen3.5-0.8B, LoRA r=8, 342s wall clock, returning a 12.8 MB\nadapter (192 tensors) to local disk with the pod terminated afterwards and\nthe account back to $0/hr.\n\nIt took four runs. Every failure was real and none was reachable without\nlive hardware:\n\n1. Default image `runpod/pytorch:2.4.0` ships torch 2.4, but\n   transformers>=4.57.1 needs DTensor from torch.distributed.tensor (2.6+).\n   The pod provisions and the job starts before dying, so only a real run\n   sees it. Default is now torch 2.8, guarded by a test that parses the\n   torch version out of the image tag.\n\n2. PRE-EXISTING: run_sft built LoraConfig without target_modules, relying\n   on peft's architecture inference — which only covers models in peft's\n   built-in mapping. Qwen3.5 is not one, so the job died with \"Please\n   specify `target_modules`\". This affected scripts/sft_from_curated.py\n   just as much; the CPU dry-run path exits before loading a model, which\n   is why no existing test caught it. New infer_lora_target_modules()\n   inspects the loaded model and selects the projection layers actually\n   present (separate q/k/v/o, fused c_attn, MLP), excluding lm_head.\n\n3. download_dir used the `remote:/path/.` form, which OpenSSH 9 rejects —\n   it runs scp over SFTP (\"unexpected filename: .\"). Now fetches the\n   directory into a staging dir and moves contents up.\n\n4. A download failure raised, discarding the job's logs. Training had\n   actually succeeded on the pod and the user would have seen a stack\n   trace and no evidence. Download failures are now reported as FAILED\n   with logs intact, matching the existing no-artifacts handling.\n\n24 RunPod tests, all red-first. Pods were terminated on all four runs,\nconfirmed by a sweep that queries the account independently of executor\nlogic.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-01T22:47:39Z",
          "url": "https://github.com/stateset/stateset-agents/commit/322521228c5e643bc6b6404b9e86466aa0fa89bb"
        },
        "date": 1785662149321,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6007.06540109579,
            "unit": "iter/sec",
            "range": "stddev: 0.000016904561738902345",
            "extra": "mean: 166.4706363639029 usec\nrounds: 1540"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6474.792855908592,
            "unit": "iter/sec",
            "range": "stddev: 0.00001431004278127096",
            "extra": "mean: 154.44509534964456 usec\nrounds: 2129"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4978.083080652218,
            "unit": "iter/sec",
            "range": "stddev: 0.00001631246654684635",
            "extra": "mean: 200.8805365034169 usec\nrounds: 3575"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 736.5216203845098,
            "unit": "iter/sec",
            "range": "stddev: 0.000025778736139993556",
            "extra": "mean: 1.3577333948159434 msec\nrounds: 656"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.9594062002619,
            "unit": "iter/sec",
            "range": "stddev: 0.000039178041502482206",
            "extra": "mean: 5.495731278103944 msec\nrounds: 169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2126876.213702301,
            "unit": "iter/sec",
            "range": "stddev: 5.01442763301862e-8",
            "extra": "mean: 470.17310812803606 nsec\nrounds: 100919"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "322521228c5e643bc6b6404b9e86466aa0fa89bb",
          "message": "fix(remote,sft): four bugs found by running RunPod on real hardware\n\nThe RunPod provider now works end-to-end, verified on live hardware: RTX\nA4000, Qwen/Qwen3.5-0.8B, LoRA r=8, 342s wall clock, returning a 12.8 MB\nadapter (192 tensors) to local disk with the pod terminated afterwards and\nthe account back to $0/hr.\n\nIt took four runs. Every failure was real and none was reachable without\nlive hardware:\n\n1. Default image `runpod/pytorch:2.4.0` ships torch 2.4, but\n   transformers>=4.57.1 needs DTensor from torch.distributed.tensor (2.6+).\n   The pod provisions and the job starts before dying, so only a real run\n   sees it. Default is now torch 2.8, guarded by a test that parses the\n   torch version out of the image tag.\n\n2. PRE-EXISTING: run_sft built LoraConfig without target_modules, relying\n   on peft's architecture inference — which only covers models in peft's\n   built-in mapping. Qwen3.5 is not one, so the job died with \"Please\n   specify `target_modules`\". This affected scripts/sft_from_curated.py\n   just as much; the CPU dry-run path exits before loading a model, which\n   is why no existing test caught it. New infer_lora_target_modules()\n   inspects the loaded model and selects the projection layers actually\n   present (separate q/k/v/o, fused c_attn, MLP), excluding lm_head.\n\n3. download_dir used the `remote:/path/.` form, which OpenSSH 9 rejects —\n   it runs scp over SFTP (\"unexpected filename: .\"). Now fetches the\n   directory into a staging dir and moves contents up.\n\n4. A download failure raised, discarding the job's logs. Training had\n   actually succeeded on the pod and the user would have seen a stack\n   trace and no evidence. Download failures are now reported as FAILED\n   with logs intact, matching the existing no-artifacts handling.\n\n24 RunPod tests, all red-first. Pods were terminated on all four runs,\nconfirmed by a sweep that queries the account independently of executor\nlogic.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-01T22:47:39Z",
          "url": "https://github.com/stateset/stateset-agents/commit/322521228c5e643bc6b6404b9e86466aa0fa89bb"
        },
        "date": 1785754284554,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6128.291842197551,
            "unit": "iter/sec",
            "range": "stddev: 0.000015337806593483403",
            "extra": "mean: 163.1776073577803 usec\nrounds: 1495"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6547.342174333864,
            "unit": "iter/sec",
            "range": "stddev: 0.000015621902020043756",
            "extra": "mean: 152.7337312413707 usec\nrounds: 2199"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5012.2516164365925,
            "unit": "iter/sec",
            "range": "stddev: 0.0000184598680989007",
            "extra": "mean: 199.51113322418146 usec\nrounds: 3663"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 746.7877964960335,
            "unit": "iter/sec",
            "range": "stddev: 0.000029358264312072018",
            "extra": "mean: 1.3390684806206676 msec\nrounds: 645"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 185.66174482029143,
            "unit": "iter/sec",
            "range": "stddev: 0.00005072260463215895",
            "extra": "mean: 5.386139190752167 msec\nrounds: 173"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2099674.695948021,
            "unit": "iter/sec",
            "range": "stddev: 5.7018744974594045e-8",
            "extra": "mean: 476.2642527101045 nsec\nrounds: 105843"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "322521228c5e643bc6b6404b9e86466aa0fa89bb",
          "message": "fix(remote,sft): four bugs found by running RunPod on real hardware\n\nThe RunPod provider now works end-to-end, verified on live hardware: RTX\nA4000, Qwen/Qwen3.5-0.8B, LoRA r=8, 342s wall clock, returning a 12.8 MB\nadapter (192 tensors) to local disk with the pod terminated afterwards and\nthe account back to $0/hr.\n\nIt took four runs. Every failure was real and none was reachable without\nlive hardware:\n\n1. Default image `runpod/pytorch:2.4.0` ships torch 2.4, but\n   transformers>=4.57.1 needs DTensor from torch.distributed.tensor (2.6+).\n   The pod provisions and the job starts before dying, so only a real run\n   sees it. Default is now torch 2.8, guarded by a test that parses the\n   torch version out of the image tag.\n\n2. PRE-EXISTING: run_sft built LoraConfig without target_modules, relying\n   on peft's architecture inference — which only covers models in peft's\n   built-in mapping. Qwen3.5 is not one, so the job died with \"Please\n   specify `target_modules`\". This affected scripts/sft_from_curated.py\n   just as much; the CPU dry-run path exits before loading a model, which\n   is why no existing test caught it. New infer_lora_target_modules()\n   inspects the loaded model and selects the projection layers actually\n   present (separate q/k/v/o, fused c_attn, MLP), excluding lm_head.\n\n3. download_dir used the `remote:/path/.` form, which OpenSSH 9 rejects —\n   it runs scp over SFTP (\"unexpected filename: .\"). Now fetches the\n   directory into a staging dir and moves contents up.\n\n4. A download failure raised, discarding the job's logs. Training had\n   actually succeeded on the pod and the user would have seen a stack\n   trace and no evidence. Download failures are now reported as FAILED\n   with logs intact, matching the existing no-artifacts handling.\n\n24 RunPod tests, all red-first. Pods were terminated on all four runs,\nconfirmed by a sweep that queries the account independently of executor\nlogic.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-01T22:47:39Z",
          "url": "https://github.com/stateset/stateset-agents/commit/322521228c5e643bc6b6404b9e86466aa0fa89bb"
        },
        "date": 1785836631477,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8498.748502462338,
            "unit": "iter/sec",
            "range": "stddev: 0.000009540528214997392",
            "extra": "mean: 117.6643831395023 usec\nrounds: 1720"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 8970.111449532347,
            "unit": "iter/sec",
            "range": "stddev: 0.000010246788536675017",
            "extra": "mean: 111.48133505655993 usec\nrounds: 2128"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6331.754105211109,
            "unit": "iter/sec",
            "range": "stddev: 0.000032741872680644264",
            "extra": "mean: 157.93411800009542 usec\nrounds: 3500"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 868.5391342607164,
            "unit": "iter/sec",
            "range": "stddev: 0.00007296420668296557",
            "extra": "mean: 1.1513585980799594 msec\nrounds: 729"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 212.19591501796143,
            "unit": "iter/sec",
            "range": "stddev: 0.00003440263398225547",
            "extra": "mean: 4.712626064999199 msec\nrounds: 200"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2328554.0702283704,
            "unit": "iter/sec",
            "range": "stddev: 3.238068848911155e-8",
            "extra": "mean: 429.4510541049735 nsec\nrounds: 41789"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "322521228c5e643bc6b6404b9e86466aa0fa89bb",
          "message": "fix(remote,sft): four bugs found by running RunPod on real hardware\n\nThe RunPod provider now works end-to-end, verified on live hardware: RTX\nA4000, Qwen/Qwen3.5-0.8B, LoRA r=8, 342s wall clock, returning a 12.8 MB\nadapter (192 tensors) to local disk with the pod terminated afterwards and\nthe account back to $0/hr.\n\nIt took four runs. Every failure was real and none was reachable without\nlive hardware:\n\n1. Default image `runpod/pytorch:2.4.0` ships torch 2.4, but\n   transformers>=4.57.1 needs DTensor from torch.distributed.tensor (2.6+).\n   The pod provisions and the job starts before dying, so only a real run\n   sees it. Default is now torch 2.8, guarded by a test that parses the\n   torch version out of the image tag.\n\n2. PRE-EXISTING: run_sft built LoraConfig without target_modules, relying\n   on peft's architecture inference — which only covers models in peft's\n   built-in mapping. Qwen3.5 is not one, so the job died with \"Please\n   specify `target_modules`\". This affected scripts/sft_from_curated.py\n   just as much; the CPU dry-run path exits before loading a model, which\n   is why no existing test caught it. New infer_lora_target_modules()\n   inspects the loaded model and selects the projection layers actually\n   present (separate q/k/v/o, fused c_attn, MLP), excluding lm_head.\n\n3. download_dir used the `remote:/path/.` form, which OpenSSH 9 rejects —\n   it runs scp over SFTP (\"unexpected filename: .\"). Now fetches the\n   directory into a staging dir and moves contents up.\n\n4. A download failure raised, discarding the job's logs. Training had\n   actually succeeded on the pod and the user would have seen a stack\n   trace and no evidence. Download failures are now reported as FAILED\n   with logs intact, matching the existing no-artifacts handling.\n\n24 RunPod tests, all red-first. Pods were terminated on all four runs,\nconfirmed by a sweep that queries the account independently of executor\nlogic.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-01T22:47:39Z",
          "url": "https://github.com/stateset/stateset-agents/commit/322521228c5e643bc6b6404b9e86466aa0fa89bb"
        },
        "date": 1785922963447,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8464.355804174173,
            "unit": "iter/sec",
            "range": "stddev: 0.000013786375222314521",
            "extra": "mean: 118.14248161766226 usec\nrounds: 1360"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9075.29978232954,
            "unit": "iter/sec",
            "range": "stddev: 0.000013717971811176603",
            "extra": "mean: 110.18919749043373 usec\nrounds: 1833"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6607.62409160394,
            "unit": "iter/sec",
            "range": "stddev: 0.000032392381678068705",
            "extra": "mean: 151.3403284049803 usec\nrounds: 3517"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 824.8858333643926,
            "unit": "iter/sec",
            "range": "stddev: 0.00002550063713192737",
            "extra": "mean: 1.2122889732769246 msec\nrounds: 711"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 164.11148654614848,
            "unit": "iter/sec",
            "range": "stddev: 0.006259667016826447",
            "extra": "mean: 6.093418693875507 msec\nrounds: 147"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2274238.8051392036,
            "unit": "iter/sec",
            "range": "stddev: 3.5356843464536846e-8",
            "extra": "mean: 439.70756181815796 nsec\nrounds: 56415"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "322521228c5e643bc6b6404b9e86466aa0fa89bb",
          "message": "fix(remote,sft): four bugs found by running RunPod on real hardware\n\nThe RunPod provider now works end-to-end, verified on live hardware: RTX\nA4000, Qwen/Qwen3.5-0.8B, LoRA r=8, 342s wall clock, returning a 12.8 MB\nadapter (192 tensors) to local disk with the pod terminated afterwards and\nthe account back to $0/hr.\n\nIt took four runs. Every failure was real and none was reachable without\nlive hardware:\n\n1. Default image `runpod/pytorch:2.4.0` ships torch 2.4, but\n   transformers>=4.57.1 needs DTensor from torch.distributed.tensor (2.6+).\n   The pod provisions and the job starts before dying, so only a real run\n   sees it. Default is now torch 2.8, guarded by a test that parses the\n   torch version out of the image tag.\n\n2. PRE-EXISTING: run_sft built LoraConfig without target_modules, relying\n   on peft's architecture inference — which only covers models in peft's\n   built-in mapping. Qwen3.5 is not one, so the job died with \"Please\n   specify `target_modules`\". This affected scripts/sft_from_curated.py\n   just as much; the CPU dry-run path exits before loading a model, which\n   is why no existing test caught it. New infer_lora_target_modules()\n   inspects the loaded model and selects the projection layers actually\n   present (separate q/k/v/o, fused c_attn, MLP), excluding lm_head.\n\n3. download_dir used the `remote:/path/.` form, which OpenSSH 9 rejects —\n   it runs scp over SFTP (\"unexpected filename: .\"). Now fetches the\n   directory into a staging dir and moves contents up.\n\n4. A download failure raised, discarding the job's logs. Training had\n   actually succeeded on the pod and the user would have seen a stack\n   trace and no evidence. Download failures are now reported as FAILED\n   with logs intact, matching the existing no-artifacts handling.\n\n24 RunPod tests, all red-first. Pods were terminated on all four runs,\nconfirmed by a sweep that queries the account independently of executor\nlogic.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-01T22:47:39Z",
          "url": "https://github.com/stateset/stateset-agents/commit/322521228c5e643bc6b6404b9e86466aa0fa89bb"
        },
        "date": 1786009550623,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8569.459874129385,
            "unit": "iter/sec",
            "range": "stddev: 0.000012985886866852121",
            "extra": "mean: 116.69346898034165 usec\nrounds: 1354"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9126.988093202282,
            "unit": "iter/sec",
            "range": "stddev: 0.000013202476986623711",
            "extra": "mean: 109.565169778713 usec\nrounds: 1767"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6613.4566199880355,
            "unit": "iter/sec",
            "range": "stddev: 0.000015584996293615958",
            "extra": "mean: 151.20685860064037 usec\nrounds: 3430"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 822.402058017704,
            "unit": "iter/sec",
            "range": "stddev: 0.000022310609987917143",
            "extra": "mean: 1.2159502645340812 msec\nrounds: 688"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 161.27647297256195,
            "unit": "iter/sec",
            "range": "stddev: 0.00675326830604951",
            "extra": "mean: 6.20053242465273 msec\nrounds: 146"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2264553.469698955,
            "unit": "iter/sec",
            "range": "stddev: 7.955715125277237e-8",
            "extra": "mean: 441.58816004151936 nsec\nrounds: 106225"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "322521228c5e643bc6b6404b9e86466aa0fa89bb",
          "message": "fix(remote,sft): four bugs found by running RunPod on real hardware\n\nThe RunPod provider now works end-to-end, verified on live hardware: RTX\nA4000, Qwen/Qwen3.5-0.8B, LoRA r=8, 342s wall clock, returning a 12.8 MB\nadapter (192 tensors) to local disk with the pod terminated afterwards and\nthe account back to $0/hr.\n\nIt took four runs. Every failure was real and none was reachable without\nlive hardware:\n\n1. Default image `runpod/pytorch:2.4.0` ships torch 2.4, but\n   transformers>=4.57.1 needs DTensor from torch.distributed.tensor (2.6+).\n   The pod provisions and the job starts before dying, so only a real run\n   sees it. Default is now torch 2.8, guarded by a test that parses the\n   torch version out of the image tag.\n\n2. PRE-EXISTING: run_sft built LoraConfig without target_modules, relying\n   on peft's architecture inference — which only covers models in peft's\n   built-in mapping. Qwen3.5 is not one, so the job died with \"Please\n   specify `target_modules`\". This affected scripts/sft_from_curated.py\n   just as much; the CPU dry-run path exits before loading a model, which\n   is why no existing test caught it. New infer_lora_target_modules()\n   inspects the loaded model and selects the projection layers actually\n   present (separate q/k/v/o, fused c_attn, MLP), excluding lm_head.\n\n3. download_dir used the `remote:/path/.` form, which OpenSSH 9 rejects —\n   it runs scp over SFTP (\"unexpected filename: .\"). Now fetches the\n   directory into a staging dir and moves contents up.\n\n4. A download failure raised, discarding the job's logs. Training had\n   actually succeeded on the pod and the user would have seen a stack\n   trace and no evidence. Download failures are now reported as FAILED\n   with logs intact, matching the existing no-artifacts handling.\n\n24 RunPod tests, all red-first. Pods were terminated on all four runs,\nconfirmed by a sweep that queries the account independently of executor\nlogic.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-01T22:47:39Z",
          "url": "https://github.com/stateset/stateset-agents/commit/322521228c5e643bc6b6404b9e86466aa0fa89bb"
        },
        "date": 1786090848364,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 9812.207857774632,
            "unit": "iter/sec",
            "range": "stddev: 0.000010657944175330225",
            "extra": "mean: 101.91386225146638 usec\nrounds: 1510"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 10340.936916028866,
            "unit": "iter/sec",
            "range": "stddev: 0.000011008666917704444",
            "extra": "mean: 96.7030364966215 usec\nrounds: 2055"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 7738.8853882374515,
            "unit": "iter/sec",
            "range": "stddev: 0.000011133689225814786",
            "extra": "mean: 129.2175746031758 usec\nrounds: 3465"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 925.2011360830787,
            "unit": "iter/sec",
            "range": "stddev: 0.000024354245967103704",
            "extra": "mean: 1.080846057143411 msec\nrounds: 805"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 216.0257216696136,
            "unit": "iter/sec",
            "range": "stddev: 0.0038770042144824018",
            "extra": "mean: 4.6290783906250965 msec\nrounds: 192"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2490704.5594897587,
            "unit": "iter/sec",
            "range": "stddev: 3.9332618547279385e-8",
            "extra": "mean: 401.4928210533562 nsec\nrounds: 117371"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "322521228c5e643bc6b6404b9e86466aa0fa89bb",
          "message": "fix(remote,sft): four bugs found by running RunPod on real hardware\n\nThe RunPod provider now works end-to-end, verified on live hardware: RTX\nA4000, Qwen/Qwen3.5-0.8B, LoRA r=8, 342s wall clock, returning a 12.8 MB\nadapter (192 tensors) to local disk with the pod terminated afterwards and\nthe account back to $0/hr.\n\nIt took four runs. Every failure was real and none was reachable without\nlive hardware:\n\n1. Default image `runpod/pytorch:2.4.0` ships torch 2.4, but\n   transformers>=4.57.1 needs DTensor from torch.distributed.tensor (2.6+).\n   The pod provisions and the job starts before dying, so only a real run\n   sees it. Default is now torch 2.8, guarded by a test that parses the\n   torch version out of the image tag.\n\n2. PRE-EXISTING: run_sft built LoraConfig without target_modules, relying\n   on peft's architecture inference — which only covers models in peft's\n   built-in mapping. Qwen3.5 is not one, so the job died with \"Please\n   specify `target_modules`\". This affected scripts/sft_from_curated.py\n   just as much; the CPU dry-run path exits before loading a model, which\n   is why no existing test caught it. New infer_lora_target_modules()\n   inspects the loaded model and selects the projection layers actually\n   present (separate q/k/v/o, fused c_attn, MLP), excluding lm_head.\n\n3. download_dir used the `remote:/path/.` form, which OpenSSH 9 rejects —\n   it runs scp over SFTP (\"unexpected filename: .\"). Now fetches the\n   directory into a staging dir and moves contents up.\n\n4. A download failure raised, discarding the job's logs. Training had\n   actually succeeded on the pod and the user would have seen a stack\n   trace and no evidence. Download failures are now reported as FAILED\n   with logs intact, matching the existing no-artifacts handling.\n\n24 RunPod tests, all red-first. Pods were terminated on all four runs,\nconfirmed by a sweep that queries the account independently of executor\nlogic.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-01T22:47:39Z",
          "url": "https://github.com/stateset/stateset-agents/commit/322521228c5e643bc6b6404b9e86466aa0fa89bb"
        },
        "date": 1786175702190,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8581.846030890734,
            "unit": "iter/sec",
            "range": "stddev: 0.000013047270712564296",
            "extra": "mean: 116.52504559047736 usec\nrounds: 1338"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9166.89786639447,
            "unit": "iter/sec",
            "range": "stddev: 0.000013911315097763729",
            "extra": "mean: 109.08815769246925 usec\nrounds: 2080"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6608.89070203696,
            "unit": "iter/sec",
            "range": "stddev: 0.00001573928229153704",
            "extra": "mean: 151.31132365251327 usec\nrounds: 3340"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 830.7275480980954,
            "unit": "iter/sec",
            "range": "stddev: 0.000022517495789134754",
            "extra": "mean: 1.2037641008648916 msec\nrounds: 694"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 165.03423838265718,
            "unit": "iter/sec",
            "range": "stddev: 0.005775554264639499",
            "extra": "mean: 6.059348713333937 msec\nrounds: 150"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2279753.8159334767,
            "unit": "iter/sec",
            "range": "stddev: 4.587659061224584e-8",
            "extra": "mean: 438.64385400339205 nsec\nrounds: 100960"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "322521228c5e643bc6b6404b9e86466aa0fa89bb",
          "message": "fix(remote,sft): four bugs found by running RunPod on real hardware\n\nThe RunPod provider now works end-to-end, verified on live hardware: RTX\nA4000, Qwen/Qwen3.5-0.8B, LoRA r=8, 342s wall clock, returning a 12.8 MB\nadapter (192 tensors) to local disk with the pod terminated afterwards and\nthe account back to $0/hr.\n\nIt took four runs. Every failure was real and none was reachable without\nlive hardware:\n\n1. Default image `runpod/pytorch:2.4.0` ships torch 2.4, but\n   transformers>=4.57.1 needs DTensor from torch.distributed.tensor (2.6+).\n   The pod provisions and the job starts before dying, so only a real run\n   sees it. Default is now torch 2.8, guarded by a test that parses the\n   torch version out of the image tag.\n\n2. PRE-EXISTING: run_sft built LoraConfig without target_modules, relying\n   on peft's architecture inference — which only covers models in peft's\n   built-in mapping. Qwen3.5 is not one, so the job died with \"Please\n   specify `target_modules`\". This affected scripts/sft_from_curated.py\n   just as much; the CPU dry-run path exits before loading a model, which\n   is why no existing test caught it. New infer_lora_target_modules()\n   inspects the loaded model and selects the projection layers actually\n   present (separate q/k/v/o, fused c_attn, MLP), excluding lm_head.\n\n3. download_dir used the `remote:/path/.` form, which OpenSSH 9 rejects —\n   it runs scp over SFTP (\"unexpected filename: .\"). Now fetches the\n   directory into a staging dir and moves contents up.\n\n4. A download failure raised, discarding the job's logs. Training had\n   actually succeeded on the pod and the user would have seen a stack\n   trace and no evidence. Download failures are now reported as FAILED\n   with logs intact, matching the existing no-artifacts handling.\n\n24 RunPod tests, all red-first. Pods were terminated on all four runs,\nconfirmed by a sweep that queries the account independently of executor\nlogic.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-01T22:47:39Z",
          "url": "https://github.com/stateset/stateset-agents/commit/322521228c5e643bc6b6404b9e86466aa0fa89bb"
        },
        "date": 1786262318895,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6026.197923257911,
            "unit": "iter/sec",
            "range": "stddev: 0.00001642989086882892",
            "extra": "mean: 165.94211022185866 usec\nrounds: 1624"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6477.49745729636,
            "unit": "iter/sec",
            "range": "stddev: 0.00001661032448834669",
            "extra": "mean: 154.38060865212435 usec\nrounds: 1988"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4998.31627886341,
            "unit": "iter/sec",
            "range": "stddev: 0.000017356927109001607",
            "extra": "mean: 200.06737153243827 usec\nrounds: 3028"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 735.0513197058705,
            "unit": "iter/sec",
            "range": "stddev: 0.00005186235651993796",
            "extra": "mean: 1.3604492274092486 msec\nrounds: 664"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 171.89317455842212,
            "unit": "iter/sec",
            "range": "stddev: 0.004467301094920169",
            "extra": "mean: 5.817566651898242 msec\nrounds: 158"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2125464.149151052,
            "unit": "iter/sec",
            "range": "stddev: 4.99944305114263e-8",
            "extra": "mean: 470.4854703851004 nsec\nrounds: 105519"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "322521228c5e643bc6b6404b9e86466aa0fa89bb",
          "message": "fix(remote,sft): four bugs found by running RunPod on real hardware\n\nThe RunPod provider now works end-to-end, verified on live hardware: RTX\nA4000, Qwen/Qwen3.5-0.8B, LoRA r=8, 342s wall clock, returning a 12.8 MB\nadapter (192 tensors) to local disk with the pod terminated afterwards and\nthe account back to $0/hr.\n\nIt took four runs. Every failure was real and none was reachable without\nlive hardware:\n\n1. Default image `runpod/pytorch:2.4.0` ships torch 2.4, but\n   transformers>=4.57.1 needs DTensor from torch.distributed.tensor (2.6+).\n   The pod provisions and the job starts before dying, so only a real run\n   sees it. Default is now torch 2.8, guarded by a test that parses the\n   torch version out of the image tag.\n\n2. PRE-EXISTING: run_sft built LoraConfig without target_modules, relying\n   on peft's architecture inference — which only covers models in peft's\n   built-in mapping. Qwen3.5 is not one, so the job died with \"Please\n   specify `target_modules`\". This affected scripts/sft_from_curated.py\n   just as much; the CPU dry-run path exits before loading a model, which\n   is why no existing test caught it. New infer_lora_target_modules()\n   inspects the loaded model and selects the projection layers actually\n   present (separate q/k/v/o, fused c_attn, MLP), excluding lm_head.\n\n3. download_dir used the `remote:/path/.` form, which OpenSSH 9 rejects —\n   it runs scp over SFTP (\"unexpected filename: .\"). Now fetches the\n   directory into a staging dir and moves contents up.\n\n4. A download failure raised, discarding the job's logs. Training had\n   actually succeeded on the pod and the user would have seen a stack\n   trace and no evidence. Download failures are now reported as FAILED\n   with logs intact, matching the existing no-artifacts handling.\n\n24 RunPod tests, all red-first. Pods were terminated on all four runs,\nconfirmed by a sweep that queries the account independently of executor\nlogic.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-01T22:47:39Z",
          "url": "https://github.com/stateset/stateset-agents/commit/322521228c5e643bc6b6404b9e86466aa0fa89bb"
        },
        "date": 1786351576048,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5930.988682252427,
            "unit": "iter/sec",
            "range": "stddev: 0.000017371741696301907",
            "extra": "mean: 168.60595316803528 usec\nrounds: 1452"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6379.22955344665,
            "unit": "iter/sec",
            "range": "stddev: 0.000016753792722821207",
            "extra": "mean: 156.75874204271383 usec\nrounds: 2105"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4929.898323709762,
            "unit": "iter/sec",
            "range": "stddev: 0.00001699408845188216",
            "extra": "mean: 202.84394004448703 usec\nrounds: 3586"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 732.5903646772448,
            "unit": "iter/sec",
            "range": "stddev: 0.0000634535783240238",
            "extra": "mean: 1.365019318047634 msec\nrounds: 676"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 173.28429874785854,
            "unit": "iter/sec",
            "range": "stddev: 0.00477891182846681",
            "extra": "mean: 5.770863299363746 msec\nrounds: 157"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2109325.8671524203,
            "unit": "iter/sec",
            "range": "stddev: 4.88758204249655e-8",
            "extra": "mean: 474.08511675343703 nsec\nrounds: 104406"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "da35472c296bfda1f4924113a5409d520bb4724e",
          "message": "chore(release): v0.21.0 — Muse Glimmer 30B starter\n\nfeat(training): first-class starter for meta-models/Muse-Glimmer-30B —\nMeta's open agentic model (Aug 2026; dense 30B, 131K+ ctx, Apache-2.0).\nPackaged starter module, muse-glimmer CLI command, init --preset scaffold,\nunified-driver preset, docs page, and unit + integration tests.\n\nVersion 0.21.0 across pyproject, package, helm chart, k8s manifests, docs.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-11T04:10:37Z",
          "url": "https://github.com/stateset/stateset-agents/commit/da35472c296bfda1f4924113a5409d520bb4724e"
        },
        "date": 1786436090790,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8444.337490717655,
            "unit": "iter/sec",
            "range": "stddev: 0.000014752889120938403",
            "extra": "mean: 118.42255252105201 usec\nrounds: 1428"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9213.28057366609,
            "unit": "iter/sec",
            "range": "stddev: 0.00001318995879213812",
            "extra": "mean: 108.53897175977204 usec\nrounds: 1983"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6686.2133552686455,
            "unit": "iter/sec",
            "range": "stddev: 0.00001547495479824493",
            "extra": "mean: 149.5614852331946 usec\nrounds: 3386"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 817.1165375276395,
            "unit": "iter/sec",
            "range": "stddev: 0.0000830392035669359",
            "extra": "mean: 1.2238156420450301 msec\nrounds: 704"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 177.88480683064432,
            "unit": "iter/sec",
            "range": "stddev: 0.00002804101112632436",
            "extra": "mean: 5.621615571430182 msec\nrounds: 14"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2306076.290691803,
            "unit": "iter/sec",
            "range": "stddev: 4.588089048326028e-8",
            "extra": "mean: 433.6369980630643 nsec\nrounds: 107957"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "62de02ccc6d8a8f4d4b626206f5e9d0aeba20085",
          "message": "fix(sft): exclude vision-tower modules from LoRA target inference\n\nOn multimodal composites, ViT-block fc1/fc2 share leaf names with\ndecoder-MLP candidates and were being adapted despite text-only SFT\nsending no gradient through the vision path (observed on the live\nMuse-Glimmer-30B run). Non-text stacks (vision_tower, visual,\nperception_encoder, projectors, audio_tower) are now skipped.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-11T20:42:45Z",
          "url": "https://github.com/stateset/stateset-agents/commit/62de02ccc6d8a8f4d4b626206f5e9d0aeba20085"
        },
        "date": 1786523116549,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6088.369699895318,
            "unit": "iter/sec",
            "range": "stddev: 0.00001569910433515164",
            "extra": "mean: 164.24758174872227 usec\nrounds: 1578"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6550.892282074449,
            "unit": "iter/sec",
            "range": "stddev: 0.000015593880420983654",
            "extra": "mean: 152.65096065407036 usec\nrounds: 1957"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5040.0330302232705,
            "unit": "iter/sec",
            "range": "stddev: 0.000017146960138567605",
            "extra": "mean: 198.41139810063916 usec\nrounds: 3685"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 739.1750371304522,
            "unit": "iter/sec",
            "range": "stddev: 0.000025547284508626667",
            "extra": "mean: 1.3528595390370528 msec\nrounds: 666"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 185.34724129457516,
            "unit": "iter/sec",
            "range": "stddev: 0.00005240733828206711",
            "extra": "mean: 5.395278575582815 msec\nrounds: 172"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2124977.7121019643,
            "unit": "iter/sec",
            "range": "stddev: 4.874766085917368e-8",
            "extra": "mean: 470.593171074171 nsec\nrounds: 103435"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "a13f173b3cd5a5259595d69aa60fa157e76fcff4",
          "message": "feat(training): GSPO GPU verification job, verified live on RunPod\n\nAdd stateset_agents/training/gpu_verify_rl.py — a runnable module\n(python -m stateset_agents.training.gpu_verify_rl) that runs a short\nreal GSPO training on a tiny GPT-2 and asserts the same convergence\nproperty as the nightly CPU e2e test (target-token probability\nstrictly increases), on CUDA when available and CPU otherwise. Prints\na GPU_VERIFY_RL_SUMMARY JSON line and exits 0/1.\n\nAdd an rl-live-smoke job to .github/workflows/gpu-verify.yml mirroring\nthe SFT job's secret gating, built from RunPodApi + SshTransport\nprimitives with unconditional pod termination.\n\nVerified live on a RunPod NVIDIA RTX A4500 (A4000 had no availability):\ntarget-token prob 2.81e-05 -> 0.1246 over 40 GSPO steps, exit 0, pod\nterminated (0 pods remaining).\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-12T19:02:32Z",
          "url": "https://github.com/stateset/stateset-agents/commit/a13f173b3cd5a5259595d69aa60fa157e76fcff4"
        },
        "date": 1786609662360,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10736.607324234417,
            "unit": "iter/sec",
            "range": "stddev: 0.000014822793325107797",
            "extra": "mean: 93.13929156585839 usec\nrounds: 1660"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11717.595773097459,
            "unit": "iter/sec",
            "range": "stddev: 0.000012200597304597664",
            "extra": "mean: 85.34173898504928 usec\nrounds: 2678"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8502.35754332815,
            "unit": "iter/sec",
            "range": "stddev: 0.00001561191624593862",
            "extra": "mean: 117.61443751382885 usec\nrounds: 4489"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1059.89076922398,
            "unit": "iter/sec",
            "range": "stddev: 0.000046240529045246384",
            "extra": "mean: 943.4934514357266 usec\nrounds: 906"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 227.62928584489762,
            "unit": "iter/sec",
            "range": "stddev: 0.0000432501455041544",
            "extra": "mean: 4.3931078388629725 msec\nrounds: 211"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2964360.859560137,
            "unit": "iter/sec",
            "range": "stddev: 3.5375069652360354e-8",
            "extra": "mean: 337.3408459280439 nsec\nrounds: 141844"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "2bbc4cf8f390ca2d8b5f9025db8678d128c5eabc",
          "message": "fix(serve-remote): fail fast on dead pod networking, retry on a fresh host\n\nFour verification attempts died the same way: a pod reaches RUNNING and\nnever publishes an IP or port mappings. That wait shared ready_timeout_s\nwith the vLLM load — necessarily long, since loading a 30B model takes\nmany minutes — so a pod that could never serve anything billed for 30\nminutes before failing.\n\nNetworking is now its own problem with its own short deadline (300s: it\nappears in about two minutes or never), and a pod that misses it is\nterminated and replaced, bounded by max_provision_attempts. Both pods are\nterminated on the way out, so a retry cannot double the bill.\n\nThe existing timeout test encoded the old single-deadline behavior; it now\nasserts the new contract, alongside tests that the vLLM timeout does not\ngovern the networking wait and that a single-attempt session does not retry.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-14T04:43:10Z",
          "url": "https://github.com/stateset/stateset-agents/commit/2bbc4cf8f390ca2d8b5f9025db8678d128c5eabc"
        },
        "date": 1786695795138,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8415.096271974244,
            "unit": "iter/sec",
            "range": "stddev: 0.000013365160634972438",
            "extra": "mean: 118.83405342971704 usec\nrounds: 1385"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 8958.105992689547,
            "unit": "iter/sec",
            "range": "stddev: 0.00004193253078108061",
            "extra": "mean: 111.63073989257006 usec\nrounds: 2053"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6604.574308669752,
            "unit": "iter/sec",
            "range": "stddev: 0.000016141383325716424",
            "extra": "mean: 151.41021256847864 usec\nrounds: 3453"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 817.4561929727877,
            "unit": "iter/sec",
            "range": "stddev: 0.000020531008225271152",
            "extra": "mean: 1.2233071430572537 msec\nrounds: 713"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 174.26222054386596,
            "unit": "iter/sec",
            "range": "stddev: 0.00003930004025189766",
            "extra": "mean: 5.7384784658375 msec\nrounds: 161"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2259984.344823118,
            "unit": "iter/sec",
            "range": "stddev: 1.1383598785313142e-7",
            "extra": "mean: 442.4809412023901 nsec\nrounds: 110327"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "df0ae2582a0fcbbc827bfa076b74693e9e20faaf",
          "message": "feat(remote): River AI provider — tokenization layer behind an injectable client\n\nRiver is a remote autograd service (you drive forward_backward/optim_step),\nso the valuable half of this integration is the pure tokenization layer:\nremote/river_batches.py turns our chat rows into their\n{input_ids, target_tokens, weights} with prompt tokens weighted 0.0 so loss\nlands only on what the model should say, plus the RL batch shape their\nppo/cispo losses take — which is exactly where our trainers' advantages\nwould plug in.\n\nNOT LIVE-VERIFIED, and labelled so everywhere: river-client is not\ninstallable from PyPI and the account has no credits. The client is\ninjectable (92 tests drive it with fakes) and every assumption is isolated\nand documented — notably whether target_tokens carries the causal shift,\nwhich is one function to flip if wrong.\n\nProbing the live API did establish what the docs omit: there is a REST\nsurface, it takes Bearer auth (401 without), and an unfunded account answers\n402 with 'Billing: insufficient_funds'. Both account states now raise named,\nactionable errors instead of a generic training failure, because no amount\nof retrying fixes either.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-14T20:24:26Z",
          "url": "https://github.com/stateset/stateset-agents/commit/df0ae2582a0fcbbc827bfa076b74693e9e20faaf"
        },
        "date": 1786779470201,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5827.797621321807,
            "unit": "iter/sec",
            "range": "stddev: 0.00001998043532471466",
            "extra": "mean: 171.59140810610188 usec\nrounds: 1431"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6308.63871577137,
            "unit": "iter/sec",
            "range": "stddev: 0.00001995821041566325",
            "extra": "mean: 158.51280205665225 usec\nrounds: 1945"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4847.427522002755,
            "unit": "iter/sec",
            "range": "stddev: 0.000021671871274928184",
            "extra": "mean: 206.29498748788754 usec\nrounds: 3117"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 731.2384746758227,
            "unit": "iter/sec",
            "range": "stddev: 0.00003293170086383706",
            "extra": "mean: 1.3675429215391415 msec\nrounds: 650"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 169.9110368208536,
            "unit": "iter/sec",
            "range": "stddev: 0.006090957012732219",
            "extra": "mean: 5.88543286363648 msec\nrounds: 154"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2129182.6145242876,
            "unit": "iter/sec",
            "range": "stddev: 6.083946372408869e-8",
            "extra": "mean: 469.66380111243996 nsec\nrounds: 102691"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "df0ae2582a0fcbbc827bfa076b74693e9e20faaf",
          "message": "feat(remote): River AI provider — tokenization layer behind an injectable client\n\nRiver is a remote autograd service (you drive forward_backward/optim_step),\nso the valuable half of this integration is the pure tokenization layer:\nremote/river_batches.py turns our chat rows into their\n{input_ids, target_tokens, weights} with prompt tokens weighted 0.0 so loss\nlands only on what the model should say, plus the RL batch shape their\nppo/cispo losses take — which is exactly where our trainers' advantages\nwould plug in.\n\nNOT LIVE-VERIFIED, and labelled so everywhere: river-client is not\ninstallable from PyPI and the account has no credits. The client is\ninjectable (92 tests drive it with fakes) and every assumption is isolated\nand documented — notably whether target_tokens carries the causal shift,\nwhich is one function to flip if wrong.\n\nProbing the live API did establish what the docs omit: there is a REST\nsurface, it takes Bearer auth (401 without), and an unfunded account answers\n402 with 'Billing: insufficient_funds'. Both account states now raise named,\nactionable errors instead of a generic training failure, because no amount\nof retrying fixes either.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-14T20:24:26Z",
          "url": "https://github.com/stateset/stateset-agents/commit/df0ae2582a0fcbbc827bfa076b74693e9e20faaf"
        },
        "date": 1786865932188,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8309.781103693633,
            "unit": "iter/sec",
            "range": "stddev: 0.000014708642946234388",
            "extra": "mean: 120.3401133581615 usec\nrounds: 1385"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 8964.623648105942,
            "unit": "iter/sec",
            "range": "stddev: 0.00001550561525289354",
            "extra": "mean: 111.54957968718311 usec\nrounds: 1920"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6559.035898613414,
            "unit": "iter/sec",
            "range": "stddev: 0.00001624112158666071",
            "extra": "mean: 152.4614311398114 usec\nrounds: 3449"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 827.300364583373,
            "unit": "iter/sec",
            "range": "stddev: 0.00002281298136733008",
            "extra": "mean: 1.2087508271600944 msec\nrounds: 729"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 163.01511847659714,
            "unit": "iter/sec",
            "range": "stddev: 0.006350434343823367",
            "extra": "mean: 6.134400350992981 msec\nrounds: 151"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2283618.305368287,
            "unit": "iter/sec",
            "range": "stddev: 3.40098968667713e-8",
            "extra": "mean: 437.90155195779374 nsec\nrounds: 55720"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "ac64d5129f290f29a9e36ffdf013cd7d92f1c739",
          "message": "security: remove a vulnerable mcp pin from the dev lock; fix numpy-stub-sensitive returns\n\nsemgrep pulls the MCP SDK transitively and every release before 1.173.0\nresolves it to mcp 1.23.3, which carries SFTY-20260716-62811 (improper\naccess control, insufficient session validation). It never shipped in the\nwheel — the [mcp] extra users install is correctly pinned >=1.25.0 — so a\nsafety ignore with that justification would have been defensible. Raised the\nsemgrep floor to 1.173.0 instead, which requires mcp==1.29.0: the fix\nremoves the package rather than annotating it, which is the standard this\nrepo already applies to its scanners.\n\nAlso: mappers.py returned Any under CI's numpy stubs after an earlier fix\nremoved a cast that this machine's numpy called redundant. An annotated\nlocal satisfies both stub versions.\n\nBoth failures shared a cause worth naming: local greens are evidence, not\nproof, when dependency versions differ from CI's.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-16T16:17:54Z",
          "url": "https://github.com/stateset/stateset-agents/commit/ac64d5129f290f29a9e36ffdf013cd7d92f1c739"
        },
        "date": 1786953474659,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8424.331500462526,
            "unit": "iter/sec",
            "range": "stddev: 0.00001571506547684156",
            "extra": "mean: 118.7037808216707 usec\nrounds: 1387"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9141.342136945017,
            "unit": "iter/sec",
            "range": "stddev: 0.000013827913733549303",
            "extra": "mean: 109.39312685371102 usec\nrounds: 1821"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6605.729155451634,
            "unit": "iter/sec",
            "range": "stddev: 0.000016622319537841016",
            "extra": "mean: 151.3837422738883 usec\nrounds: 3527"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 826.8798039057873,
            "unit": "iter/sec",
            "range": "stddev: 0.00003427056881150094",
            "extra": "mean: 1.2093656118778995 msec\nrounds: 724"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 176.52948205774615,
            "unit": "iter/sec",
            "range": "stddev: 0.00004435439931403964",
            "extra": "mean: 5.664776151514912 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2274285.006342453,
            "unit": "iter/sec",
            "range": "stddev: 3.575825948845819e-8",
            "extra": "mean: 439.69862933239773 nsec\nrounds: 55411"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "ef2034ec3cc127754ff519e0f6a3a8c67f1cf1d5",
          "message": "fix(types): Any returns in merge loader; narrow the retry re-raise\n\nCaught by CI's mypy --all pass (the targeted local run misses these\nmodules) — the lesson, again: gate with the same strictness CI uses.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-18T06:35:28Z",
          "url": "https://github.com/stateset/stateset-agents/commit/ef2034ec3cc127754ff519e0f6a3a8c67f1cf1d5"
        },
        "date": 1787039117397,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8453.663873178924,
            "unit": "iter/sec",
            "range": "stddev: 0.000015164510378474418",
            "extra": "mean: 118.29190455190867 usec\nrounds: 1362"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9011.251506922936,
            "unit": "iter/sec",
            "range": "stddev: 0.000015081178593415998",
            "extra": "mean: 110.97237705902951 usec\nrounds: 1700"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6597.373791383749,
            "unit": "iter/sec",
            "range": "stddev: 0.00001633777303899373",
            "extra": "mean: 151.5754649685019 usec\nrounds: 3140"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 813.0655489786726,
            "unit": "iter/sec",
            "range": "stddev: 0.00006522801353886364",
            "extra": "mean: 1.2299131370848808 msec\nrounds: 693"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 176.19453208855313,
            "unit": "iter/sec",
            "range": "stddev: 0.00010118689728354581",
            "extra": "mean: 5.675545024844544 msec\nrounds: 161"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2271344.346949642,
            "unit": "iter/sec",
            "range": "stddev: 4.6991932708421425e-8",
            "extra": "mean: 440.2678974427522 nsec\nrounds: 109362"
          }
        ]
      }
    ],
    "Python Benchmark": [
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "55f62fc5a9b87bd6eb7c213a02206a8f18e12cb0",
          "message": "fix(ci): isort compliance and clippy error surfaced by the hardening pass\n\n- isort starter_common.py (new file missed isort, which gates separately\n  from ruff) and cli_meta/cli_research/cli_chat (pre-existing drift)\n- rust: replace is_some/unwrap with a match guard in choose_client —\n  clippy's deny-by-default unnecessary_unwrap failed the root-crate job\n  the moment Cargo.lock changes retriggered it\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-11T06:43:26-07:00",
          "tree_id": "77faeb886637dbc8ca708d2656c3e1b826d21b17",
          "url": "https://github.com/stateset/stateset-agents/commit/55f62fc5a9b87bd6eb7c213a02206a8f18e12cb0"
        },
        "date": 1786455953648,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5901.06020196323,
            "unit": "iter/sec",
            "range": "stddev: 0.00002202036839887804",
            "extra": "mean: 169.4610740739959 usec\nrounds: 1215"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6413.651321197301,
            "unit": "iter/sec",
            "range": "stddev: 0.000018945367640796277",
            "extra": "mean: 155.91742517947173 usec\nrounds: 1811"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4873.795735353936,
            "unit": "iter/sec",
            "range": "stddev: 0.000024812568616914796",
            "extra": "mean: 205.17889019150283 usec\nrounds: 3242"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 725.0250639313449,
            "unit": "iter/sec",
            "range": "stddev: 0.000028639158918450283",
            "extra": "mean: 1.3792626624211346 msec\nrounds: 628"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.48173750896873,
            "unit": "iter/sec",
            "range": "stddev: 0.00010575457343747766",
            "extra": "mean: 5.57159749999651 msec\nrounds: 12"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2092410.642231229,
            "unit": "iter/sec",
            "range": "stddev: 5.040018442290622e-8",
            "extra": "mean: 477.91766100637693 nsec\nrounds: 103972"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "a02db743a202dce9cde0511b0ed232a2b542ac24",
          "message": "feat(training): NVIDIA Nemotron 3.5 Lightning first-class starter\n\nnvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (released 2026-08-11):\nhybrid Mamba-2 + attention MoE, 30B total / ~3B active, 256K practical\ncontext, OpenMDW-1.1. First starter built on the thin starter_common\npattern from scratch. Mamba-aware LoRA targets (q/k/v/o_proj +\nin_proj/out_proj — not the llama-style MLP list); NVFP4 variant flagged\ninference-only; trust_remote_code for the nemotron_h architecture.\n\nFull surface: CLI command, init --preset, unified-driver preset +\nforwarder, docs page + all reference lists, config/module-export tests,\nintegration round-trip row. Guardrail meta-tests (CLI reference,\nexamples README, preset registry) pass.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-11T07:51:31-07:00",
          "tree_id": "76c5c4d6f999e138b90f80b20c2341fc9f2c15d3",
          "url": "https://github.com/stateset/stateset-agents/commit/a02db743a202dce9cde0511b0ed232a2b542ac24"
        },
        "date": 1786460073256,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10361.324576586985,
            "unit": "iter/sec",
            "range": "stddev: 0.00001260386677907855",
            "extra": "mean: 96.51275689786368 usec\nrounds: 1341"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 10727.963412082241,
            "unit": "iter/sec",
            "range": "stddev: 0.000012397015135381286",
            "extra": "mean: 93.21433729665426 usec\nrounds: 2523"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 7858.576751107029,
            "unit": "iter/sec",
            "range": "stddev: 0.00001904409415937453",
            "extra": "mean: 127.24950479857961 usec\nrounds: 3647"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 958.5452413185911,
            "unit": "iter/sec",
            "range": "stddev: 0.0000729686021911779",
            "extra": "mean: 1.0432475765300164 msec\nrounds: 784"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 244.46805756082918,
            "unit": "iter/sec",
            "range": "stddev: 0.00012436179678226918",
            "extra": "mean: 4.090513950891835 msec\nrounds: 224"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2709975.6682214183,
            "unit": "iter/sec",
            "range": "stddev: 3.768856277091286e-8",
            "extra": "mean: 369.0070031722127 nsec\nrounds: 123901"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "14a1f862abe8eae726c46c75b9f303c05328a5a3",
          "message": "chore(release): v0.22.0 — architecture consolidation + Nemotron 3.5 Lightning\n\nVersion 0.22.0 across pyproject, package, helm chart, k8s manifests, docs.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-11T07:59:06-07:00",
          "tree_id": "879583bc206ec55b9466843508a6fe2e186b21e5",
          "url": "https://github.com/stateset/stateset-agents/commit/14a1f862abe8eae726c46c75b9f303c05328a5a3"
        },
        "date": 1786460520006,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 9953.811839177908,
            "unit": "iter/sec",
            "range": "stddev: 0.000011142503217137953",
            "extra": "mean: 100.46402485367763 usec\nrounds: 1529"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 10435.112674667354,
            "unit": "iter/sec",
            "range": "stddev: 0.000008315584031604334",
            "extra": "mean: 95.83030209416283 usec\nrounds: 2006"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 7768.6194808532655,
            "unit": "iter/sec",
            "range": "stddev: 0.000013164345110646054",
            "extra": "mean: 128.72299930053535 usec\nrounds: 2861"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 925.1491675885861,
            "unit": "iter/sec",
            "range": "stddev: 0.000024949271819746237",
            "extra": "mean: 1.0809067716144778 msec\nrounds: 775"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 237.80500171383636,
            "unit": "iter/sec",
            "range": "stddev: 0.000028399792455626937",
            "extra": "mean: 4.205126018347395 msec\nrounds: 218"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2499272.5482385335,
            "unit": "iter/sec",
            "range": "stddev: 4.7008987740839355e-8",
            "extra": "mean: 400.11642615960056 nsec\nrounds: 119818"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "ff20021adf758271debde4a779463afd8d4c67a9",
          "message": "chore(release): v0.23.0 — Qwen3 Coder, gpt-oss, DeepSeek V4 starters\n\nVersion 0.23.0 across pyproject, package, helm chart, k8s manifests, docs.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-11T09:57:01-07:00",
          "tree_id": "6858dc98da41291b4ecf8b5bdbc40fc5ae863c32",
          "url": "https://github.com/stateset/stateset-agents/commit/ff20021adf758271debde4a779463afd8d4c67a9"
        },
        "date": 1786467636790,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 9634.675771969598,
            "unit": "iter/sec",
            "range": "stddev: 0.000012926252024054574",
            "extra": "mean: 103.79176462889649 usec\nrounds: 1521"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9904.020581391498,
            "unit": "iter/sec",
            "range": "stddev: 0.00001550293879387914",
            "extra": "mean: 100.96909550843257 usec\nrounds: 1937"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 7366.841621211767,
            "unit": "iter/sec",
            "range": "stddev: 0.000019181305235359925",
            "extra": "mean: 135.7433824993119 usec\nrounds: 3017"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 890.2324343976447,
            "unit": "iter/sec",
            "range": "stddev: 0.00002088051260945655",
            "extra": "mean: 1.123302141509399 msec\nrounds: 742"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 227.47919511273267,
            "unit": "iter/sec",
            "range": "stddev: 0.000032146316382405504",
            "extra": "mean: 4.396006410627691 msec\nrounds: 207"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2424697.7785615507,
            "unit": "iter/sec",
            "range": "stddev: 4.340561690599275e-8",
            "extra": "mean: 412.42253316751453 nsec\nrounds: 115168"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "a33d7d4f73b2eec281a2744e05e72fbeb246300b",
          "message": "fix(remote,sft): three bugs found by training Muse Glimmer 30B on live RunPod hardware\n\nVerified end-to-end on an H100 80GB (63GB BF16 multimodal checkpoint,\n160GB pod disk, LoRA r=8 on the text stack, 3 steps/1 epoch on the smoke\ndataset, 258.7MB adapter copied back, pod terminated on completion):\n\n1. sft.py: fall back to AutoModelForImageTextToText when\n   AutoModelForCausalLM rejects a composite multimodal arch (muse_glimmer\n   registers only under the image-text-to-text auto-mapping)\n2. RunPodExecutor: container_disk_gb is now a constructor parameter; the\n   fixed 40GB default killed the 63GB download at ~29GB with an opaque\n   HF-cache 'File reconstruction error'\n3. sft.py: build_training_arguments() filters kwargs against the installed\n   transformers signature — 5.x removed warmup_ratio, crashing the job\n   after the full download\n\nAlso observed and noted: the A100-80GB-PCIe secure-cloud pool failed to\nassign pod networking twice (NET_003); the H100 pool worked immediately.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-11T13:38:56-07:00",
          "tree_id": "2ed41a15d7fd54189beafca4febcda67d05780fd",
          "url": "https://github.com/stateset/stateset-agents/commit/a33d7d4f73b2eec281a2744e05e72fbeb246300b"
        },
        "date": 1786480910295,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10753.850049123863,
            "unit": "iter/sec",
            "range": "stddev: 0.000011520096779695414",
            "extra": "mean: 92.98995201085883 usec\nrounds: 1417"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 10799.075685642285,
            "unit": "iter/sec",
            "range": "stddev: 0.000011577879643019218",
            "extra": "mean: 92.60051777667711 usec\nrounds: 1997"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8095.853745341611,
            "unit": "iter/sec",
            "range": "stddev: 0.0000131840062565565",
            "extra": "mean: 123.52001795677255 usec\nrounds: 2506"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 982.9751744398831,
            "unit": "iter/sec",
            "range": "stddev: 0.00006791928728656126",
            "extra": "mean: 1.0173196902656447 msec\nrounds: 791"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 237.435530091394,
            "unit": "iter/sec",
            "range": "stddev: 0.003324607891114718",
            "extra": "mean: 4.211669582960388 msec\nrounds: 223"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2462372.196880864,
            "unit": "iter/sec",
            "range": "stddev: 3.952422461297707e-8",
            "extra": "mean: 406.11244769036944 nsec\nrounds: 118765"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "62de02ccc6d8a8f4d4b626206f5e9d0aeba20085",
          "message": "fix(sft): exclude vision-tower modules from LoRA target inference\n\nOn multimodal composites, ViT-block fc1/fc2 share leaf names with\ndecoder-MLP candidates and were being adapted despite text-only SFT\nsending no gradient through the vision path (observed on the live\nMuse-Glimmer-30B run). Non-text stacks (vision_tower, visual,\nperception_encoder, projectors, audio_tower) are now skipped.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-11T13:42:45-07:00",
          "tree_id": "36901c700c622048aa214ba4b7129ff1dc9b5006",
          "url": "https://github.com/stateset/stateset-agents/commit/62de02ccc6d8a8f4d4b626206f5e9d0aeba20085"
        },
        "date": 1786481128740,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 13864.408005728625,
            "unit": "iter/sec",
            "range": "stddev: 0.000008325509875566507",
            "extra": "mean: 72.12713298590252 usec\nrounds: 1534"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 14618.251147380739,
            "unit": "iter/sec",
            "range": "stddev: 0.000008381370325295382",
            "extra": "mean: 68.40763576422597 usec\nrounds: 2427"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 10854.134711231845,
            "unit": "iter/sec",
            "range": "stddev: 0.0000092375275124627",
            "extra": "mean: 92.13078947372942 usec\nrounds: 4123"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1223.3125084733933,
            "unit": "iter/sec",
            "range": "stddev: 0.00006537808479177405",
            "extra": "mean: 817.4526076316579 usec\nrounds: 1022"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 329.32672719456076,
            "unit": "iter/sec",
            "range": "stddev: 0.000051517108609283894",
            "extra": "mean: 3.036498156462159 msec\nrounds: 294"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3601110.3702737265,
            "unit": "iter/sec",
            "range": "stddev: 3.3374425651190226e-8",
            "extra": "mean: 277.6921274767783 nsec\nrounds: 171292"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "52245558f99217749a909ec1274b38e383cfec4c",
          "message": "chore(release): v0.23.1 — train-remote fixes verified live on Muse Glimmer 30B\n\nVersion 0.23.1 across pyproject, package, helm chart, k8s manifests, docs.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-12T04:56:16-07:00",
          "tree_id": "cb4fab88ac9870939bdf6dfa6a1d5211abc5535b",
          "url": "https://github.com/stateset/stateset-agents/commit/52245558f99217749a909ec1274b38e383cfec4c"
        },
        "date": 1786535942966,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6020.4125643265525,
            "unit": "iter/sec",
            "range": "stddev: 0.000015501178884185818",
            "extra": "mean: 166.10157349106203 usec\nrounds: 1592"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6545.811101982855,
            "unit": "iter/sec",
            "range": "stddev: 0.000014592502116579066",
            "extra": "mean: 152.769455827572 usec\nrounds: 2128"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4971.799015254918,
            "unit": "iter/sec",
            "range": "stddev: 0.000016176446765935806",
            "extra": "mean: 201.1344378426623 usec\nrounds: 3652"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 717.5625716911154,
            "unit": "iter/sec",
            "range": "stddev: 0.00018226029939470375",
            "extra": "mean: 1.393606689439292 msec\nrounds: 644"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.96887560418213,
            "unit": "iter/sec",
            "range": "stddev: 0.00006510369655915982",
            "extra": "mean: 5.556516351190461 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2112528.8345389166,
            "unit": "iter/sec",
            "range": "stddev: 4.937611426409446e-8",
            "extra": "mean: 473.36631985819093 nsec\nrounds: 104189"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "ea78de94e18359a98c7f62ad402cfd4e4dd24b27",
          "message": "fix(sft): make vision-tower LoRA exclusion effective on real models\n\nLive rerun on published 0.23.1 showed fc1/fc2 still adapted: peft matches\ntarget_modules by leaf name model-wide, so skipping vision modules during\nthe walk changes nothing when the name never occurs in the text stack's\ncandidate set anyway — and Muse Glimmer's vision_adapter/vision_projection\nweren't in the marker list. Two-pass inference now drops names that exist\nonly in non-text stacks (kept with a warning when shared), and the marker\nset includes the names from the real weight map.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-12T05:13:18-07:00",
          "tree_id": "ea5dcdd723cf7da5df897d235e006eb66010eebc",
          "url": "https://github.com/stateset/stateset-agents/commit/ea78de94e18359a98c7f62ad402cfd4e4dd24b27"
        },
        "date": 1786536969194,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5983.06743414436,
            "unit": "iter/sec",
            "range": "stddev: 0.000017917716922215294",
            "extra": "mean: 167.1383468441569 usec\nrounds: 1505"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6567.079802341841,
            "unit": "iter/sec",
            "range": "stddev: 0.00001716052383403732",
            "extra": "mean: 152.27468374046512 usec\nrounds: 2128"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5093.4102817452795,
            "unit": "iter/sec",
            "range": "stddev: 0.000017495821444022488",
            "extra": "mean: 196.33211241277533 usec\nrounds: 3158"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 737.9806292267548,
            "unit": "iter/sec",
            "range": "stddev: 0.000031247033822413255",
            "extra": "mean: 1.355049117004311 msec\nrounds: 641"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 186.40938679952063,
            "unit": "iter/sec",
            "range": "stddev: 0.00004066446448175061",
            "extra": "mean: 5.364536717646515 msec\nrounds: 170"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2066365.0120737331,
            "unit": "iter/sec",
            "range": "stddev: 5.177381166281942e-8",
            "extra": "mean: 483.94160477796424 nsec\nrounds: 101123"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "71789cdb8200b3e2d47f6422c66238a102deb5af",
          "message": "feat(remote): --container-disk-gb and --eval-prompts for train-remote\n\nVerified end-to-end on live H100 hardware with Muse-Glimmer-30B: one\ntrain-remote invocation now sizes the pod disk from the spec, trains, and\nreturns eval_results.json with a base-vs-finetuned comparison on held-out\nprompts through the standard fetch path.\n\n- RemoteJobSpec.container_disk_gb (provider resource, ignored by the job);\n  RunPod uses it over the executor default\n- RemoteJobSpec.eval_prompts flows into the SFT job (--eval-prompts-json,\n  shell-quoted through ssh); base completions generate BEFORE LoRA is\n  applied and tuned completions after; greedy decoding keeps them\n  comparable\n- Live-hardware bug #7 fixed en route: base-eval generation ran on CPU\n  (the Trainer only moves the model to GPU later), grinding a 30B generate\n  while the H100 sat idle — the model now moves to GPU before base eval,\n  with an ordering-guard test\n- CLI: train-remote --container-disk-gb INT, --eval-prompts FILE (one\n  prompt per line)\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-12T06:55:59-07:00",
          "tree_id": "db7a524f4e1ec34208f33b97092fa27ab54ae536",
          "url": "https://github.com/stateset/stateset-agents/commit/71789cdb8200b3e2d47f6422c66238a102deb5af"
        },
        "date": 1786543141017,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5901.789332727643,
            "unit": "iter/sec",
            "range": "stddev: 0.0000177417196611274",
            "extra": "mean: 169.44013817210038 usec\nrounds: 1433"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6410.476613006324,
            "unit": "iter/sec",
            "range": "stddev: 0.000015471066040198048",
            "extra": "mean: 155.99464132995712 usec\nrounds: 2105"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5015.918808415286,
            "unit": "iter/sec",
            "range": "stddev: 0.000015991536680841782",
            "extra": "mean: 199.3652684972261 usec\nrounds: 3568"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 734.4624567738488,
            "unit": "iter/sec",
            "range": "stddev: 0.00008429960155354561",
            "extra": "mean: 1.3615399817609928 msec\nrounds: 658"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.5821061359876,
            "unit": "iter/sec",
            "range": "stddev: 0.00004562436487808045",
            "extra": "mean: 5.47698797633103 msec\nrounds: 169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2168972.8722701864,
            "unit": "iter/sec",
            "range": "stddev: 5.765378475175215e-8",
            "extra": "mean: 461.0477211516877 nsec\nrounds: 105843"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "c5d9b5760e269eb22550fbbf0b142fd004d7a416",
          "message": "chore(release): v0.24.0 — train-remote eval prompts + disk sizing\n\nVersion 0.24.0 across pyproject, package, helm chart, k8s manifests, docs.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-12T06:58:10-07:00",
          "tree_id": "71a6843d3e62daf48c56e19a152047529224d332",
          "url": "https://github.com/stateset/stateset-agents/commit/c5d9b5760e269eb22550fbbf0b142fd004d7a416"
        },
        "date": 1786543262866,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8342.966684005494,
            "unit": "iter/sec",
            "range": "stddev: 0.000012404478286427631",
            "extra": "mean: 119.86143992605466 usec\nrounds: 1623"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 8916.519015772878,
            "unit": "iter/sec",
            "range": "stddev: 0.00001120511067539907",
            "extra": "mean: 112.15138982276041 usec\nrounds: 2083"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6785.666741080868,
            "unit": "iter/sec",
            "range": "stddev: 0.000012264611356333576",
            "extra": "mean: 147.36945360813775 usec\nrounds: 3395"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 878.9904707107522,
            "unit": "iter/sec",
            "range": "stddev: 0.00001681635865240747",
            "extra": "mean: 1.1376687612908927 msec\nrounds: 775"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 196.13297928629606,
            "unit": "iter/sec",
            "range": "stddev: 0.005287994724615751",
            "extra": "mean: 5.098581603353387 msec\nrounds: 179"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2256104.6115398007,
            "unit": "iter/sec",
            "range": "stddev: 3.287531396423908e-8",
            "extra": "mean: 443.2418580614912 nsec\nrounds: 109458"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "21cdf7fa941ce2af57774952515766657d464989",
          "message": "feat(remote): chat-remote — ephemeral chat with a fine-tuned model on RunPod\n\nstateset-agents chat-remote --base-model X --adapter DIR rents a pod,\nloads base + LoRA adapter, and holds a multi-turn conversation over an\nSSH-piped JSON-lines protocol; the pod is terminated on every exit path\n(close/context-manager/atexit/startup failure). --prompt gives a scripted\nnon-interactive mode.\n\nLive-verified on an H100 with the Muse-Glimmer-30B Astra adapter: 3-turn\nconversation in which turn 2's 'I also got double charged for it'\nresolved to the order number from turn 1 via the on-pod history —\nmulti-turn state working on real hardware. Pod terminated after.\n\n39 new unit tests (session protocol/lifecycle, repl loop, CLI); pip pin\nlogic extracted to runpod.package_pin and shared.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-12T08:01:29-07:00",
          "tree_id": "cc22f06f6686dfe8d22a859150c432204fe29fb3",
          "url": "https://github.com/stateset/stateset-agents/commit/21cdf7fa941ce2af57774952515766657d464989"
        },
        "date": 1786547065804,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8422.776584907017,
            "unit": "iter/sec",
            "range": "stddev: 0.000014482755481400918",
            "extra": "mean: 118.7256945401977 usec\nrounds: 1447"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9057.370088370588,
            "unit": "iter/sec",
            "range": "stddev: 0.000013808295899900667",
            "extra": "mean: 110.40732466966017 usec\nrounds: 2193"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6584.096083360328,
            "unit": "iter/sec",
            "range": "stddev: 0.00002248322948688984",
            "extra": "mean: 151.88113711269378 usec\nrounds: 2487"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 822.9612199199232,
            "unit": "iter/sec",
            "range": "stddev: 0.00002427259417175058",
            "extra": "mean: 1.2151240857950796 msec\nrounds: 711"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 175.420396156011,
            "unit": "iter/sec",
            "range": "stddev: 0.00010464641394769843",
            "extra": "mean: 5.700591390243156 msec\nrounds: 164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2294400.9331508866,
            "unit": "iter/sec",
            "range": "stddev: 4.933922371914272e-8",
            "extra": "mean: 435.8436163232841 nsec\nrounds: 108190"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "b030301212f4a1a0823883cc33c2bc67c676d8a6",
          "message": "types: cast piped stdout in chat_session — clears full-mypy gate\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-12T08:52:12-07:00",
          "tree_id": "4f448f343dc08f79b083560f4bbd3569505283ae",
          "url": "https://github.com/stateset/stateset-agents/commit/b030301212f4a1a0823883cc33c2bc67c676d8a6"
        },
        "date": 1786550099502,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8498.74858613199,
            "unit": "iter/sec",
            "range": "stddev: 0.000017756540526664024",
            "extra": "mean: 117.66438198110376 usec\nrounds: 1343"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9196.082182895001,
            "unit": "iter/sec",
            "range": "stddev: 0.0000155992307536636",
            "extra": "mean: 108.7419599033196 usec\nrounds: 2070"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6662.639602938534,
            "unit": "iter/sec",
            "range": "stddev: 0.00001846401913017304",
            "extra": "mean: 150.09066370015773 usec\nrounds: 3259"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 832.7029742491596,
            "unit": "iter/sec",
            "range": "stddev: 0.00002243255429704629",
            "extra": "mean: 1.2009084042262375 msec\nrounds: 710"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 175.99233437007354,
            "unit": "iter/sec",
            "range": "stddev: 0.000049591248792873286",
            "extra": "mean: 5.682065662571517 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2136387.230156648,
            "unit": "iter/sec",
            "range": "stddev: 4.893408527905058e-8",
            "extra": "mean: 468.07993695350643 nsec\nrounds: 108179"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "2f242c95ce16ba808bb6343c7fc4d1a159e07f50",
          "message": "fix(remote): ssh keepalives — dead pods must fail fast, not hang\n\nObserved live: RunPod restarted a pod underneath a running training job\n(new public IP, all processes gone); without ServerAliveInterval the\nexecutor's blocking ssh read hung indefinitely — 20+ minutes of blind\nidle billing before manual intervention. Keepalives bound peer loss\ndetection to ~2 minutes. chat_session inherits via _base_opts.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-12T10:04:09-07:00",
          "tree_id": "5de3fae63823e154877a2cdf7eee5f9483d08eca",
          "url": "https://github.com/stateset/stateset-agents/commit/2f242c95ce16ba808bb6343c7fc4d1a159e07f50"
        },
        "date": 1786554441836,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5961.685305348859,
            "unit": "iter/sec",
            "range": "stddev: 0.00001767201858571203",
            "extra": "mean: 167.7378037889377 usec\nrounds: 1478"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6456.685967568988,
            "unit": "iter/sec",
            "range": "stddev: 0.00001630885510048697",
            "extra": "mean: 154.87821539143414 usec\nrounds: 2131"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4968.680648235162,
            "unit": "iter/sec",
            "range": "stddev: 0.000017578106180546574",
            "extra": "mean: 201.26067074872128 usec\nrounds: 3593"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 741.1630814178659,
            "unit": "iter/sec",
            "range": "stddev: 0.00009163612730877011",
            "extra": "mean: 1.3492307227270033 msec\nrounds: 660"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 173.65736195570062,
            "unit": "iter/sec",
            "range": "stddev: 0.005013293231984718",
            "extra": "mean: 5.758465916665811 msec\nrounds: 156"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2119588.63089613,
            "unit": "iter/sec",
            "range": "stddev: 4.907182145008326e-8",
            "extra": "mean: 471.78966023101145 nsec\nrounds: 105631"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "12fff89bcb34b7775582fc36d2ca2f7a2542d1e7",
          "message": "chore(release): v0.25.0 — chat-remote\n\nVersion 0.25.0 across pyproject, package, helm chart, k8s manifests, docs.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-12T10:15:13-07:00",
          "tree_id": "a6cde3e122232b20ecf0deac43be80db8c557073",
          "url": "https://github.com/stateset/stateset-agents/commit/12fff89bcb34b7775582fc36d2ca2f7a2542d1e7"
        },
        "date": 1786555105157,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6011.179255649183,
            "unit": "iter/sec",
            "range": "stddev: 0.00001731201015122795",
            "extra": "mean: 166.3567093029576 usec\nrounds: 1462"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6574.225669425587,
            "unit": "iter/sec",
            "range": "stddev: 0.000015726597991480855",
            "extra": "mean: 152.1091684836206 usec\nrounds: 2018"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5047.032802917393,
            "unit": "iter/sec",
            "range": "stddev: 0.0000178174429688878",
            "extra": "mean: 198.13621964611735 usec\nrounds: 3278"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 738.2430489938623,
            "unit": "iter/sec",
            "range": "stddev: 0.000029039081673691886",
            "extra": "mean: 1.3545674440997195 msec\nrounds: 644"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.16269346350157,
            "unit": "iter/sec",
            "range": "stddev: 0.00004232477982691057",
            "extra": "mean: 5.489598232144947 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2169357.216696829,
            "unit": "iter/sec",
            "range": "stddev: 4.734759093015654e-8",
            "extra": "mean: 460.9660374526282 nsec\nrounds: 106747"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "ba9036a68a7836776a3488207d1bfa8ad084e63f",
          "message": "fix(sft): reasoning-aware eval — disable thinking, configurable token budget\n\nNVIDIA Nemotron 3.5 Lightning's chat template defaults to thinking mode,\nso the fixed max_new_tokens=90 eval budget was consumed entirely by the\nreasoning preamble and the base-vs-tuned comparison was truncated garbage\n(hit for real on an H100 pod).\n\n- generate_completions now passes enable_thinking=False to\n  apply_chat_template, falling back on TypeError for templates that don't\n  accept the kwarg (Muse Glimmer's doesn't).\n- New job-level RemoteJobSpec.eval_max_new_tokens (default 90), flowed\n  through run_sft_job/run_sft, the module CLI (--eval-max-new-tokens),\n  RunPod command construction, and train-remote.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-12T10:35:29-07:00",
          "tree_id": "9954e849e2e5dd4db24611cc7b5d0d94e5151a8a",
          "url": "https://github.com/stateset/stateset-agents/commit/ba9036a68a7836776a3488207d1bfa8ad084e63f"
        },
        "date": 1786556315455,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10931.709992492579,
            "unit": "iter/sec",
            "range": "stddev: 0.00001566628563985655",
            "extra": "mean: 91.47699679983793 usec\nrounds: 1562"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11922.311755563518,
            "unit": "iter/sec",
            "range": "stddev: 0.000011373463054274318",
            "extra": "mean: 83.8763505352351 usec\nrounds: 2616"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 9139.942096136827,
            "unit": "iter/sec",
            "range": "stddev: 0.000009473724636169114",
            "extra": "mean: 109.40988350710333 usec\nrounds: 4026"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1076.5656056098037,
            "unit": "iter/sec",
            "range": "stddev: 0.00002168487833295286",
            "extra": "mean: 928.8797587338541 usec\nrounds: 916"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 267.66113405572224,
            "unit": "iter/sec",
            "range": "stddev: 0.00010578767077948366",
            "extra": "mean: 3.736067261045894 msec\nrounds: 249"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2901470.1777620786,
            "unit": "iter/sec",
            "range": "stddev: 3.48725542285405e-8",
            "extra": "mean: 344.65286173346306 nsec\nrounds: 137836"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "b1cc1f3a05e353f22f62692d64c776e1ec9b9a8f",
          "message": "feat(sft): eval prompts can assert — pass/fail gate on the finetuned model\n\neval_prompts entries may now be spec dicts {\"prompt\", \"expect\",\n\"forbid\", \"judge\", \"min_judge_score\"} alongside plain strings.\nexpect/forbid substrings match case-insensitively against the finetuned\ncompletion; rows in eval_results.json gain checks {expect_hits,\nforbid_hits, passed} and, when a domain judge is importable on the\nworker, judge_score (degrading to a logged warning otherwise). When any\nassertion fails, run_sft_job exits non-zero AFTER the adapter and\neval_results.json are saved, so a red run never destroys the artifacts.\n\n- train-remote --eval-prompts file: a line that parses as a JSON object\n  is a prompt spec; any other line stays a plain prompt (back-compat)\n- RemoteJobSpec validates spec entries at submit time, before a GPU is\n  rented; dict prompts ride the existing shlex-quoted JSON blob to pods\n- gpu-verify.yml now gates on completion content (expect: [\"number\"]),\n  not just adapter tensors\n- docs: CLI_REFERENCE train-remote file format; CHANGELOG [Unreleased]\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-12T11:49:26-07:00",
          "tree_id": "db7f067ce9baf88ed9bd1b108b3c4ca979e49759",
          "url": "https://github.com/stateset/stateset-agents/commit/b1cc1f3a05e353f22f62692d64c776e1ec9b9a8f"
        },
        "date": 1786560783618,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8499.94955114605,
            "unit": "iter/sec",
            "range": "stddev: 0.000015263968562270185",
            "extra": "mean: 117.6477570817076 usec\nrounds: 1412"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9169.112132376107,
            "unit": "iter/sec",
            "range": "stddev: 0.00001436954512273119",
            "extra": "mean: 109.06181378990918 usec\nrounds: 2132"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6639.278138264141,
            "unit": "iter/sec",
            "range": "stddev: 0.000016579780065992793",
            "extra": "mean: 150.61878402664013 usec\nrounds: 3005"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 830.0615464685093,
            "unit": "iter/sec",
            "range": "stddev: 0.000022589097541090507",
            "extra": "mean: 1.2047299435258658 msec\nrounds: 726"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 178.65770003530326,
            "unit": "iter/sec",
            "range": "stddev: 0.00007047275706134504",
            "extra": "mean: 5.597295833330425 msec\nrounds: 12"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2315183.9812175925,
            "unit": "iter/sec",
            "range": "stddev: 3.4812703753727236e-8",
            "extra": "mean: 431.93111567491235 nsec\nrounds: 57091"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "a13f173b3cd5a5259595d69aa60fa157e76fcff4",
          "message": "feat(training): GSPO GPU verification job, verified live on RunPod\n\nAdd stateset_agents/training/gpu_verify_rl.py — a runnable module\n(python -m stateset_agents.training.gpu_verify_rl) that runs a short\nreal GSPO training on a tiny GPT-2 and asserts the same convergence\nproperty as the nightly CPU e2e test (target-token probability\nstrictly increases), on CUDA when available and CPU otherwise. Prints\na GPU_VERIFY_RL_SUMMARY JSON line and exits 0/1.\n\nAdd an rl-live-smoke job to .github/workflows/gpu-verify.yml mirroring\nthe SFT job's secret gating, built from RunPodApi + SshTransport\nprimitives with unconditional pod termination.\n\nVerified live on a RunPod NVIDIA RTX A4500 (A4000 had no availability):\ntarget-token prob 2.81e-05 -> 0.1246 over 40 GSPO steps, exit 0, pod\nterminated (0 pods remaining).\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-12T12:02:32-07:00",
          "tree_id": "eb8bf9cdd04e1e78c1334710aa8a873940bd4591",
          "url": "https://github.com/stateset/stateset-agents/commit/a13f173b3cd5a5259595d69aa60fa157e76fcff4"
        },
        "date": 1786561540536,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 7481.488909420528,
            "unit": "iter/sec",
            "range": "stddev: 0.0000402311545302002",
            "extra": "mean: 133.6632336299826 usec\nrounds: 1237"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9185.683874196298,
            "unit": "iter/sec",
            "range": "stddev: 0.000013702458669070229",
            "extra": "mean: 108.86505715803278 usec\nrounds: 1907"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6697.873122626291,
            "unit": "iter/sec",
            "range": "stddev: 0.000016127808554517387",
            "extra": "mean: 149.3011261473242 usec\nrounds: 2941"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 836.3369911060641,
            "unit": "iter/sec",
            "range": "stddev: 0.000019765370966086413",
            "extra": "mean: 1.1956902667637477 msec\nrounds: 686"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.30896301452538,
            "unit": "iter/sec",
            "range": "stddev: 0.00005995476878156711",
            "extra": "mean: 5.546035999993199 msec\nrounds: 11"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2338999.8426800147,
            "unit": "iter/sec",
            "range": "stddev: 3.37929006335874e-8",
            "extra": "mean: 427.5331625735403 nsec\nrounds: 57684"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "571c5f5fce190eab6cfef380b2b465bdbd388967",
          "message": "feat(remote): survive pod death — retry on a fresh pod, COMMUNITY cloud, --resume\n\nA pod dying under a running job (observed live even on SECURE) previously\nlost the whole run. Now:\n\n- RunPodExecutor retries: on mid-job pod/ssh death (exception in the job\n  phase, or ssh's own exit 255 from keepalive-detected death) it terminates\n  the dead pod, provisions a fresh one (bounded by new ctor param\n  max_provision_attempts, default 2), re-uploads inputs, and reruns.\n  v1 restarts training FROM SCRATCH on the new pod — the dead pod's\n  checkpoints lived on its container disk; cross-pod checkpoint resume\n  needs a RunPod network volume (NOTE + follow-up left in code).\n  Training failures (the job's own non-zero exit) and never-reachable\n  pods are NOT retried. The pod still dies on every path.\n- RemoteJobSpec.cloud_type (\"SECURE\" default | \"COMMUNITY\" ~spot pricing,\n  interruptible — now usable because of the retry) -> create_pod cloudType;\n  CLI --cloud-type.\n- sft --resume / RemoteJobSpec.resume: trainer.train(resume_from_\n  checkpoint=True) when a checkpoint-* dir exists in output_dir, logged\n  fresh start otherwise (HF raises on an empty dir). Delivers value for\n  same-machine/local reruns today.\n\nVerified live on a COMMUNITY-cloud RTX A4000 pod (Qwen/Qwen3.5-0.8B,\n24-row smoke set, wheel install): SUCCEEDED, adapter + checkpoint fetched,\nzero pods left billing.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-13T04:01:25-07:00",
          "tree_id": "8791c592ac5fe671dcca83fce1128c0d8712c67f",
          "url": "https://github.com/stateset/stateset-agents/commit/571c5f5fce190eab6cfef380b2b465bdbd388967"
        },
        "date": 1786619067224,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6168.816889844823,
            "unit": "iter/sec",
            "range": "stddev: 0.00001644267424057998",
            "extra": "mean: 162.10563838362773 usec\nrounds: 1485"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6639.066874949539,
            "unit": "iter/sec",
            "range": "stddev: 0.00001586797261978531",
            "extra": "mean: 150.62357690252978 usec\nrounds: 1879"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5118.183130252845,
            "unit": "iter/sec",
            "range": "stddev: 0.000017136869234752695",
            "extra": "mean: 195.38183268377867 usec\nrounds: 3592"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 734.018329347585,
            "unit": "iter/sec",
            "range": "stddev: 0.00009717989574677123",
            "extra": "mean: 1.3623637994010673 msec\nrounds: 668"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 185.29645175132123,
            "unit": "iter/sec",
            "range": "stddev: 0.00005148996092922338",
            "extra": "mean: 5.396757415204361 msec\nrounds: 171"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2109423.3836964048,
            "unit": "iter/sec",
            "range": "stddev: 5.482937709977161e-8",
            "extra": "mean: 474.06320027024185 nsec\nrounds: 106191"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "2ada582d7f79669850f25c99e41ee95042799e5b",
          "message": "feat(serve-remote): persistent vLLM OpenAI endpoint on RunPod, with cost controls\n\nNew `stateset-agents serve-remote --base-model X [--adapter DIR]`: rents a\npod (ports 22+8000), installs vLLM, serves base model (+ LoRA adapter as\nserved-model `adapter`) behind a generated Bearer token, and prints the\nendpoint URL (from RunPod's port-8000 mapping), example curl, and stop\ncommand. The pod outlives the CLI by design, so cost is controlled by an\nON-POD self-destruct (`--max-hours`, default 1.0: nohup'ed script sleeps\nthen DELETEs its own pod via the RunPod API — the API key is copied to the\npod chmod 600, tradeoff documented), plus `--stop <name-or-id>` and\n`--list`. Startup failures terminate the pod before propagating; setup\ncommands absorb one ssh-transport death (exit 255, observed live) by\nreconnecting and retrying once.\n\nNew module stateset_agents.remote.serve_session; RunPodApi.list_pods()\nadded. 34 new unit tests (session, CLI, self-destruct script contents,\ntransport retry), CLI_REFERENCE section, CHANGELOG.\n\nLive verification status: pod provisioning with both port mappings and the\nself-destruct cost control were verified on real RunPod hardware (pods\nobserved publishing 22+8000 mappings; every pod across four attempts was\nterminated, including after the local CLI was killed). The full\nvLLM-ready + authenticated completion curl did NOT complete within the\nverification budget (SECURE capacity shortages, one host without a public\nIP, one mid-install sshd drop); treat the end-to-end serve path as not yet\nlive-verified.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-13T07:29:14-07:00",
          "tree_id": "253511becfbcb08b35aa22cb77a180baa03bf891",
          "url": "https://github.com/stateset/stateset-agents/commit/2ada582d7f79669850f25c99e41ee95042799e5b"
        },
        "date": 1786635627472,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8614.067229651582,
            "unit": "iter/sec",
            "range": "stddev: 0.000014009901863035313",
            "extra": "mean: 116.08917986590261 usec\nrounds: 1490"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9175.700948254982,
            "unit": "iter/sec",
            "range": "stddev: 0.000012355270992322121",
            "extra": "mean: 108.98349953200884 usec\nrounds: 2136"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6692.046048245499,
            "unit": "iter/sec",
            "range": "stddev: 0.00001991038512709764",
            "extra": "mean: 149.4311295515035 usec\nrounds: 2609"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 819.6682357518663,
            "unit": "iter/sec",
            "range": "stddev: 0.00002102049576808641",
            "extra": "mean: 1.2200057979344763 msec\nrounds: 678"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 175.45379334546908,
            "unit": "iter/sec",
            "range": "stddev: 0.0001339798093687338",
            "extra": "mean: 5.699506296971287 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2358054.126472242,
            "unit": "iter/sec",
            "range": "stddev: 3.739569440725795e-8",
            "extra": "mean: 424.07847588131756 nsec\nrounds: 58086"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "48515e1e6853db379950412158113f6d63ff2d88",
          "message": "chore(release): v0.26.0 — the flywheel closes — assertions, GPU-verified RL, spot pods, serve-remote\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-13T08:39:03-07:00",
          "tree_id": "b31c973b9816350a3736016d00bbf19d24bfa123",
          "url": "https://github.com/stateset/stateset-agents/commit/48515e1e6853db379950412158113f6d63ff2d88"
        },
        "date": 1786635691874,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8459.204645870705,
            "unit": "iter/sec",
            "range": "stddev: 0.000013499133142990228",
            "extra": "mean: 118.21442344324205 usec\nrounds: 1365"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 8977.79403928111,
            "unit": "iter/sec",
            "range": "stddev: 0.000018674242592798427",
            "extra": "mean: 111.38593685983848 usec\nrounds: 1758"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6607.207378226282,
            "unit": "iter/sec",
            "range": "stddev: 0.00001605773281340158",
            "extra": "mean: 151.34987336638613 usec\nrounds: 3443"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 822.3577260113742,
            "unit": "iter/sec",
            "range": "stddev: 0.00002632009089730718",
            "extra": "mean: 1.2160158144925957 msec\nrounds: 690"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 159.75769171342353,
            "unit": "iter/sec",
            "range": "stddev: 0.008050741946713595",
            "extra": "mean: 6.259479523488731 msec\nrounds: 149"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2251973.791287841,
            "unit": "iter/sec",
            "range": "stddev: 3.4511969619883664e-8",
            "extra": "mean: 444.05490146851486 nsec\nrounds: 55565"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "9c1438fa10c0aafcfaa4dda179e260bf58d8463d",
          "message": "feat(remote): --gpu-count for multi-GPU pods and --network-volume-id for durable checkpoints\n\nTwo capabilities that extend the training envelope past a single ephemeral\n80GB GPU:\n- RemoteJobSpec.gpu_count -> create_pod gpuCount, with device_map='auto'\n  in sft.py when torch reports >1 CUDA device (single-GPU path unchanged)\n- RemoteJobSpec.network_volume_id -> pod attaches a RunPod network volume\n  at /workspace, so checkpoints survive pod death and the retry path can\n  resume instead of restarting from scratch (the documented v1 gap)\n\nNOT LIVE-VERIFIED: both agents were interrupted before their live runs\ncompleted, so unit tests and the API payload shapes are green but neither\nhas been proven on real hardware. Live verification is the next step for\neach; treat these flags as experimental until then.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-13T10:00:03-07:00",
          "tree_id": "9937aeb0d129baf417c0c77cf166f7b7a7b0dafc",
          "url": "https://github.com/stateset/stateset-agents/commit/9c1438fa10c0aafcfaa4dda179e260bf58d8463d"
        },
        "date": 1786640567063,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8507.43135238018,
            "unit": "iter/sec",
            "range": "stddev: 0.00001714987332108311",
            "extra": "mean: 117.54429258135872 usec\nrounds: 1415"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9173.592248882149,
            "unit": "iter/sec",
            "range": "stddev: 0.000015836341294403684",
            "extra": "mean: 109.00855116182598 usec\nrounds: 2023"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6635.279897347419,
            "unit": "iter/sec",
            "range": "stddev: 0.000018139272584346964",
            "extra": "mean: 150.70954284833851 usec\nrounds: 3384"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 827.2801086820164,
            "unit": "iter/sec",
            "range": "stddev: 0.000026678702247088748",
            "extra": "mean: 1.208780423347967 msec\nrounds: 711"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 161.02125206459343,
            "unit": "iter/sec",
            "range": "stddev: 0.007596235421609609",
            "extra": "mean: 6.210360354165247 msec\nrounds: 144"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2222259.124591361,
            "unit": "iter/sec",
            "range": "stddev: 5.679030700634087e-8",
            "extra": "mean: 449.99252739434 nsec\nrounds: 56861"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "799e9193d16362a2e6f07bc7d68e0bf49cbb0b07",
          "message": "chore(release): v0.27.0 — cost accounting, a grader that rewards resolutions, durable checkpoints\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-13T10:45:31-07:00",
          "tree_id": "b55073b15d3d4806de69bea87f01deb98166d33b",
          "url": "https://github.com/stateset/stateset-agents/commit/799e9193d16362a2e6f07bc7d68e0bf49cbb0b07"
        },
        "date": 1786643288740,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6074.91556357486,
            "unit": "iter/sec",
            "range": "stddev: 0.000015071603379878122",
            "extra": "mean: 164.6113414309807 usec\nrounds: 1482"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6568.471228684981,
            "unit": "iter/sec",
            "range": "stddev: 0.000015065733780936534",
            "extra": "mean: 152.24242676635757 usec\nrounds: 2137"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5023.356158171967,
            "unit": "iter/sec",
            "range": "stddev: 0.00001648896750947419",
            "extra": "mean: 199.0700974632678 usec\nrounds: 3509"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 739.2746149337648,
            "unit": "iter/sec",
            "range": "stddev: 0.00002648961436597459",
            "extra": "mean: 1.3526773134088943 msec\nrounds: 619"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 172.9398302587701,
            "unit": "iter/sec",
            "range": "stddev: 0.00520975498278882",
            "extra": "mean: 5.782357936304776 msec\nrounds: 157"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2107492.8517099223,
            "unit": "iter/sec",
            "range": "stddev: 7.979535702454952e-8",
            "extra": "mean: 474.4974575779207 nsec\nrounds: 102481"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "a9524057b8f18b40a1d33839c14548c072f7593f",
          "message": "feat(training): adapter provenance manifests and lineage\n\nA LoRA adapter directory was anonymous tensors — no record of which base\nmodel it modifies, which data taught it, or which generation preceded it.\nEvery run now writes stateset_manifest.json beside the adapter (base model,\ndataset path AND content hash, hyperparameters, eval outcome, package\nversion, parent adapter), and 'stateset-agents adapters' reads them back\nwith the reconstructed family tree.\n\nDetails that matter:\n- the manifest is written as soon as the adapter is saved, so provenance\n  exists even if the eval that follows fails or the process dies\n- the dataset is hashed, not just named: two runs claiming the same file\n  are only comparable if the bytes match\n- lineage resolves by recorded path then by directory name, so a manifest\n  written on a rented pod links up after the adapter is fetched elsewhere\n- pre-manifest adapters are listed as carrying no provenance, not hidden\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-13T11:04:59-07:00",
          "tree_id": "a425f25496cbf8eb5f3dad136d9ebb485e13c7c2",
          "url": "https://github.com/stateset/stateset-agents/commit/a9524057b8f18b40a1d33839c14548c072f7593f"
        },
        "date": 1786644493039,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5989.6480055543025,
            "unit": "iter/sec",
            "range": "stddev: 0.000019307367217759178",
            "extra": "mean: 166.95471905405506 usec\nrounds: 1438"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6519.132396486518,
            "unit": "iter/sec",
            "range": "stddev: 0.000017062180594878184",
            "extra": "mean: 153.39464505107296 usec\nrounds: 2051"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4945.85561270841,
            "unit": "iter/sec",
            "range": "stddev: 0.000019640095612437965",
            "extra": "mean: 202.18948515813787 usec\nrounds: 3032"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 721.283017703521,
            "unit": "iter/sec",
            "range": "stddev: 0.00009770350823640185",
            "extra": "mean: 1.3864183343507526 msec\nrounds: 655"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 157.7576764934185,
            "unit": "iter/sec",
            "range": "stddev: 0.008866400275598804",
            "extra": "mean: 6.33883575257727 msec\nrounds: 97"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2116371.8061009464,
            "unit": "iter/sec",
            "range": "stddev: 5.4054480195872004e-8",
            "extra": "mean: 472.5067670610908 nsec\nrounds: 105186"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "3c11e424b0a58c77e87701fc185d9a352535b0b3",
          "message": "feat(sft): log device placement; --gpu-count verified on real 2-GPU hardware\n\nLive proof: meta-models/Muse-Glimmer-30B (63GB bf16) trained across two\n48GB L40S cards — a model that cannot fit on either card alone — logging\n'Model sharded across devices: 0=24 module(s), 1=36 module(s)'. Adapter\nreturned, $0.35, zero pods left. --gpu-count is no longer an unproven flag.\n\nThe first attempt succeeded and proved nothing: capacity forced a fallback\nto cards big enough to hold the whole model, so device_map='auto' was never\nobliged to split it, and an external nvidia-smi poller sampled only idle\nmoments. That gap was the real finding — a multi-GPU run was\nindistinguishable from a single-GPU one in its own logs. Placement is now\nlogged (with the single-device case) and returns counts for tests.\n\nAlso removes the duplicate placement logger I wrote before noticing the\ninterrupted agent had already added one.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-13T12:22:40-07:00",
          "tree_id": "58d71cef0507828566fa12f8d973e2fadec84538",
          "url": "https://github.com/stateset/stateset-agents/commit/3c11e424b0a58c77e87701fc185d9a352535b0b3"
        },
        "date": 1786649114026,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5990.13069279194,
            "unit": "iter/sec",
            "range": "stddev: 0.0000169835712454527",
            "extra": "mean: 166.9412657729359 usec\nrounds: 1490"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6485.197693852359,
            "unit": "iter/sec",
            "range": "stddev: 0.00001535994786755668",
            "extra": "mean: 154.19730395388711 usec\nrounds: 2099"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4968.094163144558,
            "unit": "iter/sec",
            "range": "stddev: 0.000018888888547835833",
            "extra": "mean: 201.28442963469303 usec\nrounds: 3226"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 739.7696193329147,
            "unit": "iter/sec",
            "range": "stddev: 0.00005224269269894205",
            "extra": "mean: 1.3517721921342856 msec\nrounds: 661"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 183.702830269801,
            "unit": "iter/sec",
            "range": "stddev: 0.000046480531647760154",
            "extra": "mean: 5.443574269004556 msec\nrounds: 171"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2127830.8806821643,
            "unit": "iter/sec",
            "range": "stddev: 4.083853472747542e-8",
            "extra": "mean: 469.96216150383555 nsec\nrounds: 53548"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "2bbc4cf8f390ca2d8b5f9025db8678d128c5eabc",
          "message": "fix(serve-remote): fail fast on dead pod networking, retry on a fresh host\n\nFour verification attempts died the same way: a pod reaches RUNNING and\nnever publishes an IP or port mappings. That wait shared ready_timeout_s\nwith the vLLM load — necessarily long, since loading a 30B model takes\nmany minutes — so a pod that could never serve anything billed for 30\nminutes before failing.\n\nNetworking is now its own problem with its own short deadline (300s: it\nappears in about two minutes or never), and a pod that misses it is\nterminated and replaced, bounded by max_provision_attempts. Both pods are\nterminated on the way out, so a retry cannot double the bill.\n\nThe existing timeout test encoded the old single-deadline behavior; it now\nasserts the new contract, alongside tests that the vLLM timeout does not\ngovern the networking wait and that a single-attempt session does not retry.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-13T21:43:10-07:00",
          "tree_id": "b43209ebe0765145f53ec2ba3e397c5b4932dd0a",
          "url": "https://github.com/stateset/stateset-agents/commit/2bbc4cf8f390ca2d8b5f9025db8678d128c5eabc"
        },
        "date": 1786682753830,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8434.949771728505,
            "unit": "iter/sec",
            "range": "stddev: 0.000014764565933601404",
            "extra": "mean: 118.55435148549536 usec\nrounds: 1414"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 8989.345599712102,
            "unit": "iter/sec",
            "range": "stddev: 0.00001422342055933277",
            "extra": "mean: 111.24280281670633 usec\nrounds: 1917"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6370.11493345146,
            "unit": "iter/sec",
            "range": "stddev: 0.00003561824176656449",
            "extra": "mean: 156.98303883791613 usec\nrounds: 3373"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 833.9875444374464,
            "unit": "iter/sec",
            "range": "stddev: 0.00004637967245481616",
            "extra": "mean: 1.199058675000398 msec\nrounds: 680"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 160.0066836089185,
            "unit": "iter/sec",
            "range": "stddev: 0.007238815482398331",
            "extra": "mean: 6.249738932432081 msec\nrounds: 148"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2288106.9909512573,
            "unit": "iter/sec",
            "range": "stddev: 3.44060202214958e-8",
            "extra": "mean: 437.04250017795727 nsec\nrounds: 56348"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "42d8b25bcdd4e2d7f5af33a0eec66292f8c1dd00",
          "message": "fix(remote): record chat and serve pod costs, as the docs already claimed\n\nCLI_REFERENCE said every train-remote, chat-remote and serve-remote pod\nappends a line to the cost ledger. Only training did. That made\n'stateset-agents costs' under-report actual spend, and the omission was\nworst for serve pods — the ones that deliberately outlive the command that\nstarted them.\n\nBoth sessions now record on teardown, with the same never-raise discipline\nas the training path: bookkeeping cannot break a teardown.\n\nHonest limit, now documented: a serve pod reaped by its own --max-hours\nself-destruct records nothing, because this machine never observes it end.\nserve-remote --list is the live view for that case.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-14T04:22:56-07:00",
          "tree_id": "6b0f98d49aecf49e82f29ee51bb1f55c0adb6285",
          "url": "https://github.com/stateset/stateset-agents/commit/42d8b25bcdd4e2d7f5af33a0eec66292f8c1dd00"
        },
        "date": 1786706736853,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14124.201162953435,
            "unit": "iter/sec",
            "range": "stddev: 0.000007655219428379608",
            "extra": "mean: 70.80046428557772 usec\nrounds: 1764"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 14857.762244709897,
            "unit": "iter/sec",
            "range": "stddev: 0.00000745209400462859",
            "extra": "mean: 67.30488639741492 usec\nrounds: 2676"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11092.45223563636,
            "unit": "iter/sec",
            "range": "stddev: 0.000008467653593193785",
            "extra": "mean: 90.15139112227435 usec\nrounds: 4303"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1329.687740025194,
            "unit": "iter/sec",
            "range": "stddev: 0.00001203246665201203",
            "extra": "mean: 752.0562684747719 usec\nrounds: 1069"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 315.47544785177905,
            "unit": "iter/sec",
            "range": "stddev: 0.003389391311453362",
            "extra": "mean: 3.169818782442409 msec\nrounds: 262"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3642788.6714313785,
            "unit": "iter/sec",
            "range": "stddev: 2.858142199639651e-8",
            "extra": "mean: 274.5149637261459 nsec\nrounds: 162787"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "11c2891b63779f20558b1ae85925d42ff8fc953b",
          "message": "fix(serve-remote): request an http port and survive dropped links\n\nFive verification attempts failed identically — pod RUNNING, no networking\n— and the cause was our assumption, not RunPod's infrastructure. We asked\nfor the model port as 8000/tcp and then waited for a TCP mapping on a\npublic IP. RunPod serves http ports through its proxy instead\n(https://<pod-id>-8000.proxy.runpod.net): no public IP, no mapping, ever.\nThe tell was that the proxy URL answered with a routing 404 while publicIp\nstayed empty.\n\n- request 8000/http; endpoint URL is the proxy URL\n- wait only on the ssh endpoint, which genuinely needs a TCP mapping\n- send supportPublicIp: RunPod's docs say a COMMUNITY pod 'might not have a\n  public IP address' without it — another silent hang closed\n\nWith that fixed the next attempt got networking on the first host and\nreached 'pip install vllm', where the ssh transport dropped and took the\nrun with it. Long steps now run detached with their exit code polled from a\nmarker file (the pattern already used for the server and the self-destruct),\nso a dropped link costs one poll instead of the whole install.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-14T05:51:15-07:00",
          "tree_id": "7347f0858bf5a133c953dfdc2841dd4956b738a6",
          "url": "https://github.com/stateset/stateset-agents/commit/11c2891b63779f20558b1ae85925d42ff8fc953b"
        },
        "date": 1786712030106,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8575.544566488306,
            "unit": "iter/sec",
            "range": "stddev: 0.000015610200717325527",
            "extra": "mean: 116.61067029000363 usec\nrounds: 1380"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9197.714779121827,
            "unit": "iter/sec",
            "range": "stddev: 0.000014929082031047195",
            "extra": "mean: 108.72265818352298 usec\nrounds: 2004"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6663.885364890692,
            "unit": "iter/sec",
            "range": "stddev: 0.000018302258797684182",
            "extra": "mean: 150.06260540863957 usec\nrounds: 3254"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 829.3815493207654,
            "unit": "iter/sec",
            "range": "stddev: 0.000027766191169446113",
            "extra": "mean: 1.2057176830361915 msec\nrounds: 672"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 158.26020630358707,
            "unit": "iter/sec",
            "range": "stddev: 0.008068453889352149",
            "extra": "mean: 6.318707799999464 msec\nrounds: 145"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2194521.5920547154,
            "unit": "iter/sec",
            "range": "stddev: 5.0464996174479484e-8",
            "extra": "mean: 455.6801826969982 nsec\nrounds: 53915"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "901113e4f747e3f898e3622986a53ece4111bd0f",
          "message": "chore(release): v0.28.0 — adapter lineage, verified multi-GPU, and serve-remote's real bugs\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-14T06:25:22-07:00",
          "tree_id": "8fb449bc284ec1dddfed2d757fb38afdcca14d6a",
          "url": "https://github.com/stateset/stateset-agents/commit/901113e4f747e3f898e3622986a53ece4111bd0f"
        },
        "date": 1786714076213,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6102.096422099561,
            "unit": "iter/sec",
            "range": "stddev: 0.000014934298482449211",
            "extra": "mean: 163.87810529810145 usec\nrounds: 1510"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6549.84353494936,
            "unit": "iter/sec",
            "range": "stddev: 0.00001852809792092035",
            "extra": "mean: 152.67540280376048 usec\nrounds: 2140"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5051.901667659179,
            "unit": "iter/sec",
            "range": "stddev: 0.000017415003210456923",
            "extra": "mean: 197.94526215775582 usec\nrounds: 3578"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 744.2891093553882,
            "unit": "iter/sec",
            "range": "stddev: 0.000026856707313022888",
            "extra": "mean: 1.3435639288959598 msec\nrounds: 661"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 172.10727903731572,
            "unit": "iter/sec",
            "range": "stddev: 0.0052089276732224426",
            "extra": "mean: 5.81032949677383 msec\nrounds: 155"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2110509.171855096,
            "unit": "iter/sec",
            "range": "stddev: 5.0271773181371326e-8",
            "extra": "mean: 473.81931020987696 nsec\nrounds: 103756"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "9eef1463f131cbae7ac612e5a72365bf41337c83",
          "message": "fix: repair the red CI v0.28.0 shipped with, and close the hole\n\nThree defects, none visible to the checks the release actually ran:\n- customer_support_bench.py: .lower() on Optional[str] the comprehension\n  filtered but did not narrow\n- gpu_verify_rl.py: deliberate None/stub collaborators and a method swap,\n  now annotated as deliberate rather than left as errors\n- scripts/release.py: shebang without the executable bit\n\nThe gap: the release gate ran guard tests and the allowlisted mypy surface,\nwhile CI runs pre-commit over everything and check_types.py --all. make\nrelease now runs both before tagging, so a release cannot outrun the checks\nits own CI will apply.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-14T06:52:33-07:00",
          "tree_id": "3903757be5a26e05310fc6486f317f1fb0d894d8",
          "url": "https://github.com/stateset/stateset-agents/commit/9eef1463f131cbae7ac612e5a72365bf41337c83"
        },
        "date": 1786715700447,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8542.40482472254,
            "unit": "iter/sec",
            "range": "stddev: 0.00001648796633531805",
            "extra": "mean: 117.06305431766755 usec\nrounds: 1436"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9089.665494910256,
            "unit": "iter/sec",
            "range": "stddev: 0.000018901175802603555",
            "extra": "mean: 110.0150495703003 usec\nrounds: 1513"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6686.851618301984,
            "unit": "iter/sec",
            "range": "stddev: 0.000015124088151502785",
            "extra": "mean: 149.54720952129242 usec\nrounds: 3508"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 831.2681956671232,
            "unit": "iter/sec",
            "range": "stddev: 0.000021703328766361498",
            "extra": "mean: 1.202981186111016 msec\nrounds: 720"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 161.76968798370936,
            "unit": "iter/sec",
            "range": "stddev: 0.006145765528939201",
            "extra": "mean: 6.181627797296009 msec\nrounds: 148"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2278752.7143131844,
            "unit": "iter/sec",
            "range": "stddev: 3.556509391782949e-8",
            "extra": "mean: 438.83655901706726 nsec\nrounds: 56510"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "d70d0831e903500de3a330f430ca6e1f4130f3c6",
          "message": "feat(qwen3.8): first-class Qwen3.8-27B starter + hybrid linear-attention LoRA targets\n\nTwo related changes.\n\n1. LoRA inference silently under-adapted hybrid linear-attention models.\n   `_LORA_TARGET_CANDIDATES` in training/sft.py listed only llama/GPT-style\n   projection names. On Qwen/Qwen3.8-27B, whose weight map has 432\n   Mamba-style `linear_attn` tensors against just 96 standard `self_attn`\n   ones, that adapted the minority attention layers and skipped the majority\n   entirely — no error, just a badly under-adapted model. Added\n   `in_proj_qkv`, `in_proj_a`, `in_proj_b`, `in_proj_z`, `out_proj`.\n\n   The existing two-pass vision exclusion needed no change and is pinned by\n   a new test: `out_proj` exists in BOTH the text stack (linear_attn) and\n   the `model.visual.*` tower, so it is kept, while vision-only names are\n   dropped.\n\n2. Added the Qwen3.8 27B first-class starter (`Qwen/Qwen3.8-27B`, released\n   2026-08-05, Apache-2.0, 27.8B params, multimodal, 64 text layers, 256K\n   context) following the thin starter_common pattern: CLI command\n   `qwen3-8-27b`, `init --preset qwen3.8-27b`, driver preset `qwen3.8-27b`,\n   balanced/memory/quality profiles, docs page, and the full doc/README/\n   whitepaper wiring. `Qwen/Qwen3.8-27B-FP8` is a supported variant with a\n   validation warning that it is inference-oriented.\n\n   LoRA targets cover all three groups the weight map shows — standard\n   attention, Mamba-style linear attention, and the per-layer MLP. `conv1d`\n   is excluded (LoRA targets nn.Linear); the vision tower is excluded\n   because text-only SFT sends it no gradient.\n\nNot yet trained on hardware — the README's Live-verified cell is left empty.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-14T10:30:10-07:00",
          "tree_id": "d3ab10d537de15a88b5f4dac15ae0acd6dfde543",
          "url": "https://github.com/stateset/stateset-agents/commit/d70d0831e903500de3a330f430ca6e1f4130f3c6"
        },
        "date": 1786728809364,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8369.44300470457,
            "unit": "iter/sec",
            "range": "stddev: 0.00001524881279061097",
            "extra": "mean: 119.4822641647583 usec\nrounds: 1306"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9091.218192322938,
            "unit": "iter/sec",
            "range": "stddev: 0.000014058175360030779",
            "extra": "mean: 109.99626000005676 usec\nrounds: 2100"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6682.685213521288,
            "unit": "iter/sec",
            "range": "stddev: 0.000015178594600260226",
            "extra": "mean: 149.6404466241607 usec\nrounds: 3466"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 833.9093534625512,
            "unit": "iter/sec",
            "range": "stddev: 0.00001906609164189215",
            "extra": "mean: 1.1991711039668864 msec\nrounds: 731"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 157.58467773261918,
            "unit": "iter/sec",
            "range": "stddev: 0.00783069482449612",
            "extra": "mean: 6.3457946190475685 msec\nrounds: 147"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2179442.103498878,
            "unit": "iter/sec",
            "range": "stddev: 4.149574917137463e-8",
            "extra": "mean: 458.8330189614119 nsec\nrounds: 55534"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "9d208af81080c93fb7449dbcf51a504b9936c03e",
          "message": "chore(release): v0.29.0 — Qwen3.8-27B, fine-tuned the week it shipped\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-14T12:36:17-07:00",
          "tree_id": "8c3058a7bdfcae2ec63ad2fa6306f76ac38d3a93",
          "url": "https://github.com/stateset/stateset-agents/commit/9d208af81080c93fb7449dbcf51a504b9936c03e"
        },
        "date": 1786736355097,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6074.400712762211,
            "unit": "iter/sec",
            "range": "stddev: 0.000015363966817316634",
            "extra": "mean: 164.62529347117606 usec\nrounds: 1455"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6505.602505094272,
            "unit": "iter/sec",
            "range": "stddev: 0.000017992736590818633",
            "extra": "mean: 153.71366437112334 usec\nrounds: 2178"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4849.192765289913,
            "unit": "iter/sec",
            "range": "stddev: 0.00003649916658440733",
            "extra": "mean: 206.21989027903993 usec\nrounds: 3518"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 737.653663660995,
            "unit": "iter/sec",
            "range": "stddev: 0.00002956946307771762",
            "extra": "mean: 1.3556497435896584 msec\nrounds: 663"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 170.18864986365813,
            "unit": "iter/sec",
            "range": "stddev: 0.0057241698711865295",
            "extra": "mean: 5.875832499999982 msec\nrounds: 152"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2093557.1273784568,
            "unit": "iter/sec",
            "range": "stddev: 9.477527099514406e-8",
            "extra": "mean: 477.6559411360298 nsec\nrounds: 103221"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "df0ae2582a0fcbbc827bfa076b74693e9e20faaf",
          "message": "feat(remote): River AI provider — tokenization layer behind an injectable client\n\nRiver is a remote autograd service (you drive forward_backward/optim_step),\nso the valuable half of this integration is the pure tokenization layer:\nremote/river_batches.py turns our chat rows into their\n{input_ids, target_tokens, weights} with prompt tokens weighted 0.0 so loss\nlands only on what the model should say, plus the RL batch shape their\nppo/cispo losses take — which is exactly where our trainers' advantages\nwould plug in.\n\nNOT LIVE-VERIFIED, and labelled so everywhere: river-client is not\ninstallable from PyPI and the account has no credits. The client is\ninjectable (92 tests drive it with fakes) and every assumption is isolated\nand documented — notably whether target_tokens carries the causal shift,\nwhich is one function to flip if wrong.\n\nProbing the live API did establish what the docs omit: there is a REST\nsurface, it takes Bearer auth (401 without), and an unfunded account answers\n402 with 'Billing: insufficient_funds'. Both account states now raise named,\nactionable errors instead of a generic training failure, because no amount\nof retrying fixes either.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-14T13:24:26-07:00",
          "tree_id": "dc1f7e5cd5e9223231fba210abe6a0f03917da04",
          "url": "https://github.com/stateset/stateset-agents/commit/df0ae2582a0fcbbc827bfa076b74693e9e20faaf"
        },
        "date": 1786739226597,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8468.413492250616,
            "unit": "iter/sec",
            "range": "stddev: 0.000014497907353581335",
            "extra": "mean: 118.08587298141414 usec\nrounds: 1362"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9052.312871512146,
            "unit": "iter/sec",
            "range": "stddev: 0.00001532231639191868",
            "extra": "mean: 110.46900545682915 usec\nrounds: 2016"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6615.859990748217,
            "unit": "iter/sec",
            "range": "stddev: 0.00001777679709573001",
            "extra": "mean: 151.15192906113867 usec\nrounds: 2481"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 822.5618292936164,
            "unit": "iter/sec",
            "range": "stddev: 0.000028298499409610782",
            "extra": "mean: 1.2157140829872455 msec\nrounds: 723"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 164.9241986024129,
            "unit": "iter/sec",
            "range": "stddev: 0.005974764328187943",
            "extra": "mean: 6.063391597316331 msec\nrounds: 149"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2261910.2591348453,
            "unit": "iter/sec",
            "range": "stddev: 4.58386384585752e-8",
            "extra": "mean: 442.10418868805544 nsec\nrounds: 106907"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "183530e64a3636dc6a6b50e0dbf551fd5efa0286",
          "message": "chore(release): v0.30.0 — River AI provider (code complete, not live-verified)\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-16T08:46:37-07:00",
          "tree_id": "38c1ce65142ab4c9b6e297130c68651005be772e",
          "url": "https://github.com/stateset/stateset-agents/commit/183530e64a3636dc6a6b50e0dbf551fd5efa0286"
        },
        "date": 1786895364127,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6121.355630528268,
            "unit": "iter/sec",
            "range": "stddev: 0.000015047524332124578",
            "extra": "mean: 163.36250666646873 usec\nrounds: 1500"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6627.380582052421,
            "unit": "iter/sec",
            "range": "stddev: 0.000014852120937455115",
            "extra": "mean: 150.88917674474519 usec\nrounds: 2150"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5077.711575507052,
            "unit": "iter/sec",
            "range": "stddev: 0.00001697837401072863",
            "extra": "mean: 196.93911029204955 usec\nrounds: 3663"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 731.4787343109849,
            "unit": "iter/sec",
            "range": "stddev: 0.00003336612505793633",
            "extra": "mean: 1.3670937418870397 msec\nrounds: 678"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 173.55921158356182,
            "unit": "iter/sec",
            "range": "stddev: 0.005587232749437745",
            "extra": "mean: 5.761722416666661 msec\nrounds: 156"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2133188.0819022707,
            "unit": "iter/sec",
            "range": "stddev: 5.0265894072697365e-8",
            "extra": "mean: 468.7819177708184 nsec\nrounds: 104080"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "ac64d5129f290f29a9e36ffdf013cd7d92f1c739",
          "message": "security: remove a vulnerable mcp pin from the dev lock; fix numpy-stub-sensitive returns\n\nsemgrep pulls the MCP SDK transitively and every release before 1.173.0\nresolves it to mcp 1.23.3, which carries SFTY-20260716-62811 (improper\naccess control, insufficient session validation). It never shipped in the\nwheel — the [mcp] extra users install is correctly pinned >=1.25.0 — so a\nsafety ignore with that justification would have been defensible. Raised the\nsemgrep floor to 1.173.0 instead, which requires mcp==1.29.0: the fix\nremoves the package rather than annotating it, which is the standard this\nrepo already applies to its scanners.\n\nAlso: mappers.py returned Any under CI's numpy stubs after an earlier fix\nremoved a cast that this machine's numpy called redundant. An annotated\nlocal satisfies both stub versions.\n\nBoth failures shared a cause worth naming: local greens are evidence, not\nproof, when dependency versions differ from CI's.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-16T09:17:54-07:00",
          "tree_id": "8653ddb0efe56a12d880a1fdb0de595c8343a9d7",
          "url": "https://github.com/stateset/stateset-agents/commit/ac64d5129f290f29a9e36ffdf013cd7d92f1c739"
        },
        "date": 1786897224034,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8449.122406172666,
            "unit": "iter/sec",
            "range": "stddev: 0.000013363195452521271",
            "extra": "mean: 118.3554873426181 usec\nrounds: 1422"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9087.350072280558,
            "unit": "iter/sec",
            "range": "stddev: 0.000014005453324764744",
            "extra": "mean: 110.04308099127078 usec\nrounds: 1815"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6625.71481316343,
            "unit": "iter/sec",
            "range": "stddev: 0.0000170945476048131",
            "extra": "mean: 150.92711174548012 usec\nrounds: 3499"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 829.8162393083127,
            "unit": "iter/sec",
            "range": "stddev: 0.000025668935464005092",
            "extra": "mean: 1.2050860812672728 msec\nrounds: 726"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 161.95466199727667,
            "unit": "iter/sec",
            "range": "stddev: 0.005396902695078646",
            "extra": "mean: 6.174567546668186 msec\nrounds: 150"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2264287.718649928,
            "unit": "iter/sec",
            "range": "stddev: 4.791213550991658e-8",
            "extra": "mean: 441.63998760557064 nsec\nrounds: 109854"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "491056ce8a4626c5ddce160bb6babc658c8b2f31",
          "message": "feat: the flywheel raises a ceiling — 2/12 -> 10/12 on out-of-distribution compounds\n\nThe experiment the whole week built toward. Gen-1 (trained on one-issue,\ntwo-turn conversations only) passes 2/12 compound requests — it resolves\none issue and silently drops the other. Best-of-8 rejection sampling\nharvested its occasional successes (58/240), and gen-2 trained on only\nthose hits 10/12 on the identical eval, reproduced across two independent\ntrainings with the same two near-miss failures. Untuned base: 0/12.\n$3.32, zero pods left. Full protocol, side-by-side, and limitations in\ndocs/FLYWHEEL_HEADROOM.md.\n\nAlso fixes the product bug the experiment exposed: the eval gate fails a\njob AFTER saving its artifacts, but wait() fetched only on success, so a\nfailed-assertion adapter died with the pod and had to be retrained. Fetch\nis now attempted best-effort on failure; the test that encoded the old\ncontract now encodes the new one.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T05:41:17-07:00",
          "tree_id": "d151fad6f74ac2e8df1c141c32b0fae9975ddfd9",
          "url": "https://github.com/stateset/stateset-agents/commit/491056ce8a4626c5ddce160bb6babc658c8b2f31"
        },
        "date": 1786970661191,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6059.0984889079955,
            "unit": "iter/sec",
            "range": "stddev: 0.000018590145166004322",
            "extra": "mean: 165.0410538515979 usec\nrounds: 1467"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6574.1731002277975,
            "unit": "iter/sec",
            "range": "stddev: 0.000016216240925068834",
            "extra": "mean: 152.11038479734424 usec\nrounds: 2118"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5060.707092513996,
            "unit": "iter/sec",
            "range": "stddev: 0.000018216480380272968",
            "extra": "mean: 197.60084543901795 usec\nrounds: 3442"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 737.0022572400779,
            "unit": "iter/sec",
            "range": "stddev: 0.0000340610259988547",
            "extra": "mean: 1.356847947446992 msec\nrounds: 666"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 187.90949354276754,
            "unit": "iter/sec",
            "range": "stddev: 0.0001507059467144017",
            "extra": "mean: 5.321710899999864 msec\nrounds: 170"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2085390.4264680143,
            "unit": "iter/sec",
            "range": "stddev: 5.0504950891259534e-8",
            "extra": "mean: 479.5265132647994 nsec\nrounds: 104625"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "b389486ac4f7b93e853843cce415ec2b0cdfddc5",
          "message": "feat(serve): the endpoint answered — serve-remote live-verified\n\nAn authenticated POST /v1/chat/completions to the RunPod proxy URL\nreturned a completion from Qwen3.5-0.8B under vLLM on a rented H100.\nTenth attempt; the blocker was flashinfer's `array.array[int]`\nannotation, a TypeError at import on Python 3.11 that killed the\nengine pre-listen. serve-remote now strips it post-install (no-op once\nflashinfer fixes it), with a sequence test pinning install < patch <\nlaunch. README's honesty note becomes a receipt.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T10:05:23-07:00",
          "tree_id": "e689fb9c36c4e077ce432fed70735426db1a7019",
          "url": "https://github.com/stateset/stateset-agents/commit/b389486ac4f7b93e853843cce415ec2b0cdfddc5"
        },
        "date": 1786986510572,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5936.523611663022,
            "unit": "iter/sec",
            "range": "stddev: 0.00001775950720069565",
            "extra": "mean: 168.44875307753827 usec\nrounds: 1462"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6386.973364044816,
            "unit": "iter/sec",
            "range": "stddev: 0.000018251803815901076",
            "extra": "mean: 156.56868175299678 usec\nrounds: 2099"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4906.573971309088,
            "unit": "iter/sec",
            "range": "stddev: 0.00001880131652885352",
            "extra": "mean: 203.80819811286716 usec\nrounds: 2438"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 726.3460118374926,
            "unit": "iter/sec",
            "range": "stddev: 0.00002966945695269783",
            "extra": "mean: 1.376754306766584 msec\nrounds: 665"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.88174044095567,
            "unit": "iter/sec",
            "range": "stddev: 0.00005843776706523972",
            "extra": "mean: 5.498078023531066 msec\nrounds: 170"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 1972400.9879263674,
            "unit": "iter/sec",
            "range": "stddev: 5.162180879992864e-8",
            "extra": "mean: 506.9962984815395 nsec\nrounds: 105065"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "b62758845bbfa330ad3411d3e9814f95e48e97d6",
          "message": "feat(river): align executor and batches to docs.river.ai while the SDK is uninstallable\n\n_open_session prefers the docs' canonical `with client.session(project=...)`\ncontext manager (and closes it), degrading through plain session(),\ncreate_session(), and the client-as-session test seam.\nbuild_sft_batch(shift_targets=False) emits the docs' cross_entropy field\nshape (input_ids + weights, unshifted), so the one unverified assumption —\nwho performs the causal shift — is a single argument to flip on first\ncontact with the real SDK.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T10:25:55-07:00",
          "tree_id": "010078e22c8e0e2f34bf6677b1449e96db5bbf39",
          "url": "https://github.com/stateset/stateset-agents/commit/b62758845bbfa330ad3411d3e9814f95e48e97d6"
        },
        "date": 1786987714082,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5931.350955008809,
            "unit": "iter/sec",
            "range": "stddev: 0.00001702056000304987",
            "extra": "mean: 168.59565511892978 usec\nrounds: 1299"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6393.524894224395,
            "unit": "iter/sec",
            "range": "stddev: 0.000015208564938387555",
            "extra": "mean: 156.40824373787177 usec\nrounds: 2076"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4953.728912005747,
            "unit": "iter/sec",
            "range": "stddev: 0.00001709815524938746",
            "extra": "mean: 201.86813161625022 usec\nrounds: 3533"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 733.5640368066697,
            "unit": "iter/sec",
            "range": "stddev: 0.00002883454711820784",
            "extra": "mean: 1.363207504491594 msec\nrounds: 668"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 183.15146724664265,
            "unit": "iter/sec",
            "range": "stddev: 0.000043726819432438626",
            "extra": "mean: 5.45996171929838 msec\nrounds: 171"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2122033.8780436236,
            "unit": "iter/sec",
            "range": "stddev: 5.0783811544656614e-8",
            "extra": "mean: 471.24601088929575 nsec\nrounds: 100624"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "39c2553971fd3ccf86643069f9d45bd1ed8abae9",
          "message": "fix(serve): backgrounded remote launches must not inherit ssh stdin\n\nWithout < /dev/null a nohup'd remote process holds the ssh session's\nstdin, sshd keeps the channel open until that process exits, and the\nclient blocks on the launch command. Observed live: the CLI hung 28\nminutes on 'echo armed' because the hour-long self-destruct script kept\nthe arm command's channel open, and the pod never got past provisioning.\nAll three backgrounded launches (self-destruct arm, detached installs,\nvllm serve) now redirect stdin; a test asserts it for every nohup'd\ncommand the session issues.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T10:43:52-07:00",
          "tree_id": "b69a82be5984632a4b7ed14f344335dce3e79570",
          "url": "https://github.com/stateset/stateset-agents/commit/39c2553971fd3ccf86643069f9d45bd1ed8abae9"
        },
        "date": 1786988815135,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11100.582935608796,
            "unit": "iter/sec",
            "range": "stddev: 0.000011724929914956005",
            "extra": "mean: 90.0853591023737 usec\nrounds: 1604"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11908.711931884156,
            "unit": "iter/sec",
            "range": "stddev: 0.00001307984520750624",
            "extra": "mean: 83.97213785334914 usec\nrounds: 2655"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8551.984494346678,
            "unit": "iter/sec",
            "range": "stddev: 0.000018596622803817602",
            "extra": "mean: 116.93192388983562 usec\nrounds: 4257"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1080.6284201448925,
            "unit": "iter/sec",
            "range": "stddev: 0.00002062961181772495",
            "extra": "mean: 925.3874702517247 usec\nrounds: 874"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 226.33672253695607,
            "unit": "iter/sec",
            "range": "stddev: 0.00007527062531685891",
            "extra": "mean: 4.418195990430677 msec\nrounds: 209"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2946665.467355263,
            "unit": "iter/sec",
            "range": "stddev: 3.714829987385241e-8",
            "extra": "mean: 339.3666539614134 nsec\nrounds: 139063"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "70c52a9d92eaca66c57531f1a472b62e8a3fdc42",
          "message": "fix(serve): the arm hang was shell precedence, not stdin\n\nIn `chmod && nohup script > log & echo armed` the & backgrounds the\nWHOLE `chmod && nohup` chain; the backgrounded subshell then runs the\nhour-long self-destruct script in its FOREGROUND while holding the ssh\nsession's stdout/stderr, so sshd keeps the channel open until the script\nexits and the client blocks on the arm command for the pod's whole\nlifetime. The previous stdin-redirect fix was necessary but not\nsufficient — reproduced live on a fresh pod, where the hand-run command\nwithout the chmod prefix returned instantly and the full chain hung.\n`(nohup ... &)` scopes the & to the script launch alone; verified live\nat 2.9s round-trip. Test pins the subshell grouping.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T11:00:51-07:00",
          "tree_id": "a8e240de803304f2f4d0d10f918fc30b63977e47",
          "url": "https://github.com/stateset/stateset-agents/commit/70c52a9d92eaca66c57531f1a472b62e8a3fdc42"
        },
        "date": 1786989874418,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11127.43046737924,
            "unit": "iter/sec",
            "range": "stddev: 0.000013225528769000119",
            "extra": "mean: 89.8680070777852 usec\nrounds: 1413"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11907.955625081708,
            "unit": "iter/sec",
            "range": "stddev: 0.000009741415337600776",
            "extra": "mean: 83.97747115329365 usec\nrounds: 1976"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8928.670205725308,
            "unit": "iter/sec",
            "range": "stddev: 0.000009538140719287286",
            "extra": "mean: 111.99876095308936 usec\nrounds: 2602"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1071.4736757057974,
            "unit": "iter/sec",
            "range": "stddev: 0.0000402485909439523",
            "extra": "mean: 933.2940441502527 usec\nrounds: 906"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 263.59146265229094,
            "unit": "iter/sec",
            "range": "stddev: 0.00013283320600104994",
            "extra": "mean: 3.79374957723544 msec\nrounds: 246"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2812554.3119969596,
            "unit": "iter/sec",
            "range": "stddev: 3.599963750550388e-8",
            "extra": "mean: 355.5486895788987 nsec\nrounds: 136445"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "a598ec0c7d5d7b43b5ce8a256444bea0dde1ce18",
          "message": "chore(release): v0.31.0 — The serve claim becomes a receipt: endpoint verified, flywheel raises a ceiling\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T11:39:05-07:00",
          "tree_id": "08a9698ce662e102e297e33846cc094c2816ccb2",
          "url": "https://github.com/stateset/stateset-agents/commit/a598ec0c7d5d7b43b5ce8a256444bea0dde1ce18"
        },
        "date": 1786992107209,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5933.977728938987,
            "unit": "iter/sec",
            "range": "stddev: 0.000018207313692131583",
            "extra": "mean: 168.52102344826343 usec\nrounds: 1450"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6354.988582164059,
            "unit": "iter/sec",
            "range": "stddev: 0.000018188775644286107",
            "extra": "mean: 157.35669499180605 usec\nrounds: 1777"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4929.02676152679,
            "unit": "iter/sec",
            "range": "stddev: 0.000018961480434615983",
            "extra": "mean: 202.87980739026156 usec\nrounds: 3437"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 724.6704326822532,
            "unit": "iter/sec",
            "range": "stddev: 0.0000384882450996928",
            "extra": "mean: 1.3799376308188231 msec\nrounds: 623"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.98547504367332,
            "unit": "iter/sec",
            "range": "stddev: 0.000057007409274610904",
            "extra": "mean: 5.55600389285497 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2133288.559746687,
            "unit": "iter/sec",
            "range": "stddev: 5.169776577153111e-8",
            "extra": "mean: 468.75983815276396 nsec\nrounds: 104298"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "d0f33f54c6836b375d10b1ea5b013de39849380c",
          "message": "feat(flywheel): the improvement loop as one unattended command\n\nstateset-agents flywheel: harvest the current generation's rare\nsuccesses (best-of-N rejection sampling against expect/forbid checks),\ntrain the next generation on nothing but those, measure, repeat — with\nfour stopping rules: plateau, dry harvest, perfect score, and a hard\n--max-cost ceiling checked before each rental. Every generation leaves\nits harvest set, adapter with lineage, and flywheel_report.json.\n\nUnder the hood, RemoteJobSpec gains job_kind=\"harvest\": the executors\nrun stateset_agents.training.harvest exactly as they run sft (prompts\nride the dataset upload; the current adapter ships as a tarball to\n/workspace/current_adapter). The loop reads eval_results.json from a\nFAILED training job's fetched artifacts — 10/12 fails an all-assertions\ngate while being the point, so a gate failure is a score, not an error.\n\n21 new tests: stopping discipline, spec wiring/lineage chaining, budget\ndecay, harvest filtering, executor command shapes for both providers.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T13:01:48-07:00",
          "tree_id": "4a0f2f13ad4a746205bbf3dc89341b7b9373434e",
          "url": "https://github.com/stateset/stateset-agents/commit/d0f33f54c6836b375d10b1ea5b013de39849380c"
        },
        "date": 1786997061097,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6113.747490754766,
            "unit": "iter/sec",
            "range": "stddev: 0.000014321095691086098",
            "extra": "mean: 163.56580011068567 usec\nrounds: 1806"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6619.695797143543,
            "unit": "iter/sec",
            "range": "stddev: 0.000013916404224315355",
            "extra": "mean: 151.06434353547013 usec\nrounds: 1895"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4894.83005128688,
            "unit": "iter/sec",
            "range": "stddev: 0.00004536913258726333",
            "extra": "mean: 204.29718489145378 usec\nrounds: 3786"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 735.9798246928875,
            "unit": "iter/sec",
            "range": "stddev: 0.000024716402933096047",
            "extra": "mean: 1.3587328978987758 msec\nrounds: 666"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 177.3486219751028,
            "unit": "iter/sec",
            "range": "stddev: 0.000554049850681251",
            "extra": "mean: 5.638611616279634 msec\nrounds: 172"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2135853.45799843,
            "unit": "iter/sec",
            "range": "stddev: 4.7840635791613836e-8",
            "extra": "mean: 468.19691503420324 nsec\nrounds: 103972"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "13418c83d4b074e8b875312761d22c7f003b5d3f",
          "message": "feat(serve): multiple adapters on one endpoint, and the deploy command\n\n--adapter is now repeatable as '[name=]path': each adapter is tarred to\n/workspace/<name> and served under its own model name via vLLM's\n--lora-modules, so a champion and challenger can be A/B'd through one\nURL by switching the request's model field. Bare paths keep serving as\n'adapter' — the single-adapter shape is unchanged.\n\nstateset-agents deploy = train-remote then serve-remote, glued: rent,\ntrain, release the hardware, serve the fresh adapter, print URL+token.\nA failed training job refuses to serve.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T13:06:32-07:00",
          "tree_id": "dce84e4adc1d2b176fbbb28b60b139eb948bf510",
          "url": "https://github.com/stateset/stateset-agents/commit/13418c83d4b074e8b875312761d22c7f003b5d3f"
        },
        "date": 1786997355733,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8460.3840898699,
            "unit": "iter/sec",
            "range": "stddev: 0.000014952161673605845",
            "extra": "mean: 118.1979434240293 usec\nrounds: 1361"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9128.16907658998,
            "unit": "iter/sec",
            "range": "stddev: 0.000013535643897511041",
            "extra": "mean: 109.55099446663307 usec\nrounds: 1988"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6627.5246890509325,
            "unit": "iter/sec",
            "range": "stddev: 0.000014604866724208394",
            "extra": "mean: 150.88589585370536 usec\nrounds: 3063"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 814.0859153641078,
            "unit": "iter/sec",
            "range": "stddev: 0.00006691249720034184",
            "extra": "mean: 1.2283715774062256 msec\nrounds: 717"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 173.30425375311594,
            "unit": "iter/sec",
            "range": "stddev: 0.00041822573580388217",
            "extra": "mean: 5.770198817073297 msec\nrounds: 164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2286811.46498963,
            "unit": "iter/sec",
            "range": "stddev: 4.8146489192200637e-8",
            "extra": "mean: 437.290093787655 nsec\nrounds: 108649"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "5fe7958515859c29e4414234bb7cb2c00806d7ef",
          "message": "fix(ci): document deploy+flywheel in the CLI reference; lint; Windows paths\n\nThe repo's own meta-test caught the undocumented commands. The flywheel\ntests compared paths with endswith('gen1/adapter'), which is false under\nWindows separators — compare Path.parts instead (the same lesson as last\ntime, relearned). Three ruff errors (unused loop var, zip without\nstrict, dict() literal) fixed.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T13:33:44-07:00",
          "tree_id": "2f4e70ebba8e1ac3cc30b525e4d4b1a142641590",
          "url": "https://github.com/stateset/stateset-agents/commit/5fe7958515859c29e4414234bb7cb2c00806d7ef"
        },
        "date": 1786999003062,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8404.605814444934,
            "unit": "iter/sec",
            "range": "stddev: 0.000014876547552001605",
            "extra": "mean: 118.9823796710736 usec\nrounds: 1338"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9064.228363932703,
            "unit": "iter/sec",
            "range": "stddev: 0.000013850248020442008",
            "extra": "mean: 110.32378707260739 usec\nrounds: 2104"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6652.961455073356,
            "unit": "iter/sec",
            "range": "stddev: 0.00001555465515621233",
            "extra": "mean: 150.30900250255155 usec\nrounds: 3197"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 821.9172658058898,
            "unit": "iter/sec",
            "range": "stddev: 0.00007950206648111864",
            "extra": "mean: 1.2166674695895336 msec\nrounds: 707"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 177.63169119215317,
            "unit": "iter/sec",
            "range": "stddev: 0.00003970727388218807",
            "extra": "mean: 5.629626072288247 msec\nrounds: 166"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2249808.07761702,
            "unit": "iter/sec",
            "range": "stddev: 6.11765174735558e-8",
            "extra": "mean: 444.4823582726188 nsec\nrounds: 184912"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "77bd864abd867ea0d1bc8c370d8f98d678099785",
          "message": "fix(ci): isort the appended serve tests; type-narrow the flywheel JSON reader\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T14:03:35-07:00",
          "tree_id": "ba5ba7ffe7b3c1534cde96f3afe8f4924c0bf4aa",
          "url": "https://github.com/stateset/stateset-agents/commit/77bd864abd867ea0d1bc8c370d8f98d678099785"
        },
        "date": 1787000801194,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5880.191142957517,
            "unit": "iter/sec",
            "range": "stddev: 0.00001900277802957077",
            "extra": "mean: 170.06249893724328 usec\nrounds: 1411"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6413.989503979956,
            "unit": "iter/sec",
            "range": "stddev: 0.000017818140278808536",
            "extra": "mean: 155.90920430716145 usec\nrounds: 1811"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4940.795442129953,
            "unit": "iter/sec",
            "range": "stddev: 0.000020983993873566212",
            "extra": "mean: 202.3965597670858 usec\nrounds: 3087"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 723.4905497027171,
            "unit": "iter/sec",
            "range": "stddev: 0.00003835608865188593",
            "extra": "mean: 1.3821880609372172 msec\nrounds: 640"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.3147590651705,
            "unit": "iter/sec",
            "range": "stddev: 0.00004079405722496644",
            "extra": "mean: 5.545857727811253 msec\nrounds: 169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2105126.6003354276,
            "unit": "iter/sec",
            "range": "stddev: 1.1043922471090262e-7",
            "extra": "mean: 475.0308127979863 nsec\nrounds: 106644"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "d89ff23189b3c75d09922cf891cd210715669f0a",
          "message": "fix(remote): STATESET_AGENTS_WHEEL env seam — unreleased code on rented GPUs\n\nThe flywheel's first live spin died on the pod with 'No module named\nstateset_agents.training.harvest': the pod installs the PyPI release,\nwhich predated the module. The executor always supported shipping a\nlocal wheel, but only via its constructor — unreachable from the CLI.\nSTATESET_AGENTS_WHEEL now supplies it (explicit argument still wins).\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T14:26:10-07:00",
          "tree_id": "6556289058e2c9b72f5158d7ce7ff00dbc6f69db",
          "url": "https://github.com/stateset/stateset-agents/commit/d89ff23189b3c75d09922cf891cd210715669f0a"
        },
        "date": 1787002130932,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6017.496793135149,
            "unit": "iter/sec",
            "range": "stddev: 0.00001494873235213562",
            "extra": "mean: 166.18205781859578 usec\nrounds: 1522"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6465.649947990114,
            "unit": "iter/sec",
            "range": "stddev: 0.00001539800194025258",
            "extra": "mean: 154.66349215377116 usec\nrounds: 2103"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5006.6569840267075,
            "unit": "iter/sec",
            "range": "stddev: 0.000016239733184655514",
            "extra": "mean: 199.7340746910385 usec\nrounds: 3481"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 726.9281910295269,
            "unit": "iter/sec",
            "range": "stddev: 0.00007701009788205177",
            "extra": "mean: 1.3756516975682695 msec\nrounds: 658"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.93336213715088,
            "unit": "iter/sec",
            "range": "stddev: 0.00006319718900827374",
            "extra": "mean: 5.557613041420126 msec\nrounds: 169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2106187.175607499,
            "unit": "iter/sec",
            "range": "stddev: 5.158676403372845e-8",
            "extra": "mean: 474.79160996769656 nsec\nrounds: 102586"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "eb2f87fe433b49e572c311e7d76d0fc53365c596",
          "message": "fix(harvest): move the model to the GPU before sampling\n\nSingle-GPU loads land on CPU (device_map is multi-GPU-only) and nothing\ndownstream moves the model, so generate() ground on CPU with an H100 at\n0% for an hour — the harvest module's first live run. Same fix and\nsharded-model guard as sft's eval path.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T15:27:44-07:00",
          "tree_id": "233f79bfaf5bf57858b62059e5b6c59c3dda5688",
          "url": "https://github.com/stateset/stateset-agents/commit/eb2f87fe433b49e572c311e7d76d0fc53365c596"
        },
        "date": 1787005818183,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6031.5574181362945,
            "unit": "iter/sec",
            "range": "stddev: 0.00002268544124230743",
            "extra": "mean: 165.79465810821915 usec\nrounds: 1480"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6573.035426782731,
            "unit": "iter/sec",
            "range": "stddev: 0.000014478924971384587",
            "extra": "mean: 152.1367123513991 usec\nrounds: 1846"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4991.396526523749,
            "unit": "iter/sec",
            "range": "stddev: 0.000016538300020976175",
            "extra": "mean: 200.34473211777637 usec\nrounds: 2852"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 723.8105765340687,
            "unit": "iter/sec",
            "range": "stddev: 0.000025211335109855724",
            "extra": "mean: 1.381576937972433 msec\nrounds: 661"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.97129689142133,
            "unit": "iter/sec",
            "range": "stddev: 0.00038846818405887794",
            "extra": "mean: 5.465338099414681 msec\nrounds: 171"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2127926.463187814,
            "unit": "iter/sec",
            "range": "stddev: 4.7737218584019777e-8",
            "extra": "mean: 469.94105167615396 nsec\nrounds: 102902"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "f44085e981e6baa2d5d52e124ad0bce89977d65d",
          "message": "fix(flywheel,remote): salvage failed-job artifacts for real; read the real eval format\n\nThree live-only bugs from the flywheel's first spins, each with the\nfailure it caused:\n\n1. RunPod's _run_attempt returned FAILED before its download step, so an\n   eval-gated job's artifacts (saved BEFORE the gate exits non-zero)\n   died with the pod — wait()'s fetch-on-failure was then defeated by\n   fetch()'s own success-only guard. A trained gen-2 adapter was lost.\n   Failed jobs now salvage artifacts best-effort before the pod dies,\n   and both executors' fetch() accepts any terminal status.\n2. flywheel._eval_score read an imagined {\"results\": [...]} envelope;\n   the real eval_results.json is a bare list with checks.passed nested.\n   Every real score parsed as None. The test fake now writes the real\n   on-disk shape.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T16:02:06-07:00",
          "tree_id": "add9251bdcbc6a1b4008050fd0b74814056050fd",
          "url": "https://github.com/stateset/stateset-agents/commit/f44085e981e6baa2d5d52e124ad0bce89977d65d"
        },
        "date": 1787007898043,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11664.583211264448,
            "unit": "iter/sec",
            "range": "stddev: 0.000008231579950039848",
            "extra": "mean: 85.72959546761203 usec\nrounds: 1765"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12234.596581595366,
            "unit": "iter/sec",
            "range": "stddev: 0.000007293427080277383",
            "extra": "mean: 81.73542898049541 usec\nrounds: 2443"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 9180.265590369236,
            "unit": "iter/sec",
            "range": "stddev: 0.000008248703092900852",
            "extra": "mean: 108.92931039479649 usec\nrounds: 4156"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1095.4707813773903,
            "unit": "iter/sec",
            "range": "stddev: 0.000012965050353222833",
            "extra": "mean: 912.8495410371876 usec\nrounds: 926"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 276.81192013546263,
            "unit": "iter/sec",
            "range": "stddev: 0.00002767282434462987",
            "extra": "mean: 3.612561191406183 msec\nrounds: 256"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2929124.0637407764,
            "unit": "iter/sec",
            "range": "stddev: 2.5963641334243534e-8",
            "extra": "mean: 341.398991042702 nsec\nrounds: 138658"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "0c629f20854291cd2b58771ed466c347b669d898",
          "message": "feat(river): live-verified — first real training run through RiverExecutor\n\nriver-client appeared on PyPI (0.6.2, published 2026-08-17, Python>=3.12,\ngRPC): a real run built 2 chat rows into River's wire format, opened a\nsession, created a LoRA model on Qwen/Qwen3.5-9B, took a training step,\nand saved river://.../sampler_weights/river_out — pointer + lineage\nmanifest written locally. Our blind-written batch shape matched their\nprediction-position contract exactly (target_tokens[i]=t[i+1]).\n\nHardening from first contact, per the SDK's own recovery taxonomy:\ntransient failures (RiverConnectionError — which covers heartbeat loss\nand capacity — and RiverTimeoutError) now back off and rebuild the\nsession up to 3 attempts (observed live: a slow create_model timeout\nraced into ALREADY_EXISTS); auth/model errors still fail fast. The\nexecutor prefers the SDK's pipelined train_step and reads its loss_mean\nmetric, falling back to forward_backward+optim_step for older shapes.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T20:00:20-07:00",
          "tree_id": "c859c0994e983d9c2a0112973f89a71653af2359",
          "url": "https://github.com/stateset/stateset-agents/commit/0c629f20854291cd2b58771ed466c347b669d898"
        },
        "date": 1787022181479,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6004.415732826357,
            "unit": "iter/sec",
            "range": "stddev: 0.000017314209596438706",
            "extra": "mean: 166.5440976268455 usec\nrounds: 1475"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6457.603400354701,
            "unit": "iter/sec",
            "range": "stddev: 0.000016034658441808532",
            "extra": "mean: 154.85621181769577 usec\nrounds: 1997"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4931.896657953801,
            "unit": "iter/sec",
            "range": "stddev: 0.000016653967697230413",
            "extra": "mean: 202.76175057059913 usec\nrounds: 3504"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 725.878686616527,
            "unit": "iter/sec",
            "range": "stddev: 0.00014029613397412898",
            "extra": "mean: 1.3776406697670238 msec\nrounds: 645"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.77480490620766,
            "unit": "iter/sec",
            "range": "stddev: 0.0000582904977873918",
            "extra": "mean: 5.501312464705881 msec\nrounds: 170"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2108620.1020993316,
            "unit": "iter/sec",
            "range": "stddev: 5.7173795541831535e-8",
            "extra": "mean: 474.24379526895575 nsec\nrounds: 103542"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "99817f2c1d41c089a70ff6f06574819a0bd4b974",
          "message": "chore(release): v0.32.0 — The flywheel is a product: replicated ceiling-raise, River live, honest serving\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T20:05:24-07:00",
          "tree_id": "79b4f73349e017d6e3cc8ff5a3b32fe672d2b431",
          "url": "https://github.com/stateset/stateset-agents/commit/99817f2c1d41c089a70ff6f06574819a0bd4b974"
        },
        "date": 1787022514262,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14185.595938789713,
            "unit": "iter/sec",
            "range": "stddev: 0.00000870457050788643",
            "extra": "mean: 70.49404228873857 usec\nrounds: 1608"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 14932.41022494772,
            "unit": "iter/sec",
            "range": "stddev: 0.000008604357870051673",
            "extra": "mean: 66.96842538716827 usec\nrounds: 2647"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11138.33024927888,
            "unit": "iter/sec",
            "range": "stddev: 0.000009117181237100473",
            "extra": "mean: 89.78006376357372 usec\nrounds: 3858"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1326.034812400179,
            "unit": "iter/sec",
            "range": "stddev: 0.000014475803387005245",
            "extra": "mean: 754.1280143241171 usec\nrounds: 1117"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 342.13102018614256,
            "unit": "iter/sec",
            "range": "stddev: 0.000028911869644376312",
            "extra": "mean: 2.922856861841794 msec\nrounds: 304"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3602417.1421325714,
            "unit": "iter/sec",
            "range": "stddev: 3.2197154918179384e-8",
            "extra": "mean: 277.5913950398361 nsec\nrounds: 168153"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "6377621a003c98467fee5e394cf37b470581a54d",
          "message": "chore(release): v0.32.1 — River training effect verified: 3/3 held-out tickets from a river:// checkpoint\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T20:19:10-07:00",
          "tree_id": "924e7ba5f5d7f77d4dff6d42a741fd88ddca5187",
          "url": "https://github.com/stateset/stateset-agents/commit/6377621a003c98467fee5e394cf37b470581a54d"
        },
        "date": 1787023358759,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10795.59706944504,
            "unit": "iter/sec",
            "range": "stddev: 0.000017058045167888098",
            "extra": "mean: 92.6303560208186 usec\nrounds: 1528"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11922.998895908151,
            "unit": "iter/sec",
            "range": "stddev: 0.00001405295520028469",
            "extra": "mean: 83.87151661510173 usec\nrounds: 1926"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8624.476045594953,
            "unit": "iter/sec",
            "range": "stddev: 0.000014498626282426143",
            "extra": "mean: 115.94907269882918 usec\nrounds: 3824"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1073.5917675275157,
            "unit": "iter/sec",
            "range": "stddev: 0.000031413582245525094",
            "extra": "mean: 931.4527460498344 usec\nrounds: 886"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 227.17254540944953,
            "unit": "iter/sec",
            "range": "stddev: 0.00005464448782759992",
            "extra": "mean: 4.401940376191267 msec\nrounds: 210"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2904583.6277714106,
            "unit": "iter/sec",
            "range": "stddev: 3.534720547902019e-8",
            "extra": "mean: 344.28342514870764 nsec\nrounds: 89310"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "3ce849fc98a69fd0c62cd982f364bafd783146f8",
          "message": "feat(serve): --merge — hybrid fine-tunes now actually serve\n\nFolds the (single) adapter into full base weights on the pod with peft\nmerge_and_unload and serves the merged checkpoint — sidestepping vLLM's\nLoRA mapping, which loads hybrid-Qwen3.5 adapters without error and\nsilently serves base weights (the DISPROVEN row in docs/PROOFS.md).\nSame API name 'adapter' with and without --merge. Merge runs detached\n(a 30B merge is a download plus a full-weight save), deps install\nfirst, and multiple adapters with --merge are refused before renting.\nAlso refreshes the CLI reference's stale River unverified note.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T20:24:09-07:00",
          "tree_id": "0d2b72774b1f2c64e7223b1cd10c1ba54f8e86e2",
          "url": "https://github.com/stateset/stateset-agents/commit/3ce849fc98a69fd0c62cd982f364bafd783146f8"
        },
        "date": 1787023635277,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8492.598963718106,
            "unit": "iter/sec",
            "range": "stddev: 0.000013973075858565584",
            "extra": "mean: 117.7495845820788 usec\nrounds: 1401"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9174.853593455982,
            "unit": "iter/sec",
            "range": "stddev: 0.000014347593913810657",
            "extra": "mean: 108.9935648360924 usec\nrounds: 2167"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6718.629681640045,
            "unit": "iter/sec",
            "range": "stddev: 0.000015325993601981653",
            "extra": "mean: 148.83987470431558 usec\nrounds: 2538"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 829.3756716592959,
            "unit": "iter/sec",
            "range": "stddev: 0.000022831371930609997",
            "extra": "mean: 1.205726227777267 msec\nrounds: 720"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 174.46906562069233,
            "unit": "iter/sec",
            "range": "stddev: 0.00006147231782506891",
            "extra": "mean: 5.731675104938478 msec\nrounds: 162"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2307945.1872079386,
            "unit": "iter/sec",
            "range": "stddev: 3.33165122122933e-8",
            "extra": "mean: 433.28585338274894 nsec\nrounds: 57026"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "82c3df6c4d7cab3341a3e486a1238a1ac247eed3",
          "message": "fix(serve): --merge honors STATESET_AGENTS_WHEEL — unreleased merge module on pods\n\nCaught before the pod did this time: the merge deps install pinned the\nPyPI release, which cannot contain a module that has not been released.\nSame seam as the training executor.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T20:25:01-07:00",
          "tree_id": "dfc36961237016fff55939ce94d5fa708eb86a4d",
          "url": "https://github.com/stateset/stateset-agents/commit/82c3df6c4d7cab3341a3e486a1238a1ac247eed3"
        },
        "date": 1787023725019,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6024.098650677425,
            "unit": "iter/sec",
            "range": "stddev: 0.000014921641009933926",
            "extra": "mean: 165.99993758195635 usec\nrounds: 1522"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6411.43127986187,
            "unit": "iter/sec",
            "range": "stddev: 0.000015154237440628768",
            "extra": "mean: 155.97141361258485 usec\nrounds: 2101"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4964.239113441788,
            "unit": "iter/sec",
            "range": "stddev: 0.000018368535652930293",
            "extra": "mean: 201.440739889478 usec\nrounds: 3264"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 724.1907965062622,
            "unit": "iter/sec",
            "range": "stddev: 0.00003761566911458785",
            "extra": "mean: 1.3808515722988104 msec\nrounds: 657"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 178.59031134568124,
            "unit": "iter/sec",
            "range": "stddev: 0.00034677725973032643",
            "extra": "mean: 5.59940789881031 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2098466.047590606,
            "unit": "iter/sec",
            "range": "stddev: 1.3595339836101718e-7",
            "extra": "mean: 476.53856546698444 nsec\nrounds: 107101"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "96954441e82b010314f5eec8d79e81c4a519d755",
          "message": "feat(serve): self-verifying — every adapter serve probes its own effect\n\nAfter readiness, serve-remote runs a greedy base-vs-adapter completion\nthrough the live endpoint for every served adapter: byte-identical\noutput (the silent no-op that once survived a 'successful'\nverification — docs/PROOFS.md 2026-08-18) warns loudly, or fails and\nterminates the pod with --strict. --merge verifies itself on the pod:\na pre-vs-post-merge greedy probe recorded in merge_probe.json, refusing\nto serve a merge with no observable effect. Probe transport is\nbest-effort; a completed comparison that finds no effect never is.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T20:33:13-07:00",
          "tree_id": "506be101c9782ab608b944fde4d2c9705515d177",
          "url": "https://github.com/stateset/stateset-agents/commit/96954441e82b010314f5eec8d79e81c4a519d755"
        },
        "date": 1787024180032,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6025.159246975492,
            "unit": "iter/sec",
            "range": "stddev: 0.000015525607828684725",
            "extra": "mean: 165.970716956897 usec\nrounds: 1498"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6418.241755429255,
            "unit": "iter/sec",
            "range": "stddev: 0.00003698707814991265",
            "extra": "mean: 155.8059104199511 usec\nrounds: 2121"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4972.298777235607,
            "unit": "iter/sec",
            "range": "stddev: 0.000020897738786792376",
            "extra": "mean: 201.11422197279114 usec\nrounds: 3568"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 712.4825067085213,
            "unit": "iter/sec",
            "range": "stddev: 0.00014395774502503044",
            "extra": "mean: 1.4035432317064631 msec\nrounds: 656"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.28686230739717,
            "unit": "iter/sec",
            "range": "stddev: 0.00005779405267704065",
            "extra": "mean: 5.546715868264184 msec\nrounds: 167"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2093562.6872696201,
            "unit": "iter/sec",
            "range": "stddev: 5.6096194377549826e-8",
            "extra": "mean: 477.6546726213288 nsec\nrounds: 102052"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "7b7622e131a080bd1f04894a1e7873bbebe28b80",
          "message": "feat(harvest): judge-gated sampling — semantic success criteria\n\nHarvest specs accept judge/min_judge_score alongside or instead of\nexpect/forbid. Substrings short-circuit first; the judge gates\nsemantically on top; an unavailable judge rejects rather than passes.\nFirst step toward flywheels on real transcripts, where success is not\na substring.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T20:34:41-07:00",
          "tree_id": "0637688fae972967ad37abe7042c11e96b44d322",
          "url": "https://github.com/stateset/stateset-agents/commit/7b7622e131a080bd1f04894a1e7873bbebe28b80"
        },
        "date": 1787024353102,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6003.188492975227,
            "unit": "iter/sec",
            "range": "stddev: 0.000016641307426401453",
            "extra": "mean: 166.57814445942745 usec\nrounds: 1516"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6524.976055283745,
            "unit": "iter/sec",
            "range": "stddev: 0.00001443493485251487",
            "extra": "mean: 153.25726738724623 usec\nrounds: 2128"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4980.644502652076,
            "unit": "iter/sec",
            "range": "stddev: 0.000018728109003303583",
            "extra": "mean: 200.7772286232281 usec\nrounds: 3298"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 710.7008831604238,
            "unit": "iter/sec",
            "range": "stddev: 0.00016917548797492195",
            "extra": "mean: 1.4070617100587925 msec\nrounds: 676"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.13228388398585,
            "unit": "iter/sec",
            "range": "stddev: 0.0001339643900058082",
            "extra": "mean: 5.5514757179454275 msec\nrounds: 156"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2106155.095163918,
            "unit": "iter/sec",
            "range": "stddev: 4.8919898967512e-8",
            "extra": "mean: 474.7988418783432 nsec\nrounds: 105731"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "e2d62ac22fa34cc0377f4b8518fb41a6f2f9c75f",
          "message": "feat(flywheel): --repeats N — 'reproduced' becomes a distribution\n\nRuns the whole loop N times under one shared budget; per-run best\nscores plus min/mean/max land in flywheel_repeats_report.json, and a\nrepeat that would start with no budget left is skipped loudly.\nMotivated by the live spread: two identical runs scored 7/12 and 11/12.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T20:36:36-07:00",
          "tree_id": "280653ff72f3e6a31b17736e7febde5442d5e18a",
          "url": "https://github.com/stateset/stateset-agents/commit/e2d62ac22fa34cc0377f4b8518fb41a6f2f9c75f"
        },
        "date": 1787024357443,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8521.994994381508,
            "unit": "iter/sec",
            "range": "stddev: 0.000014522187237606163",
            "extra": "mean: 117.34341555695504 usec\nrounds: 1427"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9250.021251847267,
            "unit": "iter/sec",
            "range": "stddev: 0.000013827284885638534",
            "extra": "mean: 108.1078597306245 usec\nrounds: 2153"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6631.98478003131,
            "unit": "iter/sec",
            "range": "stddev: 0.000016751495453609254",
            "extra": "mean: 150.78442324098322 usec\nrounds: 3511"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 824.5329343541289,
            "unit": "iter/sec",
            "range": "stddev: 0.000022046858974700822",
            "extra": "mean: 1.2128078313613 msec\nrounds: 676"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 175.58802516780705,
            "unit": "iter/sec",
            "range": "stddev: 0.00004850416232445149",
            "extra": "mean: 5.695149193940269 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2265167.2892573727,
            "unit": "iter/sec",
            "range": "stddev: 3.7598665041897155e-8",
            "extra": "mean: 441.4684975995069 nsec\nrounds: 56606"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "0bd21397fb9edcf6f0d6c764deb840d24743c024",
          "message": "feat(river): the zero-infrastructure flywheel — harvest via River sampling\n\njob_kind=\"harvest\" on the River executor: create a model from the\nprevious generation's river:// checkpoint pointer (the flywheel's\nadapter reference), model.sample best-of-N in-session with chat\nmessage rendering server-side, filter with the shared check/judge\nlogic, and write the exact artifacts the pod harvest writes. Training\nnow also greedy-scores itself in-session when eval_prompts are set,\nwriting eval_results.json in the sft shape — run_flywheel reads River\ngenerations exactly like pod ones. No pods, no SSH: the loop is an API\nkey and a laptop.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T20:51:25-07:00",
          "tree_id": "10d9d58fdc2af80fb53b5ea5ad6a7b81b43f0762",
          "url": "https://github.com/stateset/stateset-agents/commit/0bd21397fb9edcf6f0d6c764deb840d24743c024"
        },
        "date": 1787025291347,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10687.833474087422,
            "unit": "iter/sec",
            "range": "stddev: 0.000011237519000966514",
            "extra": "mean: 93.56433204395381 usec\nrounds: 1551"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11330.889049868238,
            "unit": "iter/sec",
            "range": "stddev: 0.000010333657306587122",
            "extra": "mean: 88.25432811131698 usec\nrounds: 2234"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8364.184833728707,
            "unit": "iter/sec",
            "range": "stddev: 0.000012152172158489758",
            "extra": "mean: 119.55737706410842 usec\nrounds: 3270"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1000.857700862708,
            "unit": "iter/sec",
            "range": "stddev: 0.000058058933911110344",
            "extra": "mean: 999.1430341576342 usec\nrounds: 849"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 258.94959899689235,
            "unit": "iter/sec",
            "range": "stddev: 0.0001796177712781422",
            "extra": "mean: 3.8617553526777266 msec\nrounds: 224"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2657186.5937044127,
            "unit": "iter/sec",
            "range": "stddev: 7.482554251389112e-8",
            "extra": "mean: 376.3378915012096 nsec\nrounds: 124425"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "8c65df802eaa1ae8252b45d44ac34a726f77c8b9",
          "message": "fix(river): type harvest checkpoints as inference-mode\n\ncreate_model given a bare river:// path tries to restore optimizer\nstate, which an inference-mode save does not carry — observed live:\n'Cannot load optimizer from an inference checkpoint'. Wrapping the URI\nas Checkpoint(checkpoint_type='inference') tells the server weights-only\nis intended; clients without a Checkpoint type get the bare URI as\nbefore.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T20:52:30-07:00",
          "tree_id": "c61cfa58d5015e11884c74c89cdc46eb09052a2e",
          "url": "https://github.com/stateset/stateset-agents/commit/8c65df802eaa1ae8252b45d44ac34a726f77c8b9"
        },
        "date": 1787025320964,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11247.827607074316,
            "unit": "iter/sec",
            "range": "stddev: 0.00000934487485945305",
            "extra": "mean: 88.90605679011745 usec\nrounds: 1620"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11618.974456040603,
            "unit": "iter/sec",
            "range": "stddev: 0.000009723501073397975",
            "extra": "mean: 86.06611571300157 usec\nrounds: 2221"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8686.993359708673,
            "unit": "iter/sec",
            "range": "stddev: 0.000011997524735856431",
            "extra": "mean: 115.1146269592102 usec\nrounds: 3509"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1030.9729326548509,
            "unit": "iter/sec",
            "range": "stddev: 0.000041996565242300844",
            "extra": "mean: 969.9575695211583 usec\nrounds: 899"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 266.14727595292277,
            "unit": "iter/sec",
            "range": "stddev: 0.00012254754513233397",
            "extra": "mean: 3.7573181856532853 msec\nrounds: 237"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2806036.408251612,
            "unit": "iter/sec",
            "range": "stddev: 3.289382326771932e-8",
            "extra": "mean: 356.3745634444853 nsec\nrounds: 135264"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "1ac172f92e92853d63b81f46f74f9f9a3f4a2a15",
          "message": "fix(river): render sample prompts with the SDK's renderers, thinking off\n\nmodel.sample takes raw text; model_input message dicts are a multimodal\nparts format ('must be a dict with a type field', observed live). The\nSDK's per-family renderers apply the chat template client-side with\nthinking disabled (the Nemotron budget lesson) and supply stop strings;\nunknown models fall back to raw text. Also restores the module logger\nthat the harvest path referenced without defining.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T20:55:20-07:00",
          "tree_id": "9875dc57c71455efb0c95cd105e132910f4991a2",
          "url": "https://github.com/stateset/stateset-agents/commit/1ac172f92e92853d63b81f46f74f9f9a3f4a2a15"
        },
        "date": 1787025535454,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5585.51426163401,
            "unit": "iter/sec",
            "range": "stddev: 0.00001675085810607561",
            "extra": "mean: 179.03454420819182 usec\nrounds: 1459"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6028.9685191498775,
            "unit": "iter/sec",
            "range": "stddev: 0.000016927710597751352",
            "extra": "mean: 165.86585198175928 usec\nrounds: 1993"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4639.715523335686,
            "unit": "iter/sec",
            "range": "stddev: 0.000018284608209086196",
            "extra": "mean: 215.53045547091173 usec\nrounds: 3144"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 671.3045761644383,
            "unit": "iter/sec",
            "range": "stddev: 0.00003380065966761381",
            "extra": "mean: 1.4896367990124448 msec\nrounds: 607"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 173.7635158200047,
            "unit": "iter/sec",
            "range": "stddev: 0.00005428650048228949",
            "extra": "mean: 5.75494801242318 msec\nrounds: 161"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2054144.914866097,
            "unit": "iter/sec",
            "range": "stddev: 5.455636841261645e-8",
            "extra": "mean: 486.82057081897096 nsec\nrounds: 99612"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "b6ce802692538c91edf3e6cdd28521bd087432b2",
          "message": "fix(serve): merge in an isolated venv; surface engine root causes\n\nInstalling the training stack into vLLM's environment downgraded its\ntransformers and crashed the engine at boot — a 30-minute readiness\ntimeout whose root cause had scrolled off the 30-line tail. The merge\nnow runs in its own venv (--system-site-packages keeps torch), leaving\nvLLM's deps untouched, and readiness-timeout evidence includes grepped\nERROR lines from the whole log, not just the tail.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T21:08:51-07:00",
          "tree_id": "b81242eaf2b43187f7dd3f40803af302d1b4edad",
          "url": "https://github.com/stateset/stateset-agents/commit/b6ce802692538c91edf3e6cdd28521bd087432b2"
        },
        "date": 1787026295196,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6049.016061504622,
            "unit": "iter/sec",
            "range": "stddev: 0.000015146188290653921",
            "extra": "mean: 165.31614230021762 usec\nrounds: 1539"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6475.099448595461,
            "unit": "iter/sec",
            "range": "stddev: 0.000014463034349214642",
            "extra": "mean: 154.43778245242456 usec\nrounds: 1949"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4997.777682483631,
            "unit": "iter/sec",
            "range": "stddev: 0.000016664176356704368",
            "extra": "mean: 200.08893222778428 usec\nrounds: 3497"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 732.2150408493212,
            "unit": "iter/sec",
            "range": "stddev: 0.00003028434442223045",
            "extra": "mean: 1.3657190090496718 msec\nrounds: 663"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.42944926491685,
            "unit": "iter/sec",
            "range": "stddev: 0.00005011036950203532",
            "extra": "mean: 5.511784355029571 msec\nrounds: 169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2145148.0876382976,
            "unit": "iter/sec",
            "range": "stddev: 1.0792922480805623e-7",
            "extra": "mean: 466.16828262935957 nsec\nrounds: 104200"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "4651989ec493189cb68197cb0eb993516aeffaf3",
          "message": "fix(merge): save processor artifacts; tail the engine's error evidence\n\nThe Qwen3.5 family is a composite multimodal arch: a merged save of\nmodel+tokenizer alone lacks the processor files, and vLLM's engine dies\nat boot loading the directory (observed live: Qwen3-VL video-processor\nerrors). AutoProcessor is saved alongside when the base has one;\ntext-only models skip. Readiness-failure evidence now tails the ERROR\nblock — the root exception sits at its end, and a head cut it off at\nexactly the interesting frame.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T21:51:38-07:00",
          "tree_id": "20785c72fcf00c9807fe03a0f31c07c6fd349464",
          "url": "https://github.com/stateset/stateset-agents/commit/4651989ec493189cb68197cb0eb993516aeffaf3"
        },
        "date": 1787028883623,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5918.423794746536,
            "unit": "iter/sec",
            "range": "stddev: 0.00001961684037869893",
            "extra": "mean: 168.96390570875403 usec\nrounds: 1559"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6097.431943264755,
            "unit": "iter/sec",
            "range": "stddev: 0.00003698486999460141",
            "extra": "mean: 164.00347052739204 usec\nrounds: 1934"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5006.102074445409,
            "unit": "iter/sec",
            "range": "stddev: 0.000017086815441784926",
            "extra": "mean: 199.75621454158681 usec\nrounds: 3631"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 726.6931629166387,
            "unit": "iter/sec",
            "range": "stddev: 0.00003096453338256094",
            "extra": "mean: 1.3760966127525176 msec\nrounds: 643"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 183.5189655197366,
            "unit": "iter/sec",
            "range": "stddev: 0.0002590483212928747",
            "extra": "mean: 5.449028100000131 msec\nrounds: 170"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2130688.549699366,
            "unit": "iter/sec",
            "range": "stddev: 5.200558935221663e-8",
            "extra": "mean: 469.33185056121744 nsec\nrounds: 104298"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "20a5f8bb1462eb9c0fdfaf1219fd024c0e206fc3",
          "message": "fix(lint): strict zips and a dict literal in the river harvest\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T21:52:17-07:00",
          "tree_id": "af300ef1727888ed6c3826743a3df75adbceccb6",
          "url": "https://github.com/stateset/stateset-agents/commit/20a5f8bb1462eb9c0fdfaf1219fd024c0e206fc3"
        },
        "date": 1787028894132,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5831.797678523639,
            "unit": "iter/sec",
            "range": "stddev: 0.00004114211169918293",
            "extra": "mean: 171.47371275972614 usec\nrounds: 1591"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6190.762471988395,
            "unit": "iter/sec",
            "range": "stddev: 0.00007249632446667932",
            "extra": "mean: 161.5309914610264 usec\nrounds: 2225"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5134.042469027167,
            "unit": "iter/sec",
            "range": "stddev: 0.000017229746861389567",
            "extra": "mean: 194.77828748648562 usec\nrounds: 3788"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 739.0680574189346,
            "unit": "iter/sec",
            "range": "stddev: 0.000029911758321103728",
            "extra": "mean: 1.3530553647418135 msec\nrounds: 658"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 183.84410811773716,
            "unit": "iter/sec",
            "range": "stddev: 0.000046498845251669365",
            "extra": "mean: 5.439391070175508 msec\nrounds: 171"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2110116.500480164,
            "unit": "iter/sec",
            "range": "stddev: 5.114443434005143e-8",
            "extra": "mean: 473.90748319936216 nsec\nrounds: 102376"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "73ea73b690b3ebd94753f6bc7f51b7a0c7e8c263",
          "message": "fix(merge): load composites as themselves; remap text-trained adapter keys\n\nTwo measured facts from merge attempt 4's corpse and a CPU repro:\nAutoModelForCausalLM extracts the text model from composite checkpoints\nand saves a config with model_type=qwen3_5_text and architectures:None\n(vLLM then guesses and dies), and an adapter trained through that\nextraction silently no-ops on the composite — probe delta exactly 0.0,\npeft warning only. The merge now loads the checkpoint in its own\narchitecture and remaps adapter keys (model.layers.* ->\nmodel.language_model.layers.*) when — and only when — the composite\nspelling exists on the model; measured 372/372 keys with real deltas.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T22:33:41-07:00",
          "tree_id": "f0fd7eebc7a10ecbab02cf457ebd6e20a562d6e1",
          "url": "https://github.com/stateset/stateset-agents/commit/73ea73b690b3ebd94753f6bc7f51b7a0c7e8c263"
        },
        "date": 1787031379660,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6131.2286819439,
            "unit": "iter/sec",
            "range": "stddev: 0.00001541039321773831",
            "extra": "mean: 163.09944578399757 usec\nrounds: 1494"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6568.3712707245495,
            "unit": "iter/sec",
            "range": "stddev: 0.000015798865383152534",
            "extra": "mean: 152.24474360288272 usec\nrounds: 1837"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5060.0232095728215,
            "unit": "iter/sec",
            "range": "stddev: 0.00001689516029276683",
            "extra": "mean: 197.62755200571942 usec\nrounds: 3663"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 746.5227557577252,
            "unit": "iter/sec",
            "range": "stddev: 0.000027148876540726136",
            "extra": "mean: 1.3395438950618375 msec\nrounds: 648"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 185.9057159004305,
            "unit": "iter/sec",
            "range": "stddev: 0.00005064593307882908",
            "extra": "mean: 5.379070757219705 msec\nrounds: 173"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2175902.008271096,
            "unit": "iter/sec",
            "range": "stddev: 3.786183670080232e-8",
            "extra": "mean: 459.5795197572197 nsec\nrounds: 73282"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "f29f3e2d18c9fe77425ead3e49c78a0a9ee64821",
          "message": "fix(runpod): retry transient 5xx from RunPod's REST API\n\nTheir /v1/pods intermittently answers 500 (observed repeatedly live;\none killed a whole serve attempt mid-provisioning). A 5xx is their\ninfrastructure hiccuping, not our request being wrong — pod create/get/\nlist now retry briefly with backoff; 4xx still raises immediately.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T22:34:37-07:00",
          "tree_id": "63be7dd705bf84761202590abac78138351be587",
          "url": "https://github.com/stateset/stateset-agents/commit/f29f3e2d18c9fe77425ead3e49c78a0a9ee64821"
        },
        "date": 1787031469030,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6048.596865729901,
            "unit": "iter/sec",
            "range": "stddev: 0.00001524333375766633",
            "extra": "mean: 165.32759947448196 usec\nrounds: 1523"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6559.093083929404,
            "unit": "iter/sec",
            "range": "stddev: 0.000014624368589301614",
            "extra": "mean: 152.4601019080099 usec\nrounds: 2149"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5046.72176663269,
            "unit": "iter/sec",
            "range": "stddev: 0.000015402870185896734",
            "extra": "mean: 198.14843104917736 usec\nrounds: 3343"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 738.2586967743641,
            "unit": "iter/sec",
            "range": "stddev: 0.00005652731324814861",
            "extra": "mean: 1.3545387333318912 msec\nrounds: 675"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.6313205246346,
            "unit": "iter/sec",
            "range": "stddev: 0.00006058608471430116",
            "extra": "mean: 5.475512070587657 msec\nrounds: 170"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2179847.822290413,
            "unit": "iter/sec",
            "range": "stddev: 5.151239366697733e-8",
            "extra": "mean: 458.7476197991098 nsec\nrounds: 104844"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "ef2034ec3cc127754ff519e0f6a3a8c67f1cf1d5",
          "message": "fix(types): Any returns in merge loader; narrow the retry re-raise\n\nCaught by CI's mypy --all pass (the targeted local run misses these\nmodules) — the lesson, again: gate with the same strictness CI uses.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-17T23:35:28-07:00",
          "tree_id": "11a2ccae29884cea06907e2838439bec6a599f44",
          "url": "https://github.com/stateset/stateset-agents/commit/ef2034ec3cc127754ff519e0f6a3a8c67f1cf1d5"
        },
        "date": 1787035116642,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5978.431816370174,
            "unit": "iter/sec",
            "range": "stddev: 0.00003729669928993789",
            "extra": "mean: 167.2679442896371 usec\nrounds: 1436"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6539.651731731284,
            "unit": "iter/sec",
            "range": "stddev: 0.00001591467863855659",
            "extra": "mean: 152.91334172244422 usec\nrounds: 2066"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5025.325792272796,
            "unit": "iter/sec",
            "range": "stddev: 0.00001655008251013919",
            "extra": "mean: 198.99207361593398 usec\nrounds: 3070"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 737.8986561631651,
            "unit": "iter/sec",
            "range": "stddev: 0.000023903008966618007",
            "extra": "mean: 1.355199649230529 msec\nrounds: 650"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.0230252745895,
            "unit": "iter/sec",
            "range": "stddev: 0.00005831514655238339",
            "extra": "mean: 5.493810458822215 msec\nrounds: 170"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2086830.9180183713,
            "unit": "iter/sec",
            "range": "stddev: 5.122806895400291e-8",
            "extra": "mean: 479.1955071039428 nsec\nrounds: 103864"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "003c3b84f74bb5dcc4dbc659696845c73c68474c",
          "message": "chore(release): v0.33.0 — Nothing rented, nothing unverified: the zero-infrastructure flywheel and self-verifying serving\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-18T04:35:37-07:00",
          "tree_id": "b756f8e6a35b5373b7d49eb7625f654075458fc4",
          "url": "https://github.com/stateset/stateset-agents/commit/003c3b84f74bb5dcc4dbc659696845c73c68474c"
        },
        "date": 1787053127487,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10858.080195407285,
            "unit": "iter/sec",
            "range": "stddev: 0.000012983508042492032",
            "extra": "mean: 92.09731204812584 usec\nrounds: 1660"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11776.46896867843,
            "unit": "iter/sec",
            "range": "stddev: 0.000011772279601288677",
            "extra": "mean: 84.91509659301732 usec\nrounds: 2671"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8613.018624265897,
            "unit": "iter/sec",
            "range": "stddev: 0.000015625833205136383",
            "extra": "mean: 116.10331332416361 usec\nrounds: 3565"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1038.6948595094173,
            "unit": "iter/sec",
            "range": "stddev: 0.00012377000095811037",
            "extra": "mean: 962.7466534996688 usec\nrounds: 886"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 227.13129684312275,
            "unit": "iter/sec",
            "range": "stddev: 0.00005111127130617422",
            "extra": "mean: 4.402739798076747 msec\nrounds: 208"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2907532.801733644,
            "unit": "iter/sec",
            "range": "stddev: 5.931728193421038e-8",
            "extra": "mean: 343.93421095842507 nsec\nrounds: 140037"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "61937e371c0265f89965e9aaada4256df548ad15",
          "message": "chore(release): v0.33.1 — First perfect score: the 35B MoE maxes the eval in one wheel-turn\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-18T07:14:44-07:00",
          "tree_id": "5980bfea9e5b1adf9a30662831768d2b142f56f1",
          "url": "https://github.com/stateset/stateset-agents/commit/61937e371c0265f89965e9aaada4256df548ad15"
        },
        "date": 1787062694071,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 13872.59976226194,
            "unit": "iter/sec",
            "range": "stddev: 0.000026373717726244715",
            "extra": "mean: 72.0845419847209 usec\nrounds: 1703"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 15069.069487256225,
            "unit": "iter/sec",
            "range": "stddev: 0.000007852201756008127",
            "extra": "mean: 66.36109819824581 usec\nrounds: 2719"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11213.387742883948,
            "unit": "iter/sec",
            "range": "stddev: 0.000008231346801387891",
            "extra": "mean: 89.17911544034526 usec\nrounds: 4825"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1326.264327133514,
            "unit": "iter/sec",
            "range": "stddev: 0.000012313552657028567",
            "extra": "mean: 753.9975098035874 usec\nrounds: 1122"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 344.4167933036497,
            "unit": "iter/sec",
            "range": "stddev: 0.000028354875538128512",
            "extra": "mean: 2.903458888888631 msec\nrounds: 306"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3629483.5524701076,
            "unit": "iter/sec",
            "range": "stddev: 2.827465003763548e-8",
            "extra": "mean: 275.5212926421537 nsec\nrounds: 172414"
          }
        ]
      }
    ]
  }
}