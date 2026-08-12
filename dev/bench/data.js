window.BENCHMARK_DATA = {
  "lastUpdate": 1786536970567,
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
      }
    ]
  }
}