window.BENCHMARK_DATA = {
  "lastUpdate": 1786009551111,
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
      }
    ]
  }
}