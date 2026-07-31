window.BENCHMARK_DATA = {
  "lastUpdate": 1785491513132,
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
      }
    ]
  }
}