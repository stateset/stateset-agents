window.BENCHMARK_DATA = {
  "lastUpdate": 1788632003158,
  "repoUrl": "https://github.com/stateset/stateset-agents",
  "entries": {
    "Python Benchmark": [
      {
        "commit": {
          "author": {
            "name": "stateset",
            "username": "stateset"
          },
          "committer": {
            "name": "stateset",
            "username": "stateset"
          },
          "id": "4a4679631cec75f62db4ab4e48728dba01b236ab",
          "message": "chore(release): v0.50.0 — Per-token GRPO and starter specs",
          "timestamp": "2026-09-05T16:55:32Z",
          "url": "https://github.com/stateset/stateset-agents/pull/65/commits/4a4679631cec75f62db4ab4e48728dba01b236ab"
        },
        "date": 1788631563076,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5541.568027551414,
            "unit": "iter/sec",
            "range": "stddev: 0.000038822677329112836",
            "extra": "mean: 180.45433982371557 usec\nrounds: 2054"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6255.361181187295,
            "unit": "iter/sec",
            "range": "stddev: 0.000030777806367361244",
            "extra": "mean: 159.86287138901795 usec\nrounds: 1454"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5082.370827217935,
            "unit": "iter/sec",
            "range": "stddev: 0.000015868553466872903",
            "extra": "mean: 196.7585668178005 usec\nrounds: 3315"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 748.6682700895639,
            "unit": "iter/sec",
            "range": "stddev: 0.00009005000148574983",
            "extra": "mean: 1.33570506451458 msec\nrounds: 589"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.05037917168914,
            "unit": "iter/sec",
            "range": "stddev: 0.000049280211522385486",
            "extra": "mean: 5.5540010779229645 msec\nrounds: 154"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2199058.944888579,
            "unit": "iter/sec",
            "range": "stddev: 4.7130340265045816e-8",
            "extra": "mean: 454.73997062441975 nsec\nrounds: 105854"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "stateset",
            "username": "stateset"
          },
          "committer": {
            "name": "stateset",
            "username": "stateset"
          },
          "id": "6a8fae4d14fe3e08e6df1f6afb529921b660ea4d",
          "message": "chore(release): v0.50.0 — Per-token GRPO and starter specs",
          "timestamp": "2026-09-05T16:55:32Z",
          "url": "https://github.com/stateset/stateset-agents/pull/65/commits/6a8fae4d14fe3e08e6df1f6afb529921b660ea4d"
        },
        "date": 1788632001680,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5950.953329006719,
            "unit": "iter/sec",
            "range": "stddev: 0.000019869390464646163",
            "extra": "mean: 168.04030290839322 usec\nrounds: 1753"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6403.00858684228,
            "unit": "iter/sec",
            "range": "stddev: 0.00005571583876352474",
            "extra": "mean: 156.17658268566555 usec\nrounds: 1802"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5033.408312914843,
            "unit": "iter/sec",
            "range": "stddev: 0.000019547882711639566",
            "extra": "mean: 198.6725371423128 usec\nrounds: 3500"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 732.4915362439338,
            "unit": "iter/sec",
            "range": "stddev: 0.00004032170183014619",
            "extra": "mean: 1.3652034877123562 msec\nrounds: 529"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.0943992763935,
            "unit": "iter/sec",
            "range": "stddev: 0.00004431696061729988",
            "extra": "mean: 5.521981927634108 msec\nrounds: 152"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2240169.1836009137,
            "unit": "iter/sec",
            "range": "stddev: 5.802808239810752e-8",
            "extra": "mean: 446.3948559423403 nsec\nrounds: 109087"
          }
        ]
      }
    ]
  }
}