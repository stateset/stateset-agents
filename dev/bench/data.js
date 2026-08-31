window.BENCHMARK_DATA = {
  "lastUpdate": 1788210391472,
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
          "id": "23ebea85a9a7c5ff827ba04faf029b93ed190dcd",
          "message": "chore(release): v0.45.0 — managed provider expansion",
          "timestamp": "2026-08-31T19:43:08Z",
          "url": "https://github.com/stateset/stateset-agents/pull/48/commits/23ebea85a9a7c5ff827ba04faf029b93ed190dcd"
        },
        "date": 1788210390189,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 12011.845502133754,
            "unit": "iter/sec",
            "range": "stddev: 0.000009170017619770086",
            "extra": "mean: 83.25115402311515 usec\nrounds: 2610"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12711.899991198272,
            "unit": "iter/sec",
            "range": "stddev: 0.000007731447060772873",
            "extra": "mean: 78.66644645508545 usec\nrounds: 2708"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 9418.50242663699,
            "unit": "iter/sec",
            "range": "stddev: 0.000008681407283646316",
            "extra": "mean: 106.17399186221417 usec\nrounds: 4178"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1102.0729115867698,
            "unit": "iter/sec",
            "range": "stddev: 0.000013413244279628006",
            "extra": "mean: 907.3809813183733 usec\nrounds: 910"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 275.718557726785,
            "unit": "iter/sec",
            "range": "stddev: 0.000037799656956096194",
            "extra": "mean: 3.626886808942763 msec\nrounds: 246"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2998409.0172077715,
            "unit": "iter/sec",
            "range": "stddev: 2.991204645911952e-8",
            "extra": "mean: 333.5102029980008 nsec\nrounds: 144384"
          }
        ]
      }
    ]
  }
}