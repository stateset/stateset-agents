window.BENCHMARK_DATA = {
  "lastUpdate": 1792036059842,
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
          "id": "082333ff3ca9c85b127a7512f3729da4f32e67d7",
          "message": "feat(flagship): v2 protocol (Qwen3.5-9B + Gemma 4 31B judge) + opt-in GSPO episode weighting",
          "timestamp": "2026-10-14T03:32:41Z",
          "url": "https://github.com/stateset/stateset-agents/pull/109/commits/082333ff3ca9c85b127a7512f3729da4f32e67d7"
        },
        "date": 1792036058220,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 15242.578787007766,
            "unit": "iter/sec",
            "range": "stddev: 0.000008479146614813065",
            "extra": "mean: 65.60569664579097 usec\nrounds: 2713"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 15636.116633577041,
            "unit": "iter/sec",
            "range": "stddev: 0.000009023794345559893",
            "extra": "mean: 63.954498641471965 usec\nrounds: 3680"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11932.251200945639,
            "unit": "iter/sec",
            "range": "stddev: 0.000007627769036811133",
            "extra": "mean: 83.80648237783909 usec\nrounds: 5164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1493.769486218477,
            "unit": "iter/sec",
            "range": "stddev: 0.00009552713723225733",
            "extra": "mean: 669.4473338932171 usec\nrounds: 1192"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 291.0296674399474,
            "unit": "iter/sec",
            "range": "stddev: 0.004091224250940888",
            "extra": "mean: 3.436075809028457 msec\nrounds: 288"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 4723491.879667333,
            "unit": "iter/sec",
            "range": "stddev: 2.0126495468088965e-8",
            "extra": "mean: 211.70778429927736 nsec\nrounds: 195772"
          }
        ]
      }
    ]
  }
}