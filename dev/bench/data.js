window.BENCHMARK_DATA = {
  "lastUpdate": 1788293683691,
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
          "id": "dc4b0206a545d4cff6ed3f8ef7d03a66e994a72a",
          "message": "release: v0.48.0 flagship evidence pipeline",
          "timestamp": "2026-09-01T19:25:02Z",
          "url": "https://github.com/stateset/stateset-agents/pull/61/commits/dc4b0206a545d4cff6ed3f8ef7d03a66e994a72a"
        },
        "date": 1788293681927,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11056.365805283578,
            "unit": "iter/sec",
            "range": "stddev: 0.000012405267948595495",
            "extra": "mean: 90.4456326437864 usec\nrounds: 2273"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12002.086823084812,
            "unit": "iter/sec",
            "range": "stddev: 0.000010958018277597086",
            "extra": "mean: 83.31884402607388 usec\nrounds: 2603"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8807.988368131711,
            "unit": "iter/sec",
            "range": "stddev: 0.000011213790260445914",
            "extra": "mean: 113.53330161267152 usec\nrounds: 4340"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1078.757945472057,
            "unit": "iter/sec",
            "range": "stddev: 0.000020849815214867416",
            "extra": "mean: 926.9920135442497 usec\nrounds: 886"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 229.3824406908354,
            "unit": "iter/sec",
            "range": "stddev: 0.00005759475366161771",
            "extra": "mean: 4.359531605768433 msec\nrounds: 208"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3052057.9910405115,
            "unit": "iter/sec",
            "range": "stddev: 3.350598603726482e-8",
            "extra": "mean: 327.6477717446905 nsec\nrounds: 145540"
          }
        ]
      }
    ]
  }
}