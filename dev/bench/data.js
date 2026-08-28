window.BENCHMARK_DATA = {
  "lastUpdate": 1787910746191,
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
          "id": "b0465b29b884fb4eec52916f4817f1e20efa7e31",
          "message": "feat: add fail-closed training backend protocol",
          "timestamp": "2026-08-28T02:20:26Z",
          "url": "https://github.com/stateset/stateset-agents/pull/35/commits/b0465b29b884fb4eec52916f4817f1e20efa7e31"
        },
        "date": 1787910745320,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8575.351684013509,
            "unit": "iter/sec",
            "range": "stddev: 0.000018231234835065855",
            "extra": "mean: 116.61329317423066 usec\nrounds: 1992"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9483.118336609181,
            "unit": "iter/sec",
            "range": "stddev: 0.000011913361690176684",
            "extra": "mean: 105.45054532743116 usec\nrounds: 2162"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6753.663123611745,
            "unit": "iter/sec",
            "range": "stddev: 0.000017906676388126966",
            "extra": "mean: 148.06779398040467 usec\nrounds: 3422"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 840.6075871294689,
            "unit": "iter/sec",
            "range": "stddev: 0.000027025764279738516",
            "extra": "mean: 1.189615719999422 msec\nrounds: 725"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 175.2730641740679,
            "unit": "iter/sec",
            "range": "stddev: 0.0005872389615664663",
            "extra": "mean: 5.705383224240753 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2233842.197616746,
            "unit": "iter/sec",
            "range": "stddev: 3.514435174451323e-8",
            "extra": "mean: 447.65919502590003 nsec\nrounds: 54834"
          }
        ]
      }
    ]
  }
}