window.BENCHMARK_DATA = {
  "lastUpdate": 1788351016173,
  "repoUrl": "https://github.com/stateset/stateset-agents",
  "entries": {
    "Python Benchmark (nightly)": [
      {
        "commit": {
          "author": {
            "name": "Dom Steil",
            "username": "domsteil",
            "email": "domsteil14@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "a8390f84442d5247c67f32c066d32540069ecd75",
          "message": "Merge pull request #61 from stateset/release/0.48.0\n\nrelease: v0.48.0 flagship evidence pipeline",
          "timestamp": "2026-09-01T20:21:29Z",
          "url": "https://github.com/stateset/stateset-agents/commit/a8390f84442d5247c67f32c066d32540069ecd75"
        },
        "date": 1788351015669,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8593.856080473091,
            "unit": "iter/sec",
            "range": "stddev: 0.000015238421565408091",
            "extra": "mean: 116.36219999915917 usec\nrounds: 1890"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9267.892826110536,
            "unit": "iter/sec",
            "range": "stddev: 0.000013458971850463495",
            "extra": "mean: 107.89939188578974 usec\nrounds: 2169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6799.9545673996345,
            "unit": "iter/sec",
            "range": "stddev: 0.00001551657552697942",
            "extra": "mean: 147.05980607491165 usec\nrounds: 3424"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 829.0458519643039,
            "unit": "iter/sec",
            "range": "stddev: 0.000132399532246641",
            "extra": "mean: 1.2062059024005067 msec\nrounds: 625"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.9753122646351,
            "unit": "iter/sec",
            "range": "stddev: 0.00006700100712798728",
            "extra": "mean: 5.556317627217689 msec\nrounds: 169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2387781.614647398,
            "unit": "iter/sec",
            "range": "stddev: 3.9701815968622414e-8",
            "extra": "mean: 418.7987686418589 nsec\nrounds: 58391"
          }
        ]
      }
    ]
  }
}