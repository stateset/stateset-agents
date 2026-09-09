window.BENCHMARK_DATA = {
  "lastUpdate": 1788912117935,
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
          "id": "dd9a47a2aed5ae801037ac9d6760e29da1e58da6",
          "message": "fix: route GSPO-token and auto-research DAPO/VAPO queries through the scenario helper",
          "timestamp": "2026-09-08T23:49:04Z",
          "url": "https://github.com/stateset/stateset-agents/pull/77/commits/dd9a47a2aed5ae801037ac9d6760e29da1e58da6"
        },
        "date": 1788912116918,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 7451.347116123635,
            "unit": "iter/sec",
            "range": "stddev: 0.00003750650736714703",
            "extra": "mean: 134.2039210381362 usec\nrounds: 1773"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9144.959445836152,
            "unit": "iter/sec",
            "range": "stddev: 0.000018144621014237885",
            "extra": "mean: 109.34985616096047 usec\nrounds: 2037"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6715.62916905455,
            "unit": "iter/sec",
            "range": "stddev: 0.000015846070333641587",
            "extra": "mean: 148.90637568375197 usec\nrounds: 3290"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 830.7344088444858,
            "unit": "iter/sec",
            "range": "stddev: 0.00002026811782178185",
            "extra": "mean: 1.2037541593960879 msec\nrounds: 596"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 177.26017933162672,
            "unit": "iter/sec",
            "range": "stddev: 0.0000531066822999849",
            "extra": "mean: 5.641424959461159 msec\nrounds: 148"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2412428.798836891,
            "unit": "iter/sec",
            "range": "stddev: 3.3364387036731605e-8",
            "extra": "mean: 414.5200059301779 nsec\nrounds: 59015"
          }
        ]
      }
    ]
  }
}