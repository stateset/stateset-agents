window.BENCHMARK_DATA = {
  "lastUpdate": 1790222717343,
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
          "id": "a32866a761ff99689e683e1e06579ad1563eb349",
          "message": "Harden checkpoint loads and expand blocking type checks",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/a32866a761ff99689e683e1e06579ad1563eb349"
        },
        "date": 1790222716345,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6074.089670495869,
            "unit": "iter/sec",
            "range": "stddev: 0.000016488465777368113",
            "extra": "mean: 164.63372361086056 usec\nrounds: 2160"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6517.768900384284,
            "unit": "iter/sec",
            "range": "stddev: 0.000016048203103037192",
            "extra": "mean: 153.42673471301512 usec\nrounds: 2175"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5035.010421510033,
            "unit": "iter/sec",
            "range": "stddev: 0.00001782134855071557",
            "extra": "mean: 198.60932079264563 usec\nrounds: 3535"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 738.268808191504,
            "unit": "iter/sec",
            "range": "stddev: 0.00003055572605660913",
            "extra": "mean: 1.35452018140878 msec\nrounds: 667"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.01974086053863,
            "unit": "iter/sec",
            "range": "stddev: 0.00013407065580501218",
            "extra": "mean: 5.493909590642633 msec\nrounds: 171"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2224726.2795082508,
            "unit": "iter/sec",
            "range": "stddev: 4.780410735350493e-8",
            "extra": "mean: 449.4934991378077 nsec\nrounds: 105286"
          }
        ]
      }
    ]
  }
}