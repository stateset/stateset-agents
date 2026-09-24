window.BENCHMARK_DATA = {
  "lastUpdate": 1790224435214,
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
          "id": "2f65db802d19c6232a734b47789c78d5f5867941",
          "message": "Harden checkpoint loads and expand blocking type checks",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/2f65db802d19c6232a734b47789c78d5f5867941"
        },
        "date": 1790224433144,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11844.774054978292,
            "unit": "iter/sec",
            "range": "stddev: 0.000008467350397130748",
            "extra": "mean: 84.42541794030302 usec\nrounds: 2486"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12191.092367404679,
            "unit": "iter/sec",
            "range": "stddev: 0.000009675701903205196",
            "extra": "mean: 82.02710387739327 usec\nrounds: 2089"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 9054.224802753228,
            "unit": "iter/sec",
            "range": "stddev: 0.000011851548913570653",
            "extra": "mean: 110.44567831979586 usec\nrounds: 3976"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1070.4441497186776,
            "unit": "iter/sec",
            "range": "stddev: 0.0000540384721778237",
            "extra": "mean: 934.1916626503205 usec\nrounds: 830"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 268.0652511190693,
            "unit": "iter/sec",
            "range": "stddev: 0.0001442410820268549",
            "extra": "mean: 3.7304350184344472 msec\nrounds: 217"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2900107.0601699646,
            "unit": "iter/sec",
            "range": "stddev: 3.5622510793757655e-8",
            "extra": "mean: 344.81485657339624 nsec\nrounds: 138851"
          }
        ]
      }
    ]
  }
}