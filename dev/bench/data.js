window.BENCHMARK_DATA = {
  "lastUpdate": 1791867696735,
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
          "id": "a9572e34b14b39d934ca34ed78cf5bf2cb58fb3a",
          "message": "Merge release chain v0.57.0–v0.64.1 into master",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/98/commits/a9572e34b14b39d934ca34ed78cf5bf2cb58fb3a"
        },
        "date": 1791867695379,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6027.376565359459,
            "unit": "iter/sec",
            "range": "stddev: 0.00001739556089005498",
            "extra": "mean: 165.90966055567202 usec\nrounds: 1909"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6491.179254085179,
            "unit": "iter/sec",
            "range": "stddev: 0.000015442735558045922",
            "extra": "mean: 154.0552125980278 usec\nrounds: 2032"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4994.930790516428,
            "unit": "iter/sec",
            "range": "stddev: 0.000018104633296430137",
            "extra": "mean: 200.20297416305334 usec\nrounds: 3406"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 728.7781335331041,
            "unit": "iter/sec",
            "range": "stddev: 0.00003090059528099991",
            "extra": "mean: 1.3721597204790117 msec\nrounds: 669"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 170.6134179968663,
            "unit": "iter/sec",
            "range": "stddev: 0.004860791643459539",
            "extra": "mean: 5.861203718563139 msec\nrounds: 167"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2221863.96486894,
            "unit": "iter/sec",
            "range": "stddev: 9.12599689620107e-8",
            "extra": "mean: 450.0725588116672 nsec\nrounds: 103853"
          }
        ]
      }
    ]
  }
}