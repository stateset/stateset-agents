window.BENCHMARK_DATA = {
  "lastUpdate": 1791868686441,
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
          "id": "274f3e63798de4f71366d7b3fd74f408c47e6fd9",
          "message": "Merge release chain v0.57.0–v0.64.1 into master",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/98/commits/274f3e63798de4f71366d7b3fd74f408c47e6fd9"
        },
        "date": 1791868685142,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5855.047789209748,
            "unit": "iter/sec",
            "range": "stddev: 0.000030611995406693184",
            "extra": "mean: 170.792798966201 usec\nrounds: 1935"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6370.571619264508,
            "unit": "iter/sec",
            "range": "stddev: 0.00001685703509728778",
            "extra": "mean: 156.9717852281914 usec\nrounds: 2058"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4902.054219574712,
            "unit": "iter/sec",
            "range": "stddev: 0.000018830234710761685",
            "extra": "mean: 203.99611167229338 usec\nrounds: 3367"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 722.6189654372002,
            "unit": "iter/sec",
            "range": "stddev: 0.00003135142226595979",
            "extra": "mean: 1.3838551820944502 msec\nrounds: 659"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 167.77203826622485,
            "unit": "iter/sec",
            "range": "stddev: 0.0058720401870384015",
            "extra": "mean: 5.960468802394682 msec\nrounds: 167"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2253361.7872823235,
            "unit": "iter/sec",
            "range": "stddev: 6.139327211297607e-8",
            "extra": "mean: 443.7813784026463 nsec\nrounds: 96815"
          }
        ]
      }
    ]
  }
}