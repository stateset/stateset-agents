window.BENCHMARK_DATA = {
  "lastUpdate": 1791849412964,
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
          "id": "fad08f95365db49b4e554f969b4eca3601c2fe4f",
          "message": "Merge release chain v0.57.0–v0.64.1 into master",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/98/commits/fad08f95365db49b4e554f969b4eca3601c2fe4f"
        },
        "date": 1791849410840,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5990.300846276984,
            "unit": "iter/sec",
            "range": "stddev: 0.000019221265974128353",
            "extra": "mean: 166.93652383444268 usec\nrounds: 1888"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6430.392205058016,
            "unit": "iter/sec",
            "range": "stddev: 0.00003399265682701628",
            "extra": "mean: 155.51150973550574 usec\nrounds: 2003"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5012.453506794394,
            "unit": "iter/sec",
            "range": "stddev: 0.00002013345224769582",
            "extra": "mean: 199.50309736429418 usec\nrounds: 3605"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 728.1142602617143,
            "unit": "iter/sec",
            "range": "stddev: 0.000041698745103511144",
            "extra": "mean: 1.3734108155505136 msec\nrounds: 656"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 168.56855766605318,
            "unit": "iter/sec",
            "range": "stddev: 0.005749726018252025",
            "extra": "mean: 5.932304421688618 msec\nrounds: 166"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2303316.9375244044,
            "unit": "iter/sec",
            "range": "stddev: 4.9955686263830395e-8",
            "extra": "mean: 434.15649132281203 nsec\nrounds: 110902"
          }
        ]
      }
    ]
  }
}