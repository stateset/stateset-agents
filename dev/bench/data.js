window.BENCHMARK_DATA = {
  "lastUpdate": 1788288839574,
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
          "id": "f0afad6b901659fb959de8b2aa4450558c16222e",
          "message": "feat: add distributed async evidence collection",
          "timestamp": "2026-09-01T17:57:43Z",
          "url": "https://github.com/stateset/stateset-agents/pull/59/commits/f0afad6b901659fb959de8b2aa4450558c16222e"
        },
        "date": 1788288838342,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6218.703086048445,
            "unit": "iter/sec",
            "range": "stddev: 0.000014515690247221999",
            "extra": "mean: 160.8052332074646 usec\nrounds: 2114"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6726.649499528161,
            "unit": "iter/sec",
            "range": "stddev: 0.000013937068351301551",
            "extra": "mean: 148.66242102701273 usec\nrounds: 2121"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5147.761790078979,
            "unit": "iter/sec",
            "range": "stddev: 0.00001531713340320998",
            "extra": "mean: 194.2591830739428 usec\nrounds: 3769"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 755.6318617871658,
            "unit": "iter/sec",
            "range": "stddev: 0.00003882515733292962",
            "extra": "mean: 1.32339575733992 msec\nrounds: 647"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 184.2272901912047,
            "unit": "iter/sec",
            "range": "stddev: 0.0000620337945030911",
            "extra": "mean: 5.428077452380296 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2146865.158095289,
            "unit": "iter/sec",
            "range": "stddev: 1.0505707718499447e-7",
            "extra": "mean: 465.79543956417166 nsec\nrounds: 107435"
          }
        ]
      }
    ]
  }
}