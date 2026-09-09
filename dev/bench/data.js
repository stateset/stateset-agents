window.BENCHMARK_DATA = {
  "lastUpdate": 1788919572915,
  "repoUrl": "https://github.com/stateset/stateset-agents",
  "entries": {
    "Python Benchmark": [
      {
        "commit": {
          "author": {
            "email": "domsteil14@gmail.com",
            "name": "Dom Steil",
            "username": "domsteil"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5ee45381e7911f163b2433f058343f668f4b14bf",
          "message": "Merge pull request #79 from stateset/release/0.54.0\n\nchore(release): v0.54.0 — Rollouts sample in inference mode",
          "timestamp": "2026-09-08T19:03:18-07:00",
          "tree_id": "8ca59b676d45ee6acf05b6fd594730ec0af3c3c2",
          "url": "https://github.com/stateset/stateset-agents/commit/5ee45381e7911f163b2433f058343f668f4b14bf"
        },
        "date": 1788919571617,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5948.682178449729,
            "unit": "iter/sec",
            "range": "stddev: 0.00002006976344282769",
            "extra": "mean: 168.10445910569854 usec\nrounds: 1834"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6425.800417157464,
            "unit": "iter/sec",
            "range": "stddev: 0.000016072323309571164",
            "extra": "mean: 155.62263610458712 usec\nrounds: 2105"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4952.089510820122,
            "unit": "iter/sec",
            "range": "stddev: 0.000017217343186421285",
            "extra": "mean: 201.9349605484794 usec\nrounds: 3574"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 749.740631119407,
            "unit": "iter/sec",
            "range": "stddev: 0.000026900090552523677",
            "extra": "mean: 1.3337945930807311 msec\nrounds: 607"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.1206013549686,
            "unit": "iter/sec",
            "range": "stddev: 0.000051277557584419565",
            "extra": "mean: 5.490866999999163 msec\nrounds: 162"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2243922.322469199,
            "unit": "iter/sec",
            "range": "stddev: 5.0663839357105723e-8",
            "extra": "mean: 445.6482249793771 nsec\nrounds: 106872"
          }
        ]
      }
    ]
  }
}