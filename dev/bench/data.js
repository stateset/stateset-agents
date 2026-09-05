window.BENCHMARK_DATA = {
  "lastUpdate": 1788631564126,
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
          "id": "4a4679631cec75f62db4ab4e48728dba01b236ab",
          "message": "chore(release): v0.50.0 — Per-token GRPO and starter specs",
          "timestamp": "2026-09-05T16:55:32Z",
          "url": "https://github.com/stateset/stateset-agents/pull/65/commits/4a4679631cec75f62db4ab4e48728dba01b236ab"
        },
        "date": 1788631563076,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5541.568027551414,
            "unit": "iter/sec",
            "range": "stddev: 0.000038822677329112836",
            "extra": "mean: 180.45433982371557 usec\nrounds: 2054"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6255.361181187295,
            "unit": "iter/sec",
            "range": "stddev: 0.000030777806367361244",
            "extra": "mean: 159.86287138901795 usec\nrounds: 1454"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5082.370827217935,
            "unit": "iter/sec",
            "range": "stddev: 0.000015868553466872903",
            "extra": "mean: 196.7585668178005 usec\nrounds: 3315"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 748.6682700895639,
            "unit": "iter/sec",
            "range": "stddev: 0.00009005000148574983",
            "extra": "mean: 1.33570506451458 msec\nrounds: 589"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.05037917168914,
            "unit": "iter/sec",
            "range": "stddev: 0.000049280211522385486",
            "extra": "mean: 5.5540010779229645 msec\nrounds: 154"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2199058.944888579,
            "unit": "iter/sec",
            "range": "stddev: 4.7130340265045816e-8",
            "extra": "mean: 454.73997062441975 nsec\nrounds: 105854"
          }
        ]
      }
    ]
  }
}