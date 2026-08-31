window.BENCHMARK_DATA = {
  "lastUpdate": 1788197174615,
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
          "id": "6a66cce8d2cfeeae3c50fbc40e3d82c5483a9ce9",
          "message": "ci: persist verified GPU evidence",
          "timestamp": "2026-08-31T13:19:43Z",
          "url": "https://github.com/stateset/stateset-agents/pull/45/commits/6a66cce8d2cfeeae3c50fbc40e3d82c5483a9ce9"
        },
        "date": 1788197173305,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6027.535531808161,
            "unit": "iter/sec",
            "range": "stddev: 0.000018794131776951015",
            "extra": "mean: 165.9052849581488 usec\nrounds: 1702"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6470.813209962205,
            "unit": "iter/sec",
            "range": "stddev: 0.00001777038341983159",
            "extra": "mean: 154.54008137036624 usec\nrounds: 1868"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5022.018939952311,
            "unit": "iter/sec",
            "range": "stddev: 0.000018924456000964734",
            "extra": "mean: 199.12310406569196 usec\nrounds: 3296"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 742.0891044594247,
            "unit": "iter/sec",
            "range": "stddev: 0.00002881630891221696",
            "extra": "mean: 1.3475470721652094 msec\nrounds: 679"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 174.3849171023535,
            "unit": "iter/sec",
            "range": "stddev: 0.00008108453336444733",
            "extra": "mean: 5.734440894409806 msec\nrounds: 161"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2201272.782422701,
            "unit": "iter/sec",
            "range": "stddev: 5.078148836095748e-8",
            "extra": "mean: 454.2826350214575 nsec\nrounds: 108261"
          }
        ]
      }
    ]
  }
}