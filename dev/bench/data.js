window.BENCHMARK_DATA = {
  "lastUpdate": 1791903836676,
  "repoUrl": "https://github.com/stateset/stateset-agents",
  "entries": {
    "Python Benchmark (nightly)": [
      {
        "commit": {
          "author": {
            "name": "Dom Steil",
            "username": "domsteil",
            "email": "domsteil14@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "055bb614a0d1a3185551916023e3f543bb1089c9",
          "message": "docs(bench): v5.1 1.5B shootout — parity with TRL (pre-release harness) (#104)\n\ndocs(bench): matched 1.5B shootout v5.1 — parity with TRL 1.12",
          "timestamp": "2026-10-13T10:01:21Z",
          "url": "https://github.com/stateset/stateset-agents/commit/055bb614a0d1a3185551916023e3f543bb1089c9"
        },
        "date": 1791903835833,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6064.986659119702,
            "unit": "iter/sec",
            "range": "stddev: 0.000016549500902225405",
            "extra": "mean: 164.88082434548076 usec\nrounds: 1947"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6547.458285069174,
            "unit": "iter/sec",
            "range": "stddev: 0.00001666247843635286",
            "extra": "mean: 152.7310227054673 usec\nrounds: 2114"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4965.513449596476,
            "unit": "iter/sec",
            "range": "stddev: 0.000028259010263000788",
            "extra": "mean: 201.38904267418013 usec\nrounds: 3515"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 726.4019251303021,
            "unit": "iter/sec",
            "range": "stddev: 0.000030224660047838025",
            "extra": "mean: 1.3766483339380193 msec\nrounds: 551"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.71143182302026,
            "unit": "iter/sec",
            "range": "stddev: 0.00005517983731556771",
            "extra": "mean: 5.473111288234169 msec\nrounds: 170"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2312958.2725554164,
            "unit": "iter/sec",
            "range": "stddev: 4.7249060470327446e-8",
            "extra": "mean: 432.3467534479876 nsec\nrounds: 111272"
          }
        ]
      }
    ]
  }
}