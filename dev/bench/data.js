window.BENCHMARK_DATA = {
  "lastUpdate": 1788893640259,
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
          "id": "646efc2deba60a778804aec85082977fed946b82",
          "message": "Merge pull request #68 from stateset/release/0.51.0\n\nchore(release): v0.51.0 — Evidence-safe benchmarking and engine rollouts",
          "timestamp": "2026-09-08T11:51:28-07:00",
          "tree_id": "8018aa752f35a9e9d4a171701314f12f71619d09",
          "url": "https://github.com/stateset/stateset-agents/commit/646efc2deba60a778804aec85082977fed946b82"
        },
        "date": 1788893639330,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8530.595669628603,
            "unit": "iter/sec",
            "range": "stddev: 0.00001332468474641278",
            "extra": "mean: 117.22510815514211 usec\nrounds: 1729"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9233.710880037872,
            "unit": "iter/sec",
            "range": "stddev: 0.00001373736690362287",
            "extra": "mean: 108.298820809072 usec\nrounds: 2076"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6691.180167338151,
            "unit": "iter/sec",
            "range": "stddev: 0.000015677041912174966",
            "extra": "mean: 149.45046688196032 usec\nrounds: 3095"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 834.3745154556722,
            "unit": "iter/sec",
            "range": "stddev: 0.000019404729531066405",
            "extra": "mean: 1.1985025686623179 msec\nrounds: 568"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 177.76177122322784,
            "unit": "iter/sec",
            "range": "stddev: 0.0001732655772277349",
            "extra": "mean: 5.625506502994001 msec\nrounds: 167"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2267775.223974553,
            "unit": "iter/sec",
            "range": "stddev: 4.9414418604196975e-8",
            "extra": "mean: 440.9608101491549 nsec\nrounds: 110461"
          }
        ]
      }
    ]
  }
}