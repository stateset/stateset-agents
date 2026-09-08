window.BENCHMARK_DATA = {
  "lastUpdate": 1788902011619,
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
          "id": "f63f26d8d22efeb003f3c4bd28e66fc8933ca97b",
          "message": "Merge pull request #73 from stateset/release/0.52.0\n\nchore(release): v0.52.0 — On-policy engine rollouts, proven live",
          "timestamp": "2026-09-08T14:11:04-07:00",
          "tree_id": "47a2398f5f5b91ab5108d224291e8830b0a366a5",
          "url": "https://github.com/stateset/stateset-agents/commit/f63f26d8d22efeb003f3c4bd28e66fc8933ca97b"
        },
        "date": 1788902010752,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6151.928460494732,
            "unit": "iter/sec",
            "range": "stddev: 0.000014112562533452809",
            "extra": "mean: 162.55065487539514 usec\nrounds: 2092"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6636.188158820316,
            "unit": "iter/sec",
            "range": "stddev: 0.000013586486156056762",
            "extra": "mean: 150.6889159962826 usec\nrounds: 1988"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5096.646679190006,
            "unit": "iter/sec",
            "range": "stddev: 0.000015642794098164307",
            "extra": "mean: 196.20744048887587 usec\nrounds: 3764"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 755.6794111984082,
            "unit": "iter/sec",
            "range": "stddev: 0.00002826566841293376",
            "extra": "mean: 1.3233124856665495 msec\nrounds: 593"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 184.39579322984918,
            "unit": "iter/sec",
            "range": "stddev: 0.00003812290010285306",
            "extra": "mean: 5.423117211538015 msec\nrounds: 156"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2226478.9627826526,
            "unit": "iter/sec",
            "range": "stddev: 5.0599889024452886e-8",
            "extra": "mean: 449.13965805012606 nsec\nrounds: 107910"
          }
        ]
      }
    ]
  }
}