window.BENCHMARK_DATA = {
  "lastUpdate": 1787854592610,
  "repoUrl": "https://github.com/stateset/stateset-agents",
  "entries": {
    "Python Benchmark (nightly)": [
      {
        "commit": {
          "author": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "committer": {
            "name": "domsteil",
            "email": "team@stateset.ai"
          },
          "id": "0a79e41d57d3905622abaf0fa2e8f2134a4a9347",
          "message": "Merge feat/polish-3: river behavioural golden, drift fixes, submit-family fold\n\n27-scenario offline golden over all five River submit modes (both episode paths\nhad zero coverage); four drift bugs it made visible; each submit family folded\ninto one implementation behind a closure-carrying mode adapter.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01JsBFXmpa8URWHqS2irLChd",
          "timestamp": "2026-08-25T19:44:44Z",
          "url": "https://github.com/stateset/stateset-agents/commit/0a79e41d57d3905622abaf0fa2e8f2134a4a9347"
        },
        "date": 1787854591391,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8482.431070268696,
            "unit": "iter/sec",
            "range": "stddev: 0.000011080826218125453",
            "extra": "mean: 117.89073105528026 usec\nrounds: 2019"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 8969.742451879001,
            "unit": "iter/sec",
            "range": "stddev: 0.00001029050630865471",
            "extra": "mean: 111.48592118054827 usec\nrounds: 2271"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6841.88436270959,
            "unit": "iter/sec",
            "range": "stddev: 0.00001195007297810662",
            "extra": "mean: 146.15856494890687 usec\nrounds: 3241"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 883.2726983737342,
            "unit": "iter/sec",
            "range": "stddev: 0.00004210583139933182",
            "extra": "mean: 1.132153186486101 msec\nrounds: 740"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 207.90581101665236,
            "unit": "iter/sec",
            "range": "stddev: 0.00010082891027706255",
            "extra": "mean: 4.809870369231307 msec\nrounds: 195"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2351470.1547995075,
            "unit": "iter/sec",
            "range": "stddev: 3.60819286160959e-8",
            "extra": "mean: 425.26586950675653 nsec\nrounds: 116037"
          }
        ]
      }
    ]
  }
}