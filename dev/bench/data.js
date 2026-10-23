window.BENCHMARK_DATA = {
  "lastUpdate": 1792759244261,
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
          "id": "4164974a853bca6c42fab94cbc9142c506a16ebd",
          "message": "fix: reward-hacking-resistant support reward, stable judge, non-finite skip, RunPod driver hardening (#108)\n\nLessons from three H100 flagship preflights: per-turn reward was hacked (judge 0.748 -> 0.605) -> conversation-level reward v2; lenient/unstable judge -> Mistral-Small-24B + checklist rubric v4 with exact detector checks; bf16 backward NaNs -> diagnose-and-skip policy; provider-exited pods and exhausted credit now fail fast. Flagship config not yet validated end to end (next preflight blocked on RunPod credit).",
          "timestamp": "2026-10-14T03:32:33Z",
          "url": "https://github.com/stateset/stateset-agents/commit/4164974a853bca6c42fab94cbc9142c506a16ebd"
        },
        "date": 1792759243320,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6043.203736651134,
            "unit": "iter/sec",
            "range": "stddev: 0.000017271230641572306",
            "extra": "mean: 165.47514258623258 usec\nrounds: 2111"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6627.099154600888,
            "unit": "iter/sec",
            "range": "stddev: 0.000015664073961605155",
            "extra": "mean: 150.89558442863287 usec\nrounds: 2055"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5000.57314206827,
            "unit": "iter/sec",
            "range": "stddev: 0.00001999337139355022",
            "extra": "mean: 199.97707694490265 usec\nrounds: 3561"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 713.9420251898721,
            "unit": "iter/sec",
            "range": "stddev: 0.00020169353257341614",
            "extra": "mean: 1.4006739549111864 msec\nrounds: 621"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 183.7172834861757,
            "unit": "iter/sec",
            "range": "stddev: 0.0000621321596976096",
            "extra": "mean: 5.44314601775204 msec\nrounds: 169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2242477.865522875,
            "unit": "iter/sec",
            "range": "stddev: 5.5684316394957116e-8",
            "extra": "mean: 445.9352822940046 nsec\nrounds: 109195"
          }
        ]
      }
    ]
  }
}