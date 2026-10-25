window.BENCHMARK_DATA = {
  "lastUpdate": 1792931662454,
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
      },
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
        "date": 1792843434700,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 13649.947761919484,
            "unit": "iter/sec",
            "range": "stddev: 0.000013233020338887552",
            "extra": "mean: 73.26035362492682 usec\nrounds: 1807"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 13971.176867903909,
            "unit": "iter/sec",
            "range": "stddev: 0.000012625167947576462",
            "extra": "mean: 71.57593160940561 usec\nrounds: 2939"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 10686.646529181344,
            "unit": "iter/sec",
            "range": "stddev: 0.000013622125958426118",
            "extra": "mean: 93.57472405111966 usec\nrounds: 4374"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1346.6105293998116,
            "unit": "iter/sec",
            "range": "stddev: 0.00006239089586569083",
            "extra": "mean: 742.6052137329589 usec\nrounds: 1034"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 278.4769615411568,
            "unit": "iter/sec",
            "range": "stddev: 0.0003072008825552042",
            "extra": "mean: 3.590961329317031 msec\nrounds: 249"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 4282108.507234963,
            "unit": "iter/sec",
            "range": "stddev: 3.060330872969248e-8",
            "extra": "mean: 233.5298132474739 nsec\nrounds: 196156"
          }
        ]
      },
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
        "date": 1792931661723,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14910.02662091312,
            "unit": "iter/sec",
            "range": "stddev: 0.000008198132250467884",
            "extra": "mean: 67.0689614059695 usec\nrounds: 2617"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 15731.972843877767,
            "unit": "iter/sec",
            "range": "stddev: 0.000008025626679242591",
            "extra": "mean: 63.56481859738008 usec\nrounds: 3280"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11967.103711822989,
            "unit": "iter/sec",
            "range": "stddev: 0.000008739233164192745",
            "extra": "mean: 83.56240775385297 usec\nrounds: 5133"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1507.2659911176172,
            "unit": "iter/sec",
            "range": "stddev: 0.00007505876907492352",
            "extra": "mean: 663.4529047248744 usec\nrounds: 1291"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 308.896941944867,
            "unit": "iter/sec",
            "range": "stddev: 0.00006070668617527674",
            "extra": "mean: 3.2373256714806953 msec\nrounds: 277"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 4727566.737197137,
            "unit": "iter/sec",
            "range": "stddev: 1.910989956440781e-8",
            "extra": "mean: 211.52530584748052 nsec\nrounds: 196967"
          }
        ]
      }
    ]
  }
}