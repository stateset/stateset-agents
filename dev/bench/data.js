window.BENCHMARK_DATA = {
  "lastUpdate": 1792068198986,
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
          "id": "082333ff3ca9c85b127a7512f3729da4f32e67d7",
          "message": "feat(flagship): v2 protocol (Qwen3.5-9B + Gemma 4 31B judge) + opt-in GSPO episode weighting",
          "timestamp": "2026-10-14T03:32:41Z",
          "url": "https://github.com/stateset/stateset-agents/pull/109/commits/082333ff3ca9c85b127a7512f3729da4f32e67d7"
        },
        "date": 1792036058220,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 15242.578787007766,
            "unit": "iter/sec",
            "range": "stddev: 0.000008479146614813065",
            "extra": "mean: 65.60569664579097 usec\nrounds: 2713"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 15636.116633577041,
            "unit": "iter/sec",
            "range": "stddev: 0.000009023794345559893",
            "extra": "mean: 63.954498641471965 usec\nrounds: 3680"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11932.251200945639,
            "unit": "iter/sec",
            "range": "stddev: 0.000007627769036811133",
            "extra": "mean: 83.80648237783909 usec\nrounds: 5164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1493.769486218477,
            "unit": "iter/sec",
            "range": "stddev: 0.00009552713723225733",
            "extra": "mean: 669.4473338932171 usec\nrounds: 1192"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 291.0296674399474,
            "unit": "iter/sec",
            "range": "stddev: 0.004091224250940888",
            "extra": "mean: 3.436075809028457 msec\nrounds: 288"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 4723491.879667333,
            "unit": "iter/sec",
            "range": "stddev: 2.0126495468088965e-8",
            "extra": "mean: 211.70778429927736 nsec\nrounds: 195772"
          }
        ]
      },
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
          "id": "df4bd0e21e0373d8ac38e57c262aa5fb70eed74b",
          "message": "feat(flagship): v2 protocol (Qwen3.5-9B + Gemma 4 31B judge) + opt-in GSPO episode weighting",
          "timestamp": "2026-10-14T03:32:41Z",
          "url": "https://github.com/stateset/stateset-agents/pull/109/commits/df4bd0e21e0373d8ac38e57c262aa5fb70eed74b"
        },
        "date": 1792037978180,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6049.641211906551,
            "unit": "iter/sec",
            "range": "stddev: 0.000016789446289732835",
            "extra": "mean: 165.29905906351246 usec\nrounds: 1964"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6577.497561326836,
            "unit": "iter/sec",
            "range": "stddev: 0.000015539681537938127",
            "extra": "mean: 152.0335037263437 usec\nrounds: 2013"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4986.860949455397,
            "unit": "iter/sec",
            "range": "stddev: 0.000017841651174742475",
            "extra": "mean: 200.52694673774846 usec\nrounds: 3586"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 732.2055679509717,
            "unit": "iter/sec",
            "range": "stddev: 0.000033688044210426634",
            "extra": "mean: 1.3657366780184872 msec\nrounds: 646"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 167.11849674805018,
            "unit": "iter/sec",
            "range": "stddev: 0.005825053688424757",
            "extra": "mean: 5.983778094339922 msec\nrounds: 159"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2310020.97114452,
            "unit": "iter/sec",
            "range": "stddev: 6.100728300752198e-8",
            "extra": "mean: 432.8965028852276 nsec\nrounds: 110534"
          }
        ]
      },
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
          "id": "d84f56d76336f4102d46e1137f316a1854513b70",
          "message": "feat(flagship): v2 protocol (Qwen3.5-9B + Gemma 4 31B judge) + opt-in GSPO episode weighting",
          "timestamp": "2026-10-14T03:32:41Z",
          "url": "https://github.com/stateset/stateset-agents/pull/109/commits/d84f56d76336f4102d46e1137f316a1854513b70"
        },
        "date": 1792054022824,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6086.275515855231,
            "unit": "iter/sec",
            "range": "stddev: 0.000018209037620722704",
            "extra": "mean: 164.30409655213944 usec\nrounds: 2030"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6518.105500991985,
            "unit": "iter/sec",
            "range": "stddev: 0.000017483414514440095",
            "extra": "mean: 153.4188116236859 usec\nrounds: 1996"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5035.983923963675,
            "unit": "iter/sec",
            "range": "stddev: 0.000017833998427707562",
            "extra": "mean: 198.57092776676885 usec\nrounds: 3018"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 737.1680373657998,
            "unit": "iter/sec",
            "range": "stddev: 0.00003319626034302658",
            "extra": "mean: 1.3565428088464135 msec\nrounds: 633"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 167.1283830098937,
            "unit": "iter/sec",
            "range": "stddev: 0.005748487132156216",
            "extra": "mean: 5.983424131739501 msec\nrounds: 167"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2220977.5197631465,
            "unit": "iter/sec",
            "range": "stddev: 7.674964817791484e-8",
            "extra": "mean: 450.2521935056073 nsec\nrounds: 106861"
          }
        ]
      }
    ],
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
        "date": 1792068198173,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 9079.656919707379,
            "unit": "iter/sec",
            "range": "stddev: 0.00002774392557227885",
            "extra": "mean: 110.13632000009845 usec\nrounds: 1875"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 10135.434093106749,
            "unit": "iter/sec",
            "range": "stddev: 0.000012882079150102066",
            "extra": "mean: 98.66375636344121 usec\nrounds: 2200"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 7352.883570762662,
            "unit": "iter/sec",
            "range": "stddev: 0.00001417317326394066",
            "extra": "mean: 136.00106548352122 usec\nrounds: 3268"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 899.7330653269538,
            "unit": "iter/sec",
            "range": "stddev: 0.0000268227924603558",
            "extra": "mean: 1.1114407578614554 msec\nrounds: 636"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 196.00888899869202,
            "unit": "iter/sec",
            "range": "stddev: 0.0001286199988406161",
            "extra": "mean: 5.101809438890667 msec\nrounds: 180"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2501251.764760059,
            "unit": "iter/sec",
            "range": "stddev: 4.414651197562349e-8",
            "extra": "mean: 399.79981787076457 nsec\nrounds: 119147"
          }
        ]
      }
    ]
  }
}