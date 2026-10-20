window.BENCHMARK_DATA = {
  "lastUpdate": 1792500347540,
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
          "id": "bf218d333e329f1ae31f5299bdc47fdfdf11d132",
          "message": "feat(flagship): v2 protocol (Qwen3.5-9B + Gemma 4 31B judge) + opt-in GSPO episode weighting",
          "timestamp": "2026-10-14T03:32:41Z",
          "url": "https://github.com/stateset/stateset-agents/pull/109/commits/bf218d333e329f1ae31f5299bdc47fdfdf11d132"
        },
        "date": 1792103207656,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14389.667078038678,
            "unit": "iter/sec",
            "range": "stddev: 0.00000960542491801684",
            "extra": "mean: 69.49431106201109 usec\nrounds: 2260"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 15259.249159515197,
            "unit": "iter/sec",
            "range": "stddev: 0.000008840866423326874",
            "extra": "mean: 65.53402395795018 usec\nrounds: 2087"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11117.542454901757,
            "unit": "iter/sec",
            "range": "stddev: 0.000010697402392673176",
            "extra": "mean: 89.94793625088403 usec\nrounds: 4204"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1322.949754063953,
            "unit": "iter/sec",
            "range": "stddev: 0.000021181739638929208",
            "extra": "mean: 755.8866063719445 usec\nrounds: 973"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 316.914765145331,
            "unit": "iter/sec",
            "range": "stddev: 0.0038247725344011554",
            "extra": "mean: 3.155422561461973 msec\nrounds: 301"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3663228.9397133347,
            "unit": "iter/sec",
            "range": "stddev: 3.571156556752344e-8",
            "extra": "mean: 272.98321138461387 nsec\nrounds: 186602"
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
          "id": "5c944d61a9e32067218c3fdb04cb7f7192e34b64",
          "message": "feat(flagship): v2 protocol (Qwen3.5-9B + Gemma 4 31B judge) + opt-in GSPO episode weighting",
          "timestamp": "2026-10-14T03:32:41Z",
          "url": "https://github.com/stateset/stateset-agents/pull/109/commits/5c944d61a9e32067218c3fdb04cb7f7192e34b64"
        },
        "date": 1792103850423,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8424.207056211497,
            "unit": "iter/sec",
            "range": "stddev: 0.000014485315389704976",
            "extra": "mean: 118.70553434019182 usec\nrounds: 2053"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9135.613015842097,
            "unit": "iter/sec",
            "range": "stddev: 0.000013825222837660991",
            "extra": "mean: 109.4617294171608 usec\nrounds: 2077"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6700.250651399253,
            "unit": "iter/sec",
            "range": "stddev: 0.00001686514865904105",
            "extra": "mean: 149.24814787207464 usec\nrounds: 3219"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 816.7755622770401,
            "unit": "iter/sec",
            "range": "stddev: 0.00008710998492699127",
            "extra": "mean: 1.2243265423027097 msec\nrounds: 721"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 161.9767078176464,
            "unit": "iter/sec",
            "range": "stddev: 0.006044064356188471",
            "extra": "mean: 6.173727157893599 msec\nrounds: 133"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2253563.310839981,
            "unit": "iter/sec",
            "range": "stddev: 3.416730477891607e-8",
            "extra": "mean: 443.74169351703966 nsec\nrounds: 57419"
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
          "id": "5162d211934e369f1988270b979fee0936080936",
          "message": "feat(flagship): v2 protocol (Qwen3.5-9B + Gemma 4 31B judge) + opt-in GSPO episode weighting",
          "timestamp": "2026-10-14T03:32:41Z",
          "url": "https://github.com/stateset/stateset-agents/pull/109/commits/5162d211934e369f1988270b979fee0936080936"
        },
        "date": 1792104961882,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6284.064051074742,
            "unit": "iter/sec",
            "range": "stddev: 0.000014879109113106236",
            "extra": "mean: 159.13268736160853 usec\nrounds: 2255"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6731.816004409485,
            "unit": "iter/sec",
            "range": "stddev: 0.00001550525408213387",
            "extra": "mean: 148.5483262384145 usec\nrounds: 2161"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5119.8646473987565,
            "unit": "iter/sec",
            "range": "stddev: 0.000021859974095549507",
            "extra": "mean: 195.31766342847925 usec\nrounds: 3809"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 744.9291594805225,
            "unit": "iter/sec",
            "range": "stddev: 0.000026752190623836178",
            "extra": "mean: 1.3424095261586373 msec\nrounds: 669"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 174.10255027509734,
            "unit": "iter/sec",
            "range": "stddev: 0.005011979341329486",
            "extra": "mean: 5.743741251463072 msec\nrounds: 171"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2224588.7462927103,
            "unit": "iter/sec",
            "range": "stddev: 1.210941781438091e-7",
            "extra": "mean: 449.52128867257176 nsec\nrounds: 109927"
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
        "date": 1792153322093,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5951.3258287308145,
            "unit": "iter/sec",
            "range": "stddev: 0.000020034832803243408",
            "extra": "mean: 168.0297850896295 usec\nrounds: 1945"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6332.962686879253,
            "unit": "iter/sec",
            "range": "stddev: 0.000026407107610527033",
            "extra": "mean: 157.90397787623448 usec\nrounds: 2034"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4734.252376932384,
            "unit": "iter/sec",
            "range": "stddev: 0.000024102296170040465",
            "extra": "mean: 211.22659300389094 usec\nrounds: 3516"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 675.1941346811807,
            "unit": "iter/sec",
            "range": "stddev: 0.00004099735183951965",
            "extra": "mean: 1.4810555196427901 msec\nrounds: 560"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 176.6304618502877,
            "unit": "iter/sec",
            "range": "stddev: 0.0000692437283726854",
            "extra": "mean: 5.661537593937799 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2236766.399403241,
            "unit": "iter/sec",
            "range": "stddev: 3.56410837336461e-8",
            "extra": "mean: 447.0739547351907 nsec\nrounds: 54723"
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
        "date": 1792237197008,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6024.093799448274,
            "unit": "iter/sec",
            "range": "stddev: 0.000018344621824071996",
            "extra": "mean: 166.00007126243398 usec\nrounds: 1726"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6513.10865466774,
            "unit": "iter/sec",
            "range": "stddev: 0.000016978396157847475",
            "extra": "mean: 153.5365142854375 usec\nrounds: 1995"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4993.3516409969325,
            "unit": "iter/sec",
            "range": "stddev: 0.00001898264398536837",
            "extra": "mean: 200.26628843634734 usec\nrounds: 3096"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 732.6134613786185,
            "unit": "iter/sec",
            "range": "stddev: 0.000024623340299558",
            "extra": "mean: 1.364976283834887 msec\nrounds: 532"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 184.4128163408058,
            "unit": "iter/sec",
            "range": "stddev: 0.00006541075043656505",
            "extra": "mean: 5.42261660465041 msec\nrounds: 172"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2320531.825593228,
            "unit": "iter/sec",
            "range": "stddev: 4.9651985858198554e-8",
            "extra": "mean: 430.9356971410452 nsec\nrounds: 112918"
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
        "date": 1792325247502,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14481.588574182435,
            "unit": "iter/sec",
            "range": "stddev: 0.000011470948232240027",
            "extra": "mean: 69.05319778127003 usec\nrounds: 2073"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 15114.040299372935,
            "unit": "iter/sec",
            "range": "stddev: 0.000011807921999988478",
            "extra": "mean: 66.16364520620532 usec\nrounds: 2858"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11418.746622744884,
            "unit": "iter/sec",
            "range": "stddev: 0.000013452585574466448",
            "extra": "mean: 87.57528589066949 usec\nrounds: 4869"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1456.3928747106131,
            "unit": "iter/sec",
            "range": "stddev: 0.00001696077962989667",
            "extra": "mean: 686.6279129515112 usec\nrounds: 942"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 302.3897679389984,
            "unit": "iter/sec",
            "range": "stddev: 0.00005729590625904831",
            "extra": "mean: 3.3069902027959217 msec\nrounds: 286"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 4616784.999379155,
            "unit": "iter/sec",
            "range": "stddev: 2.419245825886654e-8",
            "extra": "mean: 216.60094635866204 nsec\nrounds: 169809"
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
        "date": 1792420764798,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11001.010181166212,
            "unit": "iter/sec",
            "range": "stddev: 0.00001151306764059439",
            "extra": "mean: 90.90074307103227 usec\nrounds: 2273"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12034.059102933723,
            "unit": "iter/sec",
            "range": "stddev: 0.000010761947562419363",
            "extra": "mean: 83.09748119453852 usec\nrounds: 2712"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8685.821872792974,
            "unit": "iter/sec",
            "range": "stddev: 0.000012713770077955388",
            "extra": "mean: 115.13015286812974 usec\nrounds: 3696"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1056.4559347396082,
            "unit": "iter/sec",
            "range": "stddev: 0.000019559172995262213",
            "extra": "mean: 946.5610132110969 usec\nrounds: 757"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 228.02755198949782,
            "unit": "iter/sec",
            "range": "stddev: 0.00007606512820530003",
            "extra": "mean: 4.385434967288763 msec\nrounds: 214"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3003165.620904128,
            "unit": "iter/sec",
            "range": "stddev: 9.853606573840627e-8",
            "extra": "mean: 332.98196844000284 nsec\nrounds: 197317"
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
        "date": 1792500346522,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14594.908745136298,
            "unit": "iter/sec",
            "range": "stddev: 0.000008842488595860747",
            "extra": "mean: 68.51704368026601 usec\nrounds: 2152"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 15006.12920109718,
            "unit": "iter/sec",
            "range": "stddev: 0.000009485262511663611",
            "extra": "mean: 66.6394368993494 usec\nrounds: 3122"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11170.030925748484,
            "unit": "iter/sec",
            "range": "stddev: 0.000009540385303227014",
            "extra": "mean: 89.52526690815691 usec\nrounds: 4968"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1434.6819399277897,
            "unit": "iter/sec",
            "range": "stddev: 0.00001571461735974607",
            "extra": "mean: 697.0186019420667 usec\nrounds: 1236"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 291.51537097419276,
            "unit": "iter/sec",
            "range": "stddev: 0.0000788831537912701",
            "extra": "mean: 3.4303508479095868 msec\nrounds: 263"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 4454279.986908479,
            "unit": "iter/sec",
            "range": "stddev: 2.589879631198045e-8",
            "extra": "mean: 224.50317513472166 nsec\nrounds: 196928"
          }
        ]
      }
    ]
  }
}