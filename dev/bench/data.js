window.BENCHMARK_DATA = {
  "lastUpdate": 1791908616975,
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
    ],
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
          "id": "adbc7138d3654f4f0a0a111c901936041c7471c4",
          "message": "bench: verl, OpenRLHF, NeMo-RL adapters + v6 five-framework manifest",
          "timestamp": "2026-10-13T15:44:22Z",
          "url": "https://github.com/stateset/stateset-agents/pull/100/commits/adbc7138d3654f4f0a0a111c901936041c7471c4"
        },
        "date": 1791906572850,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10862.646833894805,
            "unit": "iter/sec",
            "range": "stddev: 0.000012662414720500891",
            "extra": "mean: 92.05859449279819 usec\nrounds: 2397"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11796.57680605415,
            "unit": "iter/sec",
            "range": "stddev: 0.00001253814153846672",
            "extra": "mean: 84.77035469194652 usec\nrounds: 2419"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8699.902842930524,
            "unit": "iter/sec",
            "range": "stddev: 0.000011823007723953284",
            "extra": "mean: 114.94381236827174 usec\nrounds: 4269"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1069.6131577631681,
            "unit": "iter/sec",
            "range": "stddev: 0.0000567020543942358",
            "extra": "mean: 934.9174444443569 usec\nrounds: 855"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 212.56813729250118,
            "unit": "iter/sec",
            "range": "stddev: 0.004946864751789969",
            "extra": "mean: 4.7043739138757426 msec\nrounds: 209"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2973192.572974304,
            "unit": "iter/sec",
            "range": "stddev: 4.16547376394317e-8",
            "extra": "mean: 336.3387925456931 nsec\nrounds: 59155"
          }
        ]
      },
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
          "id": "ff3a7a3e400ec8caf361ac212cf8afdb3a7e4f3c",
          "message": "bench: verl, OpenRLHF, NeMo-RL matched GSPO adapters + v6 five-framework manifest (#100)\n\nbench: verl, OpenRLHF, NeMo-RL adapters + v6 five-framework manifest",
          "timestamp": "2026-10-13T09:10:03-07:00",
          "tree_id": "9981a88b62349f878e3aa9c5d15b5dc19d8ded25",
          "url": "https://github.com/stateset/stateset-agents/commit/ff3a7a3e400ec8caf361ac212cf8afdb3a7e4f3c"
        },
        "date": 1791907999353,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8177.773503705674,
            "unit": "iter/sec",
            "range": "stddev: 0.000023017111485237298",
            "extra": "mean: 122.28267260604126 usec\nrounds: 1964"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9247.895536440834,
            "unit": "iter/sec",
            "range": "stddev: 0.00001434536054465539",
            "extra": "mean: 108.13270933474041 usec\nrounds: 2164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6698.687935865651,
            "unit": "iter/sec",
            "range": "stddev: 0.00001657467360820511",
            "extra": "mean: 149.28296549625327 usec\nrounds: 3420"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 825.5488293402661,
            "unit": "iter/sec",
            "range": "stddev: 0.00002121875124271827",
            "extra": "mean: 1.2113153873637563 msec\nrounds: 728"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 162.9362812416933,
            "unit": "iter/sec",
            "range": "stddev: 0.0063651320721798636",
            "extra": "mean: 6.137368500000556 msec\nrounds: 162"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2332223.1398061477,
            "unit": "iter/sec",
            "range": "stddev: 3.360751361697793e-8",
            "extra": "mean: 428.7754387357288 nsec\nrounds: 58293"
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
          "id": "82a985b7423db0ff76dd98ab6f74899297465bf2",
          "message": "feat(bench): runnable agent-quality gate harness (tau3-bench + BFCL V4)",
          "timestamp": "2026-10-13T16:10:10Z",
          "url": "https://github.com/stateset/stateset-agents/pull/107/commits/82a985b7423db0ff76dd98ab6f74899297465bf2"
        },
        "date": 1791908615995,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6176.29526156002,
            "unit": "iter/sec",
            "range": "stddev: 0.000016446688317089793",
            "extra": "mean: 161.90935790000074 usec\nrounds: 1886"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6535.1907222498285,
            "unit": "iter/sec",
            "range": "stddev: 0.000017240240621768906",
            "extra": "mean: 153.0177224354573 usec\nrounds: 2086"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5069.588766728475,
            "unit": "iter/sec",
            "range": "stddev: 0.000017906066307833015",
            "extra": "mean: 197.25465831922372 usec\nrounds: 3594"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 729.660851274595,
            "unit": "iter/sec",
            "range": "stddev: 0.00005650369758478317",
            "extra": "mean: 1.3704997304613067 msec\nrounds: 627"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 171.35845028358278,
            "unit": "iter/sec",
            "range": "stddev: 0.00542508497276362",
            "extra": "mean: 5.835720376468684 msec\nrounds: 170"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2310324.4091113033,
            "unit": "iter/sec",
            "range": "stddev: 4.628645453835659e-8",
            "extra": "mean: 432.8396462662415 nsec\nrounds: 110902"
          }
        ]
      }
    ]
  }
}