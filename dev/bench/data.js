window.BENCHMARK_DATA = {
  "lastUpdate": 1791948142741,
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
          "id": "838d740326ce87a607a3c33f6bccf2476e35c3cf",
          "message": "feat(bench): runnable agent-quality gate harness (tau3-bench + BFCL V4)",
          "timestamp": "2026-10-13T16:24:19Z",
          "url": "https://github.com/stateset/stateset-agents/pull/107/commits/838d740326ce87a607a3c33f6bccf2476e35c3cf"
        },
        "date": 1791909483885,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8530.063593358766,
            "unit": "iter/sec",
            "range": "stddev: 0.000015628896709951637",
            "extra": "mean: 117.2324202575193 usec\nrounds: 1787"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9154.025946586908,
            "unit": "iter/sec",
            "range": "stddev: 0.000014967187495344895",
            "extra": "mean: 109.24155184122584 usec\nrounds: 2064"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6703.467027041493,
            "unit": "iter/sec",
            "range": "stddev: 0.000016512756496410227",
            "extra": "mean: 149.17653744935924 usec\nrounds: 3191"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 820.4077624220481,
            "unit": "iter/sec",
            "range": "stddev: 0.00008181467283392365",
            "extra": "mean: 1.2189060681821815 msec\nrounds: 704"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 160.07005465940495,
            "unit": "iter/sec",
            "range": "stddev: 0.006668773884053531",
            "extra": "mean: 6.247264687500653 msec\nrounds: 160"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2317240.9451943124,
            "unit": "iter/sec",
            "range": "stddev: 7.338720063408299e-8",
            "extra": "mean: 431.5476998945162 nsec\nrounds: 105430"
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
          "id": "113197094e00462f344c81f7348b7b6da151e2e7",
          "message": "feat(bench): agent-quality harness — tau3-bench + BFCL endpoint, offline dry run (#107)\n\nfeat(bench): runnable agent-quality gate harness (tau3-bench + BFCL V4)",
          "timestamp": "2026-10-13T09:48:57-07:00",
          "tree_id": "972b79b42d033d65385361223781582edcdc3e80",
          "url": "https://github.com/stateset/stateset-agents/commit/113197094e00462f344c81f7348b7b6da151e2e7"
        },
        "date": 1791910273619,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14480.582791479597,
            "unit": "iter/sec",
            "range": "stddev: 0.000010510276584201606",
            "extra": "mean: 69.05799403242264 usec\nrounds: 2346"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 14853.695147486447,
            "unit": "iter/sec",
            "range": "stddev: 0.00000837641143142187",
            "extra": "mean: 67.32331517987433 usec\nrounds: 3195"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11374.833552397411,
            "unit": "iter/sec",
            "range": "stddev: 0.000008730640304148417",
            "extra": "mean: 87.91337432706746 usec\nrounds: 5017"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1477.4578068966402,
            "unit": "iter/sec",
            "range": "stddev: 0.000018816586601747578",
            "extra": "mean: 676.8382794636096 usec\nrounds: 1120"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 283.6950578844669,
            "unit": "iter/sec",
            "range": "stddev: 0.004701503446850607",
            "extra": "mean: 3.524911598591344 msec\nrounds: 284"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 4472781.152333844,
            "unit": "iter/sec",
            "range": "stddev: 1.1067173947979621e-7",
            "extra": "mean: 223.5745425367418 nsec\nrounds: 195810"
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
          "id": "0c9596451a38703a7d55db22c8a80b1414078f90",
          "message": "bench(flagship): pin dataset, strict 24B judge + rubric v2, 250-step budget",
          "timestamp": "2026-10-13T16:49:03Z",
          "url": "https://github.com/stateset/stateset-agents/pull/108/commits/0c9596451a38703a7d55db22c8a80b1414078f90"
        },
        "date": 1791910877793,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5938.230495237996,
            "unit": "iter/sec",
            "range": "stddev: 0.000020469334221934566",
            "extra": "mean: 168.40033420762683 usec\nrounds: 1906"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6425.646331462923,
            "unit": "iter/sec",
            "range": "stddev: 0.000018024698483779983",
            "extra": "mean: 155.62636790380753 usec\nrounds: 1832"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4973.678178114482,
            "unit": "iter/sec",
            "range": "stddev: 0.00001990859897309207",
            "extra": "mean: 201.05844491512707 usec\nrounds: 3068"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 722.3963214641412,
            "unit": "iter/sec",
            "range": "stddev: 0.0001009763512592562",
            "extra": "mean: 1.3842816889947835 msec\nrounds: 627"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 169.4374210015304,
            "unit": "iter/sec",
            "range": "stddev: 0.005845707527147497",
            "extra": "mean: 5.901883976332287 msec\nrounds: 169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2328486.064225939,
            "unit": "iter/sec",
            "range": "stddev: 4.922907524731908e-8",
            "extra": "mean: 429.4635966964359 nsec\nrounds: 111038"
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
          "id": "7729c086090b3ae562291ac647e294c392002fe2",
          "message": "bench(flagship): pin dataset, strict 24B judge + rubric v2, 250-step budget",
          "timestamp": "2026-10-13T16:49:03Z",
          "url": "https://github.com/stateset/stateset-agents/pull/108/commits/7729c086090b3ae562291ac647e294c392002fe2"
        },
        "date": 1791912604316,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6033.408632894414,
            "unit": "iter/sec",
            "range": "stddev: 0.000019817757390652186",
            "extra": "mean: 165.7437877732921 usec\nrounds: 1734"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6493.304195158075,
            "unit": "iter/sec",
            "range": "stddev: 0.000018097757414465402",
            "extra": "mean: 154.00479785710328 usec\nrounds: 2147"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4971.192363189181,
            "unit": "iter/sec",
            "range": "stddev: 0.000020150571906663525",
            "extra": "mean: 201.15898298461084 usec\nrounds: 3115"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 724.4897442299001,
            "unit": "iter/sec",
            "range": "stddev: 0.00003117856498484898",
            "extra": "mean: 1.3802817886165593 msec\nrounds: 615"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 168.5723506883693,
            "unit": "iter/sec",
            "range": "stddev: 0.0064092328712805164",
            "extra": "mean: 5.932170939756583 msec\nrounds: 166"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2270931.788372845,
            "unit": "iter/sec",
            "range": "stddev: 1.218342930459511e-7",
            "extra": "mean: 440.347880601255 nsec\nrounds: 108144"
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
          "id": "d24789eb5483cf4ef4b7e4eef0c0cc541e4d8753",
          "message": "bench(flagship): pin dataset, strict 24B judge + rubric v2, 250-step budget",
          "timestamp": "2026-10-13T16:49:03Z",
          "url": "https://github.com/stateset/stateset-agents/pull/108/commits/d24789eb5483cf4ef4b7e4eef0c0cc541e4d8753"
        },
        "date": 1791929171587,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6041.2577692063105,
            "unit": "iter/sec",
            "range": "stddev: 0.000019265211603996162",
            "extra": "mean: 165.5284442748382 usec\nrounds: 1965"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6367.873000118956,
            "unit": "iter/sec",
            "range": "stddev: 0.000023724995085562363",
            "extra": "mean: 157.038307764825 usec\nrounds: 2125"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4896.282300606534,
            "unit": "iter/sec",
            "range": "stddev: 0.000031513679033821475",
            "extra": "mean: 204.2365898461622 usec\nrounds: 3250"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 727.0151691366688,
            "unit": "iter/sec",
            "range": "stddev: 0.000047414511011358666",
            "extra": "mean: 1.375487118360269 msec\nrounds: 659"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 165.75984860186537,
            "unit": "iter/sec",
            "range": "stddev: 0.00658583872879742",
            "extra": "mean: 6.032824042943453 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2320369.8568257755,
            "unit": "iter/sec",
            "range": "stddev: 5.017227170228254e-8",
            "extra": "mean: 430.96577774371804 nsec\nrounds: 109927"
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
          "id": "99dc111de9f2bde5fae12ecf612f4ac324eacff3",
          "message": "bench(flagship): pin dataset, strict 24B judge + rubric v2, 250-step budget",
          "timestamp": "2026-10-13T16:49:03Z",
          "url": "https://github.com/stateset/stateset-agents/pull/108/commits/99dc111de9f2bde5fae12ecf612f4ac324eacff3"
        },
        "date": 1791930846759,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 15032.257077014001,
            "unit": "iter/sec",
            "range": "stddev: 0.00002202738899736585",
            "extra": "mean: 66.52360952029696 usec\nrounds: 2689"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 15842.445976544517,
            "unit": "iter/sec",
            "range": "stddev: 0.000007729036693156951",
            "extra": "mean: 63.12156604356088 usec\nrounds: 3422"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11984.89598365986,
            "unit": "iter/sec",
            "range": "stddev: 0.000008462624364961511",
            "extra": "mean: 83.43835452250853 usec\nrounds: 5506"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1534.079986347717,
            "unit": "iter/sec",
            "range": "stddev: 0.000010558859385965263",
            "extra": "mean: 651.856493076847 usec\nrounds: 1300"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 288.1014323003545,
            "unit": "iter/sec",
            "range": "stddev: 0.004656510350809776",
            "extra": "mean: 3.470999751773083 msec\nrounds: 282"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 4649300.958136025,
            "unit": "iter/sec",
            "range": "stddev: 2.8956468739796404e-8",
            "extra": "mean: 215.08609767454482 nsec\nrounds: 52497"
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
          "id": "e2e381bc0e31ce92be49a9b4d5268209ca3ed416",
          "message": "bench(flagship): pin dataset, strict 24B judge + rubric v2, 250-step budget",
          "timestamp": "2026-10-13T16:49:03Z",
          "url": "https://github.com/stateset/stateset-agents/pull/108/commits/e2e381bc0e31ce92be49a9b4d5268209ca3ed416"
        },
        "date": 1791941033614,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5992.558641189433,
            "unit": "iter/sec",
            "range": "stddev: 0.000017897711573061255",
            "extra": "mean: 166.87362775669308 usec\nrounds: 1859"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6421.536840897002,
            "unit": "iter/sec",
            "range": "stddev: 0.00001765786693289861",
            "extra": "mean: 155.72596167809473 usec\nrounds: 1931"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4955.876562669253,
            "unit": "iter/sec",
            "range": "stddev: 0.00001875268646998527",
            "extra": "mean: 201.7806511834097 usec\nrounds: 3380"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 729.1732456438167,
            "unit": "iter/sec",
            "range": "stddev: 0.000029837961614072003",
            "extra": "mean: 1.371416197692579 msec\nrounds: 607"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 167.2548455431058,
            "unit": "iter/sec",
            "range": "stddev: 0.0061605617556013",
            "extra": "mean: 5.978900023809921 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2316742.3435473577,
            "unit": "iter/sec",
            "range": "stddev: 4.712655525800506e-8",
            "extra": "mean: 431.64057616731634 nsec\nrounds: 111149"
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
          "id": "cce2cbc5b4dc11334f55b61d643f2aafd4e5dfc2",
          "message": "bench(flagship): pin dataset, strict 24B judge + rubric v2, 250-step budget",
          "timestamp": "2026-10-13T16:49:03Z",
          "url": "https://github.com/stateset/stateset-agents/pull/108/commits/cce2cbc5b4dc11334f55b61d643f2aafd4e5dfc2"
        },
        "date": 1791948140695,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14073.20326523937,
            "unit": "iter/sec",
            "range": "stddev: 0.000013293211496260922",
            "extra": "mean: 71.0570281088732 usec\nrounds: 1850"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 14631.980829758371,
            "unit": "iter/sec",
            "range": "stddev: 0.000011738194468106459",
            "extra": "mean: 68.34344656645602 usec\nrounds: 2723"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11051.425557326225,
            "unit": "iter/sec",
            "range": "stddev: 0.000011576059565324707",
            "extra": "mean: 90.48606397543698 usec\nrounds: 4830"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1389.1805626939156,
            "unit": "iter/sec",
            "range": "stddev: 0.000046723581403196366",
            "extra": "mean: 719.8488280463613 usec\nrounds: 1198"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 242.56070625064058,
            "unit": "iter/sec",
            "range": "stddev: 0.008001281911156479",
            "extra": "mean: 4.122679289062958 msec\nrounds: 128"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 4571092.250919095,
            "unit": "iter/sec",
            "range": "stddev: 2.1222809939118263e-8",
            "extra": "mean: 218.7660946459641 nsec\nrounds: 196194"
          }
        ]
      }
    ]
  }
}