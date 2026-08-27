window.BENCHMARK_DATA = {
  "lastUpdate": 1787870350727,
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
          "id": "ad74284da71ccf2141a13deed0c4bc8bbfa80876",
          "message": "release: merge v0.42.5 evidence and A+ hardening",
          "timestamp": "2026-08-25T20:06:05Z",
          "url": "https://github.com/stateset/stateset-agents/pull/31/commits/ad74284da71ccf2141a13deed0c4bc8bbfa80876"
        },
        "date": 1787863752384,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6285.266700761778,
            "unit": "iter/sec",
            "range": "stddev: 0.000016201880820761197",
            "extra": "mean: 159.10223823577755 usec\nrounds: 2019"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6512.879231946796,
            "unit": "iter/sec",
            "range": "stddev: 0.000035416883407708545",
            "extra": "mean: 153.5419227635648 usec\nrounds: 2214"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5193.514352813453,
            "unit": "iter/sec",
            "range": "stddev: 0.000017434896592427397",
            "extra": "mean: 192.54784565258313 usec\nrounds: 3680"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 761.9301734297877,
            "unit": "iter/sec",
            "range": "stddev: 0.000029656192444565473",
            "extra": "mean: 1.3124562261375656 msec\nrounds: 681"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.97198425323356,
            "unit": "iter/sec",
            "range": "stddev: 0.000054780190844914884",
            "extra": "mean: 5.465317568049096 msec\nrounds: 169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2208935.009563153,
            "unit": "iter/sec",
            "range": "stddev: 4.83615619303134e-8",
            "extra": "mean: 452.70684545751465 nsec\nrounds: 106304"
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
          "id": "01ed6b47e3f7f58a1823bd82d63afeb2d925b712",
          "message": "release: merge v0.42.5 evidence and A+ hardening",
          "timestamp": "2026-08-25T20:06:05Z",
          "url": "https://github.com/stateset/stateset-agents/pull/31/commits/01ed6b47e3f7f58a1823bd82d63afeb2d925b712"
        },
        "date": 1787864688280,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5682.00671255832,
            "unit": "iter/sec",
            "range": "stddev: 0.00003166883746565654",
            "extra": "mean: 175.99416026556412 usec\nrounds: 2109"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6627.627327957832,
            "unit": "iter/sec",
            "range": "stddev: 0.000014035022581630817",
            "extra": "mean: 150.88355915572123 usec\nrounds: 2037"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5043.177909314094,
            "unit": "iter/sec",
            "range": "stddev: 0.000015812476709347815",
            "extra": "mean: 198.28767058824755 usec\nrounds: 3655"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 747.3197908718869,
            "unit": "iter/sec",
            "range": "stddev: 0.000031780763337099724",
            "extra": "mean: 1.3381152382346448 msec\nrounds: 680"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.8769521582204,
            "unit": "iter/sec",
            "range": "stddev: 0.0002925379917302557",
            "extra": "mean: 5.559355926380143 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2238578.4779930683,
            "unit": "iter/sec",
            "range": "stddev: 4.6060619366050925e-8",
            "extra": "mean: 446.71205849192324 nsec\nrounds: 107331"
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
          "id": "0348b0fbe4dddc4386cd63a7e2a51f3aae4fe511",
          "message": "release: merge v0.42.5 evidence and A+ hardening",
          "timestamp": "2026-08-25T20:06:05Z",
          "url": "https://github.com/stateset/stateset-agents/pull/31/commits/0348b0fbe4dddc4386cd63a7e2a51f3aae4fe511"
        },
        "date": 1787865463911,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11076.601952691786,
            "unit": "iter/sec",
            "range": "stddev: 0.000011482661192168763",
            "extra": "mean: 90.28039504091636 usec\nrounds: 2339"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12168.633717688832,
            "unit": "iter/sec",
            "range": "stddev: 0.000009372915295601506",
            "extra": "mean: 82.17849457875936 usec\nrounds: 2398"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 9003.946776822468,
            "unit": "iter/sec",
            "range": "stddev: 0.000011754591036058346",
            "extra": "mean: 111.06240682965303 usec\nrounds: 4041"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1016.9611052723993,
            "unit": "iter/sec",
            "range": "stddev: 0.00008167332360004173",
            "extra": "mean: 983.3217758432795 usec\nrounds: 919"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 266.10638237190176,
            "unit": "iter/sec",
            "range": "stddev: 0.00026066889644388915",
            "extra": "mean: 3.757895587045455 msec\nrounds: 247"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2672849.0526901106,
            "unit": "iter/sec",
            "range": "stddev: 4.565570168952575e-8",
            "extra": "mean: 374.1326129111339 nsec\nrounds: 139802"
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
          "id": "15b37bbd60cfc888dde711512d8f53919304093a",
          "message": "release: merge v0.42.5 evidence and A+ hardening",
          "timestamp": "2026-08-25T20:06:05Z",
          "url": "https://github.com/stateset/stateset-agents/pull/31/commits/15b37bbd60cfc888dde711512d8f53919304093a"
        },
        "date": 1787865848257,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6087.886897300225,
            "unit": "iter/sec",
            "range": "stddev: 0.00001612776929395682",
            "extra": "mean: 164.2606074767037 usec\nrounds: 1926"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6545.679022663916,
            "unit": "iter/sec",
            "range": "stddev: 0.000015099831294229646",
            "extra": "mean: 152.77253842383288 usec\nrounds: 2043"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5036.6372348542045,
            "unit": "iter/sec",
            "range": "stddev: 0.000016764153174261165",
            "extra": "mean: 198.54517078972177 usec\nrounds: 2951"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 747.1720802956281,
            "unit": "iter/sec",
            "range": "stddev: 0.00003192658010529628",
            "extra": "mean: 1.3383797740466121 msec\nrounds: 655"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.9538893039349,
            "unit": "iter/sec",
            "range": "stddev: 0.00005446713901626424",
            "extra": "mean: 5.526269724550512 msec\nrounds: 167"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2251948.3047357635,
            "unit": "iter/sec",
            "range": "stddev: 5.02900108707365e-8",
            "extra": "mean: 444.05992708493227 nsec\nrounds: 106975"
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
          "id": "9b06df2522137ab14effc5facc090fcc0922efc6",
          "message": "release: merge v0.42.5 evidence and A+ hardening",
          "timestamp": "2026-08-25T20:06:05Z",
          "url": "https://github.com/stateset/stateset-agents/pull/31/commits/9b06df2522137ab14effc5facc090fcc0922efc6"
        },
        "date": 1787866514671,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11235.699908574195,
            "unit": "iter/sec",
            "range": "stddev: 0.000011826720277120736",
            "extra": "mean: 89.00202107007853 usec\nrounds: 2278"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12370.249758839585,
            "unit": "iter/sec",
            "range": "stddev: 0.000011144108747816384",
            "extra": "mean: 80.83911153737343 usec\nrounds: 2080"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8914.950806047664,
            "unit": "iter/sec",
            "range": "stddev: 0.00001212025407065316",
            "extra": "mean: 112.17111813131115 usec\nrounds: 4368"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1070.462541079812,
            "unit": "iter/sec",
            "range": "stddev: 0.0000820071792097575",
            "extra": "mean: 934.175612526587 usec\nrounds: 942"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 229.76703801808569,
            "unit": "iter/sec",
            "range": "stddev: 0.0000439196910645888",
            "extra": "mean: 4.35223437019407 msec\nrounds: 208"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3022883.625154564,
            "unit": "iter/sec",
            "range": "stddev: 3.917995348935544e-8",
            "extra": "mean: 330.80995632071966 nsec\nrounds: 199721"
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
          "id": "98f18cbd6e00f39c00a8b56ae08468dd0119e9a3",
          "message": "release: merge v0.42.5 evidence and A+ hardening",
          "timestamp": "2026-08-25T20:06:05Z",
          "url": "https://github.com/stateset/stateset-agents/pull/31/commits/98f18cbd6e00f39c00a8b56ae08468dd0119e9a3"
        },
        "date": 1787868647139,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5929.073654571037,
            "unit": "iter/sec",
            "range": "stddev: 0.000022136167326611913",
            "extra": "mean: 168.66041109626747 usec\nrounds: 1496"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6589.290548337366,
            "unit": "iter/sec",
            "range": "stddev: 0.000014906167745628007",
            "extra": "mean: 151.76140627951574 usec\nrounds: 2134"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5032.156045848738,
            "unit": "iter/sec",
            "range": "stddev: 0.000020526316980331905",
            "extra": "mean: 198.72197739673572 usec\nrounds: 2389"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 736.4034469272615,
            "unit": "iter/sec",
            "range": "stddev: 0.00003270559637687413",
            "extra": "mean: 1.3579512754491159 msec\nrounds: 668"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 178.7136889229087,
            "unit": "iter/sec",
            "range": "stddev: 0.00005306742184040312",
            "extra": "mean: 5.595542266666364 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2196298.3529178156,
            "unit": "iter/sec",
            "range": "stddev: 5.0934641559890135e-8",
            "extra": "mean: 455.3115466628133 nsec\nrounds: 105065"
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
          "id": "87cbbeda1b3ea1efd02d33a16cf062bdccce24ac",
          "message": "release: merge v0.42.5 evidence and A+ hardening",
          "timestamp": "2026-08-25T20:06:05Z",
          "url": "https://github.com/stateset/stateset-agents/pull/31/commits/87cbbeda1b3ea1efd02d33a16cf062bdccce24ac"
        },
        "date": 1787869556206,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8468.521368832058,
            "unit": "iter/sec",
            "range": "stddev: 0.00001552589831695374",
            "extra": "mean: 118.08436874003137 usec\nrounds: 1817"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9277.15862316608,
            "unit": "iter/sec",
            "range": "stddev: 0.000013853912932186376",
            "extra": "mean: 107.7916246363289 usec\nrounds: 2062"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6749.721379720459,
            "unit": "iter/sec",
            "range": "stddev: 0.000016352559903030893",
            "extra": "mean: 148.1542635233064 usec\nrounds: 3457"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 830.4983246560757,
            "unit": "iter/sec",
            "range": "stddev: 0.000024385119122419527",
            "extra": "mean: 1.2040963483148723 msec\nrounds: 712"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 178.30851127371358,
            "unit": "iter/sec",
            "range": "stddev: 0.00016248995178024968",
            "extra": "mean: 5.608257243900959 msec\nrounds: 164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2312857.20997065,
            "unit": "iter/sec",
            "range": "stddev: 6.241663153950766e-8",
            "extra": "mean: 432.3656452672623 nsec\nrounds: 57821"
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
          "id": "04d417449e6ce9bdc0a93f51421469369b30abe3",
          "message": "release: merge v0.42.5 evidence and A+ hardening",
          "timestamp": "2026-08-25T20:06:05Z",
          "url": "https://github.com/stateset/stateset-agents/pull/31/commits/04d417449e6ce9bdc0a93f51421469369b30abe3"
        },
        "date": 1787870349707,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6091.084363427486,
            "unit": "iter/sec",
            "range": "stddev: 0.000023142716506406594",
            "extra": "mean: 164.17438018167502 usec\nrounds: 1978"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6616.1948664507745,
            "unit": "iter/sec",
            "range": "stddev: 0.00001660989585844508",
            "extra": "mean: 151.14427857480038 usec\nrounds: 2161"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5071.508237579104,
            "unit": "iter/sec",
            "range": "stddev: 0.000028497464846961906",
            "extra": "mean: 197.18000112671655 usec\nrounds: 2662"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 767.6887140070344,
            "unit": "iter/sec",
            "range": "stddev: 0.000028457509342680997",
            "extra": "mean: 1.3026113081438853 msec\nrounds: 688"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.50484051406886,
            "unit": "iter/sec",
            "range": "stddev: 0.00005384255979575262",
            "extra": "mean: 5.479306725143612 msec\nrounds: 171"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2212671.2812239425,
            "unit": "iter/sec",
            "range": "stddev: 6.633673065084887e-8",
            "extra": "mean: 451.9424138983936 nsec\nrounds: 170911"
          }
        ]
      }
    ]
  }
}