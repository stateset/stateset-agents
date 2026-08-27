window.BENCHMARK_DATA = {
  "lastUpdate": 1787865465891,
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
      }
    ]
  }
}