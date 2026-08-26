window.BENCHMARK_DATA = {
  "lastUpdate": 1787731063016,
  "repoUrl": "https://github.com/stateset/stateset-agents",
  "entries": {
    "Python Benchmark": [
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "15a7458033960ef040dff7961cde847c2538200a",
          "message": "Merge feat/polish-2: lazy-export registry file and CLI init/evaluate extraction\n\ntraining/__init__.py 1679 -> 144 LOC via _registry.py; cli.py 1515 -> 839 via\ncli_init.py and cli_evaluate.py. --help byte-identical, export map unchanged.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01JsBFXmpa8URWHqS2irLChd",
          "timestamp": "2026-08-25T12:29:13-07:00",
          "tree_id": "b6df7e69cb631de003159bcbef4091577ffb3a61",
          "url": "https://github.com/stateset/stateset-agents/commit/15a7458033960ef040dff7961cde847c2538200a"
        },
        "date": 1787687094895,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6153.11421469922,
            "unit": "iter/sec",
            "range": "stddev: 0.000015905453900214907",
            "extra": "mean: 162.5193300672191 usec\nrounds: 1939"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6603.636265375436,
            "unit": "iter/sec",
            "range": "stddev: 0.000015637464507962435",
            "extra": "mean: 151.43172031495092 usec\nrounds: 2156"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5080.602904432998,
            "unit": "iter/sec",
            "range": "stddev: 0.000017461799359962426",
            "extra": "mean: 196.82703387967325 usec\nrounds: 3601"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 749.5002789070883,
            "unit": "iter/sec",
            "range": "stddev: 0.000043291004190152094",
            "extra": "mean: 1.3342223187137265 msec\nrounds: 684"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.29679049375636,
            "unit": "iter/sec",
            "range": "stddev: 0.00012631302787406218",
            "extra": "mean: 5.5773446766456365 msec\nrounds: 167"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2224787.873136108,
            "unit": "iter/sec",
            "range": "stddev: 4.760995951283024e-8",
            "extra": "mean: 449.4810548344004 nsec\nrounds: 106861"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "committer": {
            "email": "team@stateset.ai",
            "name": "domsteil"
          },
          "distinct": true,
          "id": "0a79e41d57d3905622abaf0fa2e8f2134a4a9347",
          "message": "Merge feat/polish-3: river behavioural golden, drift fixes, submit-family fold\n\n27-scenario offline golden over all five River submit modes (both episode paths\nhad zero coverage); four drift bugs it made visible; each submit family folded\ninto one implementation behind a closure-carrying mode adapter.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01JsBFXmpa8URWHqS2irLChd",
          "timestamp": "2026-08-25T12:44:44-07:00",
          "tree_id": "1e346da0be8cea1e8263b3208275fbb4a4b9a01d",
          "url": "https://github.com/stateset/stateset-agents/commit/0a79e41d57d3905622abaf0fa2e8f2134a4a9347"
        },
        "date": 1787688962346,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6212.4885540765645,
            "unit": "iter/sec",
            "range": "stddev: 0.000013763023277750785",
            "extra": "mean: 160.96609133288646 usec\nrounds: 2146"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6696.559387367281,
            "unit": "iter/sec",
            "range": "stddev: 0.00001358928857715323",
            "extra": "mean: 149.33041613674766 usec\nrounds: 2045"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5144.235204937641,
            "unit": "iter/sec",
            "range": "stddev: 0.0000153591847542205",
            "extra": "mean: 194.39235574612925 usec\nrounds: 3733"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 761.1858433708652,
            "unit": "iter/sec",
            "range": "stddev: 0.000020441433955890468",
            "extra": "mean: 1.3137396191862434 msec\nrounds: 688"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 183.29357579987717,
            "unit": "iter/sec",
            "range": "stddev: 0.00007652372854465828",
            "extra": "mean: 5.455728579881139 msec\nrounds: 169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2248378.573496877,
            "unit": "iter/sec",
            "range": "stddev: 4.9100135320049245e-8",
            "extra": "mean: 444.7649571952252 nsec\nrounds: 106417"
          }
        ]
      }
    ],
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
        "date": 1787731062213,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5955.5352705472615,
            "unit": "iter/sec",
            "range": "stddev: 0.00001966036723416306",
            "extra": "mean: 167.91101967701198 usec\nrounds: 1982"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6051.266709768301,
            "unit": "iter/sec",
            "range": "stddev: 0.000032907079226614556",
            "extra": "mean: 165.25465625002826 usec\nrounds: 1920"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4932.013102895031,
            "unit": "iter/sec",
            "range": "stddev: 0.000021899256051614354",
            "extra": "mean: 202.75696336106898 usec\nrounds: 3439"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 744.2662267847755,
            "unit": "iter/sec",
            "range": "stddev: 0.000033025239806714135",
            "extra": "mean: 1.3436052369593505 msec\nrounds: 671"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.00734405812403,
            "unit": "iter/sec",
            "range": "stddev: 0.00008391585257928919",
            "extra": "mean: 5.555328896342707 msec\nrounds: 164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2239569.097270928,
            "unit": "iter/sec",
            "range": "stddev: 4.9062454897270695e-8",
            "extra": "mean: 446.51446620627615 nsec\nrounds: 104516"
          }
        ]
      }
    ]
  }
}