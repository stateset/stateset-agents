window.BENCHMARK_DATA = {
  "lastUpdate": 1787687095703,
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
      }
    ]
  }
}