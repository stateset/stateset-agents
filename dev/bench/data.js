window.BENCHMARK_DATA = {
  "lastUpdate": 1787912295911,
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
          "id": "b0465b29b884fb4eec52916f4817f1e20efa7e31",
          "message": "feat: add fail-closed training backend protocol",
          "timestamp": "2026-08-28T02:20:26Z",
          "url": "https://github.com/stateset/stateset-agents/pull/35/commits/b0465b29b884fb4eec52916f4817f1e20efa7e31"
        },
        "date": 1787910745320,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8575.351684013509,
            "unit": "iter/sec",
            "range": "stddev: 0.000018231234835065855",
            "extra": "mean: 116.61329317423066 usec\nrounds: 1992"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9483.118336609181,
            "unit": "iter/sec",
            "range": "stddev: 0.000011913361690176684",
            "extra": "mean: 105.45054532743116 usec\nrounds: 2162"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6753.663123611745,
            "unit": "iter/sec",
            "range": "stddev: 0.000017906676388126966",
            "extra": "mean: 148.06779398040467 usec\nrounds: 3422"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 840.6075871294689,
            "unit": "iter/sec",
            "range": "stddev: 0.000027025764279738516",
            "extra": "mean: 1.189615719999422 msec\nrounds: 725"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 175.2730641740679,
            "unit": "iter/sec",
            "range": "stddev: 0.0005872389615664663",
            "extra": "mean: 5.705383224240753 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2233842.197616746,
            "unit": "iter/sec",
            "range": "stddev: 3.514435174451323e-8",
            "extra": "mean: 447.65919502590003 nsec\nrounds: 54834"
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
          "id": "f221fa642dc8fed1c2217015d19f6b7bd76d020e",
          "message": "Merge pull request #35 from stateset/feat/backend-shootout-foundation\n\nfeat: add fail-closed training backend protocol",
          "timestamp": "2026-08-28T03:13:38-07:00",
          "tree_id": "b0bfa82e86632b1371d92b854e5f61546b734d64",
          "url": "https://github.com/stateset/stateset-agents/commit/f221fa642dc8fed1c2217015d19f6b7bd76d020e"
        },
        "date": 1787912189831,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10920.141692113844,
            "unit": "iter/sec",
            "range": "stddev: 0.00001049376123650948",
            "extra": "mean: 91.57390336081134 usec\nrounds: 2380"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11987.695030628744,
            "unit": "iter/sec",
            "range": "stddev: 0.000009919286085855534",
            "extra": "mean: 83.41887222230669 usec\nrounds: 2700"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8670.544258414902,
            "unit": "iter/sec",
            "range": "stddev: 0.000014627473826230202",
            "extra": "mean: 115.3330137297303 usec\nrounds: 3059"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1068.735433959141,
            "unit": "iter/sec",
            "range": "stddev: 0.00001554876544892983",
            "extra": "mean: 935.6852671156323 usec\nrounds: 891"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 229.58299586496662,
            "unit": "iter/sec",
            "range": "stddev: 0.0000746991912624739",
            "extra": "mean: 4.355723280953124 msec\nrounds: 210"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2990354.3283606316,
            "unit": "iter/sec",
            "range": "stddev: 4.1178368213405694e-8",
            "extra": "mean: 334.4085316298349 nsec\nrounds: 189108"
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
          "id": "89f5dfee9f222a4099473bf1e4a20e0df887e07c",
          "message": "feat: add fail-closed OpenRLHF adapter",
          "timestamp": "2026-08-28T10:13:51Z",
          "url": "https://github.com/stateset/stateset-agents/pull/36/commits/89f5dfee9f222a4099473bf1e4a20e0df887e07c"
        },
        "date": 1787912294718,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 7757.732085843557,
            "unit": "iter/sec",
            "range": "stddev: 0.000026386045368366156",
            "extra": "mean: 128.9036523734581 usec\nrounds: 1959"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9171.299669428361,
            "unit": "iter/sec",
            "range": "stddev: 0.00001724289693123927",
            "extra": "mean: 109.03580038207706 usec\nrounds: 2094"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6764.527139583196,
            "unit": "iter/sec",
            "range": "stddev: 0.000016725236089412966",
            "extra": "mean: 147.8299930453995 usec\nrounds: 3451"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 838.1184640095793,
            "unit": "iter/sec",
            "range": "stddev: 0.00002521933654935121",
            "extra": "mean: 1.1931487527621996 msec\nrounds: 724"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 178.1499066121058,
            "unit": "iter/sec",
            "range": "stddev: 0.00004798166864976576",
            "extra": "mean: 5.613250206060154 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2286797.991162977,
            "unit": "iter/sec",
            "range": "stddev: 3.502674294858789e-8",
            "extra": "mean: 437.2926703033522 nsec\nrounds: 57383"
          }
        ]
      }
    ]
  }
}