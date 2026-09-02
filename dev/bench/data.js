window.BENCHMARK_DATA = {
  "lastUpdate": 1788363765300,
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
          "id": "a8390f84442d5247c67f32c066d32540069ecd75",
          "message": "Merge pull request #61 from stateset/release/0.48.0\n\nrelease: v0.48.0 flagship evidence pipeline",
          "timestamp": "2026-09-01T20:21:29Z",
          "url": "https://github.com/stateset/stateset-agents/commit/a8390f84442d5247c67f32c066d32540069ecd75"
        },
        "date": 1788351015669,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8593.856080473091,
            "unit": "iter/sec",
            "range": "stddev: 0.000015238421565408091",
            "extra": "mean: 116.36219999915917 usec\nrounds: 1890"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9267.892826110536,
            "unit": "iter/sec",
            "range": "stddev: 0.000013458971850463495",
            "extra": "mean: 107.89939188578974 usec\nrounds: 2169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6799.9545673996345,
            "unit": "iter/sec",
            "range": "stddev: 0.00001551657552697942",
            "extra": "mean: 147.05980607491165 usec\nrounds: 3424"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 829.0458519643039,
            "unit": "iter/sec",
            "range": "stddev: 0.000132399532246641",
            "extra": "mean: 1.2062059024005067 msec\nrounds: 625"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.9753122646351,
            "unit": "iter/sec",
            "range": "stddev: 0.00006700100712798728",
            "extra": "mean: 5.556317627217689 msec\nrounds: 169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2387781.614647398,
            "unit": "iter/sec",
            "range": "stddev: 3.9701815968622414e-8",
            "extra": "mean: 418.7987686418589 nsec\nrounds: 58391"
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
          "id": "0dc17de7fd053e71f16037220f076270b6672229",
          "message": "feat: golden path, Modal default, and npm publish contract",
          "timestamp": "2026-09-01T20:26:54Z",
          "url": "https://github.com/stateset/stateset-agents/pull/62/commits/0dc17de7fd053e71f16037220f076270b6672229"
        },
        "date": 1788363763376,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6090.782945804886,
            "unit": "iter/sec",
            "range": "stddev: 0.000016198550274867142",
            "extra": "mean: 164.18250476135654 usec\nrounds: 1890"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6588.210159690318,
            "unit": "iter/sec",
            "range": "stddev: 0.000015341070305101785",
            "extra": "mean: 151.7862933575582 usec\nrounds: 2168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5048.572172629491,
            "unit": "iter/sec",
            "range": "stddev: 0.00003166104472164119",
            "extra": "mean: 198.07580555576396 usec\nrounds: 3240"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 745.1179766682881,
            "unit": "iter/sec",
            "range": "stddev: 0.000025897700197039384",
            "extra": "mean: 1.3420693518513518 msec\nrounds: 648"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.62906271952733,
            "unit": "iter/sec",
            "range": "stddev: 0.00006608710367879341",
            "extra": "mean: 5.567027878787071 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2194770.1393878968,
            "unit": "iter/sec",
            "range": "stddev: 5.0278451341834145e-8",
            "extra": "mean: 455.62857907247263 nsec\nrounds: 103756"
          }
        ]
      }
    ]
  }
}