window.BENCHMARK_DATA = {
  "lastUpdate": 1788294232933,
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
          "id": "dc4b0206a545d4cff6ed3f8ef7d03a66e994a72a",
          "message": "release: v0.48.0 flagship evidence pipeline",
          "timestamp": "2026-09-01T19:25:02Z",
          "url": "https://github.com/stateset/stateset-agents/pull/61/commits/dc4b0206a545d4cff6ed3f8ef7d03a66e994a72a"
        },
        "date": 1788293681927,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11056.365805283578,
            "unit": "iter/sec",
            "range": "stddev: 0.000012405267948595495",
            "extra": "mean: 90.4456326437864 usec\nrounds: 2273"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12002.086823084812,
            "unit": "iter/sec",
            "range": "stddev: 0.000010958018277597086",
            "extra": "mean: 83.31884402607388 usec\nrounds: 2603"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8807.988368131711,
            "unit": "iter/sec",
            "range": "stddev: 0.000011213790260445914",
            "extra": "mean: 113.53330161267152 usec\nrounds: 4340"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1078.757945472057,
            "unit": "iter/sec",
            "range": "stddev: 0.000020849815214867416",
            "extra": "mean: 926.9920135442497 usec\nrounds: 886"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 229.3824406908354,
            "unit": "iter/sec",
            "range": "stddev: 0.00005759475366161771",
            "extra": "mean: 4.359531605768433 msec\nrounds: 208"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3052057.9910405115,
            "unit": "iter/sec",
            "range": "stddev: 3.350598603726482e-8",
            "extra": "mean: 327.6477717446905 nsec\nrounds: 145540"
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
          "id": "a8390f84442d5247c67f32c066d32540069ecd75",
          "message": "Merge pull request #61 from stateset/release/0.48.0\n\nrelease: v0.48.0 flagship evidence pipeline",
          "timestamp": "2026-09-01T13:21:29-07:00",
          "tree_id": "1e8fd3e1b94513d80c53f4dbd714ee360f44276d",
          "url": "https://github.com/stateset/stateset-agents/commit/a8390f84442d5247c67f32c066d32540069ecd75"
        },
        "date": 1788294231288,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5953.479529413267,
            "unit": "iter/sec",
            "range": "stddev: 0.00003009672004600428",
            "extra": "mean: 167.96899948332447 usec\nrounds: 1937"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6588.0404278636715,
            "unit": "iter/sec",
            "range": "stddev: 0.000024909910081963073",
            "extra": "mean: 151.79020392324364 usec\nrounds: 2192"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5440.396219699331,
            "unit": "iter/sec",
            "range": "stddev: 0.00001744822610127951",
            "extra": "mean: 183.81014169134656 usec\nrounds: 3677"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 785.2331061633392,
            "unit": "iter/sec",
            "range": "stddev: 0.000052634845379214756",
            "extra": "mean: 1.2735071816903072 msec\nrounds: 710"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 195.1297414740635,
            "unit": "iter/sec",
            "range": "stddev: 0.0001334502290960741",
            "extra": "mean: 5.124795392264275 msec\nrounds: 181"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2390033.728268912,
            "unit": "iter/sec",
            "range": "stddev: 4.3563643734549415e-8",
            "extra": "mean: 418.4041372187222 nsec\nrounds: 117289"
          }
        ]
      }
    ]
  }
}