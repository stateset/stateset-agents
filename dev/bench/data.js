window.BENCHMARK_DATA = {
  "lastUpdate": 1788232634697,
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
          "id": "f11019e28b06ebfaf465475b3b3802e01e33281c",
          "message": "feat: distributed async control plane and A+ evidence gates",
          "timestamp": "2026-08-31T22:45:21Z",
          "url": "https://github.com/stateset/stateset-agents/pull/50/commits/f11019e28b06ebfaf465475b3b3802e01e33281c"
        },
        "date": 1788232633901,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5924.439746145133,
            "unit": "iter/sec",
            "range": "stddev: 0.00002237025546599185",
            "extra": "mean: 168.79233190795335 usec\nrounds: 2103"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6153.193176788077,
            "unit": "iter/sec",
            "range": "stddev: 0.000031347862575665344",
            "extra": "mean: 162.5172445052331 usec\nrounds: 2184"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5010.927260820949,
            "unit": "iter/sec",
            "range": "stddev: 0.000018551345289264133",
            "extra": "mean: 199.56386272431504 usec\nrounds: 3635"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 734.9090750576092,
            "unit": "iter/sec",
            "range": "stddev: 0.0001359757636443832",
            "extra": "mean: 1.3607125479048008 msec\nrounds: 668"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.0767460616391,
            "unit": "iter/sec",
            "range": "stddev: 0.00006064420630064172",
            "extra": "mean: 5.4921895389182005 msec\nrounds: 167"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2214587.3919248083,
            "unit": "iter/sec",
            "range": "stddev: 4.612382595547753e-8",
            "extra": "mean: 451.55138318151916 nsec\nrounds: 105843"
          }
        ]
      }
    ]
  }
}