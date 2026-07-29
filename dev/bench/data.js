window.BENCHMARK_DATA = {
  "lastUpdate": 1785318277980,
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
          "id": "352c052cfcf49a97c763fa48f97ed5d2468878c0",
          "message": "polish(ingest): fix docstring, drop dead constant, warn about single-file grading semantics\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-28T17:34:18Z",
          "url": "https://github.com/stateset/stateset-agents/commit/352c052cfcf49a97c763fa48f97ed5d2468878c0"
        },
        "date": 1785261112997,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8602.089474046952,
            "unit": "iter/sec",
            "range": "stddev: 0.000015346275758086272",
            "extra": "mean: 116.25082522299532 usec\nrounds: 1459"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9154.697388042141,
            "unit": "iter/sec",
            "range": "stddev: 0.00001539691858107769",
            "extra": "mean: 109.23353963684252 usec\nrounds: 2094"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6747.217589772149,
            "unit": "iter/sec",
            "range": "stddev: 0.000015841874085770306",
            "extra": "mean: 148.20924131983858 usec\nrounds: 3456"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 828.0605845747955,
            "unit": "iter/sec",
            "range": "stddev: 0.00002350317526443183",
            "extra": "mean: 1.207641105769446 msec\nrounds: 728"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 160.2610250350036,
            "unit": "iter/sec",
            "range": "stddev: 0.006566298067531278",
            "extra": "mean: 6.239820316771241 msec\nrounds: 161"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2277826.287690625,
            "unit": "iter/sec",
            "range": "stddev: 6.797043664226711e-8",
            "extra": "mean: 439.015040525259 nsec\nrounds: 57124"
          }
        ]
      },
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
          "id": "3af189a6aac8af3967a1a145a6682079a15de1d6",
          "message": "Merge feat/launch-kit: five-minute offline demo script + Colab notebook\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-29T00:07:48Z",
          "url": "https://github.com/stateset/stateset-agents/commit/3af189a6aac8af3967a1a145a6682079a15de1d6"
        },
        "date": 1785318277583,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5979.664854080388,
            "unit": "iter/sec",
            "range": "stddev: 0.00001798466315662586",
            "extra": "mean: 167.23345277747845 usec\nrounds: 1440"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6423.441076542588,
            "unit": "iter/sec",
            "range": "stddev: 0.000018616818281438017",
            "extra": "mean: 155.6797965582412 usec\nrounds: 1976"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4963.110642975276,
            "unit": "iter/sec",
            "range": "stddev: 0.000018820384409604045",
            "extra": "mean: 201.48654179519195 usec\nrounds: 3577"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 720.4290734243727,
            "unit": "iter/sec",
            "range": "stddev: 0.00023218862985011172",
            "extra": "mean: 1.3880616939107682 msec\nrounds: 624"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 170.4592911961198,
            "unit": "iter/sec",
            "range": "stddev: 0.005564449685531265",
            "extra": "mean: 5.866503333335245 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2124293.7509211847,
            "unit": "iter/sec",
            "range": "stddev: 4.996213259404273e-8",
            "extra": "mean: 470.74468847180725 nsec\nrounds: 103746"
          }
        ]
      }
    ]
  }
}