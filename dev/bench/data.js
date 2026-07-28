window.BENCHMARK_DATA = {
  "lastUpdate": 1785261113629,
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
      }
    ]
  }
}