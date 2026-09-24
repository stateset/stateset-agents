window.BENCHMARK_DATA = {
  "lastUpdate": 1790259685919,
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
          "id": "a32866a761ff99689e683e1e06579ad1563eb349",
          "message": "Harden checkpoint loads and expand blocking type checks",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/a32866a761ff99689e683e1e06579ad1563eb349"
        },
        "date": 1790222716345,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6074.089670495869,
            "unit": "iter/sec",
            "range": "stddev: 0.000016488465777368113",
            "extra": "mean: 164.63372361086056 usec\nrounds: 2160"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6517.768900384284,
            "unit": "iter/sec",
            "range": "stddev: 0.000016048203103037192",
            "extra": "mean: 153.42673471301512 usec\nrounds: 2175"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5035.010421510033,
            "unit": "iter/sec",
            "range": "stddev: 0.00001782134855071557",
            "extra": "mean: 198.60932079264563 usec\nrounds: 3535"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 738.268808191504,
            "unit": "iter/sec",
            "range": "stddev: 0.00003055572605660913",
            "extra": "mean: 1.35452018140878 msec\nrounds: 667"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.01974086053863,
            "unit": "iter/sec",
            "range": "stddev: 0.00013407065580501218",
            "extra": "mean: 5.493909590642633 msec\nrounds: 171"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2224726.2795082508,
            "unit": "iter/sec",
            "range": "stddev: 4.780410735350493e-8",
            "extra": "mean: 449.4934991378077 nsec\nrounds: 105286"
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
          "id": "2f65db802d19c6232a734b47789c78d5f5867941",
          "message": "Harden checkpoint loads and expand blocking type checks",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/2f65db802d19c6232a734b47789c78d5f5867941"
        },
        "date": 1790224433144,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11844.774054978292,
            "unit": "iter/sec",
            "range": "stddev: 0.000008467350397130748",
            "extra": "mean: 84.42541794030302 usec\nrounds: 2486"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12191.092367404679,
            "unit": "iter/sec",
            "range": "stddev: 0.000009675701903205196",
            "extra": "mean: 82.02710387739327 usec\nrounds: 2089"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 9054.224802753228,
            "unit": "iter/sec",
            "range": "stddev: 0.000011851548913570653",
            "extra": "mean: 110.44567831979586 usec\nrounds: 3976"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1070.4441497186776,
            "unit": "iter/sec",
            "range": "stddev: 0.0000540384721778237",
            "extra": "mean: 934.1916626503205 usec\nrounds: 830"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 268.0652511190693,
            "unit": "iter/sec",
            "range": "stddev: 0.0001442410820268549",
            "extra": "mean: 3.7304350184344472 msec\nrounds: 217"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2900107.0601699646,
            "unit": "iter/sec",
            "range": "stddev: 3.5622510793757655e-8",
            "extra": "mean: 344.81485657339624 nsec\nrounds: 138851"
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
          "id": "a7ae0f232d3b325082b19a67163947d42a93cd90",
          "message": "Harden checkpoints, expand type checks, and replace vulnerable scanner",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/a7ae0f232d3b325082b19a67163947d42a93cd90"
        },
        "date": 1790224455909,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5955.807304243475,
            "unit": "iter/sec",
            "range": "stddev: 0.000020103589700455985",
            "extra": "mean: 167.90335027923186 usec\nrounds: 1613"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6409.941212855507,
            "unit": "iter/sec",
            "range": "stddev: 0.000021268289489079193",
            "extra": "mean: 156.0076710211386 usec\nrounds: 1684"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4838.16750243673,
            "unit": "iter/sec",
            "range": "stddev: 0.000033617151486873176",
            "extra": "mean: 206.68982615760052 usec\nrounds: 3089"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 740.3265379186929,
            "unit": "iter/sec",
            "range": "stddev: 0.000038711628487818194",
            "extra": "mean: 1.3507553069910698 msec\nrounds: 658"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.33855852687202,
            "unit": "iter/sec",
            "range": "stddev: 0.00005636419849900176",
            "extra": "mean: 5.545125835365881 msec\nrounds: 164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2213433.8810042804,
            "unit": "iter/sec",
            "range": "stddev: 5.013029392252411e-8",
            "extra": "mean: 451.78670507486737 nsec\nrounds: 107447"
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
          "id": "2e77a0b4d288c9eec3c10f162d9d04d3efac9a0f",
          "message": "Harden checkpoints, expand type checks, and replace vulnerable scanner",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/2e77a0b4d288c9eec3c10f162d9d04d3efac9a0f"
        },
        "date": 1790259683989,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6193.501195435056,
            "unit": "iter/sec",
            "range": "stddev: 0.000014887716527607379",
            "extra": "mean: 161.45956357238677 usec\nrounds: 1982"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6701.560927178752,
            "unit": "iter/sec",
            "range": "stddev: 0.000014257920918867403",
            "extra": "mean: 149.21896717291858 usec\nrounds: 2041"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5074.616309888852,
            "unit": "iter/sec",
            "range": "stddev: 0.000017661071237447584",
            "extra": "mean: 197.059233434321 usec\nrounds: 3637"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 743.2410659295617,
            "unit": "iter/sec",
            "range": "stddev: 0.000032517233123318014",
            "extra": "mean: 1.3454584869436852 msec\nrounds: 651"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.23400321595275,
            "unit": "iter/sec",
            "range": "stddev: 0.00013486687585313707",
            "extra": "mean: 5.517728363636217 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2099230.6954440856,
            "unit": "iter/sec",
            "range": "stddev: 6.525298461414047e-8",
            "extra": "mean: 476.36498559699896 nsec\nrounds: 104844"
          }
        ]
      }
    ]
  }
}