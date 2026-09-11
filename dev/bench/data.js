window.BENCHMARK_DATA = {
  "lastUpdate": 1789129007403,
  "repoUrl": "https://github.com/stateset/stateset-agents",
  "entries": {
    "Python Benchmark": [
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
          "id": "5ee45381e7911f163b2433f058343f668f4b14bf",
          "message": "Merge pull request #79 from stateset/release/0.54.0\n\nchore(release): v0.54.0 — Rollouts sample in inference mode",
          "timestamp": "2026-09-08T19:03:18-07:00",
          "tree_id": "8ca59b676d45ee6acf05b6fd594730ec0af3c3c2",
          "url": "https://github.com/stateset/stateset-agents/commit/5ee45381e7911f163b2433f058343f668f4b14bf"
        },
        "date": 1788919571617,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5948.682178449729,
            "unit": "iter/sec",
            "range": "stddev: 0.00002006976344282769",
            "extra": "mean: 168.10445910569854 usec\nrounds: 1834"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6425.800417157464,
            "unit": "iter/sec",
            "range": "stddev: 0.000016072323309571164",
            "extra": "mean: 155.62263610458712 usec\nrounds: 2105"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4952.089510820122,
            "unit": "iter/sec",
            "range": "stddev: 0.000017217343186421285",
            "extra": "mean: 201.9349605484794 usec\nrounds: 3574"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 749.740631119407,
            "unit": "iter/sec",
            "range": "stddev: 0.000026900090552523677",
            "extra": "mean: 1.3337945930807311 msec\nrounds: 607"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.1206013549686,
            "unit": "iter/sec",
            "range": "stddev: 0.000051277557584419565",
            "extra": "mean: 5.490866999999163 msec\nrounds: 162"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2243922.322469199,
            "unit": "iter/sec",
            "range": "stddev: 5.0663839357105723e-8",
            "extra": "mean: 445.6482249793771 nsec\nrounds: 106872"
          }
        ]
      }
    ],
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
          "id": "5ee45381e7911f163b2433f058343f668f4b14bf",
          "message": "Merge pull request #79 from stateset/release/0.54.0\n\nchore(release): v0.54.0 — Rollouts sample in inference mode",
          "timestamp": "2026-09-09T02:03:18Z",
          "url": "https://github.com/stateset/stateset-agents/commit/5ee45381e7911f163b2433f058343f668f4b14bf"
        },
        "date": 1788956597100,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8375.881886100715,
            "unit": "iter/sec",
            "range": "stddev: 0.000012974546608447398",
            "extra": "mean: 119.39041328405568 usec\nrounds: 2168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 8800.429030750536,
            "unit": "iter/sec",
            "range": "stddev: 0.000013546934553562472",
            "extra": "mean: 113.63082373663728 usec\nrounds: 2275"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6762.211916697525,
            "unit": "iter/sec",
            "range": "stddev: 0.000012708442729136805",
            "extra": "mean: 147.8806065705749 usec\nrounds: 2496"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 873.6242821732037,
            "unit": "iter/sec",
            "range": "stddev: 0.000016934237143726433",
            "extra": "mean: 1.1446568283477967 msec\nrounds: 769"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 199.90146908377656,
            "unit": "iter/sec",
            "range": "stddev: 0.00040808769171712516",
            "extra": "mean: 5.00246448704642 msec\nrounds: 193"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2320166.315300328,
            "unit": "iter/sec",
            "range": "stddev: 1.1041085257104055e-7",
            "extra": "mean: 431.00358513331724 nsec\nrounds: 115421"
          }
        ]
      },
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
          "id": "5ee45381e7911f163b2433f058343f668f4b14bf",
          "message": "Merge pull request #79 from stateset/release/0.54.0\n\nchore(release): v0.54.0 — Rollouts sample in inference mode",
          "timestamp": "2026-09-09T02:03:18Z",
          "url": "https://github.com/stateset/stateset-agents/commit/5ee45381e7911f163b2433f058343f668f4b14bf"
        },
        "date": 1789042713725,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6123.034052391598,
            "unit": "iter/sec",
            "range": "stddev: 0.000016975433489294485",
            "extra": "mean: 163.31772638262717 usec\nrounds: 2043"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6626.194941706217,
            "unit": "iter/sec",
            "range": "stddev: 0.00001658569065314782",
            "extra": "mean: 150.9161756932108 usec\nrounds: 2123"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5083.569345664847,
            "unit": "iter/sec",
            "range": "stddev: 0.00001821416634093378",
            "extra": "mean: 196.7121783934702 usec\nrounds: 3610"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 749.7353195562933,
            "unit": "iter/sec",
            "range": "stddev: 0.000029311581731276202",
            "extra": "mean: 1.3338040424610351 msec\nrounds: 683"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.12488100547407,
            "unit": "iter/sec",
            "range": "stddev: 0.00007409417245086077",
            "extra": "mean: 5.521052626501255 msec\nrounds: 166"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2237562.0751209115,
            "unit": "iter/sec",
            "range": "stddev: 4.9904198611610104e-8",
            "extra": "mean: 446.9149755078696 nsec\nrounds: 104080"
          }
        ]
      },
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
          "id": "5ee45381e7911f163b2433f058343f668f4b14bf",
          "message": "Merge pull request #79 from stateset/release/0.54.0\n\nchore(release): v0.54.0 — Rollouts sample in inference mode",
          "timestamp": "2026-09-09T02:03:18Z",
          "url": "https://github.com/stateset/stateset-agents/commit/5ee45381e7911f163b2433f058343f668f4b14bf"
        },
        "date": 1789129006897,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8563.214488779799,
            "unit": "iter/sec",
            "range": "stddev: 0.000014242744927424065",
            "extra": "mean: 116.7785767027416 usec\nrounds: 1923"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9306.673747016886,
            "unit": "iter/sec",
            "range": "stddev: 0.00001364321833994812",
            "extra": "mean: 107.44977498760336 usec\nrounds: 2031"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6714.857311634535,
            "unit": "iter/sec",
            "range": "stddev: 0.000015837394031103503",
            "extra": "mean: 148.92349212951171 usec\nrounds: 3367"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 832.4238173151605,
            "unit": "iter/sec",
            "range": "stddev: 0.000026167838091518536",
            "extra": "mean: 1.2013111340630878 msec\nrounds: 731"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.5319650495383,
            "unit": "iter/sec",
            "range": "stddev: 0.00004482086685916288",
            "extra": "mean: 5.570038737803988 msec\nrounds: 164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2369398.2581689255,
            "unit": "iter/sec",
            "range": "stddev: 4.6593417670343346e-8",
            "extra": "mean: 422.0480860709341 nsec\nrounds: 115835"
          }
        ]
      }
    ]
  }
}