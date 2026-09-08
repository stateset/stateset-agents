window.BENCHMARK_DATA = {
  "lastUpdate": 1788893945669,
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
          "id": "646efc2deba60a778804aec85082977fed946b82",
          "message": "Merge pull request #68 from stateset/release/0.51.0\n\nchore(release): v0.51.0 — Evidence-safe benchmarking and engine rollouts",
          "timestamp": "2026-09-08T11:51:28-07:00",
          "tree_id": "8018aa752f35a9e9d4a171701314f12f71619d09",
          "url": "https://github.com/stateset/stateset-agents/commit/646efc2deba60a778804aec85082977fed946b82"
        },
        "date": 1788893639330,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8530.595669628603,
            "unit": "iter/sec",
            "range": "stddev: 0.00001332468474641278",
            "extra": "mean: 117.22510815514211 usec\nrounds: 1729"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9233.710880037872,
            "unit": "iter/sec",
            "range": "stddev: 0.00001373736690362287",
            "extra": "mean: 108.298820809072 usec\nrounds: 2076"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6691.180167338151,
            "unit": "iter/sec",
            "range": "stddev: 0.000015677041912174966",
            "extra": "mean: 149.45046688196032 usec\nrounds: 3095"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 834.3745154556722,
            "unit": "iter/sec",
            "range": "stddev: 0.000019404729531066405",
            "extra": "mean: 1.1985025686623179 msec\nrounds: 568"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 177.76177122322784,
            "unit": "iter/sec",
            "range": "stddev: 0.0001732655772277349",
            "extra": "mean: 5.625506502994001 msec\nrounds: 167"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2267775.223974553,
            "unit": "iter/sec",
            "range": "stddev: 4.9414418604196975e-8",
            "extra": "mean: 440.9608101491549 nsec\nrounds: 110461"
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
          "id": "8e273b38598e87376df31b040baeaf8043fce01a",
          "message": "feat(training): keep engine rollouts on-policy; fix train() export shadowing",
          "timestamp": "2026-09-08T18:19:17Z",
          "url": "https://github.com/stateset/stateset-agents/pull/69/commits/8e273b38598e87376df31b040baeaf8043fce01a"
        },
        "date": 1788893650369,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5783.976757007549,
            "unit": "iter/sec",
            "range": "stddev: 0.000028190878221000557",
            "extra": "mean: 172.89142782056564 usec\nrounds: 2092"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6545.283796078054,
            "unit": "iter/sec",
            "range": "stddev: 0.000018900049923447655",
            "extra": "mean: 152.78176335137704 usec\nrounds: 2172"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5097.582720501229,
            "unit": "iter/sec",
            "range": "stddev: 0.000015646291352648996",
            "extra": "mean: 196.17141198675307 usec\nrounds: 3704"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 752.416632021282,
            "unit": "iter/sec",
            "range": "stddev: 0.000023844856790370992",
            "extra": "mean: 1.3290508973912674 msec\nrounds: 575"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.09886308821717,
            "unit": "iter/sec",
            "range": "stddev: 0.0001417967699746801",
            "extra": "mean: 5.552505900662868 msec\nrounds: 151"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2210782.1076481133,
            "unit": "iter/sec",
            "range": "stddev: 5.1988499010788194e-8",
            "extra": "mean: 452.3286110108 nsec\nrounds: 106293"
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
          "id": "b0b058a1ab96cb69a0473a2453305ffbcbb0e1f4",
          "message": "feat(training): keep engine rollouts on-policy; fix train() export shadowing",
          "timestamp": "2026-09-08T18:19:17Z",
          "url": "https://github.com/stateset/stateset-agents/pull/69/commits/b0b058a1ab96cb69a0473a2453305ffbcbb0e1f4"
        },
        "date": 1788893745374,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11029.582114589746,
            "unit": "iter/sec",
            "range": "stddev: 0.0000121561754436252",
            "extra": "mean: 90.66526633653842 usec\nrounds: 2020"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12036.637983840548,
            "unit": "iter/sec",
            "range": "stddev: 0.000009913575053873243",
            "extra": "mean: 83.07967734366706 usec\nrounds: 2560"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8702.267801820555,
            "unit": "iter/sec",
            "range": "stddev: 0.000012699187735653696",
            "extra": "mean: 114.91257483374567 usec\nrounds: 4363"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1073.6006320046947,
            "unit": "iter/sec",
            "range": "stddev: 0.000016364248413046387",
            "extra": "mean: 931.445055255544 usec\nrounds: 742"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 227.84939091466958,
            "unit": "iter/sec",
            "range": "stddev: 0.000034021002921635995",
            "extra": "mean: 4.388864047367603 msec\nrounds: 190"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2994458.7219874077,
            "unit": "iter/sec",
            "range": "stddev: 4.0721311881960544e-8",
            "extra": "mean: 333.9501702452271 nsec\nrounds: 196580"
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
          "id": "3684cfaec001eeafa090d3db8965372cf5d7abe2",
          "message": "feat(training): keep engine rollouts on-policy; fix train() export shadowing",
          "timestamp": "2026-09-08T18:53:12Z",
          "url": "https://github.com/stateset/stateset-agents/pull/69/commits/3684cfaec001eeafa090d3db8965372cf5d7abe2"
        },
        "date": 1788893944278,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8422.153866220733,
            "unit": "iter/sec",
            "range": "stddev: 0.000010826117047544945",
            "extra": "mean: 118.73447290137544 usec\nrounds: 2085"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 8909.394398314207,
            "unit": "iter/sec",
            "range": "stddev: 0.000009934634773900891",
            "extra": "mean: 112.24107445386134 usec\nrounds: 2243"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6813.892379600365,
            "unit": "iter/sec",
            "range": "stddev: 0.000011560772604061754",
            "extra": "mean: 146.75899534219678 usec\nrounds: 3650"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 876.3445005477823,
            "unit": "iter/sec",
            "range": "stddev: 0.000019827755583925415",
            "extra": "mean: 1.141103754716237 msec\nrounds: 636"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 207.3016220346612,
            "unit": "iter/sec",
            "range": "stddev: 0.00004361777203549328",
            "extra": "mean: 4.823888931427649 msec\nrounds: 175"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2347463.123404009,
            "unit": "iter/sec",
            "range": "stddev: 3.071542673269456e-8",
            "extra": "mean: 425.9917823756568 nsec\nrounds: 115447"
          }
        ]
      }
    ]
  }
}