window.BENCHMARK_DATA = {
  "lastUpdate": 1788289860609,
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
          "id": "f0afad6b901659fb959de8b2aa4450558c16222e",
          "message": "feat: add distributed async evidence collection",
          "timestamp": "2026-09-01T17:57:43Z",
          "url": "https://github.com/stateset/stateset-agents/pull/59/commits/f0afad6b901659fb959de8b2aa4450558c16222e"
        },
        "date": 1788288838342,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6218.703086048445,
            "unit": "iter/sec",
            "range": "stddev: 0.000014515690247221999",
            "extra": "mean: 160.8052332074646 usec\nrounds: 2114"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6726.649499528161,
            "unit": "iter/sec",
            "range": "stddev: 0.000013937068351301551",
            "extra": "mean: 148.66242102701273 usec\nrounds: 2121"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5147.761790078979,
            "unit": "iter/sec",
            "range": "stddev: 0.00001531713340320998",
            "extra": "mean: 194.2591830739428 usec\nrounds: 3769"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 755.6318617871658,
            "unit": "iter/sec",
            "range": "stddev: 0.00003882515733292962",
            "extra": "mean: 1.32339575733992 msec\nrounds: 647"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 184.2272901912047,
            "unit": "iter/sec",
            "range": "stddev: 0.0000620337945030911",
            "extra": "mean: 5.428077452380296 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2146865.158095289,
            "unit": "iter/sec",
            "range": "stddev: 1.0505707718499447e-7",
            "extra": "mean: 465.79543956417166 nsec\nrounds: 107435"
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
          "id": "d2ec886f00a272d5b16c1cc19e70804a0acacfc4",
          "message": "feat: add distributed async evidence collection (#59)\n\nCo-authored-by: domsteil <team@stateset.ai>",
          "timestamp": "2026-09-01T12:04:04-07:00",
          "tree_id": "8b0ac18b7d250e3c0037fac931d2acda6960a8d5",
          "url": "https://github.com/stateset/stateset-agents/commit/d2ec886f00a272d5b16c1cc19e70804a0acacfc4"
        },
        "date": 1788289604349,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8573.465474025928,
            "unit": "iter/sec",
            "range": "stddev: 0.000014654238725840317",
            "extra": "mean: 116.6389487459404 usec\nrounds: 1834"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9328.728344750236,
            "unit": "iter/sec",
            "range": "stddev: 0.000013828916794100369",
            "extra": "mean: 107.19574662743312 usec\nrounds: 1705"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6737.187292385854,
            "unit": "iter/sec",
            "range": "stddev: 0.00001863282989599986",
            "extra": "mean: 148.4298946431498 usec\nrounds: 2800"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 841.8294207048998,
            "unit": "iter/sec",
            "range": "stddev: 0.00002049620997633676",
            "extra": "mean: 1.1878891084165926 msec\nrounds: 701"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 177.1671156667799,
            "unit": "iter/sec",
            "range": "stddev: 0.00007457664507350434",
            "extra": "mean: 5.644388329269996 msec\nrounds: 164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2335612.410867215,
            "unit": "iter/sec",
            "range": "stddev: 3.512358492775768e-8",
            "extra": "mean: 428.1532309672472 nsec\nrounds: 58597"
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
          "id": "d82a692f62eba99bcc18ea4a6f7e6b9c9fae282d",
          "message": "chore(release): v0.47.2 — Distributed evidence collection",
          "timestamp": "2026-09-01T19:06:44Z",
          "url": "https://github.com/stateset/stateset-agents/pull/60/commits/d82a692f62eba99bcc18ea4a6f7e6b9c9fae282d"
        },
        "date": 1788289857370,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8254.854323009087,
            "unit": "iter/sec",
            "range": "stddev: 0.000022348756236821967",
            "extra": "mean: 121.14084160306254 usec\nrounds: 1572"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9370.472348056353,
            "unit": "iter/sec",
            "range": "stddev: 0.000014382234772464846",
            "extra": "mean: 106.71820617531863 usec\nrounds: 2008"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6815.295185841037,
            "unit": "iter/sec",
            "range": "stddev: 0.000013959097711057652",
            "extra": "mean: 146.72878763601133 usec\nrounds: 3122"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 826.199396475138,
            "unit": "iter/sec",
            "range": "stddev: 0.000021389379289251006",
            "extra": "mean: 1.2103615716331404 msec\nrounds: 698"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 177.58404273108872,
            "unit": "iter/sec",
            "range": "stddev: 0.00006802460976240834",
            "extra": "mean: 5.631136585364689 msec\nrounds: 164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2414180.0913670845,
            "unit": "iter/sec",
            "range": "stddev: 4.252996613493441e-8",
            "extra": "mean: 414.21930516945287 nsec\nrounds: 59295"
          }
        ]
      }
    ]
  }
}