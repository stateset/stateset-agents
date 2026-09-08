window.BENCHMARK_DATA = {
  "lastUpdate": 1788910860549,
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
          "id": "f63f26d8d22efeb003f3c4bd28e66fc8933ca97b",
          "message": "Merge pull request #73 from stateset/release/0.52.0\n\nchore(release): v0.52.0 — On-policy engine rollouts, proven live",
          "timestamp": "2026-09-08T14:11:04-07:00",
          "tree_id": "47a2398f5f5b91ab5108d224291e8830b0a366a5",
          "url": "https://github.com/stateset/stateset-agents/commit/f63f26d8d22efeb003f3c4bd28e66fc8933ca97b"
        },
        "date": 1788902010752,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6151.928460494732,
            "unit": "iter/sec",
            "range": "stddev: 0.000014112562533452809",
            "extra": "mean: 162.55065487539514 usec\nrounds: 2092"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6636.188158820316,
            "unit": "iter/sec",
            "range": "stddev: 0.000013586486156056762",
            "extra": "mean: 150.6889159962826 usec\nrounds: 1988"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5096.646679190006,
            "unit": "iter/sec",
            "range": "stddev: 0.000015642794098164307",
            "extra": "mean: 196.20744048887587 usec\nrounds: 3764"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 755.6794111984082,
            "unit": "iter/sec",
            "range": "stddev: 0.00002826566841293376",
            "extra": "mean: 1.3233124856665495 msec\nrounds: 593"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 184.39579322984918,
            "unit": "iter/sec",
            "range": "stddev: 0.00003812290010285306",
            "extra": "mean: 5.423117211538015 msec\nrounds: 156"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2226478.9627826526,
            "unit": "iter/sec",
            "range": "stddev: 5.0599889024452886e-8",
            "extra": "mean: 449.13965805012606 nsec\nrounds: 107910"
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
          "id": "1b2917c3042d0d8e89991cb8e5f4b1e9345d1d50",
          "message": "fix(gspo): train on the task prompts with the reward context; fail closed on zero learning signal",
          "timestamp": "2026-09-08T21:12:15Z",
          "url": "https://github.com/stateset/stateset-agents/pull/74/commits/1b2917c3042d0d8e89991cb8e5f4b1e9345d1d50"
        },
        "date": 1788910132389,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8575.22809115963,
            "unit": "iter/sec",
            "range": "stddev: 0.000013451808003053168",
            "extra": "mean: 116.61497389567043 usec\nrounds: 1992"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9280.819321882855,
            "unit": "iter/sec",
            "range": "stddev: 0.00001410397122969461",
            "extra": "mean: 107.74910762912299 usec\nrounds: 2202"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6758.9133522108095,
            "unit": "iter/sec",
            "range": "stddev: 0.000015089766951533482",
            "extra": "mean: 147.95277700562687 usec\nrounds: 3453"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 829.6454828493053,
            "unit": "iter/sec",
            "range": "stddev: 0.00005213025688405964",
            "extra": "mean: 1.2053341103788515 msec\nrounds: 607"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.50317552749277,
            "unit": "iter/sec",
            "range": "stddev: 0.00003611760407681179",
            "extra": "mean: 5.570932085526474 msec\nrounds: 152"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2376880.9875470838,
            "unit": "iter/sec",
            "range": "stddev: 3.3335561343034405e-8",
            "extra": "mean: 420.71942400111055 nsec\nrounds: 58429"
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
          "id": "8adeafe8aab5c67881d7641f538f4fc658628dba",
          "message": "Merge pull request #74 from stateset/fix/gspo-shootout-wiring\n\nfix(gspo): train on the task prompts with the reward context; fail closed on zero learning signal",
          "timestamp": "2026-09-08T16:36:04-07:00",
          "tree_id": "1571bce244692a509ab769e1470fdcf8266562af",
          "url": "https://github.com/stateset/stateset-agents/commit/8adeafe8aab5c67881d7641f538f4fc658628dba"
        },
        "date": 1788910732791,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5414.41476962764,
            "unit": "iter/sec",
            "range": "stddev: 0.000039230398566952997",
            "extra": "mean: 184.69216758375012 usec\nrounds: 1999"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6337.781797064624,
            "unit": "iter/sec",
            "range": "stddev: 0.000023063957322188626",
            "extra": "mean: 157.78391115692168 usec\nrounds: 1936"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4935.879635805391,
            "unit": "iter/sec",
            "range": "stddev: 0.000021180266394160458",
            "extra": "mean: 202.59813321740964 usec\nrounds: 3453"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 742.6476496573522,
            "unit": "iter/sec",
            "range": "stddev: 0.00003777816799754559",
            "extra": "mean: 1.3465335821925604 msec\nrounds: 584"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.51036845852678,
            "unit": "iter/sec",
            "range": "stddev: 0.00006145545400039434",
            "extra": "mean: 5.509327144738232 msec\nrounds: 152"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2190956.191050561,
            "unit": "iter/sec",
            "range": "stddev: 4.908253400431669e-8",
            "extra": "mean: 456.4217231201236 nsec\nrounds: 106191"
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
          "id": "989b6d17e91a3014bd1fcc9ba3724ec1f4172fa8",
          "message": "feat(gspo): batched group sampling and exactly on-policy old log-probs",
          "timestamp": "2026-09-08T23:36:31Z",
          "url": "https://github.com/stateset/stateset-agents/pull/75/commits/989b6d17e91a3014bd1fcc9ba3724ec1f4172fa8"
        },
        "date": 1788910858415,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11138.98460799406,
            "unit": "iter/sec",
            "range": "stddev: 0.00001089920579650132",
            "extra": "mean: 89.77478964126901 usec\nrounds: 2201"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11978.980390274362,
            "unit": "iter/sec",
            "range": "stddev: 0.00000893648725431233",
            "extra": "mean: 83.47955897915084 usec\nrounds: 2077"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8713.784790918187,
            "unit": "iter/sec",
            "range": "stddev: 0.000011235642012373342",
            "extra": "mean: 114.7606951507725 usec\nrounds: 3815"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1005.1366793975097,
            "unit": "iter/sec",
            "range": "stddev: 0.00007684872549649724",
            "extra": "mean: 994.8895712366315 usec\nrounds: 744"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 257.8076316150385,
            "unit": "iter/sec",
            "range": "stddev: 0.00021457361262496272",
            "extra": "mean: 3.8788611250004124 msec\nrounds: 200"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2718599.6877039177,
            "unit": "iter/sec",
            "range": "stddev: 8.096306950005474e-8",
            "extra": "mean: 367.8364286301315 nsec\nrounds: 126391"
          }
        ]
      }
    ]
  }
}