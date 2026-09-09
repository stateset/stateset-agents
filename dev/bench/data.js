window.BENCHMARK_DATA = {
  "lastUpdate": 1788918068461,
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
          "id": "dd9a47a2aed5ae801037ac9d6760e29da1e58da6",
          "message": "fix: route GSPO-token and auto-research DAPO/VAPO queries through the scenario helper",
          "timestamp": "2026-09-08T23:49:04Z",
          "url": "https://github.com/stateset/stateset-agents/pull/77/commits/dd9a47a2aed5ae801037ac9d6760e29da1e58da6"
        },
        "date": 1788912116918,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 7451.347116123635,
            "unit": "iter/sec",
            "range": "stddev: 0.00003750650736714703",
            "extra": "mean: 134.2039210381362 usec\nrounds: 1773"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9144.959445836152,
            "unit": "iter/sec",
            "range": "stddev: 0.000018144621014237885",
            "extra": "mean: 109.34985616096047 usec\nrounds: 2037"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6715.62916905455,
            "unit": "iter/sec",
            "range": "stddev: 0.000015846070333641587",
            "extra": "mean: 148.90637568375197 usec\nrounds: 3290"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 830.7344088444858,
            "unit": "iter/sec",
            "range": "stddev: 0.00002026811782178185",
            "extra": "mean: 1.2037541593960879 msec\nrounds: 596"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 177.26017933162672,
            "unit": "iter/sec",
            "range": "stddev: 0.0000531066822999849",
            "extra": "mean: 5.641424959461159 msec\nrounds: 148"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2412428.798836891,
            "unit": "iter/sec",
            "range": "stddev: 3.3364387036731605e-8",
            "extra": "mean: 414.5200059301779 nsec\nrounds: 59015"
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
          "id": "8e2858e4cb6596e97e1f7597a9b1afa2761e4df1",
          "message": "Merge pull request #76 from stateset/release/0.53.0\n\nchore(release): v0.53.0 — GSPO trains on the task, and evidence fails closed on zero signal",
          "timestamp": "2026-09-08T17:01:21-07:00",
          "tree_id": "e59245487c7523c9feec7ae45a71fcc7953b5af4",
          "url": "https://github.com/stateset/stateset-agents/commit/8e2858e4cb6596e97e1f7597a9b1afa2761e4df1"
        },
        "date": 1788912230077,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6146.939214916506,
            "unit": "iter/sec",
            "range": "stddev: 0.000016832040624213484",
            "extra": "mean: 162.68259129248327 usec\nrounds: 2136"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6648.2368813630865,
            "unit": "iter/sec",
            "range": "stddev: 0.000015696085708840328",
            "extra": "mean: 150.41581968947082 usec\nrounds: 1991"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5086.771795468892,
            "unit": "iter/sec",
            "range": "stddev: 0.000017735938447943204",
            "extra": "mean: 196.58833543324332 usec\nrounds: 3810"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 746.4881455312207,
            "unit": "iter/sec",
            "range": "stddev: 0.000032621449985283584",
            "extra": "mean: 1.3396060017649358 msec\nrounds: 566"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 183.27453253601772,
            "unit": "iter/sec",
            "range": "stddev: 0.00009626926105158358",
            "extra": "mean: 5.456295461037264 msec\nrounds: 154"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2255888.682972276,
            "unit": "iter/sec",
            "range": "stddev: 4.816600217801088e-8",
            "extra": "mean: 443.2842841706342 nsec\nrounds: 105731"
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
          "id": "999506de3e2c03a6bfc9a2f20a293563d4b47a3d",
          "message": "fix: route GSPO-token and auto-research DAPO/VAPO queries through the scenario helper",
          "timestamp": "2026-09-09T00:01:44Z",
          "url": "https://github.com/stateset/stateset-agents/pull/77/commits/999506de3e2c03a6bfc9a2f20a293563d4b47a3d"
        },
        "date": 1788912354854,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8619.461570128095,
            "unit": "iter/sec",
            "range": "stddev: 0.00001409774361625366",
            "extra": "mean: 116.016527466824 usec\nrounds: 1875"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9367.561241071811,
            "unit": "iter/sec",
            "range": "stddev: 0.000013410571616575286",
            "extra": "mean: 106.75137042238141 usec\nrounds: 2130"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6700.228197055528,
            "unit": "iter/sec",
            "range": "stddev: 0.000016093079041311653",
            "extra": "mean: 149.24864804447384 usec\nrounds: 3401"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 821.1669214780734,
            "unit": "iter/sec",
            "range": "stddev: 0.000021865462926005134",
            "extra": "mean: 1.2177792040137625 msec\nrounds: 598"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 178.2944885544249,
            "unit": "iter/sec",
            "range": "stddev: 0.00003621161467933462",
            "extra": "mean: 5.608698328859151 msec\nrounds: 149"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2155513.7119188206,
            "unit": "iter/sec",
            "range": "stddev: 5.170602992506803e-8",
            "extra": "mean: 463.9265315133663 nsec\nrounds: 103264"
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
          "id": "78cafaeddd0dea85dcf339bfee775864bcb19f8d",
          "message": "Merge pull request #77 from stateset/fix/query-wiring-everywhere\n\nfix: route GSPO-token and auto-research DAPO/VAPO queries through the scenario helper",
          "timestamp": "2026-09-08T17:14:24-07:00",
          "tree_id": "e3d401902b6934f088628bdd8ba3ad6a4c6eebfe",
          "url": "https://github.com/stateset/stateset-agents/commit/78cafaeddd0dea85dcf339bfee775864bcb19f8d"
        },
        "date": 1788913040133,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11503.496609460364,
            "unit": "iter/sec",
            "range": "stddev: 0.000010152898708730897",
            "extra": "mean: 86.93009038465833 usec\nrounds: 2080"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12177.64846322174,
            "unit": "iter/sec",
            "range": "stddev: 0.000008919612173457262",
            "extra": "mean: 82.11766032006464 usec\nrounds: 2311"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8980.155308267977,
            "unit": "iter/sec",
            "range": "stddev: 0.000010749321379759",
            "extra": "mean: 111.35664870732313 usec\nrounds: 3675"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1065.6371674393372,
            "unit": "iter/sec",
            "range": "stddev: 0.000020308226040858045",
            "extra": "mean: 938.4057074538237 usec\nrounds: 711"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 268.0750389914694,
            "unit": "iter/sec",
            "range": "stddev: 0.000046265330673791014",
            "extra": "mean: 3.7302988139519457 msec\nrounds: 215"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2909789.888859253,
            "unit": "iter/sec",
            "range": "stddev: 3.704561071073902e-8",
            "extra": "mean: 343.6674255514846 nsec\nrounds: 140865"
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
          "id": "26359f190fee6ec8a67516f8b064061f011f780a",
          "message": "fix(agent): always sample rollouts in inference mode",
          "timestamp": "2026-09-09T00:14:51Z",
          "url": "https://github.com/stateset/stateset-agents/pull/78/commits/26359f190fee6ec8a67516f8b064061f011f780a"
        },
        "date": 1788915235093,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5746.626833174792,
            "unit": "iter/sec",
            "range": "stddev: 0.00001641081764199254",
            "extra": "mean: 174.01512731383292 usec\nrounds: 1728"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6252.067295060677,
            "unit": "iter/sec",
            "range": "stddev: 0.00001601138712381649",
            "extra": "mean: 159.94709474577002 usec\nrounds: 2037"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5089.340233229609,
            "unit": "iter/sec",
            "range": "stddev: 0.000016793327067042652",
            "extra": "mean: 196.4891231815753 usec\nrounds: 3572"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 751.0083359806891,
            "unit": "iter/sec",
            "range": "stddev: 0.00008771362371366083",
            "extra": "mean: 1.3315431428522964 msec\nrounds: 567"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.5411418937905,
            "unit": "iter/sec",
            "range": "stddev: 0.00025980194083799333",
            "extra": "mean: 5.5083932466671595 msec\nrounds: 150"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2224074.3851540224,
            "unit": "iter/sec",
            "range": "stddev: 4.832896645168631e-8",
            "extra": "mean: 449.62524935097787 nsec\nrounds: 106987"
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
          "id": "1792b088bfb23c17a20eb206ac157136a60bd3ef",
          "message": "fix(agent): always sample rollouts in inference mode",
          "timestamp": "2026-09-09T00:14:51Z",
          "url": "https://github.com/stateset/stateset-agents/pull/78/commits/1792b088bfb23c17a20eb206ac157136a60bd3ef"
        },
        "date": 1788916480486,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14606.572210121345,
            "unit": "iter/sec",
            "range": "stddev: 0.00000839279093804972",
            "extra": "mean: 68.46233227170636 usec\nrounds: 2823"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 15495.65809351793,
            "unit": "iter/sec",
            "range": "stddev: 0.000007549005821490018",
            "extra": "mean: 64.53420654772418 usec\nrounds: 3147"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11369.197156334032,
            "unit": "iter/sec",
            "range": "stddev: 0.000007520431964019156",
            "extra": "mean: 87.95695828380263 usec\nrounds: 4866"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1346.0467761628192,
            "unit": "iter/sec",
            "range": "stddev: 0.000014024443004160097",
            "extra": "mean: 742.9162327112464 usec\nrounds: 1070"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 341.90275639082614,
            "unit": "iter/sec",
            "range": "stddev: 0.000014154950805749402",
            "extra": "mean: 2.9248082424258333 msec\nrounds: 297"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3742233.9422563147,
            "unit": "iter/sec",
            "range": "stddev: 2.7932638754123223e-8",
            "extra": "mean: 267.22006572284664 nsec\nrounds: 190986"
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
          "id": "edb2a9752f2bdd59822b3c2e20be9d53ff55478a",
          "message": "fix(agent): always sample rollouts in inference mode",
          "timestamp": "2026-09-09T00:14:51Z",
          "url": "https://github.com/stateset/stateset-agents/pull/78/commits/edb2a9752f2bdd59822b3c2e20be9d53ff55478a"
        },
        "date": 1788917284266,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11939.954005136015,
            "unit": "iter/sec",
            "range": "stddev: 0.000008029657861417848",
            "extra": "mean: 83.7524164305697 usec\nrounds: 2471"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12057.316071399628,
            "unit": "iter/sec",
            "range": "stddev: 0.00002080972624837425",
            "extra": "mean: 82.93719714058378 usec\nrounds: 2029"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 9049.011760649582,
            "unit": "iter/sec",
            "range": "stddev: 0.000012294390789215759",
            "extra": "mean: 110.50930493300797 usec\nrounds: 3791"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1093.6325772442358,
            "unit": "iter/sec",
            "range": "stddev: 0.00001748092292790005",
            "extra": "mean: 914.3838806629429 usec\nrounds: 905"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 275.673114323934,
            "unit": "iter/sec",
            "range": "stddev: 0.00008611592517283523",
            "extra": "mean: 3.6274846839976362 msec\nrounds: 250"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3012384.0123281656,
            "unit": "iter/sec",
            "range": "stddev: 3.138473518122944e-8",
            "extra": "mean: 331.9629887516018 nsec\nrounds: 148567"
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
          "id": "05592a470466768b24d804f97b8857110a6ba0ee",
          "message": "fix(agent): always sample rollouts in inference mode",
          "timestamp": "2026-09-09T00:14:51Z",
          "url": "https://github.com/stateset/stateset-agents/pull/78/commits/05592a470466768b24d804f97b8857110a6ba0ee"
        },
        "date": 1788918067318,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6182.910455052411,
            "unit": "iter/sec",
            "range": "stddev: 0.000019319689014754254",
            "extra": "mean: 161.73612852226296 usec\nrounds: 2023"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6491.909613138389,
            "unit": "iter/sec",
            "range": "stddev: 0.000027021096010144023",
            "extra": "mean: 154.03788093047234 usec\nrounds: 2192"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5008.482429496346,
            "unit": "iter/sec",
            "range": "stddev: 0.000028067172915145334",
            "extra": "mean: 199.66127745816215 usec\nrounds: 3651"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 759.7411178014032,
            "unit": "iter/sec",
            "range": "stddev: 0.000028493254965271328",
            "extra": "mean: 1.3162378296621304 msec\nrounds: 681"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.3693743045359,
            "unit": "iter/sec",
            "range": "stddev: 0.000050241398602537325",
            "extra": "mean: 5.48337682142899 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2227811.8828766304,
            "unit": "iter/sec",
            "range": "stddev: 5.0557152836371095e-8",
            "extra": "mean: 448.8709337113169 nsec\nrounds: 110291"
          }
        ]
      }
    ]
  }
}