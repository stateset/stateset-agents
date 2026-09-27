window.BENCHMARK_DATA = {
  "lastUpdate": 1790484817231,
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
          "id": "2aa7c5d581dadc8a16e1a0e67a37fc080a37e703",
          "message": "Harden checkpoints, expand type checks, and replace vulnerable scanner",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/2aa7c5d581dadc8a16e1a0e67a37fc080a37e703"
        },
        "date": 1790270624909,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10956.854151817493,
            "unit": "iter/sec",
            "range": "stddev: 0.000012555800776057891",
            "extra": "mean: 91.26707229502757 usec\nrounds: 2310"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11857.795504855527,
            "unit": "iter/sec",
            "range": "stddev: 0.000011752459276198106",
            "extra": "mean: 84.33270750794448 usec\nrounds: 2677"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8685.967411069312,
            "unit": "iter/sec",
            "range": "stddev: 0.000012065282087566338",
            "extra": "mean: 115.12822379757145 usec\nrounds: 3387"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1088.6456152157205,
            "unit": "iter/sec",
            "range": "stddev: 0.000013135590790602224",
            "extra": "mean: 918.5725694599386 usec\nrounds: 943"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 230.53108907000836,
            "unit": "iter/sec",
            "range": "stddev: 0.0000309899086238969",
            "extra": "mean: 4.337809724641161 msec\nrounds: 207"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2955163.928147765,
            "unit": "iter/sec",
            "range": "stddev: 3.934942737906506e-8",
            "extra": "mean: 338.3907032956981 nsec\nrounds: 198531"
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
          "id": "d1c67faf4616dfe3a561c06dc5429e2025079e01",
          "message": "Harden checkpoints, expand type checks, and replace vulnerable scanner",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/d1c67faf4616dfe3a561c06dc5429e2025079e01"
        },
        "date": 1790270800494,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6072.350503146401,
            "unit": "iter/sec",
            "range": "stddev: 0.000016811609414617645",
            "extra": "mean: 164.68087596093935 usec\nrounds: 1822"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6404.84291171014,
            "unit": "iter/sec",
            "range": "stddev: 0.00002292392831166379",
            "extra": "mean: 156.13185425229932 usec\nrounds: 1976"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5033.474400881881,
            "unit": "iter/sec",
            "range": "stddev: 0.000017337533240525673",
            "extra": "mean: 198.66992863315184 usec\nrounds: 3489"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 754.8089231728798,
            "unit": "iter/sec",
            "range": "stddev: 0.00002255461059434731",
            "extra": "mean: 1.324838603916401 msec\nrounds: 664"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 185.72744463005802,
            "unit": "iter/sec",
            "range": "stddev: 0.00015070917955984517",
            "extra": "mean: 5.384233880953104 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2180113.3897464336,
            "unit": "iter/sec",
            "range": "stddev: 4.9679389209710104e-8",
            "extra": "mean: 458.69173810097504 nsec\nrounds: 104298"
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
          "id": "2c1f429f08911fd4ea6e82b6a3fb1094ca41b26e",
          "message": "Harden checkpoints, expand type checks, and replace vulnerable scanner",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/2c1f429f08911fd4ea6e82b6a3fb1094ca41b26e"
        },
        "date": 1790271479130,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6150.289415256982,
            "unit": "iter/sec",
            "range": "stddev: 0.000017709314264834912",
            "extra": "mean: 162.59397444278096 usec\nrounds: 1839"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6772.257474236947,
            "unit": "iter/sec",
            "range": "stddev: 0.000015677410761050928",
            "extra": "mean: 147.66124941412883 usec\nrounds: 2137"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5148.039092943246,
            "unit": "iter/sec",
            "range": "stddev: 0.00001697835731090279",
            "extra": "mean: 194.2487191619748 usec\nrounds: 3194"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 757.8650877259353,
            "unit": "iter/sec",
            "range": "stddev: 0.000027226152647850425",
            "extra": "mean: 1.3194960636075999 msec\nrounds: 676"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.5978187328321,
            "unit": "iter/sec",
            "range": "stddev: 0.00005248036887192492",
            "extra": "mean: 5.506674072287214 msec\nrounds: 166"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2154703.2478297716,
            "unit": "iter/sec",
            "range": "stddev: 5.126128399159628e-8",
            "extra": "mean: 464.10103154910325 nsec\nrounds: 104745"
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
          "id": "8850a888819b01506937e2dbd9920cd90be06368",
          "message": "Harden checkpoints, expand type checks, and replace vulnerable scanner",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/8850a888819b01506937e2dbd9920cd90be06368"
        },
        "date": 1790443123708,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11208.4767023227,
            "unit": "iter/sec",
            "range": "stddev: 0.000011822308953676066",
            "extra": "mean: 89.21818963970125 usec\nrounds: 2220"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12187.635289980477,
            "unit": "iter/sec",
            "range": "stddev: 0.000011013304147807608",
            "extra": "mean: 82.05037123338484 usec\nrounds: 2489"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8857.549393615638,
            "unit": "iter/sec",
            "range": "stddev: 0.000011860500604744742",
            "extra": "mean: 112.8980438676167 usec\nrounds: 4354"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1077.8703424389157,
            "unit": "iter/sec",
            "range": "stddev: 0.00001820959683193092",
            "extra": "mean: 927.7553715201801 usec\nrounds: 934"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 230.35803585298396,
            "unit": "iter/sec",
            "range": "stddev: 0.00005948370649701549",
            "extra": "mean: 4.341068442857391 msec\nrounds: 210"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2975767.79317934,
            "unit": "iter/sec",
            "range": "stddev: 4.138178446356147e-8",
            "extra": "mean: 336.0477259993428 nsec\nrounds: 193499"
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
          "id": "aae1a2da0c092100c672a9a2c9a95324180da54f",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/aae1a2da0c092100c672a9a2c9a95324180da54f"
        },
        "date": 1790443539172,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10044.479885668557,
            "unit": "iter/sec",
            "range": "stddev: 0.000012575127697610067",
            "extra": "mean: 99.55717084234476 usec\nrounds: 1756"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 10406.444032174148,
            "unit": "iter/sec",
            "range": "stddev: 0.000014023638749711585",
            "extra": "mean: 96.09430434721482 usec\nrounds: 2277"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 7814.336349951159,
            "unit": "iter/sec",
            "range": "stddev: 0.000013786451713038242",
            "extra": "mean: 127.96992031271476 usec\nrounds: 3451"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 938.5809647477876,
            "unit": "iter/sec",
            "range": "stddev: 0.000024022654402343324",
            "extra": "mean: 1.0654381854725947 msec\nrounds: 771"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 234.85498615218322,
            "unit": "iter/sec",
            "range": "stddev: 0.00009767879259180711",
            "extra": "mean: 4.25794664351734 msec\nrounds: 216"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2550524.4910376994,
            "unit": "iter/sec",
            "range": "stddev: 4.312727466175402e-8",
            "extra": "mean: 392.07621942620233 nsec\nrounds: 117371"
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
          "id": "400e2cdf765a20b366cddd2595c370b9416d69d3",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/400e2cdf765a20b366cddd2595c370b9416d69d3"
        },
        "date": 1790446470108,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5745.387644139182,
            "unit": "iter/sec",
            "range": "stddev: 0.00001626678020984892",
            "extra": "mean: 174.05265961820538 usec\nrounds: 1939"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6513.936345341239,
            "unit": "iter/sec",
            "range": "stddev: 0.000017183256430912218",
            "extra": "mean: 153.51700523066964 usec\nrounds: 2103"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4602.994957769136,
            "unit": "iter/sec",
            "range": "stddev: 0.000043652987556192883",
            "extra": "mean: 217.24985779359946 usec\nrounds: 3073"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 734.9913282082542,
            "unit": "iter/sec",
            "range": "stddev: 0.000034644266970647306",
            "extra": "mean: 1.3605602700616592 msec\nrounds: 648"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.5375724104543,
            "unit": "iter/sec",
            "range": "stddev: 0.00013558923246973723",
            "extra": "mean: 5.569864773005982 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2194886.6725132517,
            "unit": "iter/sec",
            "range": "stddev: 5.473153857798348e-8",
            "extra": "mean: 455.6043883828186 nsec\nrounds: 106406"
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
          "id": "92d22e55f8eaece02a586b0401f0e7f9c02049d3",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/92d22e55f8eaece02a586b0401f0e7f9c02049d3"
        },
        "date": 1790446641466,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5980.124816189234,
            "unit": "iter/sec",
            "range": "stddev: 0.000017612000712801146",
            "extra": "mean: 167.2205899938454 usec\nrounds: 1639"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6456.887073946372,
            "unit": "iter/sec",
            "range": "stddev: 0.00001712803198489272",
            "extra": "mean: 154.873391550398 usec\nrounds: 1941"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5002.01496513726,
            "unit": "iter/sec",
            "range": "stddev: 0.000017074907243977002",
            "extra": "mean: 199.91943386210139 usec\nrounds: 3591"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 748.5460048130398,
            "unit": "iter/sec",
            "range": "stddev: 0.00003163512710367107",
            "extra": "mean: 1.3359232346043508 msec\nrounds: 682"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.91532675327775,
            "unit": "iter/sec",
            "range": "stddev: 0.00008149931819727793",
            "extra": "mean: 5.527447662650187 msec\nrounds: 166"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2219660.393052873,
            "unit": "iter/sec",
            "range": "stddev: 4.862388201666251e-8",
            "extra": "mean: 450.51936914755754 nsec\nrounds: 105175"
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
          "id": "fd44c65d493c17e1c65c48093d9dfe2bdcd3c570",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/fd44c65d493c17e1c65c48093d9dfe2bdcd3c570"
        },
        "date": 1790448498168,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 9887.482384091445,
            "unit": "iter/sec",
            "range": "stddev: 0.00001094920653283536",
            "extra": "mean: 101.13798044373348 usec\nrounds: 2250"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 10486.008965686915,
            "unit": "iter/sec",
            "range": "stddev: 0.000008584817982803338",
            "extra": "mean: 95.36516736465448 usec\nrounds: 1918"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 7789.049931555699,
            "unit": "iter/sec",
            "range": "stddev: 0.000009805845034188464",
            "extra": "mean: 128.38536262923546 usec\nrounds: 2708"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 906.9260080760938,
            "unit": "iter/sec",
            "range": "stddev: 0.000026172719895557818",
            "extra": "mean: 1.102625783244819 msec\nrounds: 752"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 230.28788928583023,
            "unit": "iter/sec",
            "range": "stddev: 0.00006698794738038993",
            "extra": "mean: 4.342390748819681 msec\nrounds: 211"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2513496.1099282173,
            "unit": "iter/sec",
            "range": "stddev: 3.5071856282098624e-8",
            "extra": "mean: 397.8522170971488 nsec\nrounds: 120788"
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
          "id": "44ceb3996630013826197ba2a8d7d6f8cb8c3ea9",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/44ceb3996630013826197ba2a8d7d6f8cb8c3ea9"
        },
        "date": 1790449006629,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8387.465962382259,
            "unit": "iter/sec",
            "range": "stddev: 0.000011840547409762373",
            "extra": "mean: 119.22552109123242 usec\nrounds: 2015"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 8883.210111779,
            "unit": "iter/sec",
            "range": "stddev: 0.000011210872915735285",
            "extra": "mean: 112.57191796848475 usec\nrounds: 2048"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6742.651052668259,
            "unit": "iter/sec",
            "range": "stddev: 0.000014662886262971946",
            "extra": "mean: 148.30961771398086 usec\nrounds: 3534"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 878.6139295838791,
            "unit": "iter/sec",
            "range": "stddev: 0.000014994687921282426",
            "extra": "mean: 1.1381563236467358 msec\nrounds: 757"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 206.32289423260525,
            "unit": "iter/sec",
            "range": "stddev: 0.00006430899445323029",
            "extra": "mean: 4.846771870467343 msec\nrounds: 193"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2357936.334389063,
            "unit": "iter/sec",
            "range": "stddev: 3.155517519691977e-8",
            "extra": "mean: 424.0996609686233 nsec\nrounds: 114000"
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
          "id": "709b775c5ebe5ac5c7985c54cd63a24505259c62",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/709b775c5ebe5ac5c7985c54cd63a24505259c62"
        },
        "date": 1790451273710,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14700.419318924527,
            "unit": "iter/sec",
            "range": "stddev: 0.000008431102811893696",
            "extra": "mean: 68.02527045692186 usec\nrounds: 2603"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 16009.804625141633,
            "unit": "iter/sec",
            "range": "stddev: 0.000007808404530358678",
            "extra": "mean: 62.46172413807039 usec\nrounds: 3306"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 12017.827385767729,
            "unit": "iter/sec",
            "range": "stddev: 0.000008495475995796275",
            "extra": "mean: 83.20971569156195 usec\nrounds: 5691"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1545.2030298165687,
            "unit": "iter/sec",
            "range": "stddev: 0.00002045201868289872",
            "extra": "mean: 647.164146525593 usec\nrounds: 1324"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 310.6017429062616,
            "unit": "iter/sec",
            "range": "stddev: 0.00008090920693670487",
            "extra": "mean: 3.2195569498198084 msec\nrounds: 279"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 4176156.2392488364,
            "unit": "iter/sec",
            "range": "stddev: 2.2737717315240024e-7",
            "extra": "mean: 239.45464266918074 nsec\nrounds: 197356"
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
          "id": "ff0c002d0e824deb84fbad4dcabbae79b054b8f5",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/ff0c002d0e824deb84fbad4dcabbae79b054b8f5"
        },
        "date": 1790467682056,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6002.244298365609,
            "unit": "iter/sec",
            "range": "stddev: 0.00001755420104544109",
            "extra": "mean: 166.60434835554705 usec\nrounds: 1642"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6450.6286276083565,
            "unit": "iter/sec",
            "range": "stddev: 0.000016858755745050095",
            "extra": "mean: 155.0236508299442 usec\nrounds: 2168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5007.315619617888,
            "unit": "iter/sec",
            "range": "stddev: 0.000018274426923093945",
            "extra": "mean: 199.70780273609174 usec\nrounds: 3655"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 755.3719787287293,
            "unit": "iter/sec",
            "range": "stddev: 0.00005232174386689248",
            "extra": "mean: 1.3238510669709684 msec\nrounds: 657"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 168.63114620944572,
            "unit": "iter/sec",
            "range": "stddev: 0.005532814536573667",
            "extra": "mean: 5.930102608434893 msec\nrounds: 166"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2227074.3657346778,
            "unit": "iter/sec",
            "range": "stddev: 6.319850986180148e-8",
            "extra": "mean: 449.0195816474747 nsec\nrounds: 105175"
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
          "id": "bddc22aef7abf5de51a392cb358addf3b1004b31",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/bddc22aef7abf5de51a392cb358addf3b1004b31"
        },
        "date": 1790469636799,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6101.582601804752,
            "unit": "iter/sec",
            "range": "stddev: 0.00001624181850679266",
            "extra": "mean: 163.89190563514057 usec\nrounds: 1473"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6563.5214253814775,
            "unit": "iter/sec",
            "range": "stddev: 0.000016673358236208694",
            "extra": "mean: 152.3572386208641 usec\nrounds: 2175"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5020.709567784835,
            "unit": "iter/sec",
            "range": "stddev: 0.00001945827658666655",
            "extra": "mean: 199.17503422553193 usec\nrounds: 3623"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 738.070913126844,
            "unit": "iter/sec",
            "range": "stddev: 0.00003766066132953432",
            "extra": "mean: 1.3548833617673552 msec\nrounds: 633"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 164.05885966882195,
            "unit": "iter/sec",
            "range": "stddev: 0.0057690431353321",
            "extra": "mean: 6.095373343558854 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2207542.590318262,
            "unit": "iter/sec",
            "range": "stddev: 4.805706196584381e-8",
            "extra": "mean: 452.99239271113214 nsec\nrounds: 102691"
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
          "id": "4d13c8669bad4b676988f1e36b63cc8ac3807b72",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/4d13c8669bad4b676988f1e36b63cc8ac3807b72"
        },
        "date": 1790470946955,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5772.696756477276,
            "unit": "iter/sec",
            "range": "stddev: 0.000048814758173434405",
            "extra": "mean: 173.22926219499513 usec\nrounds: 1476"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6515.852663386304,
            "unit": "iter/sec",
            "range": "stddev: 0.000023999545453418843",
            "extra": "mean: 153.47185574332764 usec\nrounds: 2142"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5090.350798145955,
            "unit": "iter/sec",
            "range": "stddev: 0.00001890007752839757",
            "extra": "mean: 196.4501150616628 usec\nrounds: 3572"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 750.414704615521,
            "unit": "iter/sec",
            "range": "stddev: 0.000024754795849776374",
            "extra": "mean: 1.3325964881143357 msec\nrounds: 631"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 165.46468945999715,
            "unit": "iter/sec",
            "range": "stddev: 0.0068945275022133995",
            "extra": "mean: 6.043585512193287 msec\nrounds: 164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2140480.1248732395,
            "unit": "iter/sec",
            "range": "stddev: 7.469161190284696e-8",
            "extra": "mean: 467.184903227831 nsec\nrounds: 103542"
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
          "id": "37d0a044cda80e21d201b092c44e9aecc005cb31",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/37d0a044cda80e21d201b092c44e9aecc005cb31"
        },
        "date": 1790472524747,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5280.872780776254,
            "unit": "iter/sec",
            "range": "stddev: 0.00004212177796965933",
            "extra": "mean: 189.3626378655171 usec\nrounds: 1574"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6630.204724827275,
            "unit": "iter/sec",
            "range": "stddev: 0.00001510994763086345",
            "extra": "mean: 150.82490533896015 usec\nrounds: 2229"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5128.247747194068,
            "unit": "iter/sec",
            "range": "stddev: 0.00001683012512693815",
            "extra": "mean: 194.99837942641363 usec\nrounds: 3276"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 750.3961472773246,
            "unit": "iter/sec",
            "range": "stddev: 0.000045352471534581355",
            "extra": "mean: 1.332629443299139 msec\nrounds: 679"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 169.3171854269035,
            "unit": "iter/sec",
            "range": "stddev: 0.005127666921128668",
            "extra": "mean: 5.9060750240956095 msec\nrounds: 166"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2205863.998499648,
            "unit": "iter/sec",
            "range": "stddev: 5.79018949582503e-8",
            "extra": "mean: 453.3371054063918 nsec\nrounds: 105742"
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
          "id": "2a03fe8b2a08950f7f48ca2d6063a59a63acc66b",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/2a03fe8b2a08950f7f48ca2d6063a59a63acc66b"
        },
        "date": 1790473456214,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6064.3213917054345,
            "unit": "iter/sec",
            "range": "stddev: 0.00001725893628195846",
            "extra": "mean: 164.89891208070947 usec\nrounds: 1490"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6529.451689087503,
            "unit": "iter/sec",
            "range": "stddev: 0.000016694257900039272",
            "extra": "mean: 153.1522166970426 usec\nrounds: 2192"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5046.977897309402,
            "unit": "iter/sec",
            "range": "stddev: 0.000017735575151552187",
            "extra": "mean: 198.13837515181328 usec\nrounds: 3292"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 753.1761398529189,
            "unit": "iter/sec",
            "range": "stddev: 0.000027113898708356187",
            "extra": "mean: 1.3277106736218185 msec\nrounds: 671"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 169.10107196855375,
            "unit": "iter/sec",
            "range": "stddev: 0.004935311594528002",
            "extra": "mean: 5.913623067900842 msec\nrounds: 162"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2221544.1961729745,
            "unit": "iter/sec",
            "range": "stddev: 4.901243082464718e-8",
            "extra": "mean: 450.13734217968164 nsec\nrounds: 106531"
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
          "id": "131284956a0a459954f1b8049fd5de76ae9d1b20",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/131284956a0a459954f1b8049fd5de76ae9d1b20"
        },
        "date": 1790474683414,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6096.303007973903,
            "unit": "iter/sec",
            "range": "stddev: 0.000016946546091821492",
            "extra": "mean: 164.0338412792163 usec\nrounds: 1594"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6566.8258854551295,
            "unit": "iter/sec",
            "range": "stddev: 0.00001643561458657739",
            "extra": "mean: 152.28057168607154 usec\nrounds: 2218"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5034.50077432846,
            "unit": "iter/sec",
            "range": "stddev: 0.000016031333975378192",
            "extra": "mean: 198.6294261983479 usec\nrounds: 3672"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 729.8859387787871,
            "unit": "iter/sec",
            "range": "stddev: 0.0001415903248709831",
            "extra": "mean: 1.3700770858432423 msec\nrounds: 664"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 169.34660526572964,
            "unit": "iter/sec",
            "range": "stddev: 0.005193781852859441",
            "extra": "mean: 5.905048987730538 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2147469.9515461405,
            "unit": "iter/sec",
            "range": "stddev: 5.10757546883722e-8",
            "extra": "mean: 465.6642572716874 nsec\nrounds: 103221"
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
          "id": "167a1483c90b7f7658ac15ac8a8aa2c31eb93bdc",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/167a1483c90b7f7658ac15ac8a8aa2c31eb93bdc"
        },
        "date": 1790475660694,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14903.56835332929,
            "unit": "iter/sec",
            "range": "stddev: 0.000007675118422871415",
            "extra": "mean: 67.09802486842764 usec\nrounds: 2091"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 15972.134928821144,
            "unit": "iter/sec",
            "range": "stddev: 0.000007210127812136297",
            "extra": "mean: 62.60903783097499 usec\nrounds: 3780"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 12140.319816016856,
            "unit": "iter/sec",
            "range": "stddev: 0.000008449951555188633",
            "extra": "mean: 82.37015294116792 usec\nrounds: 6205"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1536.3163564884574,
            "unit": "iter/sec",
            "range": "stddev: 0.000016339338554958127",
            "extra": "mean: 650.907604919139 usec\nrounds: 1301"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 294.85332639220036,
            "unit": "iter/sec",
            "range": "stddev: 0.0038330978621465804",
            "extra": "mean: 3.3915167661016854 msec\nrounds: 295"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 4711668.891513399,
            "unit": "iter/sec",
            "range": "stddev: 1.8907090984267866e-8",
            "extra": "mean: 212.239022525795 nsec\nrounds: 185943"
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
          "id": "33b74b730976bd27e6384ea17d2459cbd7997fa7",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/33b74b730976bd27e6384ea17d2459cbd7997fa7"
        },
        "date": 1790476604257,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6041.834378278398,
            "unit": "iter/sec",
            "range": "stddev: 0.000016731106977538244",
            "extra": "mean: 165.51264688671372 usec\nrounds: 1365"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6556.514748768685,
            "unit": "iter/sec",
            "range": "stddev: 0.000014549325117220215",
            "extra": "mean: 152.5200565113958 usec\nrounds: 2035"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5031.312869974778,
            "unit": "iter/sec",
            "range": "stddev: 0.000025507730752954016",
            "extra": "mean: 198.7552803499205 usec\nrounds: 3542"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 744.6213909901596,
            "unit": "iter/sec",
            "range": "stddev: 0.000039435531540855104",
            "extra": "mean: 1.3429643737070873 msec\nrounds: 677"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 168.163218481532,
            "unit": "iter/sec",
            "range": "stddev: 0.005283990644111343",
            "extra": "mean: 5.946603597562698 msec\nrounds: 164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2181713.518748493,
            "unit": "iter/sec",
            "range": "stddev: 7.095346188000497e-8",
            "extra": "mean: 458.355320900993 nsec\nrounds: 104954"
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
          "id": "d89f864fc877e29eb9f9acd8e82c7186db312e19",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/d89f864fc877e29eb9f9acd8e82c7186db312e19"
        },
        "date": 1790477465960,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11088.151626409855,
            "unit": "iter/sec",
            "range": "stddev: 0.000011675870713598015",
            "extra": "mean: 90.18635690534673 usec\nrounds: 1810"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12123.95460360193,
            "unit": "iter/sec",
            "range": "stddev: 0.000011314342330220673",
            "extra": "mean: 82.48133820155579 usec\nrounds: 2602"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8868.104726598758,
            "unit": "iter/sec",
            "range": "stddev: 0.000013472928137536518",
            "extra": "mean: 112.7636660627864 usec\nrounds: 4414"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1086.007608117104,
            "unit": "iter/sec",
            "range": "stddev: 0.00001774547060774191",
            "extra": "mean: 920.8038622618658 usec\nrounds: 893"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 213.13493607486373,
            "unit": "iter/sec",
            "range": "stddev: 0.004499639722191566",
            "extra": "mean: 4.691863372641778 msec\nrounds: 212"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3002337.309955051,
            "unit": "iter/sec",
            "range": "stddev: 4.2102394318630067e-8",
            "extra": "mean: 333.0738344036937 nsec\nrounds: 196194"
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
          "id": "6746f4a03a8a85aa5e6626068859256b4e139e25",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/6746f4a03a8a85aa5e6626068859256b4e139e25"
        },
        "date": 1790478285276,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 12604.172969536976,
            "unit": "iter/sec",
            "range": "stddev: 0.000010643975249754385",
            "extra": "mean: 79.33880330085123 usec\nrounds: 1515"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 14045.667202139133,
            "unit": "iter/sec",
            "range": "stddev: 0.00001057228073635008",
            "extra": "mean: 71.19633304765343 usec\nrounds: 2333"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 10781.45572227014,
            "unit": "iter/sec",
            "range": "stddev: 0.000009912082897672689",
            "extra": "mean: 92.75185334522158 usec\nrounds: 4214"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1263.630908648288,
            "unit": "iter/sec",
            "range": "stddev: 0.00003348957247760164",
            "extra": "mean: 791.3703227390225 usec\nrounds: 1007"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 304.5132008797043,
            "unit": "iter/sec",
            "range": "stddev: 0.00417598175268543",
            "extra": "mean: 3.2839298825506176 msec\nrounds: 298"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3610755.7683922215,
            "unit": "iter/sec",
            "range": "stddev: 3.5536305053466416e-8",
            "extra": "mean: 276.9503295553204 nsec\nrounds: 187126"
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
          "id": "0e454c109eb8f8d58b1d0b60cfd3ce29592be8d8",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/0e454c109eb8f8d58b1d0b60cfd3ce29592be8d8"
        },
        "date": 1790479217881,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 15303.216217530298,
            "unit": "iter/sec",
            "range": "stddev: 0.000008087206349993928",
            "extra": "mean: 65.3457407766656 usec\nrounds: 2087"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 15935.049806124145,
            "unit": "iter/sec",
            "range": "stddev: 0.000007790397949486645",
            "extra": "mean: 62.75474580667334 usec\nrounds: 3875"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 12071.32032019956,
            "unit": "iter/sec",
            "range": "stddev: 0.000008222805650674636",
            "extra": "mean: 82.8409795676326 usec\nrounds: 5873"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1530.5398443674972,
            "unit": "iter/sec",
            "range": "stddev: 0.000012328800738703638",
            "extra": "mean: 653.364238559405 usec\nrounds: 1333"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 299.923513686151,
            "unit": "iter/sec",
            "range": "stddev: 0.003649250276438704",
            "extra": "mean: 3.3341833979926965 msec\nrounds: 299"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 4589579.014345754,
            "unit": "iter/sec",
            "range": "stddev: 1.976108232533131e-8",
            "extra": "mean: 217.8849077168683 nsec\nrounds: 193536"
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
          "id": "ac31631a37951516311d6b4f340c723c6803f835",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/ac31631a37951516311d6b4f340c723c6803f835"
        },
        "date": 1790479997998,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6131.8190342645485,
            "unit": "iter/sec",
            "range": "stddev: 0.0000179485491693644",
            "extra": "mean: 163.08374308048053 usec\nrounds: 1409"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6593.220044766488,
            "unit": "iter/sec",
            "range": "stddev: 0.000016950547873424422",
            "extra": "mean: 151.67095792499322 usec\nrounds: 2044"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5075.792877262773,
            "unit": "iter/sec",
            "range": "stddev: 0.000016046908323018453",
            "extra": "mean: 197.01355515894707 usec\nrounds: 3363"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 757.2495893495641,
            "unit": "iter/sec",
            "range": "stddev: 0.00003226090118642349",
            "extra": "mean: 1.3205685603064443 msec\nrounds: 655"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 166.94119601973875,
            "unit": "iter/sec",
            "range": "stddev: 0.005677287127106353",
            "extra": "mean: 5.990133195653889 msec\nrounds: 138"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2149221.3224683385,
            "unit": "iter/sec",
            "range": "stddev: 5.065595228076222e-8",
            "extra": "mean: 465.2847938673527 nsec\nrounds: 103649"
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
          "id": "ab26cfa6857172bfd611011f51d6794fdb436261",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/ab26cfa6857172bfd611011f51d6794fdb436261"
        },
        "date": 1790481021749,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6101.185298640651,
            "unit": "iter/sec",
            "range": "stddev: 0.000017747877733931555",
            "extra": "mean: 163.90257811425604 usec\nrounds: 1517"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6576.602828184472,
            "unit": "iter/sec",
            "range": "stddev: 0.00001396766980341121",
            "extra": "mean: 152.054187568456 usec\nrounds: 1834"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5061.920354620303,
            "unit": "iter/sec",
            "range": "stddev: 0.000016242874709708912",
            "extra": "mean: 197.5534836472176 usec\nrounds: 3608"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 757.3894865669972,
            "unit": "iter/sec",
            "range": "stddev: 0.000023382546893174306",
            "extra": "mean: 1.320324638427024 msec\nrounds: 661"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 167.05613661876959,
            "unit": "iter/sec",
            "range": "stddev: 0.005993825753235172",
            "extra": "mean: 5.986011769696613 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2038828.6641621666,
            "unit": "iter/sec",
            "range": "stddev: 6.276409249311478e-8",
            "extra": "mean: 490.4777029956751 nsec\nrounds: 99325"
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
          "id": "9c0123ea1ccee0edf56bef66303f1505d5ed4e4e",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/9c0123ea1ccee0edf56bef66303f1505d5ed4e4e"
        },
        "date": 1790482049025,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8448.223193479345,
            "unit": "iter/sec",
            "range": "stddev: 0.000010788518434995487",
            "extra": "mean: 118.3680848739694 usec\nrounds: 1461"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 8945.628378596695,
            "unit": "iter/sec",
            "range": "stddev: 0.000010670040758299645",
            "extra": "mean: 111.78644558861839 usec\nrounds: 2233"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6826.829892743995,
            "unit": "iter/sec",
            "range": "stddev: 0.00001186225779429309",
            "extra": "mean: 146.48087263209325 usec\nrounds: 3431"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 881.7150320410254,
            "unit": "iter/sec",
            "range": "stddev: 0.000016546724347631233",
            "extra": "mean: 1.1341532849736773 msec\nrounds: 772"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 192.25850801503114,
            "unit": "iter/sec",
            "range": "stddev: 0.0055257874827236595",
            "extra": "mean: 5.201330283504635 msec\nrounds: 194"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2318133.476572427,
            "unit": "iter/sec",
            "range": "stddev: 3.747638974585688e-8",
            "extra": "mean: 431.38154472390073 nsec\nrounds: 114260"
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
          "id": "718eec4507b54a7e5c6ff91805ac69837e721da0",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/718eec4507b54a7e5c6ff91805ac69837e721da0"
        },
        "date": 1790483032874,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6050.539264102225,
            "unit": "iter/sec",
            "range": "stddev: 0.000017955035112705217",
            "extra": "mean: 165.27452452593565 usec\nrounds: 1529"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6483.825828566987,
            "unit": "iter/sec",
            "range": "stddev: 0.000017166738512795176",
            "extra": "mean: 154.2299294336556 usec\nrounds: 2154"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4988.282286063315,
            "unit": "iter/sec",
            "range": "stddev: 0.00002051047513655753",
            "extra": "mean: 200.4698095763114 usec\nrounds: 3634"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 753.7901553313612,
            "unit": "iter/sec",
            "range": "stddev: 0.000029604584845873326",
            "extra": "mean: 1.326629159225894 msec\nrounds: 672"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 169.53480425465318,
            "unit": "iter/sec",
            "range": "stddev: 0.004938493382221445",
            "extra": "mean: 5.898493848483936 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2126494.0509224343,
            "unit": "iter/sec",
            "range": "stddev: 4.858523690539358e-8",
            "extra": "mean: 470.2576052663859 nsec\nrounds: 105966"
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
          "id": "14e80ee5cc390ca372469feb0af1583dbdc09ad3",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/14e80ee5cc390ca372469feb0af1583dbdc09ad3"
        },
        "date": 1790483997292,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10786.853589610422,
            "unit": "iter/sec",
            "range": "stddev: 0.00002535235459795068",
            "extra": "mean: 92.70543923606884 usec\nrounds: 1152"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11763.591436883462,
            "unit": "iter/sec",
            "range": "stddev: 0.000016761631028259",
            "extra": "mean: 85.00805263132557 usec\nrounds: 2546"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8482.94450114353,
            "unit": "iter/sec",
            "range": "stddev: 0.000023370170395122767",
            "extra": "mean: 117.88359570962615 usec\nrounds: 4242"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1069.6593798820547,
            "unit": "iter/sec",
            "range": "stddev: 0.000053533602992075675",
            "extra": "mean: 934.8770447937027 usec\nrounds: 826"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 209.39498154427537,
            "unit": "iter/sec",
            "range": "stddev: 0.006438614564791037",
            "extra": "mean: 4.775663641148705 msec\nrounds: 209"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3005212.326699597,
            "unit": "iter/sec",
            "range": "stddev: 3.668273279060863e-8",
            "extra": "mean: 332.755190412195 nsec\nrounds: 140037"
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
          "id": "374dfb53b6b9dcfe1ead0c158886ca7312e84d42",
          "message": "Harden verification and add executable product-use benchmark",
          "timestamp": "2026-09-24T03:29:37Z",
          "url": "https://github.com/stateset/stateset-agents/pull/83/commits/374dfb53b6b9dcfe1ead0c158886ca7312e84d42"
        },
        "date": 1790484815533,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6193.574979617095,
            "unit": "iter/sec",
            "range": "stddev: 0.000017326901060534346",
            "extra": "mean: 161.4576401013915 usec\nrounds: 1581"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6653.526078912521,
            "unit": "iter/sec",
            "range": "stddev: 0.000016940772650406418",
            "extra": "mean: 150.29624715372634 usec\nrounds: 2108"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5139.972687123779,
            "unit": "iter/sec",
            "range": "stddev: 0.00001826653541536629",
            "extra": "mean: 194.55356299949113 usec\nrounds: 3254"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 754.490818367026,
            "unit": "iter/sec",
            "range": "stddev: 0.00006164629130904873",
            "extra": "mean: 1.325397176024407 msec\nrounds: 659"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 167.3243267206212,
            "unit": "iter/sec",
            "range": "stddev: 0.0060236143767711595",
            "extra": "mean: 5.976417294477953 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2164043.7515652543,
            "unit": "iter/sec",
            "range": "stddev: 4.8553523896731794e-8",
            "extra": "mean: 462.0978662176767 nsec\nrounds: 102260"
          }
        ]
      }
    ]
  }
}