window.BENCHMARK_DATA = {
  "lastUpdate": 1790451275275,
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
      }
    ]
  }
}