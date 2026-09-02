window.BENCHMARK_DATA = {
  "lastUpdate": 1788367683077,
  "repoUrl": "https://github.com/stateset/stateset-agents",
  "entries": {
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
          "id": "a8390f84442d5247c67f32c066d32540069ecd75",
          "message": "Merge pull request #61 from stateset/release/0.48.0\n\nrelease: v0.48.0 flagship evidence pipeline",
          "timestamp": "2026-09-01T20:21:29Z",
          "url": "https://github.com/stateset/stateset-agents/commit/a8390f84442d5247c67f32c066d32540069ecd75"
        },
        "date": 1788351015669,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8593.856080473091,
            "unit": "iter/sec",
            "range": "stddev: 0.000015238421565408091",
            "extra": "mean: 116.36219999915917 usec\nrounds: 1890"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9267.892826110536,
            "unit": "iter/sec",
            "range": "stddev: 0.000013458971850463495",
            "extra": "mean: 107.89939188578974 usec\nrounds: 2169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6799.9545673996345,
            "unit": "iter/sec",
            "range": "stddev: 0.00001551657552697942",
            "extra": "mean: 147.05980607491165 usec\nrounds: 3424"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 829.0458519643039,
            "unit": "iter/sec",
            "range": "stddev: 0.000132399532246641",
            "extra": "mean: 1.2062059024005067 msec\nrounds: 625"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.9753122646351,
            "unit": "iter/sec",
            "range": "stddev: 0.00006700100712798728",
            "extra": "mean: 5.556317627217689 msec\nrounds: 169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2387781.614647398,
            "unit": "iter/sec",
            "range": "stddev: 3.9701815968622414e-8",
            "extra": "mean: 418.7987686418589 nsec\nrounds: 58391"
          }
        ]
      }
    ],
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
          "id": "0dc17de7fd053e71f16037220f076270b6672229",
          "message": "feat: golden path, Modal default, and npm publish contract",
          "timestamp": "2026-09-01T20:26:54Z",
          "url": "https://github.com/stateset/stateset-agents/pull/62/commits/0dc17de7fd053e71f16037220f076270b6672229"
        },
        "date": 1788363763376,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6090.782945804886,
            "unit": "iter/sec",
            "range": "stddev: 0.000016198550274867142",
            "extra": "mean: 164.18250476135654 usec\nrounds: 1890"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6588.210159690318,
            "unit": "iter/sec",
            "range": "stddev: 0.000015341070305101785",
            "extra": "mean: 151.7862933575582 usec\nrounds: 2168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5048.572172629491,
            "unit": "iter/sec",
            "range": "stddev: 0.00003166104472164119",
            "extra": "mean: 198.07580555576396 usec\nrounds: 3240"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 745.1179766682881,
            "unit": "iter/sec",
            "range": "stddev: 0.000025897700197039384",
            "extra": "mean: 1.3420693518513518 msec\nrounds: 648"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.62906271952733,
            "unit": "iter/sec",
            "range": "stddev: 0.00006608710367879341",
            "extra": "mean: 5.567027878787071 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2194770.1393878968,
            "unit": "iter/sec",
            "range": "stddev: 5.0278451341834145e-8",
            "extra": "mean: 455.62857907247263 nsec\nrounds: 103756"
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
          "id": "9bc7ab7ec7e9ac8f4c026c519807d7edf0fc68d9",
          "message": "feat: golden path, Modal default, and npm publish contract",
          "timestamp": "2026-09-01T20:26:54Z",
          "url": "https://github.com/stateset/stateset-agents/pull/62/commits/9bc7ab7ec7e9ac8f4c026c519807d7edf0fc68d9"
        },
        "date": 1788364299960,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6129.494430975177,
            "unit": "iter/sec",
            "range": "stddev: 0.000013979748513797317",
            "extra": "mean: 163.14559239119893 usec\nrounds: 2024"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6559.709133934565,
            "unit": "iter/sec",
            "range": "stddev: 0.000015592767773747183",
            "extra": "mean: 152.44578373556513 usec\nrounds: 2238"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5068.325743029808,
            "unit": "iter/sec",
            "range": "stddev: 0.000017139408699774768",
            "extra": "mean: 197.3038140603424 usec\nrounds: 3044"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 745.2117063665386,
            "unit": "iter/sec",
            "range": "stddev: 0.000027619885406181584",
            "extra": "mean: 1.341900551825392 msec\nrounds: 685"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.92945254911857,
            "unit": "iter/sec",
            "range": "stddev: 0.0001329692294287387",
            "extra": "mean: 5.496636119047372 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2245398.203894378,
            "unit": "iter/sec",
            "range": "stddev: 6.938675646777773e-8",
            "extra": "mean: 445.3553041351944 nsec\nrounds: 106417"
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
          "id": "e8b5ae5d3016881bbbcbee4cf281511af9b479be",
          "message": "feat: execute official agent benchmark pipelines",
          "timestamp": "2026-09-01T20:26:54Z",
          "url": "https://github.com/stateset/stateset-agents/pull/55/commits/e8b5ae5d3016881bbbcbee4cf281511af9b479be"
        },
        "date": 1788364900663,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6045.019297455378,
            "unit": "iter/sec",
            "range": "stddev: 0.000015499338084176823",
            "extra": "mean: 165.42544378988916 usec\nrounds: 2037"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6535.6751521703845,
            "unit": "iter/sec",
            "range": "stddev: 0.000013829027371878299",
            "extra": "mean: 153.00638062892668 usec\nrounds: 1910"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5024.422392485853,
            "unit": "iter/sec",
            "range": "stddev: 0.0000199894194022328",
            "extra": "mean: 199.0278527329877 usec\nrounds: 3531"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 749.5552697502259,
            "unit": "iter/sec",
            "range": "stddev: 0.00002729926968021855",
            "extra": "mean: 1.3341244339903444 msec\nrounds: 659"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.10561235655507,
            "unit": "iter/sec",
            "range": "stddev: 0.00024686467475968136",
            "extra": "mean: 5.552297826345911 msec\nrounds: 167"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2228621.657349432,
            "unit": "iter/sec",
            "range": "stddev: 4.6755627654983164e-8",
            "extra": "mean: 448.707835492064 nsec\nrounds: 108602"
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
          "id": "29ab0c272b11305c55c63b1325279a363116bbbc",
          "message": "feat: golden path, Modal default, and npm publish contract",
          "timestamp": "2026-09-01T20:26:54Z",
          "url": "https://github.com/stateset/stateset-agents/pull/62/commits/29ab0c272b11305c55c63b1325279a363116bbbc"
        },
        "date": 1788365333303,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6152.910967622422,
            "unit": "iter/sec",
            "range": "stddev: 0.000016121466047204785",
            "extra": "mean: 162.52469851459838 usec\nrounds: 2020"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6569.308997134462,
            "unit": "iter/sec",
            "range": "stddev: 0.000019282307694689826",
            "extra": "mean: 152.22301164950542 usec\nrounds: 2146"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5090.522766019335,
            "unit": "iter/sec",
            "range": "stddev: 0.000017684432832761652",
            "extra": "mean: 196.44347859030907 usec\nrounds: 3433"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 740.741542105771,
            "unit": "iter/sec",
            "range": "stddev: 0.000030221914836487822",
            "extra": "mean: 1.349998539513812 msec\nrounds: 658"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 177.7032184435145,
            "unit": "iter/sec",
            "range": "stddev: 0.000055246325802877094",
            "extra": "mean: 5.627360093750156 msec\nrounds: 160"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2250654.915685736,
            "unit": "iter/sec",
            "range": "stddev: 4.6243321457347246e-8",
            "extra": "mean: 444.31511602715744 nsec\nrounds: 105408"
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
          "id": "9864a8bb53452c56993ac32024f84e598d01d63a",
          "message": "feat: golden path, Modal default, and npm publish contract",
          "timestamp": "2026-09-01T20:26:54Z",
          "url": "https://github.com/stateset/stateset-agents/pull/62/commits/9864a8bb53452c56993ac32024f84e598d01d63a"
        },
        "date": 1788366523589,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6165.1110847487735,
            "unit": "iter/sec",
            "range": "stddev: 0.000015007219363620397",
            "extra": "mean: 162.20307894756283 usec\nrounds: 1976"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6509.814815818247,
            "unit": "iter/sec",
            "range": "stddev: 0.000014765633121740932",
            "extra": "mean: 153.61420075577152 usec\nrounds: 2117"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5092.798653326098,
            "unit": "iter/sec",
            "range": "stddev: 0.00001732327282638199",
            "extra": "mean: 196.35569125571098 usec\nrounds: 3728"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 762.6841400077528,
            "unit": "iter/sec",
            "range": "stddev: 0.00002581828768631684",
            "extra": "mean: 1.3111587714277562 msec\nrounds: 700"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.68557331573498,
            "unit": "iter/sec",
            "range": "stddev: 0.00005942959600097894",
            "extra": "mean: 5.473885988094433 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2181868.5062673506,
            "unit": "iter/sec",
            "range": "stddev: 4.844148592068208e-8",
            "extra": "mean: 458.32276194808736 nsec\nrounds: 104734"
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
          "id": "cb6d9ddd310af8c23d6e1354fd8f991ca875095a",
          "message": "Merge PR #62: feat: golden path, Modal default, and npm publish contract\n\nfeat: golden path, Modal default, and npm publish contract",
          "timestamp": "2026-09-02T09:44:44-07:00",
          "tree_id": "c15841a7a06aaa57148842b02378ed0b4895b17b",
          "url": "https://github.com/stateset/stateset-agents/commit/cb6d9ddd310af8c23d6e1354fd8f991ca875095a"
        },
        "date": 1788367681278,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6097.600456777272,
            "unit": "iter/sec",
            "range": "stddev: 0.000015958279307798815",
            "extra": "mean: 163.99893812139408 usec\nrounds: 1810"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6602.106381077601,
            "unit": "iter/sec",
            "range": "stddev: 0.000016339680640969947",
            "extra": "mean: 151.4668110871578 usec\nrounds: 1948"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5051.009840764006,
            "unit": "iter/sec",
            "range": "stddev: 0.00001703944348586178",
            "extra": "mean: 197.98021218045022 usec\nrounds: 3596"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 750.1499347215237,
            "unit": "iter/sec",
            "range": "stddev: 0.00006151101251886526",
            "extra": "mean: 1.3330668359935638 msec\nrounds: 689"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.57936007273457,
            "unit": "iter/sec",
            "range": "stddev: 0.00006300036268967509",
            "extra": "mean: 5.5377314417174555 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2153945.632640379,
            "unit": "iter/sec",
            "range": "stddev: 5.4224377258828355e-8",
            "extra": "mean: 464.26427150538905 nsec\nrounds: 103542"
          }
        ]
      }
    ]
  }
}