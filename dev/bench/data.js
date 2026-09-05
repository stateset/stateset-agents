window.BENCHMARK_DATA = {
  "lastUpdate": 1788626697336,
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
          "id": "5ea30e23c787bed3244a62a2159d3931ca9f7d67",
          "message": "test(api): stabilize py310 coverage by exercising SSE router helpers (#63)\n\n* test(api): increase coverage for SSE routers by exercising error extraction and content-length helpers (stabilize py310 coverage)\n\nCo-authored-by: Dom Steil <domsteil@users.noreply.github.com>\n\n* test(api): fix ruff findings in new coverage tests (remove unused import, avoid setattr)\n\nCo-authored-by: Dom Steil <domsteil@users.noreply.github.com>\n\n* test(api): format and sort imports per black/isort (py310, 88)\n\nCo-authored-by: Dom Steil <domsteil@users.noreply.github.com>\n\n---------\n\nCo-authored-by: Cursor Agent <cursoragent@cursor.com>\nCo-authored-by: Dom Steil <domsteil@users.noreply.github.com>",
          "timestamp": "2026-09-02T17:56:04Z",
          "url": "https://github.com/stateset/stateset-agents/commit/5ea30e23c787bed3244a62a2159d3931ca9f7d67"
        },
        "date": 1788437437333,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6017.4914839285875,
            "unit": "iter/sec",
            "range": "stddev: 0.00001700104742951689",
            "extra": "mean: 166.18220444030254 usec\nrounds: 1937"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6471.876598346864,
            "unit": "iter/sec",
            "range": "stddev: 0.00001599035316979853",
            "extra": "mean: 154.51468902473107 usec\nrounds: 2132"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4976.961115012196,
            "unit": "iter/sec",
            "range": "stddev: 0.000017328771824677688",
            "extra": "mean: 200.92582137796137 usec\nrounds: 3527"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 733.4958254813532,
            "unit": "iter/sec",
            "range": "stddev: 0.00014183131316289764",
            "extra": "mean: 1.3633342757523599 msec\nrounds: 631"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 178.33558133767878,
            "unit": "iter/sec",
            "range": "stddev: 0.000054775671605182854",
            "extra": "mean: 5.6074059506190075 msec\nrounds: 162"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2271564.2480511954,
            "unit": "iter/sec",
            "range": "stddev: 5.993922735179844e-8",
            "extra": "mean: 440.22527685840845 nsec\nrounds: 108720"
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
          "id": "5ea30e23c787bed3244a62a2159d3931ca9f7d67",
          "message": "test(api): stabilize py310 coverage by exercising SSE router helpers (#63)\n\n* test(api): increase coverage for SSE routers by exercising error extraction and content-length helpers (stabilize py310 coverage)\n\nCo-authored-by: Dom Steil <domsteil@users.noreply.github.com>\n\n* test(api): fix ruff findings in new coverage tests (remove unused import, avoid setattr)\n\nCo-authored-by: Dom Steil <domsteil@users.noreply.github.com>\n\n* test(api): format and sort imports per black/isort (py310, 88)\n\nCo-authored-by: Dom Steil <domsteil@users.noreply.github.com>\n\n---------\n\nCo-authored-by: Cursor Agent <cursoragent@cursor.com>\nCo-authored-by: Dom Steil <domsteil@users.noreply.github.com>",
          "timestamp": "2026-09-02T17:56:04Z",
          "url": "https://github.com/stateset/stateset-agents/commit/5ea30e23c787bed3244a62a2159d3931ca9f7d67"
        },
        "date": 1788523958672,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8404.695980969245,
            "unit": "iter/sec",
            "range": "stddev: 0.000014970540057092768",
            "extra": "mean: 118.98110321471476 usec\nrounds: 1773"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9041.7584985293,
            "unit": "iter/sec",
            "range": "stddev: 0.00001615767901508015",
            "extra": "mean: 110.5979550507411 usec\nrounds: 1980"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6689.71097998931,
            "unit": "iter/sec",
            "range": "stddev: 0.000015771716100303267",
            "extra": "mean: 149.48328903763763 usec\nrounds: 3439"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 817.3619026879062,
            "unit": "iter/sec",
            "range": "stddev: 0.00003434714957524251",
            "extra": "mean: 1.2234482628949133 msec\nrounds: 601"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 174.23620938064332,
            "unit": "iter/sec",
            "range": "stddev: 0.00005475537213406634",
            "extra": "mean: 5.739335144828366 msec\nrounds: 145"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2349312.763523336,
            "unit": "iter/sec",
            "range": "stddev: 5.338675719030506e-8",
            "extra": "mean: 425.6563942981647 nsec\nrounds: 112702"
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
          "id": "5ea30e23c787bed3244a62a2159d3931ca9f7d67",
          "message": "test(api): stabilize py310 coverage by exercising SSE router helpers (#63)\n\n* test(api): increase coverage for SSE routers by exercising error extraction and content-length helpers (stabilize py310 coverage)\n\nCo-authored-by: Dom Steil <domsteil@users.noreply.github.com>\n\n* test(api): fix ruff findings in new coverage tests (remove unused import, avoid setattr)\n\nCo-authored-by: Dom Steil <domsteil@users.noreply.github.com>\n\n* test(api): format and sort imports per black/isort (py310, 88)\n\nCo-authored-by: Dom Steil <domsteil@users.noreply.github.com>\n\n---------\n\nCo-authored-by: Cursor Agent <cursoragent@cursor.com>\nCo-authored-by: Dom Steil <domsteil@users.noreply.github.com>",
          "timestamp": "2026-09-02T17:56:04Z",
          "url": "https://github.com/stateset/stateset-agents/commit/5ea30e23c787bed3244a62a2159d3931ca9f7d67"
        },
        "date": 1788607336725,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6088.685398335821,
            "unit": "iter/sec",
            "range": "stddev: 0.00001550800636570098",
            "extra": "mean: 164.2390655088409 usec\nrounds: 2015"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6512.119909255886,
            "unit": "iter/sec",
            "range": "stddev: 0.000015390092119244094",
            "extra": "mean: 153.55982597597253 usec\nrounds: 1948"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4952.67584243663,
            "unit": "iter/sec",
            "range": "stddev: 0.000026479623133837036",
            "extra": "mean: 201.91105410767557 usec\nrounds: 3031"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 751.90373351982,
            "unit": "iter/sec",
            "range": "stddev: 0.00003070939466123598",
            "extra": "mean: 1.329957487135739 msec\nrounds: 583"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 184.49321800402782,
            "unit": "iter/sec",
            "range": "stddev: 0.000029333801448920618",
            "extra": "mean: 5.420253442477046 msec\nrounds: 113"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2242198.9337716447,
            "unit": "iter/sec",
            "range": "stddev: 5.5384517366618535e-8",
            "extra": "mean: 445.99075708143397 nsec\nrounds: 106633"
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
          "id": "0c4bfb26f83467166b1296519bfca66164d231e8",
          "message": "feat(api): add standard SSE headers for streaming endpoints; add header assertions in tests",
          "timestamp": "2026-09-02T16:45:43Z",
          "url": "https://github.com/stateset/stateset-agents/pull/53/commits/0c4bfb26f83467166b1296519bfca66164d231e8"
        },
        "date": 1788367743742,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8332.967181225657,
            "unit": "iter/sec",
            "range": "stddev: 0.0000192955045040395",
            "extra": "mean: 120.00527282202913 usec\nrounds: 1917"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 8736.458835059884,
            "unit": "iter/sec",
            "range": "stddev: 0.00002194914669465447",
            "extra": "mean: 114.46285261334326 usec\nrounds: 1703"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6346.368897827282,
            "unit": "iter/sec",
            "range": "stddev: 0.00003193138999664316",
            "extra": "mean: 157.57041799797616 usec\nrounds: 3067"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 819.9364625634147,
            "unit": "iter/sec",
            "range": "stddev: 0.00002585808573858338",
            "extra": "mean: 1.219606695955004 msec\nrounds: 717"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 175.95989970336504,
            "unit": "iter/sec",
            "range": "stddev: 0.00010548340738236328",
            "extra": "mean: 5.683113037037473 msec\nrounds: 162"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2322380.5096284216,
            "unit": "iter/sec",
            "range": "stddev: 3.504733245624405e-8",
            "extra": "mean: 430.59265949489 nsec\nrounds: 57353"
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
          "id": "322e21e241fd3dd63213394c88461456c5a143ba",
          "message": "feat: execute official agent benchmark pipelines",
          "timestamp": "2026-09-02T16:45:43Z",
          "url": "https://github.com/stateset/stateset-agents/pull/55/commits/322e21e241fd3dd63213394c88461456c5a143ba"
        },
        "date": 1788367969544,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6049.492252034048,
            "unit": "iter/sec",
            "range": "stddev: 0.00001738538088370655",
            "extra": "mean: 165.30312931035914 usec\nrounds: 1856"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6447.447949609352,
            "unit": "iter/sec",
            "range": "stddev: 0.000032248501470454375",
            "extra": "mean: 155.10012764982298 usec\nrounds: 2123"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5049.735702799869,
            "unit": "iter/sec",
            "range": "stddev: 0.000017374538160919673",
            "extra": "mean: 198.03016610266187 usec\nrounds: 3546"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 750.2677801153905,
            "unit": "iter/sec",
            "range": "stddev: 0.00003109166475065154",
            "extra": "mean: 1.3328574497044254 msec\nrounds: 676"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.30346120214165,
            "unit": "iter/sec",
            "range": "stddev: 0.00030203223979202167",
            "extra": "mean: 5.485359374999362 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2226440.6098576384,
            "unit": "iter/sec",
            "range": "stddev: 5.101678504832449e-8",
            "extra": "mean: 449.14739498213754 nsec\nrounds: 104625"
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
          "id": "437f7f94e80037e3a295022aa0eceb21e26ce4b3",
          "message": "Merge PR #53: feat(api): standard SSE headers for streaming endpoints\n\nfeat(api): add standard SSE headers for streaming endpoints; add header assertions in tests",
          "timestamp": "2026-09-02T09:57:58-07:00",
          "tree_id": "27272963716b861f461ca268a3e7d3c4e3814de2",
          "url": "https://github.com/stateset/stateset-agents/commit/437f7f94e80037e3a295022aa0eceb21e26ce4b3"
        },
        "date": 1788368463040,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14435.448440246315,
            "unit": "iter/sec",
            "range": "stddev: 0.000012538996711795437",
            "extra": "mean: 69.27391304394676 usec\nrounds: 2277"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 14937.030130813162,
            "unit": "iter/sec",
            "range": "stddev: 0.000013127575063455203",
            "extra": "mean: 66.9477125802357 usec\nrounds: 2178"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11358.450183537125,
            "unit": "iter/sec",
            "range": "stddev: 0.000012299022202076901",
            "extra": "mean: 88.0401801162446 usec\nrounds: 3953"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1492.5437912619575,
            "unit": "iter/sec",
            "range": "stddev: 0.0000206219364455969",
            "extra": "mean: 669.9970921151279 usec\nrounds: 1205"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 314.5031125819076,
            "unit": "iter/sec",
            "range": "stddev: 0.00003727020707813908",
            "extra": "mean: 3.1796187700354315 msec\nrounds: 287"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 4521739.936220426,
            "unit": "iter/sec",
            "range": "stddev: 2.2380889829453683e-8",
            "extra": "mean: 221.1538067436641 nsec\nrounds: 198492"
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
          "id": "1f2125b3421deaaf795a56160c6194e4c3188643",
          "message": "feat: execute official agent benchmark pipelines",
          "timestamp": "2026-09-02T16:45:43Z",
          "url": "https://github.com/stateset/stateset-agents/pull/55/commits/1f2125b3421deaaf795a56160c6194e4c3188643"
        },
        "date": 1788368845661,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8560.931684218323,
            "unit": "iter/sec",
            "range": "stddev: 0.000014839862682950406",
            "extra": "mean: 116.8097161484717 usec\nrounds: 1994"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9031.715034972785,
            "unit": "iter/sec",
            "range": "stddev: 0.000018747439004799942",
            "extra": "mean: 110.7209423822364 usec\nrounds: 2048"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6296.42458241135,
            "unit": "iter/sec",
            "range": "stddev: 0.000034154026586067134",
            "extra": "mean: 158.82029347154173 usec\nrounds: 3278"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 829.7603338158808,
            "unit": "iter/sec",
            "range": "stddev: 0.00009375836816145403",
            "extra": "mean: 1.2051672745083213 msec\nrounds: 510"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 177.0837919065549,
            "unit": "iter/sec",
            "range": "stddev: 0.000044617919921740275",
            "extra": "mean: 5.647044200000464 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2401340.884015376,
            "unit": "iter/sec",
            "range": "stddev: 3.7216708603565464e-8",
            "extra": "mean: 416.4340042917442 nsec\nrounds: 59902"
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
          "id": "cee042a2f053e91c4f115a9c0371d2bbdf54ea05",
          "message": "Merge PR #55: feat: execute official agent benchmark pipelines\n\nfeat: execute official agent benchmark pipelines",
          "timestamp": "2026-09-02T10:17:20-07:00",
          "tree_id": "9e4a20340e31d1b01489024552a8b4f72d63f3dc",
          "url": "https://github.com/stateset/stateset-agents/commit/cee042a2f053e91c4f115a9c0371d2bbdf54ea05"
        },
        "date": 1788369603341,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6121.092146675506,
            "unit": "iter/sec",
            "range": "stddev: 0.00001624632050937851",
            "extra": "mean: 163.36953864403446 usec\nrounds: 2109"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6540.295477884375,
            "unit": "iter/sec",
            "range": "stddev: 0.000014991076895508106",
            "extra": "mean: 152.89829081597938 usec\nrounds: 2156"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5036.356560574684,
            "unit": "iter/sec",
            "range": "stddev: 0.000017639246360779475",
            "extra": "mean: 198.55623563830693 usec\nrounds: 3586"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 741.4766587356102,
            "unit": "iter/sec",
            "range": "stddev: 0.00017908806611926525",
            "extra": "mean: 1.3486601206101783 msec\nrounds: 655"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.90377700191854,
            "unit": "iter/sec",
            "range": "stddev: 0.00005164432701598607",
            "extra": "mean: 5.497411964070724 msec\nrounds: 167"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2236362.754049993,
            "unit": "iter/sec",
            "range": "stddev: 4.940861877811029e-8",
            "extra": "mean: 447.15464796085826 nsec\nrounds: 107090"
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
          "id": "012984136e1bcb03497c4c7afe8a057505c3dc32",
          "message": "chore(release): v0.49.0 — Declarative policy objectives",
          "timestamp": "2026-09-02T17:56:12Z",
          "url": "https://github.com/stateset/stateset-agents/pull/64/commits/012984136e1bcb03497c4c7afe8a057505c3dc32"
        },
        "date": 1788625834581,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6101.308265162769,
            "unit": "iter/sec",
            "range": "stddev: 0.000016423724451979087",
            "extra": "mean: 163.8992748014056 usec\nrounds: 2016"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6616.0953255454,
            "unit": "iter/sec",
            "range": "stddev: 0.000014674138832680326",
            "extra": "mean: 151.14655258047762 usec\nrounds: 2054"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4997.122821087064,
            "unit": "iter/sec",
            "range": "stddev: 0.000017513085363676143",
            "extra": "mean: 200.1151534199157 usec\nrounds: 3611"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 739.2679570864041,
            "unit": "iter/sec",
            "range": "stddev: 0.000030609492004284025",
            "extra": "mean: 1.3526894956210338 msec\nrounds: 571"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.19884565165378,
            "unit": "iter/sec",
            "range": "stddev: 0.00012451421452635325",
            "extra": "mean: 5.518798954836902 msec\nrounds: 155"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2215001.213520734,
            "unit": "iter/sec",
            "range": "stddev: 5.8328453549194e-8",
            "extra": "mean: 451.4670212800943 nsec\nrounds: 108614"
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
          "id": "b4b8a8b19a16f414dc171f5b0255dbb886da32f2",
          "message": "chore(release): v0.49.0 — Declarative policy objectives",
          "timestamp": "2026-09-02T17:56:12Z",
          "url": "https://github.com/stateset/stateset-agents/pull/64/commits/b4b8a8b19a16f414dc171f5b0255dbb886da32f2"
        },
        "date": 1788626696029,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 9925.50232811602,
            "unit": "iter/sec",
            "range": "stddev: 0.000014577840223607092",
            "extra": "mean: 100.75056827776818 usec\nrounds: 2131"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 10648.534675513858,
            "unit": "iter/sec",
            "range": "stddev: 0.000010466389872990943",
            "extra": "mean: 93.90963456216042 usec\nrounds: 2170"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 7854.741920852791,
            "unit": "iter/sec",
            "range": "stddev: 0.000012390276615258958",
            "extra": "mean: 127.31163035989726 usec\nrounds: 3195"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 902.566672625278,
            "unit": "iter/sec",
            "range": "stddev: 0.00008527129580593242",
            "extra": "mean: 1.107951390550816 msec\nrounds: 635"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 226.63765855731197,
            "unit": "iter/sec",
            "range": "stddev: 0.00022449642574987923",
            "extra": "mean: 4.412329382352495 msec\nrounds: 170"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2567410.493103137,
            "unit": "iter/sec",
            "range": "stddev: 3.629318031902177e-8",
            "extra": "mean: 389.4975122545892 nsec\nrounds: 122400"
          }
        ]
      }
    ]
  }
}