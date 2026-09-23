window.BENCHMARK_DATA = {
  "lastUpdate": 1790204473459,
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
          "id": "bc183b317672f6c7a931df49d29d7cca66aac42e",
          "message": "chore(release): v0.55.0 — Evidence-backed A+ release gates",
          "timestamp": "2026-09-09T02:03:34Z",
          "url": "https://github.com/stateset/stateset-agents/pull/80/commits/bc183b317672f6c7a931df49d29d7cca66aac42e"
        },
        "date": 1789140832985,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 9877.225442802666,
            "unit": "iter/sec",
            "range": "stddev: 0.000011757378932458836",
            "extra": "mean: 101.24300652959984 usec\nrounds: 1838"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 10385.97802837691,
            "unit": "iter/sec",
            "range": "stddev: 0.00001099194322289513",
            "extra": "mean: 96.2836621902884 usec\nrounds: 1936"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 7516.8505815884055,
            "unit": "iter/sec",
            "range": "stddev: 0.000023083373068839885",
            "extra": "mean: 133.03443897759203 usec\nrounds: 3130"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 913.3389677703934,
            "unit": "iter/sec",
            "range": "stddev: 0.000016443176229578834",
            "extra": "mean: 1.0948837565106413 msec\nrounds: 768"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 227.06408014370902,
            "unit": "iter/sec",
            "range": "stddev: 0.000035261539963049154",
            "extra": "mean: 4.404043120193644 msec\nrounds: 208"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2498937.3497056314,
            "unit": "iter/sec",
            "range": "stddev: 4.0474665255464143e-8",
            "extra": "mean: 400.17009634827286 nsec\nrounds: 121848"
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
          "id": "978602e104bafa2908b3b029a5072e1e63c49fc1",
          "message": "Release v0.56.0: formal verification foundations",
          "timestamp": "2026-09-09T02:03:34Z",
          "url": "https://github.com/stateset/stateset-agents/pull/82/commits/978602e104bafa2908b3b029a5072e1e63c49fc1"
        },
        "date": 1790203304676,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5986.950319015899,
            "unit": "iter/sec",
            "range": "stddev: 0.000015520683807118968",
            "extra": "mean: 167.02994792252997 usec\nrounds: 1997"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6531.380379334476,
            "unit": "iter/sec",
            "range": "stddev: 0.000014325847845050881",
            "extra": "mean: 153.10699146600558 usec\nrounds: 1992"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4955.468674566053,
            "unit": "iter/sec",
            "range": "stddev: 0.000016988544588738465",
            "extra": "mean: 201.79725989037138 usec\nrounds: 3640"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 739.7458139219018,
            "unit": "iter/sec",
            "range": "stddev: 0.000043314755905884516",
            "extra": "mean: 1.351815692877411 msec\nrounds: 674"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.312078802548,
            "unit": "iter/sec",
            "range": "stddev: 0.00020204114742418432",
            "extra": "mean: 5.545940164635653 msec\nrounds: 164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2204335.732861379,
            "unit": "iter/sec",
            "range": "stddev: 7.128550818371993e-8",
            "extra": "mean: 453.651403954665 nsec\nrounds: 27102"
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
          "id": "40e6197661e41f8b7688a8349695754f5c3a0744",
          "message": "Release v0.56.0: formal verification foundations",
          "timestamp": "2026-09-09T02:03:34Z",
          "url": "https://github.com/stateset/stateset-agents/pull/82/commits/40e6197661e41f8b7688a8349695754f5c3a0744"
        },
        "date": 1790203429674,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8495.030644460368,
            "unit": "iter/sec",
            "range": "stddev: 0.0000156048126284709",
            "extra": "mean: 117.71587906538073 usec\nrounds: 1968"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9351.678484993035,
            "unit": "iter/sec",
            "range": "stddev: 0.000012575828101875718",
            "extra": "mean: 106.93267541273312 usec\nrounds: 2058"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6741.961063255371,
            "unit": "iter/sec",
            "range": "stddev: 0.00001752535849404163",
            "extra": "mean: 148.32479609681218 usec\nrounds: 3433"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 832.6238490467779,
            "unit": "iter/sec",
            "range": "stddev: 0.00003631488818849185",
            "extra": "mean: 1.2010225279336417 msec\nrounds: 716"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 178.90649092668426,
            "unit": "iter/sec",
            "range": "stddev: 0.00004758788814936262",
            "extra": "mean: 5.589512123457831 msec\nrounds: 162"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2358893.401421022,
            "unit": "iter/sec",
            "range": "stddev: 6.212558602212571e-8",
            "extra": "mean: 423.92759223354034 nsec\nrounds: 193125"
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
          "id": "6b4ff821288ccbb71b8419d56e4c84def1295b77",
          "message": "Release v0.56.0: formal verification foundations",
          "timestamp": "2026-09-09T02:03:34Z",
          "url": "https://github.com/stateset/stateset-agents/pull/82/commits/6b4ff821288ccbb71b8419d56e4c84def1295b77"
        },
        "date": 1790203538465,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5902.544312404497,
            "unit": "iter/sec",
            "range": "stddev: 0.000020581375941061205",
            "extra": "mean: 169.41846550790805 usec\nrounds: 1783"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6447.636766860079,
            "unit": "iter/sec",
            "range": "stddev: 0.00001694751863475512",
            "extra": "mean: 155.09558558569174 usec\nrounds: 2109"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4915.341929498615,
            "unit": "iter/sec",
            "range": "stddev: 0.000020208817570341255",
            "extra": "mean: 203.44464624091862 usec\nrounds: 3525"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 733.4638064594087,
            "unit": "iter/sec",
            "range": "stddev: 0.00003596480344845762",
            "extra": "mean: 1.3633937914772103 msec\nrounds: 657"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.5413127908806,
            "unit": "iter/sec",
            "range": "stddev: 0.0000628505329980545",
            "extra": "mean: 5.538898463413141 msec\nrounds: 164"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2177849.5355078867,
            "unit": "iter/sec",
            "range": "stddev: 5.504791834352095e-8",
            "extra": "mean: 459.16854387591763 nsec\nrounds: 103638"
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
          "id": "32ab81129712d4f661e051b2f0cdaf5442d14f4d",
          "message": "Release v0.56.0: formal verification foundations",
          "timestamp": "2026-09-09T02:03:34Z",
          "url": "https://github.com/stateset/stateset-agents/pull/82/commits/32ab81129712d4f661e051b2f0cdaf5442d14f4d"
        },
        "date": 1790203857098,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6074.4076658488775,
            "unit": "iter/sec",
            "range": "stddev: 0.000016166739976812614",
            "extra": "mean: 164.6251050324021 usec\nrounds: 1828"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6474.88792046663,
            "unit": "iter/sec",
            "range": "stddev: 0.00001602421326035964",
            "extra": "mean: 154.44282778070578 usec\nrounds: 2131"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5029.366144024288,
            "unit": "iter/sec",
            "range": "stddev: 0.000017171070623758535",
            "extra": "mean: 198.83221291974618 usec\nrounds: 2771"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 741.5679077676299,
            "unit": "iter/sec",
            "range": "stddev: 0.000024003290848775654",
            "extra": "mean: 1.3484941696173691 msec\nrounds: 678"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.2114200343771,
            "unit": "iter/sec",
            "range": "stddev: 0.00005973277590302771",
            "extra": "mean: 5.488130216049762 msec\nrounds: 162"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2204601.8643002957,
            "unit": "iter/sec",
            "range": "stddev: 4.4566683880055175e-8",
            "extra": "mean: 453.5966408235727 nsec\nrounds: 56234"
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
          "id": "973bbfef3c915eed0af04fdabdf5e0d6690fd48b",
          "message": "Release v0.56.0: formal verification foundations",
          "timestamp": "2026-09-09T02:03:34Z",
          "url": "https://github.com/stateset/stateset-agents/pull/82/commits/973bbfef3c915eed0af04fdabdf5e0d6690fd48b"
        },
        "date": 1790204471998,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11189.644160368796,
            "unit": "iter/sec",
            "range": "stddev: 0.000010467259473820746",
            "extra": "mean: 89.36834680961304 usec\nrounds: 2226"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12159.040489875895,
            "unit": "iter/sec",
            "range": "stddev: 0.000010425853785582422",
            "extra": "mean: 82.24333168662776 usec\nrounds: 2632"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8741.85642506181,
            "unit": "iter/sec",
            "range": "stddev: 0.000012923624718520763",
            "extra": "mean: 114.39217843170303 usec\nrounds: 4349"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1063.2701546928606,
            "unit": "iter/sec",
            "range": "stddev: 0.00008928425961351217",
            "extra": "mean: 940.4947515797271 usec\nrounds: 950"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 231.13983555564792,
            "unit": "iter/sec",
            "range": "stddev: 0.00002541935106997297",
            "extra": "mean: 4.3263853571412865 msec\nrounds: 210"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3023422.626769832,
            "unit": "iter/sec",
            "range": "stddev: 3.862947747291564e-8",
            "extra": "mean: 330.75098107219674 nsec\nrounds: 198492"
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
        "date": 1789213217911,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14180.60137837482,
            "unit": "iter/sec",
            "range": "stddev: 0.000009382614944988402",
            "extra": "mean: 70.51887104907858 usec\nrounds: 2784"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 14855.759939761841,
            "unit": "iter/sec",
            "range": "stddev: 0.000009716118201964414",
            "extra": "mean: 67.31395795670291 usec\nrounds: 2545"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 10928.406377847974,
            "unit": "iter/sec",
            "range": "stddev: 0.00001040213383019982",
            "extra": "mean: 91.5046499393556 usec\nrounds: 4125"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1285.3012202487953,
            "unit": "iter/sec",
            "range": "stddev: 0.000032663706080402726",
            "extra": "mean: 778.027737191777 usec\nrounds: 1054"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 333.81450082146995,
            "unit": "iter/sec",
            "range": "stddev: 0.00005671850187122995",
            "extra": "mean: 2.995675734694396 msec\nrounds: 294"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3511029.026033313,
            "unit": "iter/sec",
            "range": "stddev: 4.4490182750658275e-8",
            "extra": "mean: 284.81678521746056 nsec\nrounds: 176992"
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
        "date": 1789303624625,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10910.731994210191,
            "unit": "iter/sec",
            "range": "stddev: 0.000009904529508800991",
            "extra": "mean: 91.65287906720214 usec\nrounds: 2059"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11953.704366566579,
            "unit": "iter/sec",
            "range": "stddev: 0.000009079497856430507",
            "extra": "mean: 83.65607591876781 usec\nrounds: 2068"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8601.051084400948,
            "unit": "iter/sec",
            "range": "stddev: 0.000010962206086888131",
            "extra": "mean: 116.26485997898813 usec\nrounds: 2871"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1037.306400217397,
            "unit": "iter/sec",
            "range": "stddev: 0.0000441584101049191",
            "extra": "mean: 964.0353127970884 usec\nrounds: 844"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 259.26033998059296,
            "unit": "iter/sec",
            "range": "stddev: 0.00027412548691936253",
            "extra": "mean: 3.8571267787230994 msec\nrounds: 235"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2786475.2207709025,
            "unit": "iter/sec",
            "range": "stddev: 3.78032332892955e-8",
            "extra": "mean: 358.8763296891409 nsec\nrounds: 137081"
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
        "date": 1789394288555,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6033.067226330515,
            "unit": "iter/sec",
            "range": "stddev: 0.000016388131450401765",
            "extra": "mean: 165.75316708483436 usec\nrounds: 1993"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6583.719229017209,
            "unit": "iter/sec",
            "range": "stddev: 0.00001565757496873939",
            "extra": "mean: 151.88983084099047 usec\nrounds: 2140"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5034.717189893766,
            "unit": "iter/sec",
            "range": "stddev: 0.000017533022634212053",
            "extra": "mean: 198.62088818162601 usec\nrounds: 3613"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 739.9784593913145,
            "unit": "iter/sec",
            "range": "stddev: 0.00003717660479475343",
            "extra": "mean: 1.3513906888892036 msec\nrounds: 675"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 174.62295692360394,
            "unit": "iter/sec",
            "range": "stddev: 0.0008232806841865565",
            "extra": "mean: 5.726623907975007 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2230898.7837004396,
            "unit": "iter/sec",
            "range": "stddev: 5.0396197285012727e-8",
            "extra": "mean: 448.2498297575288 nsec\nrounds: 105731"
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
        "date": 1789476251750,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5857.191640541669,
            "unit": "iter/sec",
            "range": "stddev: 0.00002287156341002219",
            "extra": "mean: 170.73028532621493 usec\nrounds: 1840"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6489.893321723703,
            "unit": "iter/sec",
            "range": "stddev: 0.00001507888673428631",
            "extra": "mean: 154.08573768889036 usec\nrounds: 2051"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4963.306455109877,
            "unit": "iter/sec",
            "range": "stddev: 0.000017383254992071782",
            "extra": "mean: 201.47859275754962 usec\nrounds: 3590"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 741.3831001870512,
            "unit": "iter/sec",
            "range": "stddev: 0.000027481067406549302",
            "extra": "mean: 1.3488303142433373 msec\nrounds: 681"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 177.7709443935647,
            "unit": "iter/sec",
            "range": "stddev: 0.000044588451384022364",
            "extra": "mean: 5.625216220858418 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2249825.835912331,
            "unit": "iter/sec",
            "range": "stddev: 6.351275913592676e-8",
            "extra": "mean: 444.4788498903908 nsec\nrounds: 106068"
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
        "date": 1789562475478,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6066.244292791495,
            "unit": "iter/sec",
            "range": "stddev: 0.000016243824119213925",
            "extra": "mean: 164.84664179916027 usec\nrounds: 1823"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6512.493018430115,
            "unit": "iter/sec",
            "range": "stddev: 0.00001679551119655009",
            "extra": "mean: 153.55102833431636 usec\nrounds: 2047"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4994.652413402929,
            "unit": "iter/sec",
            "range": "stddev: 0.000016980886127005625",
            "extra": "mean: 200.21413248228131 usec\nrounds: 3525"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 745.2130131932325,
            "unit": "iter/sec",
            "range": "stddev: 0.000028237566465968422",
            "extra": "mean: 1.3418981986304923 msec\nrounds: 584"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 178.96482186396352,
            "unit": "iter/sec",
            "range": "stddev: 0.0002100410141366951",
            "extra": "mean: 5.5876903046350055 msec\nrounds: 151"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2203805.9131267844,
            "unit": "iter/sec",
            "range": "stddev: 4.708404434129918e-8",
            "extra": "mean: 453.760466855808 nsec\nrounds: 105065"
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
        "date": 1789648840713,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6014.616757277331,
            "unit": "iter/sec",
            "range": "stddev: 0.000016409662979510068",
            "extra": "mean: 166.26163234591115 usec\nrounds: 2059"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6532.774686845856,
            "unit": "iter/sec",
            "range": "stddev: 0.000014350613988910183",
            "extra": "mean: 153.074313432784 usec\nrounds: 2144"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5011.640896639177,
            "unit": "iter/sec",
            "range": "stddev: 0.00001634710066288545",
            "extra": "mean: 199.5354457001505 usec\nrounds: 3628"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 748.5272123757625,
            "unit": "iter/sec",
            "range": "stddev: 0.00002021376229981489",
            "extra": "mean: 1.3359567741379554 msec\nrounds: 580"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.40784493261094,
            "unit": "iter/sec",
            "range": "stddev: 0.00004961413571426067",
            "extra": "mean: 5.573892269736696 msec\nrounds: 152"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2234597.342031083,
            "unit": "iter/sec",
            "range": "stddev: 4.926206830164025e-8",
            "extra": "mean: 447.5079161649204 nsec\nrounds: 107331"
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
        "date": 1789733922599,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6165.826014273864,
            "unit": "iter/sec",
            "range": "stddev: 0.000014895805705480412",
            "extra": "mean: 162.18427144797857 usec\nrounds: 2133"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6663.693918767282,
            "unit": "iter/sec",
            "range": "stddev: 0.000014454259005056967",
            "extra": "mean: 150.0669166666932 usec\nrounds: 2172"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5176.2815973199495,
            "unit": "iter/sec",
            "range": "stddev: 0.000014973867175092326",
            "extra": "mean: 193.18887143190892 usec\nrounds: 3469"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 761.3333371440377,
            "unit": "iter/sec",
            "range": "stddev: 0.00004910967538774169",
            "extra": "mean: 1.313485107260985 msec\nrounds: 606"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.20029079582767,
            "unit": "iter/sec",
            "range": "stddev: 0.00004058569853128583",
            "extra": "mean: 5.48846544444099 msec\nrounds: 153"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2182730.453696757,
            "unit": "iter/sec",
            "range": "stddev: 4.584453822959401e-8",
            "extra": "mean: 458.1417729827158 nsec\nrounds: 106758"
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
        "date": 1789819377113,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5896.907625046795,
            "unit": "iter/sec",
            "range": "stddev: 0.00001927136379487791",
            "extra": "mean: 169.58040783148004 usec\nrounds: 1660"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6442.914198535775,
            "unit": "iter/sec",
            "range": "stddev: 0.000015970313106271604",
            "extra": "mean: 155.2092685367844 usec\nrounds: 1996"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4966.693349752903,
            "unit": "iter/sec",
            "range": "stddev: 0.000016558302121705813",
            "extra": "mean: 201.3412001869918 usec\nrounds: 3202"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 731.4030119330477,
            "unit": "iter/sec",
            "range": "stddev: 0.00009401217625266574",
            "extra": "mean: 1.3672352775210332 msec\nrounds: 645"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.74956983365962,
            "unit": "iter/sec",
            "range": "stddev: 0.00006523657486563952",
            "extra": "mean: 5.5632956503061495 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2232145.82803786,
            "unit": "iter/sec",
            "range": "stddev: 5.609556072237335e-8",
            "extra": "mean: 447.99940373028295 nsec\nrounds: 95511"
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
        "date": 1789906811022,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6032.053346296019,
            "unit": "iter/sec",
            "range": "stddev: 0.000016583285151438798",
            "extra": "mean: 165.7810272208633 usec\nrounds: 2094"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6501.223706322217,
            "unit": "iter/sec",
            "range": "stddev: 0.00001630354366059352",
            "extra": "mean: 153.81719583461407 usec\nrounds: 2017"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5045.000514381617,
            "unit": "iter/sec",
            "range": "stddev: 0.000015606963492997672",
            "extra": "mean: 198.21603529064723 usec\nrounds: 3542"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 741.9347852912649,
            "unit": "iter/sec",
            "range": "stddev: 0.000026604481517643058",
            "extra": "mean: 1.3478273560221672 msec\nrounds: 573"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.31173219232133,
            "unit": "iter/sec",
            "range": "stddev: 0.00004608160347072293",
            "extra": "mean: 5.545950825503664 msec\nrounds: 149"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2237062.840503752,
            "unit": "iter/sec",
            "range": "stddev: 4.631948703877217e-8",
            "extra": "mean: 447.01471138594184 nsec\nrounds: 108484"
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
        "date": 1789999145468,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6134.838188871186,
            "unit": "iter/sec",
            "range": "stddev: 0.000015209364881841379",
            "extra": "mean: 163.00348423435773 usec\nrounds: 1776"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6560.073242706281,
            "unit": "iter/sec",
            "range": "stddev: 0.000015379821848855226",
            "extra": "mean: 152.4373224204219 usec\nrounds: 2016"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5044.885272269653,
            "unit": "iter/sec",
            "range": "stddev: 0.00001777276141191178",
            "extra": "mean: 198.2205632101735 usec\nrounds: 2642"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 760.626785007122,
            "unit": "iter/sec",
            "range": "stddev: 0.00006354378461716151",
            "extra": "mean: 1.314705213793696 msec\nrounds: 580"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.25045774126974,
            "unit": "iter/sec",
            "range": "stddev: 0.00003693604972184314",
            "extra": "mean: 5.486954668830742 msec\nrounds: 154"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2245170.787014097,
            "unit": "iter/sec",
            "range": "stddev: 4.925510322486486e-8",
            "extra": "mean: 445.4004148744169 nsec\nrounds: 106417"
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
        "date": 1790080676822,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 12876.553206181825,
            "unit": "iter/sec",
            "range": "stddev: 0.000010089419169821038",
            "extra": "mean: 77.6605341497689 usec\nrounds: 2533"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 14305.815416315738,
            "unit": "iter/sec",
            "range": "stddev: 0.000010458825314379101",
            "extra": "mean: 69.90164285633821 usec\nrounds: 2296"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 10989.39029140845,
            "unit": "iter/sec",
            "range": "stddev: 0.000009294945769498134",
            "extra": "mean: 90.99685910525938 usec\nrounds: 3932"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1314.3554967549194,
            "unit": "iter/sec",
            "range": "stddev: 0.00001779923876162068",
            "extra": "mean: 760.8291687210591 usec\nrounds: 1055"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 342.005143839838,
            "unit": "iter/sec",
            "range": "stddev: 0.000024777346034688323",
            "extra": "mean: 2.9239326308738294 msec\nrounds: 298"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3728099.8061696636,
            "unit": "iter/sec",
            "range": "stddev: 2.8074267542561604e-8",
            "extra": "mean: 268.2331621983649 nsec\nrounds: 81427"
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
        "date": 1790167813641,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5989.831538355505,
            "unit": "iter/sec",
            "range": "stddev: 0.00001734584473566195",
            "extra": "mean: 166.949603439857 usec\nrounds: 1977"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6501.267514611999,
            "unit": "iter/sec",
            "range": "stddev: 0.000015634408095185638",
            "extra": "mean: 153.81615934930204 usec\nrounds: 2027"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5036.427376697033,
            "unit": "iter/sec",
            "range": "stddev: 0.00001696629257642263",
            "extra": "mean: 198.55344378177367 usec\nrounds: 3522"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 748.7738699358731,
            "unit": "iter/sec",
            "range": "stddev: 0.000028490248920888635",
            "extra": "mean: 1.3355166895522712 msec\nrounds: 670"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.49751511309415,
            "unit": "iter/sec",
            "range": "stddev: 0.00006729411119780445",
            "extra": "mean: 5.5711077636363955 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2168906.7021923,
            "unit": "iter/sec",
            "range": "stddev: 5.12959489394402e-8",
            "extra": "mean: 461.06178702348706 nsec\nrounds: 105175"
          }
        ]
      }
    ]
  }
}