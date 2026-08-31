window.BENCHMARK_DATA = {
  "lastUpdate": 1788181971427,
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
          "id": "b0465b29b884fb4eec52916f4817f1e20efa7e31",
          "message": "feat: add fail-closed training backend protocol",
          "timestamp": "2026-08-28T02:20:26Z",
          "url": "https://github.com/stateset/stateset-agents/pull/35/commits/b0465b29b884fb4eec52916f4817f1e20efa7e31"
        },
        "date": 1787910745320,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8575.351684013509,
            "unit": "iter/sec",
            "range": "stddev: 0.000018231234835065855",
            "extra": "mean: 116.61329317423066 usec\nrounds: 1992"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9483.118336609181,
            "unit": "iter/sec",
            "range": "stddev: 0.000011913361690176684",
            "extra": "mean: 105.45054532743116 usec\nrounds: 2162"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6753.663123611745,
            "unit": "iter/sec",
            "range": "stddev: 0.000017906676388126966",
            "extra": "mean: 148.06779398040467 usec\nrounds: 3422"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 840.6075871294689,
            "unit": "iter/sec",
            "range": "stddev: 0.000027025764279738516",
            "extra": "mean: 1.189615719999422 msec\nrounds: 725"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 175.2730641740679,
            "unit": "iter/sec",
            "range": "stddev: 0.0005872389615664663",
            "extra": "mean: 5.705383224240753 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2233842.197616746,
            "unit": "iter/sec",
            "range": "stddev: 3.514435174451323e-8",
            "extra": "mean: 447.65919502590003 nsec\nrounds: 54834"
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
          "id": "f221fa642dc8fed1c2217015d19f6b7bd76d020e",
          "message": "Merge pull request #35 from stateset/feat/backend-shootout-foundation\n\nfeat: add fail-closed training backend protocol",
          "timestamp": "2026-08-28T03:13:38-07:00",
          "tree_id": "b0bfa82e86632b1371d92b854e5f61546b734d64",
          "url": "https://github.com/stateset/stateset-agents/commit/f221fa642dc8fed1c2217015d19f6b7bd76d020e"
        },
        "date": 1787912189831,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10920.141692113844,
            "unit": "iter/sec",
            "range": "stddev: 0.00001049376123650948",
            "extra": "mean: 91.57390336081134 usec\nrounds: 2380"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11987.695030628744,
            "unit": "iter/sec",
            "range": "stddev: 0.000009919286085855534",
            "extra": "mean: 83.41887222230669 usec\nrounds: 2700"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8670.544258414902,
            "unit": "iter/sec",
            "range": "stddev: 0.000014627473826230202",
            "extra": "mean: 115.3330137297303 usec\nrounds: 3059"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1068.735433959141,
            "unit": "iter/sec",
            "range": "stddev: 0.00001554876544892983",
            "extra": "mean: 935.6852671156323 usec\nrounds: 891"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 229.58299586496662,
            "unit": "iter/sec",
            "range": "stddev: 0.0000746991912624739",
            "extra": "mean: 4.355723280953124 msec\nrounds: 210"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2990354.3283606316,
            "unit": "iter/sec",
            "range": "stddev: 4.1178368213405694e-8",
            "extra": "mean: 334.4085316298349 nsec\nrounds: 189108"
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
          "id": "89f5dfee9f222a4099473bf1e4a20e0df887e07c",
          "message": "feat: add fail-closed OpenRLHF adapter",
          "timestamp": "2026-08-28T10:13:51Z",
          "url": "https://github.com/stateset/stateset-agents/pull/36/commits/89f5dfee9f222a4099473bf1e4a20e0df887e07c"
        },
        "date": 1787912294718,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 7757.732085843557,
            "unit": "iter/sec",
            "range": "stddev: 0.000026386045368366156",
            "extra": "mean: 128.9036523734581 usec\nrounds: 1959"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9171.299669428361,
            "unit": "iter/sec",
            "range": "stddev: 0.00001724289693123927",
            "extra": "mean: 109.03580038207706 usec\nrounds: 2094"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6764.527139583196,
            "unit": "iter/sec",
            "range": "stddev: 0.000016725236089412966",
            "extra": "mean: 147.8299930453995 usec\nrounds: 3451"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 838.1184640095793,
            "unit": "iter/sec",
            "range": "stddev: 0.00002521933654935121",
            "extra": "mean: 1.1931487527621996 msec\nrounds: 724"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 178.1499066121058,
            "unit": "iter/sec",
            "range": "stddev: 0.00004798166864976576",
            "extra": "mean: 5.613250206060154 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2286797.991162977,
            "unit": "iter/sec",
            "range": "stddev: 3.502674294858789e-8",
            "extra": "mean: 437.2926703033522 nsec\nrounds: 57383"
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
          "id": "5ebc895f66bb6993ff880f8b42e743b6beecaaf0",
          "message": "feat: add fail-closed OpenRLHF adapter",
          "timestamp": "2026-08-28T10:13:51Z",
          "url": "https://github.com/stateset/stateset-agents/pull/36/commits/5ebc895f66bb6993ff880f8b42e743b6beecaaf0"
        },
        "date": 1787912840642,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6156.511010059387,
            "unit": "iter/sec",
            "range": "stddev: 0.000015048708335121681",
            "extra": "mean: 162.42966159989922 usec\nrounds: 1974"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6603.674608379714,
            "unit": "iter/sec",
            "range": "stddev: 0.0000147217474779071",
            "extra": "mean: 151.43084105492613 usec\nrounds: 2158"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5039.804829221884,
            "unit": "iter/sec",
            "range": "stddev: 0.000024530007370055118",
            "extra": "mean: 198.4203821151531 usec\nrounds: 3232"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 747.6275440305005,
            "unit": "iter/sec",
            "range": "stddev: 0.000027622582298256394",
            "extra": "mean: 1.3375644169139969 msec\nrounds: 674"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 177.53700574918489,
            "unit": "iter/sec",
            "range": "stddev: 0.00006329124960565722",
            "extra": "mean: 5.632628509082486 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2236458.579631571,
            "unit": "iter/sec",
            "range": "stddev: 4.6973411239378646e-8",
            "extra": "mean: 447.13548871749623 nsec\nrounds: 103864"
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
          "id": "b17c30b3f5b85f84f7b65e7bd45ea1fb48ff5755",
          "message": "feat: add fail-closed OpenRLHF adapter",
          "timestamp": "2026-08-28T10:13:51Z",
          "url": "https://github.com/stateset/stateset-agents/pull/36/commits/b17c30b3f5b85f84f7b65e7bd45ea1fb48ff5755"
        },
        "date": 1787913429741,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 14643.157754927746,
            "unit": "iter/sec",
            "range": "stddev: 0.000008857234186083591",
            "extra": "mean: 68.29128093381894 usec\nrounds: 2570"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 15409.40357358729,
            "unit": "iter/sec",
            "range": "stddev: 0.000008156131836640024",
            "extra": "mean: 64.89543837466003 usec\nrounds: 3002"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 11380.820581792635,
            "unit": "iter/sec",
            "range": "stddev: 0.000009085755975575409",
            "extra": "mean: 87.86712634762284 usec\nrounds: 3989"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1339.0640825601492,
            "unit": "iter/sec",
            "range": "stddev: 0.000012168715013503767",
            "extra": "mean: 746.7902492673131 usec\nrounds: 1023"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 343.3384196605334,
            "unit": "iter/sec",
            "range": "stddev: 0.00002969304464402804",
            "extra": "mean: 2.912578210701625 msec\nrounds: 299"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3788068.344928887,
            "unit": "iter/sec",
            "range": "stddev: 3.10091871444157e-8",
            "extra": "mean: 263.9867892929405 nsec\nrounds: 173401"
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
          "id": "15df35b944cc48965615f564c3b987916c2ce50c",
          "message": "Merge pull request #36 from stateset/feat/openrlhf-backend-adapter\n\nfeat: add fail-closed OpenRLHF adapter",
          "timestamp": "2026-08-28T03:44:05-07:00",
          "tree_id": "252572d28e34e7f2b53196967d6fc7309c903404",
          "url": "https://github.com/stateset/stateset-agents/commit/15df35b944cc48965615f564c3b987916c2ce50c"
        },
        "date": 1787914029707,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8105.95110312474,
            "unit": "iter/sec",
            "range": "stddev: 0.000021855363349209603",
            "extra": "mean: 123.36615250670742 usec\nrounds: 1895"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 8750.41991146274,
            "unit": "iter/sec",
            "range": "stddev: 0.000015048823504042768",
            "extra": "mean: 114.28022999102424 usec\nrounds: 2274"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6744.157412780431,
            "unit": "iter/sec",
            "range": "stddev: 0.000014275353956706028",
            "extra": "mean: 148.27649160515776 usec\nrounds: 3395"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 878.1079939348293,
            "unit": "iter/sec",
            "range": "stddev: 0.000020016407212246957",
            "extra": "mean: 1.1388120902065462 msec\nrounds: 776"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 208.4316774825712,
            "unit": "iter/sec",
            "range": "stddev: 0.00006129749976875094",
            "extra": "mean: 4.7977352198953485 msec\nrounds: 191"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2346292.5053812396,
            "unit": "iter/sec",
            "range": "stddev: 3.363172295376427e-8",
            "extra": "mean: 426.2043192425892 nsec\nrounds: 111038"
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
          "id": "3681832f608a307a4d65b8bb2c001d428a3cb91f",
          "message": "feat: add fail-closed verl adapter",
          "timestamp": "2026-08-28T10:45:30Z",
          "url": "https://github.com/stateset/stateset-agents/pull/37/commits/3681832f608a307a4d65b8bb2c001d428a3cb91f"
        },
        "date": 1787914509318,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8499.909439065475,
            "unit": "iter/sec",
            "range": "stddev: 0.000015059416843122082",
            "extra": "mean: 117.6483122754241 usec\nrounds: 1947"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9355.802388548169,
            "unit": "iter/sec",
            "range": "stddev: 0.00001211079545281096",
            "extra": "mean: 106.8855410225461 usec\nrounds: 2072"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6797.317134707037,
            "unit": "iter/sec",
            "range": "stddev: 0.00001485951010006405",
            "extra": "mean: 147.11686687296216 usec\nrounds: 3553"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 834.2808061812269,
            "unit": "iter/sec",
            "range": "stddev: 0.000021238711468396847",
            "extra": "mean: 1.198637188571224 msec\nrounds: 700"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 178.95118584085566,
            "unit": "iter/sec",
            "range": "stddev: 0.00006085138207470015",
            "extra": "mean: 5.588116084848507 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2356165.0540340436,
            "unit": "iter/sec",
            "range": "stddev: 3.300408761602344e-8",
            "extra": "mean: 424.4184838782314 nsec\nrounds: 55754"
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
          "id": "a80b82f89559ab0ffd1c1a85b562dca3e71cefdf",
          "message": "Merge pull request #37 from stateset/feat/verl-backend-adapter\n\nfeat: add fail-closed verl adapter",
          "timestamp": "2026-08-28T04:02:33-07:00",
          "tree_id": "8271ff650346ee75d3636080abbc704c051efa4b",
          "url": "https://github.com/stateset/stateset-agents/commit/a80b82f89559ab0ffd1c1a85b562dca3e71cefdf"
        },
        "date": 1787915108309,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11896.690414617766,
            "unit": "iter/sec",
            "range": "stddev: 0.000009088509340430792",
            "extra": "mean: 84.05699107470045 usec\nrounds: 2577"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12578.593253289027,
            "unit": "iter/sec",
            "range": "stddev: 0.000008344194206647927",
            "extra": "mean: 79.50014599116813 usec\nrounds: 2507"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8895.952153416845,
            "unit": "iter/sec",
            "range": "stddev: 0.00003155558516822687",
            "extra": "mean: 112.41067653628399 usec\nrounds: 4019"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1084.9492203238124,
            "unit": "iter/sec",
            "range": "stddev: 0.00005929698515464857",
            "extra": "mean: 921.7021232583967 usec\nrounds: 933"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 278.0961366089893,
            "unit": "iter/sec",
            "range": "stddev: 0.00005832232850683119",
            "extra": "mean: 3.595878792829212 msec\nrounds: 251"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2981673.1532093333,
            "unit": "iter/sec",
            "range": "stddev: 3.57104389025308e-8",
            "extra": "mean: 335.382165856659 nsec\nrounds: 142858"
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
          "id": "e6d592e2ef6170fc38c8b320346e063e4580eee0",
          "message": "feat: add NeMo RL backend adapter",
          "timestamp": "2026-08-28T11:05:06Z",
          "url": "https://github.com/stateset/stateset-agents/pull/38/commits/e6d592e2ef6170fc38c8b320346e063e4580eee0"
        },
        "date": 1787916059314,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6056.0908158887705,
            "unit": "iter/sec",
            "range": "stddev: 0.00001938131973853286",
            "extra": "mean: 165.1230191886156 usec\nrounds: 1824"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6493.319691147605,
            "unit": "iter/sec",
            "range": "stddev: 0.000019479751745761212",
            "extra": "mean: 154.00443033219327 usec\nrounds: 2110"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4985.579754925263,
            "unit": "iter/sec",
            "range": "stddev: 0.00002422255396485367",
            "extra": "mean: 200.5784781623638 usec\nrounds: 2221"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 746.7913325924686,
            "unit": "iter/sec",
            "range": "stddev: 0.00004096668496271042",
            "extra": "mean: 1.339062140060629 msec\nrounds: 664"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.3599012461807,
            "unit": "iter/sec",
            "range": "stddev: 0.00005872308393554086",
            "extra": "mean: 5.513898017856685 msec\nrounds: 168"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2151625.3244688823,
            "unit": "iter/sec",
            "range": "stddev: 5.108106837013868e-8",
            "extra": "mean: 464.7649331078797 nsec\nrounds: 101958"
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
          "id": "7eb3b33c0219b5cb3af4230962958dcabc036aa9",
          "message": "Merge pull request #38 from stateset/feat/nemo-rl-backend-adapter\n\nfeat: add NeMo RL backend adapter",
          "timestamp": "2026-08-28T04:27:23-07:00",
          "tree_id": "4ccc6afea9e4b7683493b21bad26bf20be98c5e3",
          "url": "https://github.com/stateset/stateset-agents/commit/7eb3b33c0219b5cb3af4230962958dcabc036aa9"
        },
        "date": 1787916600014,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5766.2781202898595,
            "unit": "iter/sec",
            "range": "stddev: 0.000033189120233657805",
            "extra": "mean: 173.4220894551184 usec\nrounds: 2001"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6408.300150463707,
            "unit": "iter/sec",
            "range": "stddev: 0.00002480067559605144",
            "extra": "mean: 156.04762207145362 usec\nrounds: 1622"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5169.099620515805,
            "unit": "iter/sec",
            "range": "stddev: 0.000016361054711487593",
            "extra": "mean: 193.45728916329412 usec\nrounds: 3645"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 756.2003064509171,
            "unit": "iter/sec",
            "range": "stddev: 0.00002668543827630529",
            "extra": "mean: 1.3224009451851593 msec\nrounds: 675"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 183.61952850705924,
            "unit": "iter/sec",
            "range": "stddev: 0.00006865788947642492",
            "extra": "mean: 5.446043828402245 msec\nrounds: 169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2207967.358008944,
            "unit": "iter/sec",
            "range": "stddev: 5.2328584136509555e-8",
            "extra": "mean: 452.905246254075 nsec\nrounds: 102902"
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
          "id": "a31e08852948bff37dad1ad69833f33314a9dc99",
          "message": "feat: add backend conformance evidence runner",
          "timestamp": "2026-08-28T11:27:32Z",
          "url": "https://github.com/stateset/stateset-agents/pull/39/commits/a31e08852948bff37dad1ad69833f33314a9dc99"
        },
        "date": 1787918416953,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6222.3110113565335,
            "unit": "iter/sec",
            "range": "stddev: 0.000017846466410371166",
            "extra": "mean: 160.7119924052123 usec\nrounds: 1975"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6649.304321006326,
            "unit": "iter/sec",
            "range": "stddev: 0.00001588585592335863",
            "extra": "mean: 150.39167283121986 usec\nrounds: 2271"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5139.772178648715,
            "unit": "iter/sec",
            "range": "stddev: 0.000018260024533212696",
            "extra": "mean: 194.5611527596749 usec\nrounds: 3823"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 744.4309747411256,
            "unit": "iter/sec",
            "range": "stddev: 0.000046552075340468846",
            "extra": "mean: 1.3433078874072215 msec\nrounds: 675"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 182.60267493148555,
            "unit": "iter/sec",
            "range": "stddev: 0.00012013630912640914",
            "extra": "mean: 5.476371035502139 msec\nrounds: 169"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2261931.2479254436,
            "unit": "iter/sec",
            "range": "stddev: 5.286881759588484e-8",
            "extra": "mean: 442.1000863386815 nsec\nrounds: 105397"
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
          "id": "5f12ac10d015c9eccae67f803bbc8a5d93d96fec",
          "message": "feat: add backend conformance evidence runner",
          "timestamp": "2026-08-28T11:27:32Z",
          "url": "https://github.com/stateset/stateset-agents/pull/39/commits/5f12ac10d015c9eccae67f803bbc8a5d93d96fec"
        },
        "date": 1787918858589,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 9737.424662875006,
            "unit": "iter/sec",
            "range": "stddev: 0.000013720462466340909",
            "extra": "mean: 102.69655834284491 usec\nrounds: 1834"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 10518.693432127146,
            "unit": "iter/sec",
            "range": "stddev: 0.000012984115586789053",
            "extra": "mean: 95.06884162492173 usec\nrounds: 2191"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 7959.583342706457,
            "unit": "iter/sec",
            "range": "stddev: 0.00001899825177022058",
            "extra": "mean: 125.63471691220147 usec\nrounds: 2720"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 930.598838404554,
            "unit": "iter/sec",
            "range": "stddev: 0.00009540223051547378",
            "extra": "mean: 1.0745768839712173 msec\nrounds: 836"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 199.87784391942628,
            "unit": "iter/sec",
            "range": "stddev: 0.00017646324606733745",
            "extra": "mean: 5.003055768417808 msec\nrounds: 190"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2701686.6190163763,
            "unit": "iter/sec",
            "range": "stddev: 4.670577722176741e-8",
            "extra": "mean: 370.1391541718031 nsec\nrounds: 126551"
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
          "id": "fe04c3b495d718376bdfffa5e51d5efa66da69df",
          "message": "feat: add backend conformance evidence runner",
          "timestamp": "2026-08-28T11:27:32Z",
          "url": "https://github.com/stateset/stateset-agents/pull/39/commits/fe04c3b495d718376bdfffa5e51d5efa66da69df"
        },
        "date": 1787919347607,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8624.534141892622,
            "unit": "iter/sec",
            "range": "stddev: 0.000015369814502995647",
            "extra": "mean: 115.94829164657392 usec\nrounds: 1975"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9382.602757576005,
            "unit": "iter/sec",
            "range": "stddev: 0.00001887619567691688",
            "extra": "mean: 106.58023427375177 usec\nrounds: 1558"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6850.405434013335,
            "unit": "iter/sec",
            "range": "stddev: 0.000014945603783633371",
            "extra": "mean: 145.97676146799188 usec\nrounds: 3488"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 826.5906460437322,
            "unit": "iter/sec",
            "range": "stddev: 0.0000907320242674477",
            "extra": "mean: 1.2097886720425013 msec\nrounds: 744"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 176.09152894567737,
            "unit": "iter/sec",
            "range": "stddev: 0.00006150497753985962",
            "extra": "mean: 5.678864883435085 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2367915.4718948933,
            "unit": "iter/sec",
            "range": "stddev: 4.824390285357526e-8",
            "extra": "mean: 422.31237215565096 nsec\nrounds: 110583"
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
          "id": "45ccf36043865d105eaaf286c916bfac0b3049e7",
          "message": "Merge pull request #39 from stateset/feat/backend-conformance-runner\n\nfeat: add backend conformance evidence runner",
          "timestamp": "2026-08-28T05:22:12-07:00",
          "tree_id": "fc197616e3332d8ee6df27950d37a87a95f84abe",
          "url": "https://github.com/stateset/stateset-agents/commit/45ccf36043865d105eaaf286c916bfac0b3049e7"
        },
        "date": 1787919869103,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5590.787310360125,
            "unit": "iter/sec",
            "range": "stddev: 0.000038666656503205076",
            "extra": "mean: 178.86568464998288 usec\nrounds: 1557"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6507.526577118507,
            "unit": "iter/sec",
            "range": "stddev: 0.000014924594286057229",
            "extra": "mean: 153.66821604942166 usec\nrounds: 1944"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5016.825145793062,
            "unit": "iter/sec",
            "range": "stddev: 0.000015863299874705664",
            "extra": "mean: 199.32925125735466 usec\nrounds: 3379"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 725.1906061036009,
            "unit": "iter/sec",
            "range": "stddev: 0.00005080020100490068",
            "extra": "mean: 1.3789478125936174 msec\nrounds: 667"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.23550081750608,
            "unit": "iter/sec",
            "range": "stddev: 0.00010149033514242051",
            "extra": "mean: 5.548296509090795 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2256812.8293582923,
            "unit": "iter/sec",
            "range": "stddev: 4.916034131255517e-8",
            "extra": "mean: 443.1027628836825 nsec\nrounds: 105397"
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
          "id": "cf0be0803e767db6ae70525d8c34297659b80590",
          "message": "feat: gate backend conformance roster",
          "timestamp": "2026-08-28T12:22:49Z",
          "url": "https://github.com/stateset/stateset-agents/pull/40/commits/cf0be0803e767db6ae70525d8c34297659b80590"
        },
        "date": 1787921870619,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8451.44993799217,
            "unit": "iter/sec",
            "range": "stddev: 0.000018969616844836004",
            "extra": "mean: 118.32289220630136 usec\nrounds: 1809"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 9287.976091591549,
            "unit": "iter/sec",
            "range": "stddev: 0.000016179903649638813",
            "extra": "mean: 107.66608248543027 usec\nrounds: 1964"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6401.217094467924,
            "unit": "iter/sec",
            "range": "stddev: 0.000033533462292799506",
            "extra": "mean: 156.22029142930063 usec\nrounds: 3325"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 821.8613415338319,
            "unit": "iter/sec",
            "range": "stddev: 0.00007884594487178743",
            "extra": "mean: 1.2167502587890429 msec\nrounds: 711"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 178.15764857311788,
            "unit": "iter/sec",
            "range": "stddev: 0.00006787425029183237",
            "extra": "mean: 5.613006278479191 msec\nrounds: 158"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2348809.5187324905,
            "unit": "iter/sec",
            "range": "stddev: 3.342850758028149e-8",
            "extra": "mean: 425.7475934190012 nsec\nrounds: 55504"
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
          "id": "9f2b7907e52212fd5f607a6fe67a080d3bb42b43",
          "message": "Merge pull request #40 from stateset/feat/backend-conformance-suite\n\nfeat: gate backend conformance roster",
          "timestamp": "2026-08-28T06:04:38-07:00",
          "tree_id": "8e1b0d075fb41c4dbae513160db88f8c6af911bf",
          "url": "https://github.com/stateset/stateset-agents/commit/9f2b7907e52212fd5f607a6fe67a080d3bb42b43"
        },
        "date": 1787922415656,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 11052.782931971597,
            "unit": "iter/sec",
            "range": "stddev: 0.000011077382022886726",
            "extra": "mean: 90.47495152622344 usec\nrounds: 2228"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 12117.371222075217,
            "unit": "iter/sec",
            "range": "stddev: 0.000010270016275274006",
            "extra": "mean: 82.52615040613902 usec\nrounds: 2214"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8777.65808097927,
            "unit": "iter/sec",
            "range": "stddev: 0.000011391935864145556",
            "extra": "mean: 113.92560416165539 usec\nrounds: 4325"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1085.6081968184737,
            "unit": "iter/sec",
            "range": "stddev: 0.000014685760411169366",
            "extra": "mean: 921.1426396103489 usec\nrounds: 924"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 230.66500132340767,
            "unit": "iter/sec",
            "range": "stddev: 0.00004087688777274436",
            "extra": "mean: 4.335291415093933 msec\nrounds: 212"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2917830.956411401,
            "unit": "iter/sec",
            "range": "stddev: 4.8810767714038803e-8",
            "extra": "mean: 342.7203340216412 nsec\nrounds: 192382"
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
          "id": "a67073f026868b57e35b8e97b684e7691a45702d",
          "message": "feat: bind conformance execution envelopes",
          "timestamp": "2026-08-28T13:06:44Z",
          "url": "https://github.com/stateset/stateset-agents/pull/41/commits/a67073f026868b57e35b8e97b684e7691a45702d"
        },
        "date": 1787923088755,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 5959.965476722008,
            "unit": "iter/sec",
            "range": "stddev: 0.000018142641380866305",
            "extra": "mean: 167.7862068003122 usec\nrounds: 1794"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6417.770264243004,
            "unit": "iter/sec",
            "range": "stddev: 0.000019201956044128556",
            "extra": "mean: 155.81735693649873 usec\nrounds: 2076"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4923.352788620826,
            "unit": "iter/sec",
            "range": "stddev: 0.0000205878610477218",
            "extra": "mean: 203.11361849007966 usec\nrounds: 2477"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 737.2964651484325,
            "unit": "iter/sec",
            "range": "stddev: 0.000038377669663499744",
            "extra": "mean: 1.3563065161294108 msec\nrounds: 651"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.3940041265311,
            "unit": "iter/sec",
            "range": "stddev: 0.00009368489064045073",
            "extra": "mean: 5.574322312883295 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2249855.200132552,
            "unit": "iter/sec",
            "range": "stddev: 5.0584155264899705e-8",
            "extra": "mean: 444.47304872824003 nsec\nrounds: 105854"
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
          "id": "983a741b5d3b4a00d032bd50e0c2015bd8f947b4",
          "message": "Merge pull request #41 from stateset/feat/conformance-execution-envelope\n\nfeat: bind conformance execution envelopes",
          "timestamp": "2026-08-28T06:26:33-07:00",
          "tree_id": "579a320950bdde8e9f07fa0bcdbe784d273801b3",
          "url": "https://github.com/stateset/stateset-agents/commit/983a741b5d3b4a00d032bd50e0c2015bd8f947b4"
        },
        "date": 1787923743678,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6035.26871971086,
            "unit": "iter/sec",
            "range": "stddev: 0.00001891272649415657",
            "extra": "mean: 165.69270507111875 usec\nrounds: 1814"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6527.000678866276,
            "unit": "iter/sec",
            "range": "stddev: 0.000018802012904655454",
            "extra": "mean: 153.20972820454762 usec\nrounds: 2145"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 4995.729371378917,
            "unit": "iter/sec",
            "range": "stddev: 0.000020623050268798458",
            "extra": "mean: 200.17097117572263 usec\nrounds: 3608"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 736.936172582916,
            "unit": "iter/sec",
            "range": "stddev: 0.000034380351456784167",
            "extra": "mean: 1.3569696226133958 msec\nrounds: 681"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 176.92521311742254,
            "unit": "iter/sec",
            "range": "stddev: 0.00006411870493500231",
            "extra": "mean: 5.652105668717298 msec\nrounds: 163"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2230139.916051114,
            "unit": "iter/sec",
            "range": "stddev: 5.1156642421985916e-8",
            "extra": "mean: 448.40235933299186 nsec\nrounds: 107551"
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
          "id": "5a364ef5d2bc0cb4a4156f513cfea95137c85565",
          "message": "feat: add safe RunPod conformance launcher",
          "timestamp": "2026-08-28T13:30:43Z",
          "url": "https://github.com/stateset/stateset-agents/pull/42/commits/5a364ef5d2bc0cb4a4156f513cfea95137c85565"
        },
        "date": 1788143826980,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6081.20356270761,
            "unit": "iter/sec",
            "range": "stddev: 0.000015820220203947796",
            "extra": "mean: 164.44113236603405 usec\nrounds: 1798"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6621.394774138952,
            "unit": "iter/sec",
            "range": "stddev: 0.00001548020142628545",
            "extra": "mean: 151.0255820881848 usec\nrounds: 2144"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5015.660546381554,
            "unit": "iter/sec",
            "range": "stddev: 0.000017510487949664337",
            "extra": "mean: 199.37553404036277 usec\nrounds: 3290"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 751.3722732538033,
            "unit": "iter/sec",
            "range": "stddev: 0.0000310656575895759",
            "extra": "mean: 1.3308981920100924 msec\nrounds: 651"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.72288391300674,
            "unit": "iter/sec",
            "range": "stddev: 0.00007717108615958134",
            "extra": "mean: 5.564121709086535 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2202206.3640813665,
            "unit": "iter/sec",
            "range": "stddev: 5.626018970637929e-8",
            "extra": "mean: 454.0900509190665 nsec\nrounds: 107447"
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
          "id": "a53d731b87d13f977f1dc50f87df13ba93b41a78",
          "message": "feat: add safe RunPod conformance launcher",
          "timestamp": "2026-08-28T13:30:43Z",
          "url": "https://github.com/stateset/stateset-agents/pull/42/commits/a53d731b87d13f977f1dc50f87df13ba93b41a78"
        },
        "date": 1788179462016,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10791.689223179988,
            "unit": "iter/sec",
            "range": "stddev: 0.00001673828070410798",
            "extra": "mean: 92.66389897997172 usec\nrounds: 1960"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 11943.748187738469,
            "unit": "iter/sec",
            "range": "stddev: 0.000014999405567277385",
            "extra": "mean: 83.72581071548433 usec\nrounds: 2557"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8681.567983034281,
            "unit": "iter/sec",
            "range": "stddev: 0.000014610855919504378",
            "extra": "mean: 115.18656560130877 usec\nrounds: 3849"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 1077.9250395715715,
            "unit": "iter/sec",
            "range": "stddev: 0.00002751389209875027",
            "extra": "mean: 927.7082944445345 usec\nrounds: 900"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 227.34006965462586,
            "unit": "iter/sec",
            "range": "stddev: 0.00017715092516929067",
            "extra": "mean: 4.3986966376811445 msec\nrounds: 207"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 3032522.2614867673,
            "unit": "iter/sec",
            "range": "stddev: 6.281714626571508e-8",
            "extra": "mean: 329.758502583828 nsec\nrounds: 144509"
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
          "id": "ab455977eb13accb49af51328d409fc091cf2281",
          "message": "Merge pull request #42 from stateset/feat/runpod-conformance-launcher\n\nfeat: add safe RunPod conformance launcher",
          "timestamp": "2026-08-31T05:46:05-07:00",
          "tree_id": "759eaed6408cb61f7aab67b26524ff017120ef49",
          "url": "https://github.com/stateset/stateset-agents/commit/ab455977eb13accb49af51328d409fc091cf2281"
        },
        "date": 1788180526173,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6142.672950024202,
            "unit": "iter/sec",
            "range": "stddev: 0.00001563638148446243",
            "extra": "mean: 162.79557908028622 usec\nrounds: 1979"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6596.371336139099,
            "unit": "iter/sec",
            "range": "stddev: 0.000016886299242098532",
            "extra": "mean: 151.59849999974483 usec\nrounds: 2224"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5076.908294062684,
            "unit": "iter/sec",
            "range": "stddev: 0.000017069601995628526",
            "extra": "mean: 196.97027050291112 usec\nrounds: 3719"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 750.2315543218868,
            "unit": "iter/sec",
            "range": "stddev: 0.00003380641252831704",
            "extra": "mean: 1.3329218082594139 msec\nrounds: 678"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 180.07526215203694,
            "unit": "iter/sec",
            "range": "stddev: 0.000053889817216741386",
            "extra": "mean: 5.553233620483103 msec\nrounds: 166"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2216924.996397639,
            "unit": "iter/sec",
            "range": "stddev: 4.727580893338915e-8",
            "extra": "mean: 451.07525136165447 nsec\nrounds: 106519"
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
          "id": "deba85df92a9c8b2da7f99b0a3de6e17f98a5ef4",
          "message": "fix: harden RunPod provisioning retries",
          "timestamp": "2026-08-31T12:49:07Z",
          "url": "https://github.com/stateset/stateset-agents/pull/43/commits/deba85df92a9c8b2da7f99b0a3de6e17f98a5ef4"
        },
        "date": 1788181420476,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 10227.625464699866,
            "unit": "iter/sec",
            "range": "stddev: 0.000009470624125137362",
            "extra": "mean: 97.77440555009075 usec\nrounds: 2054"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 10738.98114145976,
            "unit": "iter/sec",
            "range": "stddev: 0.000010176541083678185",
            "extra": "mean: 93.11870342516208 usec\nrounds: 2131"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 8006.705785518268,
            "unit": "iter/sec",
            "range": "stddev: 0.000010018383119158991",
            "extra": "mean: 124.89530985498433 usec\nrounds: 3237"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 945.0348144533588,
            "unit": "iter/sec",
            "range": "stddev: 0.000018485313664100512",
            "extra": "mean: 1.058162074778626 msec\nrounds: 789"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 237.1974228411336,
            "unit": "iter/sec",
            "range": "stddev: 0.00005551383357071108",
            "extra": "mean: 4.2158974074088675 msec\nrounds: 216"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2505894.3376754285,
            "unit": "iter/sec",
            "range": "stddev: 8.358542238112221e-8",
            "extra": "mean: 399.0591243075482 nsec\nrounds: 122760"
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
          "id": "c9d84870993177cbf48607735bf3ed10c8366c9d",
          "message": "Merge pull request #43 from stateset/fix/runpod-provisioning-resilience\n\nfix: harden RunPod provisioning retries",
          "timestamp": "2026-08-31T06:09:49-07:00",
          "tree_id": "f5249fc8c133844d842f415c256aa17a958c70d3",
          "url": "https://github.com/stateset/stateset-agents/commit/c9d84870993177cbf48607735bf3ed10c8366c9d"
        },
        "date": 1788181970176,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6104.1196015247015,
            "unit": "iter/sec",
            "range": "stddev: 0.00001708158583904082",
            "extra": "mean: 163.82378873281212 usec\nrounds: 1775"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6602.620872564329,
            "unit": "iter/sec",
            "range": "stddev: 0.000017010387851608728",
            "extra": "mean: 151.45500844297598 usec\nrounds: 2132"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5083.243288405619,
            "unit": "iter/sec",
            "range": "stddev: 0.00001729548433099014",
            "extra": "mean: 196.72479621050252 usec\nrounds: 2797"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 754.1280986064063,
            "unit": "iter/sec",
            "range": "stddev: 0.00003631244403008772",
            "extra": "mean: 1.3260346642008878 msec\nrounds: 676"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 181.78722514589109,
            "unit": "iter/sec",
            "range": "stddev: 0.00007517646738524033",
            "extra": "mean: 5.500936598803698 msec\nrounds: 167"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2192292.654372574,
            "unit": "iter/sec",
            "range": "stddev: 4.8771586061143777e-8",
            "extra": "mean: 456.14347975188394 nsec\nrounds: 105065"
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
          "id": "983a741b5d3b4a00d032bd50e0c2015bd8f947b4",
          "message": "Merge pull request #41 from stateset/feat/conformance-execution-envelope\n\nfeat: bind conformance execution envelopes",
          "timestamp": "2026-08-28T13:26:33Z",
          "url": "https://github.com/stateset/stateset-agents/commit/983a741b5d3b4a00d032bd50e0c2015bd8f947b4"
        },
        "date": 1787945393578,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6059.364402474052,
            "unit": "iter/sec",
            "range": "stddev: 0.00001657325304334011",
            "extra": "mean: 165.03381106963923 usec\nrounds: 1879"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6544.171362320496,
            "unit": "iter/sec",
            "range": "stddev: 0.000016200839557221626",
            "extra": "mean: 152.80773449144678 usec\nrounds: 2015"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5027.682071190219,
            "unit": "iter/sec",
            "range": "stddev: 0.00001871981233423116",
            "extra": "mean: 198.8988137754834 usec\nrounds: 3528"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 750.934235853406,
            "unit": "iter/sec",
            "range": "stddev: 0.00003951916191138492",
            "extra": "mean: 1.3316745358713615 msec\nrounds: 683"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 185.37252607207583,
            "unit": "iter/sec",
            "range": "stddev: 0.0000722566745154632",
            "extra": "mean: 5.394542660605401 msec\nrounds: 165"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2232411.3411483965,
            "unit": "iter/sec",
            "range": "stddev: 5.0907045976233865e-8",
            "extra": "mean: 447.94612066680344 nsec\nrounds: 108614"
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
          "id": "983a741b5d3b4a00d032bd50e0c2015bd8f947b4",
          "message": "Merge pull request #41 from stateset/feat/conformance-execution-envelope\n\nfeat: bind conformance execution envelopes",
          "timestamp": "2026-08-28T13:26:33Z",
          "url": "https://github.com/stateset/stateset-agents/commit/983a741b5d3b4a00d032bd50e0c2015bd8f947b4"
        },
        "date": 1788009196999,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 8429.614888992895,
            "unit": "iter/sec",
            "range": "stddev: 0.00001090564587310825",
            "extra": "mean: 118.62938143304339 usec\nrounds: 2079"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 8831.657325333026,
            "unit": "iter/sec",
            "range": "stddev: 0.00001245165879831509",
            "extra": "mean: 113.22903087867392 usec\nrounds: 2105"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 6798.361931328686,
            "unit": "iter/sec",
            "range": "stddev: 0.00001200055713211038",
            "extra": "mean: 147.09425742570872 usec\nrounds: 3737"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 874.6568914167384,
            "unit": "iter/sec",
            "range": "stddev: 0.00006270273309946821",
            "extra": "mean: 1.1433054604763193 msec\nrounds: 797"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 207.23413859238119,
            "unit": "iter/sec",
            "range": "stddev: 0.00004405089846078646",
            "extra": "mean: 4.825459776040801 msec\nrounds: 192"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2322585.9217210617,
            "unit": "iter/sec",
            "range": "stddev: 3.2803701667962196e-8",
            "extra": "mean: 430.55457739922457 nsec\nrounds: 113547"
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
          "id": "983a741b5d3b4a00d032bd50e0c2015bd8f947b4",
          "message": "Merge pull request #41 from stateset/feat/conformance-execution-envelope\n\nfeat: bind conformance execution envelopes",
          "timestamp": "2026-08-28T13:26:33Z",
          "url": "https://github.com/stateset/stateset-agents/commit/983a741b5d3b4a00d032bd50e0c2015bd8f947b4"
        },
        "date": 1788094261314,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance/test_benchmarks.py::test_helpfulness_reward_throughput",
            "value": 6003.840261683485,
            "unit": "iter/sec",
            "range": "stddev: 0.000015741276372048338",
            "extra": "mean: 166.56006096331396 usec\nrounds: 2034"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_safety_reward_throughput",
            "value": 6464.606419521869,
            "unit": "iter/sec",
            "range": "stddev: 0.00001461370415686679",
            "extra": "mean: 154.68845821459328 usec\nrounds: 2106"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_throughput",
            "value": 5002.871198009008,
            "unit": "iter/sec",
            "range": "stddev: 0.000015829144113389737",
            "extra": "mean: 199.88521799201422 usec\nrounds: 3546"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_composite_reward_large_batch",
            "value": 744.258683166006,
            "unit": "iter/sec",
            "range": "stddev: 0.00003045541613062185",
            "extra": "mean: 1.3436188554040036 msec\nrounds: 657"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_trajectory_turn_construction",
            "value": 179.57184386701772,
            "unit": "iter/sec",
            "range": "stddev: 0.0000665982947017985",
            "extra": "mean: 5.568801759036076 msec\nrounds: 166"
          },
          {
            "name": "tests/performance/test_benchmarks.py::test_serving_manifest_build_throughput",
            "value": 2206352.244172218,
            "unit": "iter/sec",
            "range": "stddev: 5.195960500065391e-8",
            "extra": "mean: 453.23678603059204 nsec\nrounds: 104189"
          }
        ]
      }
    ]
  }
}