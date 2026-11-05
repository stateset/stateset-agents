# Final training-seed display acceptance

The broader [seed-feature baseline](../finetune-studio-seeds/README.md) passed
1,839 source tests and 1,728 installed-wheel tests. Final inspection then found
numeric zero rendered as not set in Configure. Only `tui.py` and the seed test
module changed from that baseline; the exact hash delta is retained.

The formatter now distinguishes absent values from numeric zero. Seed zero,
checkpoint interval zero and automatic GPU count zero are visible in the field
row and selected help, in narrow and two-column layouts. Twelve cases cover
72x22, 74x44, 110x32 and 140x40 without mutating settings or creating a workspace.

The final code passed 613 targeted source tests and
613 matching installed-wheel tests, including all
63 seed cases. Counts overlap. Buildr job `d808523574bf` also passed typing for
429 modules, repository lint/hygiene, API/release governance, serial formatting,
workflow lint, actual wheel/source identities and installed CLI acceptance.
The selected source/wheel case identities match without skips or failures.
Source/test Python and packaged assets stayed frozen through verification.

Coverage retains actual tiny CPU SFT seed comparisons, native initialization
controls, provider seed boundary tests, exact recovery, separate training/eval
seed binding, recorded judge replay and real terminal edit/review/save/exit.
Provider/judge boundaries are injected; no live provider, GPU or quality gain
is certified. Earlier development corrections and the missing curation-script
harness repair remain in the broader baseline, with their actual queue handles.

The idle Herdr split was refreshed to the final code, its private draft bytes
and permissions preserved, and the new Training seed setting is visible. No
synthetic live run or paid training was launched. Automatic checkpoint selection
and a reviewed independent quality study remain separate work.
