# Local SFT source and row identity

Packaged local SFT previously hashed the dataset path after training, allowing a
later edit to mislabel an adapter or deletion to erase its source hash. It now
captures source bytes and fingerprints parsed rows before model loading, then
carries that identity through adapter publication. Source path/hash and nonblank
source-line count remain distinct from parsed input count, parsed-content hash
and the effective row count after truncation filtering.

`load_sft_dataset` returns rows and their source/content identity. Studio uses
strict text normalization and checks the captured source hash against its
reviewed plan before importing model/trainer dependencies. Industry execution
checks the capture against its prepared manifest before GPU detection or output
setup. Packaged workers pass the identity from their single captured read. Direct
`run_sft` calls without an identity check the supplied rows against a captured
dataset file before model loading; callers without a source file remain
explicitly unbound. The trainer privately copies rows and rejects changes to
captured input, including an integer-to-boolean tool argument change.

The new tests include five real tiny CPU LoRA fits using randomly initialized
local GPT-2 fixtures and locally built tokenizers, with no checkpoint downloads
or provider calls. Four fits verify source replacement/deletion with both an
explicit captured identity and the direct-file API; the fifth verifies unbound
in-memory lineage. Additional checks reject input changes before ML loading,
recheck reviewed Studio and industry data, and verify identity handoff. The
regression suite also covers seeded training and exact checkpoint recovery.
Final counts and source hashes are retained in `checks.json` and `tests.xml`.

The full-corpus review verified all 1,536 retained industry training rows in both
general and Studio modes. The modes share the same source hash while normalized
Studio row hashes agree with its existing text snapshot. Twelve held-out
corpus/mode combinations were rejected. This corpus review did not load a model
or train. Previous evidence bundles remain unchanged and were verified again.

These are local input-consistency records, not authenticated provider receipts
or independent data review. Caller-constructed identities remain declarations.
In-memory calls cannot establish file provenance; effective input counts do not
prove a quality gain. The tiny CPU test fits are not training runs of the target
industry or River models. No paid job, target-model update, production-quality
certification or A+ result is claimed. Other provider staging paths and native
RL retain their separate protocols.
