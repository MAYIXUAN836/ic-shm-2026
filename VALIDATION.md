# Code release validation — 30 September 2026

The following checks passed locally without weights or model loading:

- Python syntax compilation for the entry point, pipeline, evaluation utilities, and tests.
- Bash syntax validation and help output for `reproduce.sh`.
- `python3 run.py --help`.
- Three CPU-only tests: recognition JSON normalization/rejection; recursive image inventory and empty-input rejection; route selection, normal bypass, mixed A/B merge, neutral-extent replacement, and rejection of a missing required expert result.
- Four additional CPU-only download tests cover dataset extraction/path traversal rejection, adapter allowlisting without code replacement, corrupt-cache rejection, and the 32-file manifest inventory. All seven tests pass.
- Final adapter SHA-256 values were computed directly from the server's final reproduction bundle. The manifest covers four adapters and the three descriptor processor/tokenizer sets. Weights were not copied into Git.
- The two base models' required configuration/tokenizer files and index/shard naming were inspected in the server's existing final bundle.

The inference code originates from the server's `zlc/ICSHM_upload_2026/repro_package` final snapshot. The evaluation utilities originate from `zlc/code/evaluation`. Two historical absolute model-path defaults were replaced by package-relative defaults. Prompts, category parsing, routing, generation, and merge behavior are preserved.

Not performed in this release: dependency installation into a fresh CUDA environment, fresh base-model downloads, GPU model loading, new inference, retraining, evaluation against official labels, or full Drive archive re-download verification. The CPU tests establish interface behavior, not model accuracy or successful GPU reproduction.

## New Drive delivery

All seven final reproduction-bundle parts and all 25 prepared dataset ZIPs were copied to the author's new Drive folder. The original folders remain intact. Copies are owned by the destination account and use the original filenames. Public, cookie-free listing returned exactly the expected 32 files after copying. The 635,040-byte `dataset_manifests.zip` was downloaded anonymously from its new file ID and matched the authoritative server SHA-256.

The download manifest records all 32 new file IDs, byte sizes, and SHA-256 values calculated from the server originals (1,786,093,167 bytes total). This verifies the source inventory and the sample download; it is not a claim that all 1.79 GB were re-downloaded. The script will verify every archive and every restored adapter file on the recipient's machine. No model loading or inference was performed.
