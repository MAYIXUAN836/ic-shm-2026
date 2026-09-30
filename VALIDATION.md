# Code release validation — 30 September 2026

The following checks passed locally without weights or model loading:

- Python syntax compilation for the entry point, pipeline, evaluation utilities, and tests.
- Bash syntax validation and help output for `reproduce.sh`.
- `python3 run.py --help`.
- Three CPU-only tests: recognition JSON normalization/rejection; recursive image inventory and empty-input rejection; route selection, normal bypass, mixed A/B merge, neutral-extent replacement, and rejection of a missing required expert result.
- Final adapter SHA-256 values were computed directly from the server's final reproduction bundle. The manifest covers four adapters and the three descriptor processor/tokenizer sets. Weights were not copied into Git.
- The two base models' required configuration/tokenizer files and index/shard naming were inspected in the server's existing final bundle.

The inference code originates from the server's `zlc/ICSHM_upload_2026/repro_package` final snapshot. The evaluation utilities originate from `zlc/code/evaluation`. Two historical absolute model-path defaults were replaced by package-relative defaults. Prompts, category parsing, routing, generation, and merge behavior are preserved.

Not performed in this release: dependency installation into a fresh CUDA environment, fresh base-model downloads, GPU model loading, new inference, retraining, evaluation against official labels, or independent Drive download verification. The CPU tests establish interface behavior, not model accuracy or successful GPU reproduction.
