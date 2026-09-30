# IC-SHM 2026 — Damage Recognition and Description

Final inference code for **Project 3: Apparent Damage Diagnosis of Structures Based on Image-Text Multi-modal Data**, the 4th International Competition for Structural Health Monitoring (IC-SHM, 2026).

Given a directory of images, the system recognizes visible structural-damage categories and generates a category-conditioned English description for each image. It exports the competition JSON format, together with JSONL/CSV copies and intermediate audit records.

**For judges:** start with [Quick start](#quick-start), obtain the separate final-adapter bundle, then run `reproduce.sh`. This is a **code-only inference release** of the frozen final system. It does not retrain the models. Model weights, organizer data, and saved competition predictions are delivered separately and are not stored in Git.

## Final system

| Component | Base model | Frozen adapter / behavior |
| --- | --- | --- |
| Damage recognition (Stage 1) | Qwen3.5-9B | `stage1_UNIFORM_V1/final_adapter` |
| Descriptor A (Expert A) | Qwen3-VL-8B-Instruct | `expert_a_v010_validated` |
| Descriptor B (Expert B), B subset exactly `crack` or exactly `exposed_rebar` | Qwen3-VL-8B-Instruct | `expert_b_v07_validated`, legacy category interface |
| Descriptor B, all other nonempty B subsets | Qwen3-VL-8B-Instruct | `expert_b_v08_validated`, canonical category interface |

The adapters are loaded separately on their respective base models; there is no weight fusion. The code preserves the final V0.10 routing/merge behavior with the UNIFORM_V1 recognition adapter. Earlier SAM experiments are not dependencies of this system.

The nine ordered output categories are:

```text
crack, void, spalling, corrosion, exposed_rebar,
honeycomb, looseness, pothole, efflorescence
```

Descriptor A receives `spalling`, `void`, `pothole`, `honeycomb`, and `looseness`. Descriptor B receives `crack`, `corrosion`, `exposed_rebar`, and `efflorescence`. An image may reach both descriptors. B's adapter is selected using **only its assigned subset**, not the full image-level category list. For example, `[spalling, crack]` uses A and B-v07; `[crack, corrosion]` uses B-v08.

The final merger keeps Stage 1's category list, joins A then B descriptions, removes repeated sentences, and applies the frozen neutral-extent substitutions for two concrete-spalling phrases. If Stage 1 returns an empty list, both descriptors are bypassed and the output description is `No visible structural damage is observed in the image.`

## Requirements

- Linux x86-64, Python **3.12** with `venv`, and an NVIDIA GPU/driver compatible with CUDA 12.8 PyTorch.
- The recorded server environment used Python 3.12.13, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, Transformers 5.12.1, PEFT 0.19.1, and an RTX PRO 6000 Blackwell with approximately 96 GB VRAM.
- The models run sequentially in BF16. Use a GPU with adequate free memory for a 9B model plus inference activations. A minimum VRAM threshold has **not** been benchmarked; 10 GB free is insufficient for the unquantized model weights alone.
- Allow roughly 50 GB of disk space for both downloaded base models, the adapters, environment, and outputs; more may be needed for download caches and datasets.
- Internet access is needed for environment/base-model setup. Inference uses local model files.

The runtime versions are recorded in [requirements.txt](requirements.txt). Evaluation uses [requirements-evaluation.txt](requirements-evaluation.txt). The script installs the CUDA 12.8 PyTorch wheels explicitly.

## Quick start

```bash
git clone https://github.com/MAYIXUAN836/ic-shm-2026-damage-recognition-description.git
cd ic-shm-2026-damage-recognition-description

# Point to models/adapters/ from the separately supplied final bundle.
ADAPTERS_SOURCE=/absolute/path/to/repro_package/models/adapters \
  bash reproduce.sh setup

# Validate files and adapter SHA-256 hashes without loading any model.
bash reproduce.sh check

# Run recognition + description. Choose a new or empty output directory.
bash reproduce.sh run \
  --input-dir /absolute/path/to/images \
  --output-dir outputs/submission \
  --gpu 0
```

This repository is private. Judges must be granted GitHub access, or receive the code ZIP directly. Extracting the ZIP gives the same layout and does not require Git.

`setup` installs dependencies, downloads the pinned public base models, copies the final adapter inference artifacts, and checks them. **It never starts training or inference.** Only the explicit `run` action starts inference. If Python 3.12 has another executable name, set `PYTHON=/path/to/python3.12`.

### Reuse an existing complete model bundle

On a machine that already has the final `models/` directory, skip base-model downloads:

```bash
MODELS_SOURCE=/absolute/path/to/final/models bash reproduce.sh setup
```

This creates a local `models` symlink. It refuses to replace an existing `models/` path. Alternatively, place the files manually in the layout below and run setup without either source variable; existing Hugging Face download files are reused when possible.

If using an already configured environment, the direct entry point is:

```bash
python run.py --input-dir /path/to/images --output-dir /path/to/new-output --gpu 0
```

## Model weights

Two public base models are downloaded at the exact revisions recorded in the original server's Hugging Face download metadata:

| Model | Revision | Local directory |
| --- | --- | --- |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | `models/base/Qwen3.5-9B` |
| [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | `models/base/Qwen3_VL_8B_Instruct` |

The four trained adapters are **not** the public base models. Obtain the final bundle from the submitting team through the competition delivery channel. The team's existing [resource folder](https://drive.google.com/drive/folders/16huZE_ZjYZgiDI-DT-lD_Q9SlT_A5sK2) contains the separate delivery. The 30 September server delivery record reports public-reader permission and an owner-performed signed-out download check; this code release has not independently repeated that access test. In its `repro_package/` folder, download all seven `repro_package.tar.part000`–`006` files, then restore them with `cat repro_package.tar.part* | tar -xf -` in an empty directory. Set `ADAPTERS_SOURCE` to the restored `repro_package/models/adapters/`. If the folder is inaccessible, request the adapter bundle from the team; the code archive alone is insufficient for inference.

Required layout:

```text
models/
├── base/
│   ├── Qwen3.5-9B/
│   └── Qwen3_VL_8B_Instruct/
└── adapters/
    ├── stage1_UNIFORM_V1/final_adapter/
    ├── expert_a_v010_validated/
    ├── expert_b_v07_validated/
    └── expert_b_v08_validated/
```

All adapters require `adapter_config.json` and `adapter_model.safetensors`. The three descriptor adapters also require their tokenizer/processor files and chat template. [config/adapter_checksums.json](config/adapter_checksums.json) lists every required adapter file and its SHA-256 digest, measured from the final server bundle. `check` verifies these hashes and the presence of all base-model shards listed in the model index. It does not load weights or verify base-model shard hashes. Historical `checkpoint-*` folders, optimizer states, and training argument binaries are not needed.

## Input and output

`--input-dir` is scanned recursively for `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, and `.tiff` files (case-insensitive). Images are sorted by their relative paths. Each `image_id` is the relative path inside the input directory, including subdirectories. An empty input directory or nonempty output directory is rejected.

The primary submission artifact is `final_predictions.json`, a JSON array. A schema illustration (not a measured prediction):

```json
[
  {
    "image_id": "example.jpg",
    "damage_categories": ["crack"],
    "description": "A thin irregular crack is visible on the concrete surface."
  }
]
```

The output directory contains:

```text
final_predictions.json        # Competition JSON array
final_predictions.jsonl       # One JSON record per line
final_predictions.csv         # Tabular copy; categories separated by semicolons
final_run_decision.json        # Component provenance and validation counts
intermediate/                 # Inventory, raw recognition output, expert diagnostics,
                              # routing inputs and selected/merged predictions
```

The runner checks output IDs/count/order, the three-field schema, valid unique category strings, nonempty descriptions, successful recognition JSON parsing, and the no-damage bypass. Invalid generation is reported as a failure rather than accepted as a successful submission. After a failed run, use a fresh output directory for the full pipeline.

## Fixed inference settings

[config/final_config.json](config/final_config.json) records the final paths, categories, routing policy, and generation settings. Recognition downsizes images to a maximum side of 768 pixels and generates up to 96 new tokens. Descriptors use a batch size of 4, processor bounds of 50,176–100,352 pixels, and up to 180 new tokens. Generation is greedy (`do_sample=False`). The prompts and parsing rules are in `pipeline/` and are part of the frozen system.

The routing/category/normal-description entries in the config document the fixed implementation; editing these metadata entries alone does not redefine the Python routing/merge logic. Keep the code and config together. Hardware and library differences may affect numerical results even with greedy decoding.

## Data and evaluation

Input images and reference annotations are supplied separately by the competition/team. The existing [dataset resource folder](https://drive.google.com/drive/folders/11gWf6Go-Ie_fqKz6AKAvMs6GSMx6oSXJ) contains the team's dataset delivery location; access is not verified by this code release. Its recorded roles include updated organizer images, official unlabeled inputs, selected DACL10K/MCDS data, normal variants, and split manifests. Use the **updated** organizer data and the matching manifests, not an older copy with the same filenames.

[Evaluation instructions](evaluation/README.md) describe the portable scoring utilities:

| Evidence | Meaning and limitation |
| --- | --- |
| 240-image recognition regression | Historical selection-involved check; V3 validation 120 + V3 holdout 120. Not an untouched final test. |
| 120-image integrated description comparison | Internal development comparison linked to the V3 validation images. |
| 116-image controlled description comparison | Reference categories are provided to isolate the descriptor subsystem; V4 holdout selection. |
| 110 official test images | Unlabeled competition inputs. Output completeness is not an accuracy score. |

The description scorer uses **METEOR without synonym matching**, plus token F1, matching the supplied project development utility. Do not relabel this as an independently verified official competition scorer. This repository supplies inference/evaluation code, not a from-scratch training reproduction or new performance claims.

## Code map

| File | Purpose |
| --- | --- |
| `reproduce.sh` | Environment setup, pinned downloads, adapter placement/checking, explicit inference launch |
| `run.py` | Complete image-to-answer pipeline and output validation |
| `pipeline/stage1_infer.py` | Recognition prompt, final adapter inference, strict category parsing |
| `pipeline/expert_a_infer.py`, `legacy_runner.py` | Descriptor A's frozen prompt and common expert inference machinery |
| `pipeline/expert_b_infer.py`, `prompt_config.py` | Descriptor B's legacy/canonical interfaces |
| `pipeline/prepare_routes.py`, `combine_b_routes.py` | B route preparation and final adapter-output selection |
| `pipeline/merge.py` | Category-preserving deterministic description merge |
| `evaluation/` | Category and description scoring utilities |
| `config/` | Frozen model configuration and adapter checksums |
| `tests/test_pipeline.py` | CPU-only parsing, routing, merge, and input-validation checks |

The runtime was copied from the final server reproduction package on 30 September 2026. The only portability changes to existing inference code replace two historical absolute default base-model paths with package-relative paths; `run.py` already passes these paths explicitly. No prompts, adapter selection, generation settings, or merge rules were changed.

## Validation of this code release

```bash
python3 -m compileall -q run.py pipeline evaluation tests
bash -n reproduce.sh
python3 -m unittest discover -s tests -v
python3 run.py --help
bash reproduce.sh --help
```

These checks require no weights or GPU. This release was prepared without loading models, running new inference, or retraining, as requested by the author. Previous server inference records do not constitute a fresh run of this code archive. See [VALIDATION.md](VALIDATION.md) for the checks actually performed.

## Troubleshooting

- **Missing adapter/checksum mismatch:** use the four final adapters with their original processor files. A base-model download cannot replace a trained adapter; a historical adapter may have the same filename but different weights.
- **CUDA out of memory:** stop this run and choose a GPU with sufficient free memory. The release does not change precision, quantize weights, or overwrite other GPU jobs.
- **Download/access errors:** confirm access to Hugging Face and the team's private resources. The repository contains no account tokens or proxy configuration.
- **Output directory not empty:** choose a new output directory; the full pipeline does not overwrite an existing run.
- **Evaluation reference mismatch:** use the matching split/reference files and image IDs. Official test labels are not bundled.

## Third-party resources

Base models remain subject to the licenses and terms in their original model repositories. Data remain subject to organizer/source permissions. Relevant source resources include [DACL10K](https://doi.org/10.1109/WACV57701.2024.00843), [MCDS](https://doi.org/10.5281/zenodo.2601506), and [LoRA](https://openreview.net/forum?id=nZeVKeeFYf9). No new open-source license is asserted for this private competition submission.
