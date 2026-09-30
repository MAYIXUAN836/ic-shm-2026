#!/usr/bin/env bash
# Environment, public base-model download, final-adapter setup, and inference.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
VENV="$ROOT/.venv"
PYTHON="${PYTHON:-python3.12}"
ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

check_models() {
  "$PYTHON" - "$ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
config = json.loads((root / 'config/final_config.json').read_text())
for name, relative in config['base_models'].items():
    model = root / relative
    for filename in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'preprocessor_config.json'):
        if not (model / filename).is_file():
            raise SystemExit(f'Missing {name} model file: {model / filename}')
    index = model / 'model.safetensors.index.json'
    if not index.is_file():
        raise SystemExit(f'Missing model index: {index}')
    for shard in set(json.loads(index.read_text())['weight_map'].values()):
        if not (model / shard).is_file() or (model / shard).stat().st_size == 0:
            raise SystemExit(f'Missing or empty model shard: {model / shard}')
checksums = json.loads((root / 'config/adapter_checksums.json').read_text())
for relative, expected in checksums.items():
    path = root / relative
    if not path.is_file():
        raise SystemExit(f'Missing final adapter file: {path}. See README.md, Model weights.')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit(f'Final adapter checksum mismatch: {path}')
print(f'Model files present; {len(checksums)} final adapter files match SHA-256. No model was loaded.')
PY
}

case "$ACTION" in
  setup)
    if [[ "$(uname -s)" != Linux ]]; then
      echo 'Inference setup requires Linux and an NVIDIA CUDA GPU. See README.md.' >&2
      exit 1
    fi
    "$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
    "$PYTHON" -m venv "$VENV"
    PYTHON="$VENV/bin/python"
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
    "$PYTHON" -m pip install -r requirements.txt -r requirements-evaluation.txt
    if [[ -n "${MODELS_SOURCE:-}" ]]; then
      # MODELS_SOURCE is an existing models/ directory with base/ and adapters/.
      if [[ -e "$ROOT/models" || -L "$ROOT/models" ]]; then
        echo 'models/ already exists; refusing to replace it. Unset MODELS_SOURCE to use it.' >&2
        exit 1
      fi
      ln -s "$(cd "$MODELS_SOURCE" && pwd)" "$ROOT/models"
    else
      mkdir -p models/base models/adapters
      "$VENV/bin/hf" download Qwen/Qwen3.5-9B \
        --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
        --local-dir models/base/Qwen3.5-9B
      "$VENV/bin/hf" download Qwen/Qwen3-VL-8B-Instruct \
        --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
        --local-dir models/base/Qwen3_VL_8B_Instruct
      if [[ -n "${ADAPTERS_SOURCE:-}" ]]; then
        # Copy only inference artifacts; never bring historical checkpoints along.
        "$PYTHON" - "$ADAPTERS_SOURCE" "$ROOT" <<'PY'
import json, shutil, sys
from pathlib import Path
source, root = map(Path, sys.argv[1:])
for relative in json.loads((root / 'config/adapter_checksums.json').read_text()):
    src = source / Path(relative).relative_to('models/adapters')
    dst = root / relative
    if not src.is_file():
        raise SystemExit(f'Missing adapter source file: {src}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
PY
      fi
    fi
    check_models
    echo 'Setup complete. Run: bash reproduce.sh run --input-dir /path/to/images --output-dir outputs/run1 --gpu 0'
    ;;
  check)
    if [[ -x "$VENV/bin/python" ]]; then PYTHON="$VENV/bin/python"; fi
    check_models
    ;;
  run)
    if [[ ! -x "$VENV/bin/python" ]]; then
      echo 'Run bash reproduce.sh setup first, or use python run.py in your existing environment.' >&2
      exit 1
    fi
    PYTHON="$VENV/bin/python"
    check_models
    exec "$PYTHON" run.py "$@"
    ;;
  help|-h|--help)
    cat <<'HELP'
Usage:
  bash reproduce.sh setup        Install Python dependencies and download pinned base models.
  bash reproduce.sh check        Check model files and final adapter hashes; no model loading.
  bash reproduce.sh run --input-dir IMAGES --output-dir EMPTY_DIR [--gpu 0]

Setup environment variables:
  PYTHON=python3.12               Python 3.12 executable (must already be installed).
  ADAPTERS_SOURCE=/path/adapters  Copy the four final adapters from the supplied model bundle.
  MODELS_SOURCE=/path/models     Link an existing complete models/ bundle; skip model downloads.

The private final adapters are supplied separately. No training or inference is
started by setup/check. See README.md for the precise directory layout.
HELP
    ;;
  *) echo "Unknown action: $ACTION (use --help)" >&2; exit 2 ;;
esac
