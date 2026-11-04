#!/usr/bin/env bash
# Install tmu (CAIR's standard Tsetlin Machine library) for the baseline in
# baselines.py. Run from the repository root with your venv active:
#
#     bash experiments/relaxed_tm/classification/install_tmu.sh
#
# Why not a line in requirements.txt:
# - The PyPI release (0.7.1) and the latest tag (v0.8.3) fail under NumPy 2,
#   so a later commit of the main branch is pinned below.
# - tmu's build passes the Intel-only compiler flag -mrdrnd, which makes the
#   build fail on ARM (Apple Silicon, Linux ARM). Its C code does not use
#   that instruction, so the flag is removed before building.
# - tmu's dependencies (pandas) would otherwise upgrade NumPy. The installed
#   NumPy version is kept, because torch may only work with that version
#   (e.g. torch 2.2.2 on Intel macOS needs NumPy < 2).
set -euo pipefail

TMU_REPO="https://github.com/cair/tmu.git"
TMU_COMMIT="acff83ce6d72000a33026ff3b6a7eaec8926c1bf"
PYTHON="${PYTHON:-python}"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

git clone --quiet "$TMU_REPO" "$workdir/tmu"
git -C "$workdir/tmu" checkout --quiet "$TMU_COMMIT"
# Remove -mrdrnd (portable in-place edit; BSD and GNU sed differ)
"$PYTHON" - "$workdir/tmu/pyproject.toml" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
open(path, "w").write(text.replace('"-O3", "-mrdrnd"', '"-O3"'))
PY
numpy_version="$("$PYTHON" -c 'import numpy; print(numpy.__version__)')"
echo "Keeping NumPy $numpy_version"
"$PYTHON" -m pip install "$workdir/tmu" "numpy==$numpy_version"
"$PYTHON" - <<'PY'
import logging
logging.disable(logging.WARNING)  # tmu warns about CUDA on import
from tmu.models.classification.vanilla_classifier import TMClassifier  # noqa: F401
print("tmu installed and importable")
PY
