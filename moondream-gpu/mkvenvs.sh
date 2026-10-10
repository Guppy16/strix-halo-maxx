set -e
UV=~/.local/bin/uv
for spec in "714 https://download.pytorch.org/whl/rocm7.14 torch==2.14.1+rocm7.14" "100 https://download.pytorch.org/whl/nightly/rocm10.0 torch==2.15.0.dev20261001+rocm10.0"; do
  set -- $spec; tag=$1; idx=$2; tv=$3
  echo "=== venv-rocm$tag ($tv) ==="
  $UV venv --python 3.14 -q venv-rocm$tag
  $UV pip install --python venv-rocm$tag/bin/python --index-url $idx --index-strategy unsafe-best-match --extra-index-url https://pypi.org/simple "$tv" "triton-rocm" 2>&1 | tail -3
  $UV pip install --python venv-rocm$tag/bin/python "transformers<5" einops accelerate pillow 2>&1 | tail -1
  venv-rocm$tag/bin/python -c "import torch; print(\"torch\", torch.__version__, \"hip\", torch.version.hip)"
done
echo DONE
