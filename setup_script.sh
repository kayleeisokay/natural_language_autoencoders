#!/usr/bin/env bash
set -e  # exit immediately on any failure

# --- Check HF_TOKEN early, before doing any real work ---
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "Get a token from https://huggingface.co/settings/tokens"
    echo "Then run: export HF_TOKEN=hf_..."
    exit 1
fi

# Install uv
wget -qO- https://astral.sh/uv/install.sh | sh

# Make sure uv is on PATH (in case .zshrc wasn't updated by installer)
export PATH="$HOME/.local/bin:$PATH"
source ~/.zshrc

uv --version

# --- Python + base deps ---
# NOTE: this project is pinned to Python 3.10, not 3.11 — sgl_kernel and the
# notebook's hardcoded UV_PYTHON_INCLUDE path both assume 3.10.
uv python install 3.10
rm -rf .venv
uv sync --python 3.10
uv add ipykernel matplotlib accelerate
uv add "transformers==4.57.1"   # sglang requires this exact version
uv run python -m ipykernel install --user --name nla-venv --display-name "NLA (3.10)"

# Verify torch version and CUDA build
TORCH_VERSION=$(uv run python -c "import torch; print(torch.__version__)")
echo "torch version: $TORCH_VERSION"
if [[ "$TORCH_VERSION" != *"cu"* ]]; then
    echo "WARNING: torch does not appear to be a CUDA build (got: $TORCH_VERSION)"
    exit 1
fi

# Verify GPU visibility
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

# Confirm venv Python version
VENV_PY_VERSION=$(.venv/bin/python3 --version)
echo "venv python version: $VENV_PY_VERSION"

# --- libnuma (no-sudo build into ~/.local) ---
if [ ! -f "$HOME/.local/lib/libnuma.so.1" ]; then
    echo "Building libnuma into ~/.local..."
    cd ~
    wget https://github.com/numactl/numactl/releases/download/v2.0.16/numactl-2.0.16.tar.gz
    tar -xzf numactl-2.0.16.tar.gz
    cd numactl-2.0.16
    ./configure --prefix=$HOME/.local
    make
    make install
    cd ~
    rm -rf numactl-2.0.16 numactl-2.0.16.tar.gz
else
    echo "libnuma already present at ~/.local/lib, skipping build."
fi

export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/.local/lib:$LIBRARY_PATH
export CPATH=$HOME/.local/include:$CPATH

# Persist LD_LIBRARY_PATH in .zshrc, only if not already added
if ! grep -q 'HOME/.local/lib:\$LD_LIBRARY_PATH' ~/.zshrc 2>/dev/null; then
    echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.zshrc
fi
source ~/.zshrc

# --- Python.h headers for the correct 3.10 uv-managed install (needed for triton/sgl_kernel builds) ---
cd ~/natural_language_autoencoders
UV_PY310_DIR=$(find ~/.local/share/uv/python -maxdepth 1 -name "cpython-3.10*" | head -1)
if [ -n "$UV_PY310_DIR" ]; then
    export CPATH="${UV_PY310_DIR}/include/python3.10:${CPATH}"
    echo "Using Python.h from: ${UV_PY310_DIR}/include/python3.10"
else
    echo "WARNING: could not find a uv-managed cpython-3.10 install with headers."
fi
rm -rf ~/.triton/cache

# --- sglang: intentionally NOT managed via uv add/pyproject.toml (see comment in pyproject.toml).
# Install it LAST, after uv sync, via uv pip install directly into the venv.
# Do not run `uv sync` or bare `uv run` again after this without re-checking sglang survives it —
# use /home/kaylee/natural_language_autoencoders/.venv/bin/python directly where possible.
uv pip install "sglang[all]>=0.5.6" --python /home/kaylee/natural_language_autoencoders/.venv/bin/python

# Final verification
uv run python -c "import transformers, sglang, torch; print('transformers:', transformers.__version__); print('sglang: ok'); print('torch:', torch.__version__)"


git config --global user.name "Kaylee Vo"
git config --global user.email "kayleeyvo@gmail.com"

echo "Setup complete."