#!/usr/bin/env bash
# =============================================================================
# RunPod Setup — FLUX.1-dev text-to-image
# Modello: black-forest-labs/FLUX.1-dev (gated — richiede HF_TOKEN + licenza accettata)
# GPU: ≥16GB VRAM consigliata (usa cpu_offload, minimo ~12GB)
# Network Volume: 40GB su /workspace (modello ~24GB + venv ~3GB + output)
#
# COME USARLO:
#   1. Accetta la licenza FLUX.1-dev su: https://huggingface.co/black-forest-labs/FLUX.1-dev
#   2. Avvia pod RunPod (GPU ≥16GB, CUDA 12.x) + Network Volume 40GB su /workspace
#   3. Connettiti via web terminal e lancia:
#        export HF_TOKEN="hf_..."   # il tuo token HuggingFace
#        cd /workspace && git clone -b master https://github.com/CeciliaPagano/Hunyuan3D-2.git
#        bash /workspace/Hunyuan3D-2/run/setup_runpod_flux.sh
#
# Dopo il setup, per generare:
#   source /workspace/venv-flux/bin/activate
#   export HF_HOME="/workspace/models"
#   cd /workspace/Hunyuan3D-2
#   python run/flux_text2image.py --prompt "a silver ring with ruby" --style realistic --seed 1234
# =============================================================================
set -euo pipefail

REPO_DIR="/workspace/Hunyuan3D-2"
VENV_DIR="/workspace/venv-flux"
MODEL_ID="black-forest-labs/FLUX.1-dev"
MODEL_DIR_NAME="models--black-forest-labs--FLUX.1-dev"

export HF_HOME="/workspace/models"
export HUGGINGFACE_HUB_CACHE="/workspace/models/hub"

echo "============================================================"
echo "  FLUX.1-dev Setup"
echo "  $(date)"
echo "============================================================"

# ── Check HF_TOKEN ───────────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    echo ""
    echo "ERRORE: HF_TOKEN non impostato."
    echo "  Accetta la licenza su https://huggingface.co/black-forest-labs/FLUX.1-dev"
    echo "  Poi: export HF_TOKEN=\"hf_...\""
    exit 1
fi

# ── 0. Check GPU e spazio ────────────────────────────────────────────────────
echo ""
echo "[0/5] Stato sistema:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace | grep -v Filesystem
echo ""

# ── 1. Clone / aggiorna repo ─────────────────────────────────────────────────
echo "[1/5] Repository..."
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/CeciliaPagano/Hunyuan3D-2.git "$REPO_DIR"
else
    echo "  Repo già presente, aggiorno..."
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" reset --hard origin/master
fi
cd "$REPO_DIR"
echo "  Commit: $(git -C "$REPO_DIR" rev-parse --short HEAD)"

# ── 2. Venv Python ───────────────────────────────────────────────────────────
echo ""
echo "[2/5] Ambiente Python..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Venv creato in $VENV_DIR"
else
    echo "  Venv già presente, skip."
fi
source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip

# ── 3. Dipendenze ────────────────────────────────────────────────────────────
echo ""
echo "[3/5] Installazione dipendenze..."
echo "  torch cu124..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo "  diffusers, transformers, accelerate..."
pip install -q \
    "diffusers>=0.30.0" \
    transformers \
    accelerate \
    Pillow \
    sentencepiece \
    protobuf \
    huggingface_hub \
    safetensors

echo "  Python: $(python3 --version)"
echo "  Torch:  $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA:   $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# ── 4. Download modello FLUX.1-schnell ───────────────────────────────────────
echo ""
echo "[4/5] Modello FLUX.1-schnell..."
mkdir -p "$HF_HOME/hub"

if [ -d "$HF_HOME/hub/$MODEL_DIR_NAME" ]; then
    echo "  Modello già presente sul volume. Skip download."
    du -sh "$HF_HOME/hub/$MODEL_DIR_NAME" 2>/dev/null || true
else
    echo "  Download $MODEL_ID (~24GB)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '$MODEL_ID',
    cache_dir='$HF_HOME/hub',
)
print('  Download completato.')
"
fi

# ── 5. Test generazione ───────────────────────────────────────────────────────
echo ""
echo "[5/5] Test generazione..."
mkdir -p "$REPO_DIR/run/outputs/flux"

python3 run/flux_text2image.py \
    --prompt "a wooden treasure chest, closed, dusty" \
    --style realistic \
    --seed 1234 \
    --output run/outputs/flux/setup_test.png \
    --show-prompt

echo ""
echo "============================================================"
echo "  SETUP COMPLETATO"
echo ""
echo "  Test output: $REPO_DIR/run/outputs/flux/setup_test.png"
echo ""
echo "  Per generare:"
echo "    source $VENV_DIR/bin/activate"
echo "    export HF_HOME=$HF_HOME"
echo "    cd $REPO_DIR"
echo "    python run/flux_text2image.py --prompt \"descrizione\" --style realistic --seed 1234"
echo ""
echo "  Varianti multipli:"
echo "    python run/flux_text2image.py --prompt \"descrizione\" --seeds 1 2 3 4"
echo "============================================================"
