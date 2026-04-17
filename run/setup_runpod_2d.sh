#!/usr/bin/env bash
# =============================================================================
# RunPod Setup — Generazione 2D con HunyuanDiT v1.1 Distilled
# GPU: qualsiasi ≥8GB VRAM (usa cpu_offload)
# Network Volume: 40GB su /workspace
#
# COME USARLO:
#   1. Avvia pod RunPod (GPU ≥8GB, CUDA 12.x) + Network Volume 40GB su /workspace
#   2. Connettiti via web terminal e lancia:
#        cd /workspace && git clone https://github.com/CeciliaPagano/Hunyuan3D-2.git
#        bash /workspace/Hunyuan3D-2/run/setup_runpod_2d.sh
#
# Dopo il setup, per generare:
#   source /workspace/venv-2d/bin/activate
#   export HF_HOME="/workspace/models"
#   cd /workspace/Hunyuan3D-2
#   python run/text2image_asset.py --prompt "..." --category prop --style realistic --seed 1234 --output run/outputs/result.png
# =============================================================================
set -euo pipefail

REPO_DIR="/workspace/Hunyuan3D-2"
VENV_DIR="/workspace/venv-2d"
MODEL_ID="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"

export HF_HOME="/workspace/models"
export HUGGINGFACE_HUB_CACHE="/workspace/models/hub"

echo "============================================================"
echo "  HunyuanDiT 2D Setup"
echo "  $(date)"
echo "============================================================"

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

# Torch cu124 PER PRIMO (evita mismatch cu130 del sistema)
echo "  torch cu124..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Dipendenze pipeline 2D
echo "  diffusers, transformers, accelerate..."
pip install -q \
    diffusers \
    transformers \
    accelerate \
    Pillow \
    pyyaml \
    huggingface_hub \
    safetensors \
    sentencepiece \
    tiktoken \
    protobuf

echo "  Python: $(python3 --version)"
echo "  Torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA disponibile: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# ── 4. Download modello HunyuanDiT ───────────────────────────────────────────
echo ""
echo "[4/5] Modello HunyuanDiT..."
mkdir -p "$HF_HOME/hub"

if [ -d "$HF_HOME/hub/models--Tencent-Hunyuan--HunyuanDiT-v1.1-Diffusers-Distilled" ]; then
    echo "  Modello già presente sul volume. Skip download."
    du -sh "$HF_HOME/hub/models--Tencent-Hunyuan--HunyuanDiT-v1.1-Diffusers-Distilled" 2>/dev/null || true
else
    echo "  Download $MODEL_ID (~14.5GB)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '$MODEL_ID',
    cache_dir='$HF_HOME/hub',
    ignore_patterns=['*.msgpack', '*.h5'],
)
print('  Download completato.')
"
fi

# ── 5. Test generazione ───────────────────────────────────────────────────────
echo ""
echo "[5/5] Test generazione..."
mkdir -p "$REPO_DIR/run/outputs"

python3 run/text2image_asset.py \
    --prompt "a wooden treasure chest, closed, dusty" \
    --category prop \
    --style realistic \
    --seed 1234 \
    --output run/outputs/setup_test.png \
    --steps 20

echo ""
echo "============================================================"
echo "  SETUP COMPLETATO"
echo ""
echo "  Test output: $REPO_DIR/run/outputs/setup_test.png"
echo ""
echo "  Per generare da zero:"
echo "    source $VENV_DIR/bin/activate"
echo "    export HF_HOME=$HF_HOME"
echo "    cd $REPO_DIR"
echo "    python run/text2image_asset.py \\"
echo "        --prompt \"descrizione oggetto\" \\"
echo "        --category prop --style realistic \\"
echo "        --seed 1234 --output run/outputs/mio_asset.png"
echo "============================================================"
