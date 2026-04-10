#!/usr/bin/env bash
# =============================================================================
# RunPod Setup — Hunyuan3D 2.1 PBR (A100 40GB)
# Repo shape/texture: CeciliaPagano/Hunyuan3D-2.1  (fork personale)
# Repo benchmark:     CeciliaPagano/Hunyuan3D-2     (questo repo)
# GPU consigliata: A100 40GB  (shape ~10GB + texture ~21GB = ~31GB)
#   Alternativa: RTX 3090 24GB con SEQUENTIAL=1 (più lento ma funziona)
# Network Volume: 50 GB su /workspace  (modelli ~28GB + venv ~5GB)
#
# COME USARLO:
#   1. Avvia pod RunPod A100 40GB + Network Volume su /workspace
#   2. Connettiti via SSH e lancia:
#        cd /workspace && git clone https://github.com/CeciliaPagano/Hunyuan3D-2.git
#        bash /workspace/Hunyuan3D-2/run/setup_runpod_21.sh
#   Per GPU <40GB:
#        SEQUENTIAL=1 bash /workspace/Hunyuan3D-2/run/setup_runpod_21.sh
#
# Il venv è su /workspace/venv-21 — persiste tra i restart del pod.
# Per rilanciare senza reinstallare:
#   source /workspace/venv-21/bin/activate
#   cd /workspace/Hunyuan3D-2.1
#   python run/benchmark_runpod.py --input_dir benchmark/inputs --seed 1234 [--sequential]
# =============================================================================
set -euo pipefail

export HF_HOME="/workspace/models"
export HUGGINGFACE_HUB_CACHE="/workspace/models/hub"
export U2NET_HOME="/workspace/models/u2net"

FORK_DIR="/workspace/Hunyuan3D-2"       # fork con benchmark scripts
REPO_DIR="/workspace/Hunyuan3D-2.1"     # fork 2.1 (hy3dgen PBR)
VENV_DIR="/workspace/venv-21"           # venv persistente sul volume
BENCH_INPUTS="$REPO_DIR/benchmark/inputs"
BENCH_OUTPUTS="$REPO_DIR/benchmark/outputs"
SEQUENTIAL="${SEQUENTIAL:-0}"

echo "============================================================"
echo "  Hunyuan3D BENCHMARK — Variante: 2.1 PBR"
echo "  $(date)  |  Sequential: $SEQUENTIAL"
echo "============================================================"

# ── 0. Check GPU e spazio ────────────────────────────────────────────────────
echo ""
echo "[0/6] Stato sistema:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
df -h /workspace | grep -v Filesystem
if [ "$VRAM_MB" -lt 35000 ] && [ "$SEQUENTIAL" -eq 0 ]; then
    echo ""
    echo "  ⚠️  VRAM < 35GB. Riavvia con: SEQUENTIAL=1 bash $0"
fi
echo ""

# ── 1. Fork (benchmark scripts) ──────────────────────────────────────────────
echo "[1/6] Fork con benchmark scripts..."
if [ ! -d "$FORK_DIR" ]; then
    git clone https://github.com/CeciliaPagano/Hunyuan3D-2.git "$FORK_DIR"
else
    git -C "$FORK_DIR" fetch origin
    git -C "$FORK_DIR" reset --hard origin/master
fi

# ── 2. Fork 2.1 (hy3dgen con pipeline PBR) ───────────────────────────────────
echo ""
echo "[2/6] Fork Hunyuan3D-2.1 (pipeline PBR)..."
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$REPO_DIR"
else
    echo "  Repo 2.1 già presente, aggiorno..."
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" reset --hard origin/main || git -C "$REPO_DIR" reset --hard origin/master
fi
cd "$REPO_DIR"

# ── 3. Venv persistente sul volume + dipendenze ───────────────────────────────
echo ""
echo "[3/6] Ambiente Python (venv su volume)..."

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "  venv esistente trovato: $VENV_DIR — riuso."
else
    echo "  Creo nuovo venv in $VENV_DIR (--system-site-packages)..."
    # --system-site-packages: eredita torch/torchvision/transformers dal sistema RunPod
    # così installiamo solo il delta (~2 min invece di 40 min)
    python3 -m venv "$VENV_DIR" --system-site-packages
fi

source "$VENV_DIR/bin/activate"
echo "  Python: $(python --version)"
pip install --upgrade pip setuptools wheel -q

echo "  Installazione dipendenze (solo delta rispetto al sistema)..."
pip install -q --prefer-binary \
    "numpy>=1.26,<2.0" "pymeshlab>=2023.12" "open3d>=0.19.0" "bpy>=4.2" \
    einops omegaconf trimesh "rembg[gpu]" huggingface_hub accelerate safetensors \
    diffusers invisible-watermark PyYAML imageio scipy tqdm

# Estensioni custom del repo 2.1
[ -d "hy3dpaint/custom_rasterizer" ]        && pip install -q --prefer-binary -e hy3dpaint/custom_rasterizer
[ -d "hy3dgen/texgen/custom_rasterizer" ]   && pip install -q --prefer-binary -e hy3dgen/texgen/custom_rasterizer
# hy3dshape e hy3dpaint come pacchetti editabili (necessario per gli import)
[ -d "hy3dshape" ] && pip install -q --no-deps -e hy3dshape
[ -d "hy3dpaint" ] && pip install -q --no-deps -e hy3dpaint
# Anche hy3dgen dal fork Hunyuan3D-2 (per benchmark 2.0 full con lo stesso venv)
[ -d "$FORK_DIR" ] && pip install -q --no-deps -e "$FORK_DIR"

if [ -f "compile_mesh_painter.sh" ]; then
    echo "  Compilazione mesh painter renderer..."
    bash compile_mesh_painter.sh || echo "  WARNING: compile_mesh_painter.sh fallito (continuo comunque)"
fi

# ── 3b. RealESRGAN checkpoint (richiesto da hy3dpaint texture pipeline) ──────
REALESRGAN_CKPT="$REPO_DIR/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
if [ ! -f "$REALESRGAN_CKPT" ]; then
    echo ""
    echo "[3b] Download RealESRGAN_x4plus.pth..."
    mkdir -p "$(dirname "$REALESRGAN_CKPT")"
    wget -q --show-progress \
        -O "$REALESRGAN_CKPT" \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
        || curl -L -o "$REALESRGAN_CKPT" \
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    echo "  RealESRGAN OK"
else
    echo "[3b] RealESRGAN checkpoint già presente. Skip."
fi

# ── 4. Download modelli 2.1 sul volume ───────────────────────────────────────
echo ""
echo "[4/6] Modelli su Network Volume..."
if [ -d "$HF_HOME/hub/models--tencent--Hunyuan3D-2.1" ]; then
    echo "  Modelli 2.1 già presenti sul volume. Skip download."
    du -sh "$HF_HOME/hub/models--tencent--Hunyuan3D-2.1" 2>/dev/null || true
else
    echo "  Download modelli 2.1 (--clean rimuove varianti precedenti)..."
    bash "$FORK_DIR/run/download_models_to_volume.sh" --v21 --clean
fi

# ── 5. Struttura benchmark + verifica input ───────────────────────────────────
echo ""
echo "[5/6] Setup benchmark..."
mkdir -p "$BENCH_INPUTS" "$BENCH_OUTPUTS/2.1"

# Copia il benchmark script dal fork nel repo 2.1
mkdir -p run
cp "$FORK_DIR/run/benchmark_runpod_21.py" run/benchmark_runpod.py

N_IMGS=$(find "$BENCH_INPUTS" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" 2>/dev/null | wc -l)
if [ "$N_IMGS" -eq 0 ]; then
    echo ""
    echo "  ATTENZIONE: nessuna immagine in $BENCH_INPUTS"
    echo "  Carica le 6 immagini dalla tua macchina locale:"
    echo ""
    echo "    rsync -avz -e 'ssh -p <PORT>' benchmark/inputs/ root@<HOST>:$BENCH_INPUTS/"
    echo ""
    echo "  Poi rilanciare con:"
    echo "    source $VENV_DIR/bin/activate"
    SEQ_FLAG=""; [ "$SEQUENTIAL" -eq 1 ] && SEQ_FLAG=" --sequential"
    echo "    cd $REPO_DIR && python run/benchmark_runpod.py --input_dir benchmark/inputs --seed 1234$SEQ_FLAG"
    exit 0
fi
echo "  Trovate $N_IMGS immagini."

# ── 6. Esegui benchmark ──────────────────────────────────────────────────────
echo ""
echo "[6/6] Avvio benchmark 2.1 PBR..."
SEQ_FLAG=""; [ "$SEQUENTIAL" -eq 1 ] && SEQ_FLAG="--sequential"
source "$VENV_DIR/bin/activate"
python run/benchmark_runpod.py \
    --input_dir benchmark/inputs \
    --seed 1234 \
    $SEQ_FLAG

echo ""
echo "============================================================"
echo "  COMPLETATO — Variante 2.1 PBR"
echo "  Risultati in: $BENCH_OUTPUTS/2.1/"
echo ""
echo "  Scarica i risultati:"
echo "  rsync -avz -e 'ssh -p <PORT>' root@<HOST>:$BENCH_OUTPUTS/2.1/ benchmark/outputs/2.1/"
echo "============================================================"
