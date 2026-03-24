#!/usr/bin/env bash
# =============================================================================
# RunPod Setup — Hunyuan3D 2.1 PBR (A100 40GB)
# Repo shape/texture: Tencent-Hunyuan/Hunyuan3D-2.1
# GPU consigliata: A100 40GB  (shape ~10GB + texture ~21GB = ~31GB)
#   Alternativa: RTX 3090 24GB con SEQUENTIAL=1 (più lento ma funziona)
# Network Volume: 40 GB montato su /workspace
#
# COME USARLO:
#   1. Avvia pod RunPod A100 40GB + Network Volume 40GB su /workspace
#   2. Connettiti via SSH e lancia:
#        cd /workspace && git clone https://github.com/CeciliaPagano/Hunyuan3D-2.git
#        bash /workspace/Hunyuan3D-2/run/setup_runpod_21.sh
#   Per GPU <40GB:
#        SEQUENTIAL=1 bash /workspace/Hunyuan3D-2/run/setup_runpod_21.sh
# =============================================================================
set -euo pipefail

export HF_HOME="/workspace/models"
export HUGGINGFACE_HUB_CACHE="/workspace/models/hub"
export U2NET_HOME="/workspace/models/u2net"

FORK_DIR="/workspace/Hunyuan3D-2"       # il tuo fork (benchmark scripts)
REPO_DIR="/workspace/Hunyuan3D-2.1"     # repo ufficiale 2.1 (hy3dgen con PBR)
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
    git -C "$FORK_DIR" pull
fi

# ── 2. Repo ufficiale 2.1 (hy3dgen con pipeline PBR) ────────────────────────
echo ""
echo "[2/6] Repo Hunyuan3D-2.1 (pipeline PBR)..."
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$REPO_DIR"
else
    git -C "$REPO_DIR" pull
fi
cd "$REPO_DIR"

# ── 3. Dipendenze ────────────────────────────────────────────────────────────
echo ""
echo "[3/6] Dipendenze Python..."
[ -f "requirements.txt" ] && pip install -q -r requirements.txt
[ -d "hy3dgen/texgen/custom_rasterizer" ] && pip install -q -e hy3dgen/texgen/custom_rasterizer
pip install -q rembg[gpu]

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
    echo "  ⚠️  Nessuna immagine in $BENCH_INPUTS"
    echo "  Carica le immagini dalla tua macchina locale:"
    echo ""
    echo "    rsync -avz -e 'ssh -p <PORT>' benchmark/inputs/ root@<HOST>:$BENCH_INPUTS/"
    echo ""
    echo "  Poi esegui:"
    SEQ_FLAG=""; [ "$SEQUENTIAL" -eq 1 ] && SEQ_FLAG=" --sequential"
    echo "    cd $REPO_DIR && python run/benchmark_runpod.py --input_dir benchmark/inputs --seed 1234$SEQ_FLAG"
    exit 0
fi
echo "  Trovate $N_IMGS immagini."

# ── 6. Esegui benchmark ──────────────────────────────────────────────────────
echo ""
echo "[6/6] Avvio benchmark 2.1 PBR..."
SEQ_FLAG=""; [ "$SEQUENTIAL" -eq 1 ] && SEQ_FLAG="--sequential"
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
