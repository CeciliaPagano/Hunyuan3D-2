#!/usr/bin/env bash
# =============================================================================
# RunPod Setup — Hunyuan3D 2.0 FULL (RTX 3090 24GB)
# Repo: Tencent-Hunyuan/Hunyuan3D-2  (stesso repo della baseline mini)
# GPU consigliata: RTX 3090 24GB  (shape ~6GB + texture ~16GB = ~22GB)
# Network Volume: 40 GB montato su /workspace
#
# COME USARLO:
#   1. Avvia pod RunPod con RTX 3090 24GB + Network Volume 40GB su /workspace
#   2. Connettiti via SSH e lancia:
#        cd /workspace && git clone https://github.com/CeciliaPagano/Hunyuan3D-2.git
#        bash /workspace/Hunyuan3D-2/run/setup_runpod_20.sh
# =============================================================================
set -euo pipefail

REPO_DIR="/workspace/Hunyuan3D-2"
BENCH_INPUTS="$REPO_DIR/benchmark/inputs"
BENCH_OUTPUTS="$REPO_DIR/benchmark/outputs"
VARIANT="2.0"

export HF_HOME="/workspace/models"
export HUGGINGFACE_HUB_CACHE="/workspace/models/hub"
export U2NET_HOME="/workspace/models/u2net"

echo "============================================================"
echo "  Hunyuan3D BENCHMARK — Variante: $VARIANT"
echo "  $(date)"
echo "============================================================"

# ── 0. Check GPU e spazio ────────────────────────────────────────────────────
echo ""
echo "[0/5] Stato sistema:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace | grep -v Filesystem
echo ""

# ── 1. Clone / aggiorna il tuo fork ─────────────────────────────────────────
echo "[1/5] Repository..."
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/CeciliaPagano/Hunyuan3D-2.git "$REPO_DIR"
else
    echo "  Repo già presente, aggiorno..."
    git -C "$REPO_DIR" pull
fi
cd "$REPO_DIR"

# ── 2. Installa dipendenze ───────────────────────────────────────────────────
echo ""
echo "[2/5] Installazione dipendenze Python..."
pip install -q -r requirements.txt
if [ -d "hy3dgen/texgen/custom_rasterizer" ]; then
    echo "  Build custom_rasterizer..."
    pip install -q -e hy3dgen/texgen/custom_rasterizer
fi
pip install -q rembg[gpu]

# ── 3. Download modelli (cancella varianti precedenti, scarica 2.0) ──────────
echo ""
echo "[3/5] Modelli su Network Volume..."

# Se ci sono già i modelli 2.0 sul volume, salta il download
if [ -d "$HF_HOME/hub/models--tencent--Hunyuan3D-2" ]; then
    echo "  Modelli 2.0 già presenti sul volume. Skip download."
    du -sh "$HF_HOME/hub/models--tencent--Hunyuan3D-2" 2>/dev/null || true
else
    echo "  Modelli 2.0 non trovati. Avvio download (--clean rimuove varianti precedenti)..."
    bash run/download_models_to_volume.sh --v20 --clean
fi

# ── 4. Struttura benchmark + input ──────────────────────────────────────────
echo ""
echo "[4/5] Setup benchmark..."
mkdir -p "$BENCH_INPUTS" "$BENCH_OUTPUTS/$VARIANT"

N_IMGS=$(find "$BENCH_INPUTS" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" 2>/dev/null | wc -l)
if [ "$N_IMGS" -eq 0 ]; then
    echo ""
    echo "  ⚠️  Nessuna immagine in $BENCH_INPUTS"
    echo "  Carica le immagini dalla tua macchina locale:"
    echo ""
    echo "    rsync -avz -e 'ssh -p <PORT>' benchmark/inputs/ root@<HOST>:$BENCH_INPUTS/"
    echo ""
    echo "  Poi esegui:"
    echo "    cd $REPO_DIR && python run/benchmark_runpod.py --variant $VARIANT --input_dir benchmark/inputs --seed 1234"
    exit 0
fi
echo "  Trovate $N_IMGS immagini."

# ── 5. Esegui benchmark ──────────────────────────────────────────────────────
echo ""
echo "[5/5] Avvio benchmark $VARIANT..."
echo ""
python run/benchmark_runpod.py \
    --variant "$VARIANT" \
    --input_dir benchmark/inputs \
    --seed 1234

echo ""
echo "============================================================"
echo "  COMPLETATO — Variante $VARIANT"
echo "  Risultati in: $BENCH_OUTPUTS/$VARIANT/"
echo ""
echo "  Scarica i risultati sulla tua macchina:"
echo "  rsync -avz -e 'ssh -p <PORT>' root@<HOST>:$BENCH_OUTPUTS/$VARIANT/ benchmark/outputs/$VARIANT/"
echo "============================================================"
