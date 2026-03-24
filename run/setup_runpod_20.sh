#!/usr/bin/env bash
# =============================================================================
# RunPod Setup — Hunyuan3D 2.0 FULL (RTX 3090 24GB)
# Repo: Tencent-Hunyuan/Hunyuan3D-2  (stesso repo della baseline mini)
# GPU consigliata: RTX 3090 24GB  (shape ~6GB + texture ~16GB = ~22GB)
#
# COME USARLO:
#   1. Avvia pod RunPod con RTX 3090 24GB, template PyTorch 2.x/CUDA 12.x
#   2. Copia questo script sul pod:
#        scp -P <PORT> run/setup_runpod_20.sh root@<HOST>:/root/
#   3. Sul pod:
#        bash /root/setup_runpod_20.sh
# =============================================================================
set -euo pipefail

REPO_DIR="/workspace/Hunyuan3D-2"
BENCH_INPUTS="/workspace/Hunyuan3D-2/benchmark/inputs"
BENCH_OUTPUTS="/workspace/Hunyuan3D-2/benchmark/outputs"
VARIANT="2.0"

# Usa i modelli già scaricati sul Network Volume (evita ri-download)
export HF_HOME="/workspace/models"
export HUGGINGFACE_HUB_CACHE="/workspace/models/hub"
export U2NET_HOME="/workspace/models/u2net"

echo "============================================================"
echo "  Hunyuan3D BENCHMARK — Variante: $VARIANT"
echo "  $(date)"
echo "============================================================"

# ── 0. Check GPU ─────────────────────────────────────────────────────────────
echo ""
echo "[0/5] GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── 1. Clone repo ────────────────────────────────────────────────────────────
echo "[1/5] Clone repository..."
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$REPO_DIR"
else
    echo "  Repo già presente, aggiorno..."
    git -C "$REPO_DIR" pull
fi
cd "$REPO_DIR"

# ── 2. Installa dipendenze ───────────────────────────────────────────────────
echo ""
echo "[2/5] Installazione dipendenze Python..."
pip install -q -r requirements.txt

# custom_rasterizer (necessario per la texture)
if [ -d "hy3dgen/texgen/custom_rasterizer" ]; then
    echo "  Build custom_rasterizer..."
    pip install -q -e hy3dgen/texgen/custom_rasterizer
fi

# rembg
pip install -q rembg[gpu]

# ── 3. Struttura benchmark ───────────────────────────────────────────────────
echo ""
echo "[3/5] Setup struttura benchmark..."
mkdir -p "$BENCH_INPUTS"
mkdir -p "$BENCH_OUTPUTS/$VARIANT"

# ── 4. Istruzioni per caricare gli input ────────────────────────────────────
echo ""
echo "============================================================"
echo "  [4/5] AZIONE RICHIESTA: carica le immagini di test"
echo "============================================================"
echo ""
echo "  Da eseguire sulla tua macchina locale (NON sul pod):"
echo ""
echo "  scp -P <PORT> benchmark/inputs/*.png root@<HOST>:$BENCH_INPUTS/"
echo ""
echo "  Oppure con rsync:"
echo "  rsync -avz -e 'ssh -p <PORT>' benchmark/inputs/ root@<HOST>:$BENCH_INPUTS/"
echo ""
echo "  Attendo 30 secondi, poi controllo se ci sono immagini..."
sleep 30

N_IMGS=$(find "$BENCH_INPUTS" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" 2>/dev/null | wc -l)
echo "  Trovate $N_IMGS immagini in $BENCH_INPUTS"
if [ "$N_IMGS" -eq 0 ]; then
    echo ""
    echo "  ATTENZIONE: nessuna immagine trovata."
    echo "  Carica le immagini e poi lancia manualmente:"
    echo "  python run/benchmark_runpod.py --variant $VARIANT --input_dir benchmark/inputs"
    exit 0
fi

# ── 5. Esegui benchmark ──────────────────────────────────────────────────────
echo ""
echo "[5/5] Avvio benchmark $VARIANT..."
echo "  Comando: python run/benchmark_runpod.py --variant $VARIANT --input_dir benchmark/inputs --seed 1234"
echo ""
python run/benchmark_runpod.py \
    --variant "$VARIANT" \
    --input_dir benchmark/inputs \
    --seed 1234

echo ""
echo "============================================================"
echo "  COMPLETATO. Risultati in: $BENCH_OUTPUTS/$VARIANT/"
echo "============================================================"
echo ""
echo "  Per scaricare i risultati sulla tua macchina:"
echo "  rsync -avz -e 'ssh -p <PORT>' root@<HOST>:$BENCH_OUTPUTS/$VARIANT/ benchmark/outputs/$VARIANT/"
echo ""
