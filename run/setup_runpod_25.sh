#!/usr/bin/env bash
# =============================================================================
# RunPod Setup — Hunyuan3D 2.5 (RTX 4090 24GB)
# Network Volume: 40 GB montato su /workspace
#
# ⚠️  Verificare esistenza repo 2.5 su GitHub/HuggingFace prima di eseguire.
#     Se non esiste un repo dedicato, usa il repo Hunyuan3D-2 con variante 2.5.
#
# COME USARLO:
#   1. Avvia pod RunPod RTX 4090 24GB + Network Volume 40GB su /workspace
#   2. Connettiti via SSH e lancia:
#        cd /workspace && git clone https://github.com/CeciliaPagano/Hunyuan3D-2.git
#        bash /workspace/Hunyuan3D-2/run/setup_runpod_25.sh
# =============================================================================
set -euo pipefail

export HF_HOME="/workspace/models"
export HUGGINGFACE_HUB_CACHE="/workspace/models/hub"
export U2NET_HOME="/workspace/models/u2net"

FORK_DIR="/workspace/Hunyuan3D-2"
REPO_DIR="/workspace/Hunyuan3D-2.5"
BENCH_INPUTS="$REPO_DIR/benchmark/inputs"
BENCH_OUTPUTS="$REPO_DIR/benchmark/outputs"

echo "============================================================"
echo "  Hunyuan3D BENCHMARK — Variante: 2.5"
echo "  $(date)"
echo "============================================================"

# ── 0. Check GPU e spazio ────────────────────────────────────────────────────
echo ""
echo "[0/6] Stato sistema:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace | grep -v Filesystem
echo ""

# ── 1. Fork (benchmark scripts) ──────────────────────────────────────────────
echo "[1/6] Fork con benchmark scripts..."
if [ ! -d "$FORK_DIR" ]; then
    git clone https://github.com/CeciliaPagano/Hunyuan3D-2.git "$FORK_DIR"
else
    git -C "$FORK_DIR" pull
fi

# ── 2. Repo ufficiale 2.5 ────────────────────────────────────────────────────
echo ""
echo "[2/6] Repo Hunyuan3D-2.5..."
REPO_URL_PRIMARY="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.5.git"
REPO_URL_FALLBACK="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git"

if [ ! -d "$REPO_DIR" ]; then
    if git clone "$REPO_URL_PRIMARY" "$REPO_DIR" 2>/dev/null; then
        echo "  Clonato repo dedicato 2.5"
    else
        echo "  ⚠️  Repo 2.5 dedicato non trovato. Uso Hunyuan3D-2 come base."
        git clone "$REPO_URL_FALLBACK" "$REPO_DIR"
    fi
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

# ── 4. Download modelli 2.5 sul volume ───────────────────────────────────────
echo ""
echo "[4/6] Modelli su Network Volume..."
if [ -d "$HF_HOME/hub/models--tencent--Hunyuan3D-2.5" ]; then
    echo "  Modelli 2.5 già presenti sul volume. Skip download."
    du -sh "$HF_HOME/hub/models--tencent--Hunyuan3D-2.5" 2>/dev/null || true
else
    echo "  Download modelli 2.5 (--clean rimuove varianti precedenti)..."
    bash "$FORK_DIR/run/download_models_to_volume.sh" --v25 --clean
fi

# ── 5. Struttura benchmark + verifica input ───────────────────────────────────
echo ""
echo "[5/6] Setup benchmark..."
mkdir -p "$BENCH_INPUTS" "$BENCH_OUTPUTS/2.5"

mkdir -p run
cp "$FORK_DIR/run/benchmark_runpod_25.py" run/benchmark_runpod.py

N_IMGS=$(find "$BENCH_INPUTS" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" 2>/dev/null | wc -l)
if [ "$N_IMGS" -eq 0 ]; then
    echo ""
    echo "  ⚠️  Nessuna immagine in $BENCH_INPUTS"
    echo "  Carica le immagini:"
    echo "    rsync -avz -e 'ssh -p <PORT>' benchmark/inputs/ root@<HOST>:$BENCH_INPUTS/"
    echo ""
    echo "  Poi esegui:"
    echo "    cd $REPO_DIR && python run/benchmark_runpod.py --input_dir benchmark/inputs --seed 1234"
    exit 0
fi
echo "  Trovate $N_IMGS immagini."

# ── 6. Esegui benchmark ──────────────────────────────────────────────────────
echo ""
echo "[6/6] Avvio benchmark 2.5..."
python run/benchmark_runpod.py \
    --input_dir benchmark/inputs \
    --seed 1234

echo ""
echo "============================================================"
echo "  COMPLETATO — Variante 2.5"
echo "  Risultati in: $BENCH_OUTPUTS/2.5/"
echo ""
echo "  Scarica i risultati:"
echo "  rsync -avz -e 'ssh -p <PORT>' root@<HOST>:$BENCH_OUTPUTS/2.5/ benchmark/outputs/2.5/"
echo "============================================================"
