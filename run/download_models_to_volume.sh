#!/usr/bin/env bash
# =============================================================================
# Download modelli Hunyuan3D sul Network Volume (una tantum)
#
# Scarica su /workspace/models/ tutti i modelli necessari per il benchmark.
# La prossima volta che avvii un pod con lo stesso volume, i modelli sono già lì.
#
# Uso:
#   bash run/download_models_to_volume.sh            # scarica tutto (default)
#   bash run/download_models_to_volume.sh --mini     # solo mini (baseline)
#   bash run/download_models_to_volume.sh --v20      # solo 2.0
#   bash run/download_models_to_volume.sh --v21      # solo 2.1
#   bash run/download_models_to_volume.sh --v25      # solo 2.5 (se disponibile)
# =============================================================================
set -euo pipefail

MODELS_DIR="/workspace/models"
export HF_HOME="$MODELS_DIR"
export HUGGINGFACE_HUB_CACHE="$MODELS_DIR/hub"

# Parsing argomenti
DOWNLOAD_MINI=1
DOWNLOAD_V20=1
DOWNLOAD_V21=1
DOWNLOAD_V25=1

if [ $# -gt 0 ]; then
    DOWNLOAD_MINI=0; DOWNLOAD_V20=0; DOWNLOAD_V21=0; DOWNLOAD_V25=0
    for arg in "$@"; do
        case $arg in
            --mini) DOWNLOAD_MINI=1 ;;
            --v20)  DOWNLOAD_V20=1 ;;
            --v21)  DOWNLOAD_V21=1 ;;
            --v25)  DOWNLOAD_V25=1 ;;
        esac
    done
fi

mkdir -p "$MODELS_DIR"
pip install -q huggingface_hub

echo "============================================================"
echo "  Download modelli Hunyuan3D → $MODELS_DIR"
echo "  HF_HOME=$HF_HOME"
echo "  $(date)"
echo "============================================================"

# ── Funzione download con progress ────────────────────────────────────────────
download_model() {
    local REPO="$1"
    local SUBFOLDER="${2:-}"
    local DESC="$3"

    echo ""
    echo "  [$DESC]  $REPO${SUBFOLDER:+ / $SUBFOLDER}"

    if [ -n "$SUBFOLDER" ]; then
        huggingface-cli download "$REPO" \
            --include "${SUBFOLDER}/**" \
            --cache-dir "$MODELS_DIR/hub" \
            --local-dir-use-symlinks False \
            --quiet
    else
        huggingface-cli download "$REPO" \
            --cache-dir "$MODELS_DIR/hub" \
            --local-dir-use-symlinks False \
            --quiet
    fi
    echo "  OK"
}

# ── Mini (baseline, shape + texture turbo) ────────────────────────────────────
if [ "$DOWNLOAD_MINI" -eq 1 ]; then
    echo ""
    echo "=== MINI (baseline) ==="
    download_model "tencent/Hunyuan3D-2mini" "hunyuan3d-dit-v2-mini-turbo" "mini shape turbo"
    download_model "tencent/Hunyuan3D-2"     "hunyuan3d-paint-v2-0-turbo"  "mini texture turbo"
fi

# ── 2.0 full (shape 2B + texture v2-0) ───────────────────────────────────────
if [ "$DOWNLOAD_V20" -eq 1 ]; then
    echo ""
    echo "=== 2.0 FULL ==="
    # shape: repo Hunyuan3D-2, no subfolder = modello 2B principale
    download_model "tencent/Hunyuan3D-2" "" "2.0 shape (2B)"
    # texture non-turbo per qualità massima
    download_model "tencent/Hunyuan3D-2" "hunyuan3d-paint-v2-0" "2.0 texture RGB"
fi

# ── 2.1 PBR ──────────────────────────────────────────────────────────────────
if [ "$DOWNLOAD_V21" -eq 1 ]; then
    echo ""
    echo "=== 2.1 PBR ==="
    download_model "tencent/Hunyuan3D-2.1" "hunyuan3d-dit-v2-1"   "2.1 shape"
    download_model "tencent/Hunyuan3D-2.1" "hunyuan3d-paint-v2-1" "2.1 texture PBR"
fi

# ── 2.5 ──────────────────────────────────────────────────────────────────────
if [ "$DOWNLOAD_V25" -eq 1 ]; then
    echo ""
    echo "=== 2.5 ==="
    # ⚠️  Verifica che repo e subfolder esistano su HuggingFace prima di eseguire
    python3 - << 'EOF'
try:
    from huggingface_hub import model_info
    info = model_info("tencent/Hunyuan3D-2.5")
    print(f"  Repo 2.5 trovato: {info.modelId}")
except Exception:
    print("  ⚠️  tencent/Hunyuan3D-2.5 non trovato su HuggingFace.")
    print("  Controlla il nome corretto e aggiorna questo script.")
    import sys; sys.exit(1)
EOF
    download_model "tencent/Hunyuan3D-2.5" "" "2.5 shape+texture"
fi

# ── Rembg model (scaricato automaticamente, ma lo preloaidiamo) ───────────────
echo ""
echo "=== REMBG model ==="
python3 -c "
import os; os.environ['U2NET_HOME'] = '$MODELS_DIR/u2net'
from rembg import remove
from PIL import Image
import io
img = Image.new('RGB', (64, 64), color='red')
buf = io.BytesIO(); img.save(buf, format='PNG')
remove(buf.getvalue())
print('  rembg OK')
" 2>/dev/null || echo "  rembg: scaricato al primo utilizzo"

# ── Riepilogo spazio usato ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  DOWNLOAD COMPLETATO"
du -sh "$MODELS_DIR" 2>/dev/null || true
echo "  I modelli sono in: $MODELS_DIR"
echo "  Imposta HF_HOME=$MODELS_DIR nei prossimi script per riutilizzarli."
echo "============================================================"