#!/usr/bin/env bash
# =============================================================================
# Download modelli Hunyuan3D sul Network Volume — UN MODELLO ALLA VOLTA
#
# Scarica SOLO la variante indicata, cancellando le altre prima se necessario.
# Network Volume consigliato: 40 GB (sufficiente per una variante alla volta).
#
# Uso:
#   bash run/download_models_to_volume.sh --v20      # 2.0 full full  (~20-25 GB)
#   bash run/download_models_to_volume.sh --v21      # 2.1 PBR   (~25-30 GB)
#   bash run/download_models_to_volume.sh --v25      # 2.5       (~25-30 GB)
#   bash run/download_models_to_volume.sh --2.0 mini     # 2.0 mini      (~15 GB)
#
# Il flag --clean cancella i modelli delle ALTRE varianti prima di scaricare:
#   bash run/download_models_to_volume.sh --v21 --clean
#
# =============================================================================
set -euo pipefail

MODELS_DIR="/workspace/models"
HUB_DIR="$MODELS_DIR/hub"
export HF_HOME="$MODELS_DIR"
export HUGGINGFACE_HUB_CACHE="$HUB_DIR"
export U2NET_HOME="$MODELS_DIR/u2net"

# ── Parsing argomenti ─────────────────────────────────────────────────────────
VARIANT=""
CLEAN=0

for arg in "$@"; do
    case $arg in
        --2.0 mini)  VARIANT="mini" ;;
        --v20)   VARIANT="2.0" ;;
        --v21)   VARIANT="2.1" ;;
        --v25)   VARIANT="2.5" ;;
        --clean) CLEAN=1 ;;
        *)
            echo "Argomento non riconosciuto: $arg"
            echo "Uso: $0 --v20|--v21|--v25|--mini [--clean]"
            exit 1 ;;
    esac
done

if [ -z "$VARIANT" ]; then
    echo "ERRORE: specifica la variante da scaricare."
    echo "Uso: $0 --v20|--v21|--v25|--mini [--clean]"
    echo ""
    echo "Spazio attuale su /workspace:"
    df -h /workspace 2>/dev/null || true
    exit 1
fi

mkdir -p "$HUB_DIR" "$MODELS_DIR/u2net"
pip install -q huggingface_hub

echo "============================================================"
echo "  Download Hunyuan3D — variante: $VARIANT"
echo "  Destinazione: $MODELS_DIR"
echo "  $(date)"
echo "============================================================"
echo ""
df -h /workspace | grep -v Filesystem || true
echo ""

# ── Funzione: mostra e cancella modelli di una variante ───────────────────────
clean_variant() {
    local LABEL="$1"
    shift
    local DIRS=("$@")
    echo "  Pulizia modelli $LABEL..."
    for d in "${DIRS[@]}"; do
        local full="$HUB_DIR/$d"
        if [ -d "$full" ]; then
            SIZE=$(du -sh "$full" 2>/dev/null | cut -f1)
            echo "    Cancello $d  ($SIZE)"
            rm -rf "$full"
        fi
    done
}

# ── Pulizia modelli delle ALTRE varianti ──────────────────────────────────────
if [ "$CLEAN" -eq 1 ]; then
    echo "--- Pulizia modelli precedenti ---"
    case "$VARIANT" in
        "mini")
            clean_variant "2.0 shape" "models--tencent--Hunyuan3D-2"
            clean_variant "2.1"       "models--tencent--Hunyuan3D-2.1"
            clean_variant "2.5"       "models--tencent--Hunyuan3D-2.5"
            ;;
        "2.0")
            clean_variant "mini"      "models--tencent--Hunyuan3D-2mini"
            clean_variant "2.1"       "models--tencent--Hunyuan3D-2.1"
            clean_variant "2.5"       "models--tencent--Hunyuan3D-2.5"
            # NB: Hunyuan3D-2 è condiviso tra 2.0 mini (texture) e 2.0 full (shape+texture)
            # quindi qui non lo cancelliamo — lo sovrascriviamo col download
            ;;
        "2.1")
            clean_variant "mini"      "models--tencent--Hunyuan3D-2mini"
            clean_variant "2.0/mini"  "models--tencent--Hunyuan3D-2"
            clean_variant "2.5"       "models--tencent--Hunyuan3D-2.5"
            ;;
        "2.5")
            clean_variant "mini"      "models--tencent--Hunyuan3D-2mini"
            clean_variant "2.0/mini"  "models--tencent--Hunyuan3D-2"
            clean_variant "2.1"       "models--tencent--Hunyuan3D-2.1"
            ;;
    esac
    echo ""
    echo "  Spazio dopo pulizia:"
    df -h /workspace | grep -v Filesystem || true
    echo ""
fi

# ── Funzione download ─────────────────────────────────────────────────────────
download_model() {
    local REPO="$1"
    local SUBFOLDER="${2:-}"
    local DESC="$3"

    echo "  Scarico [$DESC]  →  $REPO${SUBFOLDER:+ / $SUBFOLDER}"

    python3 - << PYEOF
import os
from huggingface_hub import snapshot_download
repo = "$REPO"
subfolder = "$SUBFOLDER"
hub_dir = "$HUB_DIR"
kwargs = dict(repo_id=repo, cache_dir=hub_dir, local_dir_use_symlinks=False)
if subfolder:
    kwargs['allow_patterns'] = [f"{subfolder}/**", f"{subfolder}/*"]
snapshot_download(**kwargs)
PYEOF
    echo "  OK"
    echo ""
}

# ── Download variante selezionata ─────────────────────────────────────────────
echo "--- Download modelli $VARIANT ---"

case "$VARIANT" in

    "mini")
        download_model "tencent/Hunyuan3D-2mini" "hunyuan3d-dit-v2-mini-turbo" "mini shape turbo"
        download_model "tencent/Hunyuan3D-2"     "hunyuan3d-paint-v2-0-turbo"  "mini texture turbo"
        ;;

    "2.0")
        # shape: intero repo Hunyuan3D-2 (contiene il modello 2B principale)
        download_model "tencent/Hunyuan3D-2" "" "2.0 shape (2B)"
        # texture qualità piena (non turbo)
        download_model "tencent/Hunyuan3D-2" "hunyuan3d-paint-v2-0" "2.0 texture RGB"
        ;;

    "2.1")
        download_model "tencent/Hunyuan3D-2.1" "hunyuan3d-dit-v2-1"   "2.1 shape"
        download_model "tencent/Hunyuan3D-2.1" "hunyuan3d-paint-v2-1" "2.1 texture PBR"
        ;;

    "2.5")
        # ⚠️  Verifica che il repo esista su HuggingFace prima di eseguire
        python3 - << 'EOF'
try:
    from huggingface_hub import model_info
    info = model_info("tencent/Hunyuan3D-2.5")
    print(f"  Repo trovato: {info.modelId}")
except Exception:
    print("  ERRORE: tencent/Hunyuan3D-2.5 non trovato su HuggingFace.")
    print("  Verifica il nome del repo e aggiorna questo script.")
    import sys; sys.exit(1)
EOF
        download_model "tencent/Hunyuan3D-2.5" "" "2.5 shape+texture"
        ;;
esac

# ── Preload rembg ─────────────────────────────────────────────────────────────
echo "--- Preload rembg ---"
python3 - << EOF
import os, io
os.environ['U2NET_HOME'] = '$MODELS_DIR/u2net'
try:
    from rembg import remove
    from PIL import Image
    img = Image.new('RGB', (64, 64), color='red')
    buf = io.BytesIO(); img.save(buf, format='PNG')
    remove(buf.getvalue())
    print("  rembg OK")
except Exception as e:
    print(f"  rembg: verrà scaricato al primo utilizzo ({e})")
EOF

# ── Riepilogo finale ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  DOWNLOAD COMPLETATO — variante $VARIANT"
echo ""
du -sh "$MODELS_DIR" 2>/dev/null && echo ""
df -h /workspace | grep -v Filesystem || true
echo ""
echo "  Prossimo step:"
echo "  bash run/setup_runpod_${VARIANT//.}.sh"
echo "============================================================"
