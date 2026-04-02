#!/usr/bin/env bash
# =============================================================================
# RunPod Setup — Microsoft TRELLIS.2 (4B PBR)
# GPU consigliata: RTX 4090 24GB  (NON RTX 5090 — bug Blackwell issue #99)
# Template RunPod: PyTorch con CUDA 12.4 DEVEL (serve gcc + nvcc per compilare)
# Network Volume: 40GB su /workspace  (modello ~20GB + build ~10GB)
#
# PRIMA VOLTA:
#   cd /workspace
#   git clone https://github.com/CeciliaPagano/Hunyuan3D-2.git
#   bash /workspace/Hunyuan3D-2/run/setup_runpod_trellis.sh
#
# AD OGNI RIAVVIO POD:
#   source /workspace/activate_trellis.sh
# =============================================================================
set -euo pipefail

FORK_DIR="/workspace/Hunyuan3D-2"        # nostro fork con gli script
TRELLIS_DIR="/workspace/TRELLIS.2"       # repo Microsoft ufficiale
MODELS_DIR="/workspace/models"
BENCH_INPUTS="$FORK_DIR/benchmark/inputs"
BENCH_OUTPUTS="$FORK_DIR/benchmark/outputs/trellis2"

export HF_HOME="$MODELS_DIR"
export HUGGINGFACE_HUB_CACHE="$MODELS_DIR/hub"
export OPENCV_IO_ENABLE_OPENEXR="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
unset HF_HUB_ENABLE_HF_TRANSFER 2>/dev/null || true

echo "============================================================"
echo "  TRELLIS.2 — Setup RunPod"
echo "  $(date)"
echo "============================================================"

# ── 0. Check sistema ──────────────────────────────────────────────────────────
echo ""
echo "[0/6] Stato sistema:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
df -h /workspace | grep -v Filesystem
nvcc --version 2>/dev/null | grep "release" || echo "  nvcc: non trovato nel PATH"
gcc --version | head -1

# Avviso GPU Blackwell
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -qi "5090\|5080\|5070\|GB200\|B200"; then
    echo ""
    echo "  ⚠️  WARNING: GPU Blackwell rilevata ($GPU_NAME)"
    echo "  TRELLIS.2 ha bug noto su Blackwell (issue #99): produce point cloud invece di mesh."
    echo "  Usa RTX 4090 o A100 per risultati corretti."
    echo "  Premi Ctrl+C per fermare, o Invio per continuare comunque."
    read -r || true
fi

# Check GCC version (serve <=11 per CUDA 12.4)
GCC_MAJOR=$(gcc -dumpversion | cut -d. -f1)
if [ "$GCC_MAJOR" -gt 11 ]; then
    echo ""
    echo "  GCC $GCC_MAJOR rilevato — TRELLIS.2 richiede GCC ≤11 con CUDA 12.4."
    echo "  Installo gcc-11..."
    apt-get install -y gcc-11 g++-11 -q 2>/dev/null || true
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 2>/dev/null || true
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 2>/dev/null || true
    echo "  gcc ora: $(gcc --version | head -1)"
fi

# ── 1. Fork con benchmark scripts ────────────────────────────────────────────
echo ""
echo "[1/6] Fork Hunyuan3D-2 (benchmark scripts)..."
if [ ! -d "$FORK_DIR" ]; then
    git clone https://github.com/CeciliaPagano/Hunyuan3D-2.git "$FORK_DIR"
else
    git -C "$FORK_DIR" fetch origin
    git -C "$FORK_DIR" reset --hard origin/master
fi

# ── 2. Clone TRELLIS.2 (--recursive obbligatorio per i submodules) ────────────
echo ""
echo "[2/6] Clone TRELLIS.2 (Microsoft)..."
if [ ! -d "$TRELLIS_DIR" ]; then
    git clone --recursive https://github.com/microsoft/TRELLIS.2.git "$TRELLIS_DIR"
else
    echo "  Repo già presente, aggiorno..."
    git -C "$TRELLIS_DIR" fetch origin
    git -C "$TRELLIS_DIR" reset --hard origin/main || git -C "$TRELLIS_DIR" reset --hard origin/master
    git -C "$TRELLIS_DIR" submodule update --init --recursive
fi
cd "$TRELLIS_DIR"

# ── 3. Setup ufficiale TRELLIS.2 (conda + compilazioni CUDA) ─────────────────
echo ""
echo "[3/6] Setup ambiente TRELLIS.2 (40-70 min, compilazione CUDA)..."
echo "  Flags: --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm"
echo ""

# Assicura conda disponibile
if ! command -v conda &>/dev/null; then
    echo "  conda non trovato — installo Miniconda..."
    wget -q --show-progress \
        -O /tmp/miniconda.sh \
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    bash /tmp/miniconda.sh -b -p /workspace/miniconda
    eval "$(/workspace/miniconda/bin/conda shell.bash hook)"
    conda init bash
    echo "  Miniconda installato in /workspace/miniconda"
fi
eval "$(conda shell.bash hook)" 2>/dev/null || true

# Lancia il setup ufficiale
bash setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

# ── 4. Download modello TRELLIS.2-4B ─────────────────────────────────────────
echo ""
echo "[4/6] Download modello TRELLIS.2-4B (~20GB)..."
mkdir -p "$MODELS_DIR/hub"

conda run -n trellis2 python -c "
import os
os.environ['HF_HOME'] = '$MODELS_DIR'
os.environ['HUGGINGFACE_HUB_CACHE'] = '$MODELS_DIR/hub'
from huggingface_hub import snapshot_download
print('  Download in corso...')
snapshot_download('microsoft/TRELLIS.2-4B', cache_dir='$MODELS_DIR/hub')
print('  Download completato.')
"

# ── 5. Setup benchmark ────────────────────────────────────────────────────────
echo ""
echo "[5/6] Setup benchmark..."
mkdir -p "$BENCH_INPUTS" "$BENCH_OUTPUTS"

# Copia benchmark script nel repo TRELLIS.2
cp "$FORK_DIR/run/benchmark_trellis.py" "$TRELLIS_DIR/run/benchmark_trellis.py" 2>/dev/null || \
    cp "$FORK_DIR/run/benchmark_trellis.py" "$TRELLIS_DIR/benchmark_trellis.py"

N_IMGS=$(find "$BENCH_INPUTS" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
if [ "$N_IMGS" -eq 0 ]; then
    echo ""
    echo "  ATTENZIONE: nessuna immagine in $BENCH_INPUTS"
    echo "  Carica le immagini dalla macchina locale:"
    echo ""
    echo "    rsync -avz -e 'ssh -p <PORT>' benchmark/inputs/ root@<HOST>:$BENCH_INPUTS/"
fi

# ── 6. activate_trellis.sh (per riavvii veloci) ───────────────────────────────
cat > /workspace/activate_trellis.sh << 'ACTIVATE'
#!/usr/bin/env bash
# source /workspace/activate_trellis.sh

# Inizializza conda
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/workspace/miniconda")
eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate trellis2

export HF_HOME="/workspace/models"
export HUGGINGFACE_HUB_CACHE="/workspace/models/hub"
export OPENCV_IO_ENABLE_OPENEXR="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
unset HF_HUB_ENABLE_HF_TRANSFER

echo "Ambiente TRELLIS.2 pronto."
echo "  Python : $(which python)"
echo "  GPU    : $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null)"
echo ""
echo "  Benchmark:"
echo "    cd /workspace/TRELLIS.2"
echo "    python benchmark_trellis.py --input_dir /workspace/Hunyuan3D-2/benchmark/inputs --seed 1234"
echo ""
echo "  Test singola immagine:"
echo "    python benchmark_trellis.py --input_dir /workspace/Hunyuan3D-2/benchmark/inputs --only T1_apple --seed 1234"
ACTIVATE
chmod +x /workspace/activate_trellis.sh

# ── Riepilogo ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  SETUP COMPLETATO"
echo ""
echo "  AD OGNI RIAVVIO POD:"
echo "    source /workspace/activate_trellis.sh"
echo ""
if [ "$N_IMGS" -eq 0 ]; then
    echo "  Carica le immagini poi lancia:"
fi
echo "  BENCHMARK:"
echo "    cd /workspace/TRELLIS.2"
echo "    python benchmark_trellis.py --input_dir /workspace/Hunyuan3D-2/benchmark/inputs --seed 1234"
echo ""
echo "  Spazio volume:"
df -h /workspace | grep -v Filesystem
echo "============================================================"
