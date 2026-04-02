#!/usr/bin/env bash
# =============================================================================
# RunPod Setup — Hunyuan3D 2.0 full + 2.1 PBR (setup combinato)
# GPU consigliata: RTX 5090 32GB | Network Volume: 80GB su /workspace
#
# PRIMA VOLTA:
#   cd /workspace
#   git clone https://github.com/CeciliaPagano/Hunyuan3D-2.git
#   bash /workspace/Hunyuan3D-2/run/setup_runpod_combined.sh
#
# AD OGNI RIAVVIO POD (venv + modelli già presenti):
#   source /workspace/activate.sh
# =============================================================================
set -euo pipefail

FORK_DIR="/workspace/Hunyuan3D-2"
REPO_21_DIR="/workspace/Hunyuan3D-2.1"
VENV_DIR="/workspace/venv"
COMPILED_DIR="/workspace/compiled"
MODELS_DIR="/workspace/models"

export HF_HOME="$MODELS_DIR"
export HUGGINGFACE_HUB_CACHE="$MODELS_DIR/hub"
export U2NET_HOME="$MODELS_DIR/u2net"
unset HF_HUB_ENABLE_HF_TRANSFER 2>/dev/null || true

echo "============================================================"
echo "  Hunyuan3D — Setup combinato 2.0 full + 2.1 PBR"
echo "  $(date)"
echo "============================================================"

# ── 0. Check sistema ──────────────────────────────────────────────────────────
echo ""
echo "[0/7] Stato sistema:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
df -h /workspace | grep -v Filesystem
if [ "$VRAM_MB" -lt 30000 ]; then
    echo "  WARNING: VRAM < 30GB. Aggiungi --sequential per 2.1."
fi

# ── 1. Fork Hunyuan3D-2 (benchmark + setup scripts) ──────────────────────────
echo ""
echo "[1/7] Fork Hunyuan3D-2 (benchmark scripts)..."
if [ ! -d "$FORK_DIR" ]; then
    git clone https://github.com/CeciliaPagano/Hunyuan3D-2.git "$FORK_DIR"
else
    git -C "$FORK_DIR" fetch origin
    git -C "$FORK_DIR" reset --hard origin/master
fi

# ── 2. Repo Hunyuan3D-2.1 (Tencent ufficiale) ────────────────────────────────
echo ""
echo "[2/7] Repo Hunyuan3D-2.1 (Tencent)..."
if [ ! -d "$REPO_21_DIR" ]; then
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$REPO_21_DIR"
else
    echo "  Repo 2.1 già presente, aggiorno..."
    git -C "$REPO_21_DIR" fetch origin
    git -C "$REPO_21_DIR" reset --hard origin/main || git -C "$REPO_21_DIR" reset --hard origin/master
fi

# ── 3. Venv condiviso sul volume ──────────────────────────────────────────────
echo ""
echo "[3/7] Venv Python condiviso..."
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "  Creo nuovo venv in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "  Python: $(python --version) — $(which python)"
pip install --upgrade pip setuptools wheel -q

# ── 4. Dipendenze Python ──────────────────────────────────────────────────────
echo ""
echo "[4/7] Dipendenze Python..."

# Pacchetti core (versioni compatibili Python 3.12)
pip install -q --prefer-binary \
    "numpy>=1.26,<2.0" \
    "open3d>=0.19" \
    trimesh \
    omegaconf einops timm \
    diffusers transformers accelerate safetensors \
    pytorch-lightning \
    basicsr realesrgan \
    "rembg[gpu]" \
    scipy scikit-image imageio \
    huggingface_hub \
    pybind11 \
    opencv-python-headless

# Installa hy3dgen (2.0) come package editable
if [ -f "$FORK_DIR/setup.py" ] || [ -f "$FORK_DIR/pyproject.toml" ]; then
    pip install -q --no-build-isolation -e "$FORK_DIR" || true
fi

# Installa hy3dshape (2.1 shape) come package editable
if [ -f "$REPO_21_DIR/hy3dshape/setup.py" ] || [ -f "$REPO_21_DIR/hy3dshape/pyproject.toml" ]; then
    pip install -q --no-build-isolation -e "$REPO_21_DIR/hy3dshape" || true
fi

# ── 5. Compilazioni C++ per 2.1 (persistenti su volume) ──────────────────────
echo ""
echo "[5/7] Compilazioni C++ (salvate su volume per riuso)..."
mkdir -p "$COMPILED_DIR"

EXT=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# 5a. custom_rasterizer (differentiable rasterizer)
RAST_SO=$(find "$COMPILED_DIR" -name "custom_rasterizer*.so" 2>/dev/null | head -1)
if [ -z "$RAST_SO" ]; then
    echo "  Compilo custom_rasterizer..."
    cd "$REPO_21_DIR/hy3dpaint/custom_rasterizer"
    pip install --no-build-isolation -q -e .
    find . -name "*.so" -exec cp {} "$COMPILED_DIR/" \; 2>/dev/null || true
    cd /workspace
    echo "  custom_rasterizer OK"
else
    echo "  custom_rasterizer già compilato: $(basename $RAST_SO)"
    cp "$COMPILED_DIR"/custom_rasterizer*.so \
       "$REPO_21_DIR/hy3dpaint/custom_rasterizer/" 2>/dev/null || true
    cd "$REPO_21_DIR/hy3dpaint/custom_rasterizer"
    pip install --no-build-isolation -q -e .
    cd /workspace
fi

# 5b. mesh_inpaint_processor (inpainting C++ extension)
INPAINT_SO="$COMPILED_DIR/mesh_inpaint_processor${EXT}"
INPAINT_DEST="$REPO_21_DIR/hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor${EXT}"

if [ ! -f "$INPAINT_SO" ]; then
    echo "  Compilo mesh_inpaint_processor..."
    cd "$REPO_21_DIR/hy3dpaint/DifferentiableRenderer"
    printf 'import pybind11,sysconfig,subprocess\ninc=pybind11.get_include()\npy_inc=sysconfig.get_path("include")\next=sysconfig.get_config_var("EXT_SUFFIX")\ncmd="c++ -O3 -Wall -shared -std=c++11 -fPIC -I"+inc+" -I"+py_inc+" mesh_inpaint_processor.cpp -o mesh_inpaint_processor"+ext\nsubprocess.run(cmd,shell=True,check=True)\n' > /tmp/compile_inpaint.py
    python /tmp/compile_inpaint.py
    cp "$INPAINT_DEST" "$COMPILED_DIR/"
    cd /workspace
    echo "  mesh_inpaint_processor OK"
else
    echo "  mesh_inpaint_processor già compilato."
    cp "$INPAINT_SO" "$INPAINT_DEST"
fi

# ── 6. RealESRGAN checkpoint ──────────────────────────────────────────────────
REALESRGAN_CKPT="$REPO_21_DIR/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
if [ ! -f "$REALESRGAN_CKPT" ]; then
    echo ""
    echo "[6a/7] Download RealESRGAN_x4plus.pth..."
    mkdir -p "$(dirname "$REALESRGAN_CKPT")"
    wget -q --show-progress -O "$REALESRGAN_CKPT" \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
        || curl -L -o "$REALESRGAN_CKPT" \
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    echo "  RealESRGAN OK"
else
    echo "[6/7] RealESRGAN già presente. Skip."
fi

# ── 7. Download modelli su Network Volume ─────────────────────────────────────
echo ""
echo "[7/7] Modelli su Network Volume..."

if [ -d "$MODELS_DIR/hub/models--tencent--Hunyuan3D-2" ]; then
    echo "  Modelli 2.0 già presenti. Skip."
    du -sh "$MODELS_DIR/hub/models--tencent--Hunyuan3D-2" 2>/dev/null || true
else
    echo "  Download modelli 2.0 full..."
    bash "$FORK_DIR/run/download_models_to_volume.sh" --v20
fi

if [ -d "$MODELS_DIR/hub/models--tencent--Hunyuan3D-2.1" ]; then
    echo "  Modelli 2.1 già presenti. Skip."
    du -sh "$MODELS_DIR/hub/models--tencent--Hunyuan3D-2.1" 2>/dev/null || true
else
    echo "  Download modelli 2.1..."
    bash "$FORK_DIR/run/download_models_to_volume.sh" --v21
fi

# ── Crea activate.sh (usato ad ogni riavvio pod) ──────────────────────────────
cat > /workspace/activate.sh << 'ACTIVATE'
#!/usr/bin/env bash
# Esegui con: source /workspace/activate.sh
source /workspace/venv/bin/activate

export HF_HOME="/workspace/models"
export HUGGINGFACE_HUB_CACHE="/workspace/models/hub"
export U2NET_HOME="/workspace/models/u2net"
unset HF_HUB_ENABLE_HF_TRANSFER

# Ricollega i binari compilati nel repo 2.1 (la dir viene reclonata ad ogni setup)
EXT=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
REPO21="/workspace/Hunyuan3D-2.1"
COMPILED="/workspace/compiled"

if [ -f "$COMPILED/mesh_inpaint_processor${EXT}" ]; then
    cp "$COMPILED/mesh_inpaint_processor${EXT}" \
       "$REPO21/hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor${EXT}" 2>/dev/null && \
    echo "  mesh_inpaint_processor ricollegato" || true
fi

SO=$(find "$COMPILED" -name "custom_rasterizer*.so" 2>/dev/null | head -1)
if [ -n "$SO" ]; then
    cp "$SO" "$REPO21/hy3dpaint/custom_rasterizer/" 2>/dev/null || true
    echo "  custom_rasterizer ricollegato"
fi

echo ""
echo "Ambiente Hunyuan3D pronto."
echo "  Python : $(which python)"
echo "  GPU    : $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null)"
echo ""
echo "  Test comparativo:"
echo "    cd /workspace/Hunyuan3D-2"
echo "    python run/test_compare.py --image benchmark/inputs/T1_apple.png"
echo ""
echo "  Benchmark completo 2.0 full:"
echo "    python run/benchmark_runpod.py --variant '2.0 full' --input_dir benchmark/inputs --seed 1234"
echo ""
echo "  Benchmark completo 2.1:"
echo "    cd /workspace/Hunyuan3D-2.1"
echo "    python run/benchmark_runpod.py --input_dir benchmark/inputs --seed 1234"
ACTIVATE
chmod +x /workspace/activate.sh

# ── Riepilogo ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  SETUP COMPLETATO"
echo ""
echo "  AD OGNI RIAVVIO POD:"
echo "    source /workspace/activate.sh"
echo ""
echo "  TEST RAPIDO SU SINGOLA IMMAGINE:"
echo "    cd /workspace/Hunyuan3D-2"
echo "    python run/test_compare.py --image benchmark/inputs/T1_apple.png"
echo ""
echo "  Spazio volume:"
df -h /workspace | grep -v Filesystem
echo "============================================================"
