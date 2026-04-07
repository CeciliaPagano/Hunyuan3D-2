#!/usr/bin/env bash
# =============================================================================
# Patch per far girare TRELLIS.2 con Python 3.13 (miniconda base env)
# Da eseguire DOPO setup_runpod_trellis.sh e DOPO aver attivato il base env.
#
# USO:
#   eval "$(/workspace/miniconda/bin/conda shell.bash hook)"
#   conda activate base
#   bash /workspace/Hunyuan3D-2/run/setup_trellis_patches.sh
# =============================================================================
set -euo pipefail

TRELLIS_DIR="/workspace/TRELLIS.2"

echo "============================================================"
echo "  TRELLIS.2 — Patch compatibilità Python 3.13"
echo "  $(date)"
echo "============================================================"

# ── 1. rembg[gpu] — sostituisce RMBG-2.0 (incompatibile con Python 3.13) ──────
echo ""
echo "[1/5] Installo rembg[gpu]..."
pip install "rembg[gpu]" -q --root-user-action=ignore
python3 -c "import rembg; print('  rembg OK')"

# ── 2. BiRefNet.py — wrapper rembg invece di transformers ─────────────────────
echo ""
echo "[2/5] Patch BiRefNet.py..."
cat > "$TRELLIS_DIR/trellis2/pipelines/rembg/BiRefNet.py" << 'BIREFNET'
import io
from PIL import Image
from rembg import remove

class BiRefNet:
    def __init__(self, model_name='ZhengPeng7/BiRefNet'):
        pass
    def to(self, device): pass
    def cuda(self): pass
    def cpu(self): pass
    def __call__(self, image):
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        result = remove(buf.getvalue())
        return Image.open(io.BytesIO(result)).convert('RGBA')
BIREFNET
echo "  BiRefNet.py OK"

# ── 3. image_feature_extractor.py — self.model.layer → self.model.model.layer ─
echo ""
echo "[3/5] Patch image_feature_extractor.py (DINOv3 layer path)..."
python3 - << 'PY'
path = '/workspace/TRELLIS.2/trellis2/modules/image_feature_extractor.py'
with open(path) as f:
    c = f.read()
c = c.replace('self.model.layer', 'self.model.model.layer')
with open(path, 'w') as f:
    f.write(c)
print('  image_feature_extractor.py OK')
PY

# ── 4. sparse/attention/full_attn.py — aggiunge backend naive (sdpa-based) ────
echo ""
echo "[4/5] Patch sparse/attention/full_attn.py (naive backend)..."
python3 - << 'PY'
path = '/workspace/TRELLIS.2/trellis2/modules/sparse/attention/full_attn.py'
with open(path) as f:
    c = f.read()

naive_block = """    elif config.ATTN == 'naive':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        out_list = []; offset_q = 0; offset_kv = 0
        for i in range(len(q_seqlen)):
            ql = q_seqlen[i]; kl = kv_seqlen[i]
            qi = q[offset_q:offset_q+ql].unsqueeze(0).permute(0,2,1,3)
            ki = k[offset_kv:offset_kv+kl].unsqueeze(0).permute(0,2,1,3)
            vi = v[offset_kv:offset_kv+kl].unsqueeze(0).permute(0,2,1,3)
            oi = torch.nn.functional.scaled_dot_product_attention(qi,ki,vi)
            out_list.append(oi.permute(0,2,1,3).squeeze(0))
            offset_q += ql; offset_kv += kl
        out = torch.cat(out_list, dim=0)
"""

target = "    else:\n        raise ValueError"
if "config.ATTN == 'naive'" not in c:
    c = c.replace(target, naive_block + target)

with open(path, 'w') as f:
    f.write(c)
print('  sparse/attention/full_attn.py OK')
PY

# ── 5. sparse/config.py — accetta 'naive' come valore valido ──────────────────
echo ""
echo "[5/5] Patch sparse/config.py (naive accettato)..."
python3 - << 'PY'
path = '/workspace/TRELLIS.2/trellis2/modules/sparse/config.py'
with open(path) as f:
    c = f.read()
c = c.replace(
    "in ['xformers', 'flash_attn', 'flash_attn_3']",
    "in ['xformers', 'flash_attn', 'flash_attn_3', 'naive']"
)
with open(path, 'w') as f:
    f.write(c)
print('  sparse/config.py OK')
PY

# ── Aggiorna activate_trellis.sh con ATTN_BACKEND=naive ──────────────────────
echo ""
echo "Aggiorno /workspace/activate_trellis.sh..."
cat > /workspace/activate_trellis.sh << 'ACTIVATE'
#!/usr/bin/env bash
# source /workspace/activate_trellis.sh

eval "$(/workspace/miniconda/bin/conda shell.bash hook)"
conda activate base

export PYTHONPATH="/workspace/TRELLIS.2"
export HF_HOME="/workspace/models"
export HUGGINGFACE_HUB_CACHE="/workspace/models/hub"
export OPENCV_IO_ENABLE_OPENEXR="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export ATTN_BACKEND="naive"
unset HF_HUB_ENABLE_HF_TRANSFER

echo "Ambiente TRELLIS.2 pronto (base env, Python 3.13)."
echo "  Python : $(which python)"
echo "  GPU    : $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null)"
echo ""
echo "  Benchmark tutti i soggetti:"
echo "    cd /workspace/TRELLIS.2"
echo "    python /workspace/Hunyuan3D-2/run/benchmark_trellis.py --input_dir /workspace/Hunyuan3D-2/benchmark/inputs --seed 1234"
echo ""
echo "  Singola immagine:"
echo "    python /workspace/Hunyuan3D-2/run/benchmark_trellis.py --input_dir /workspace/Hunyuan3D-2/benchmark/inputs --only T1_apple --seed 1234"
ACTIVATE
chmod +x /workspace/activate_trellis.sh

echo ""
echo "============================================================"
echo "  PATCH COMPLETATE"
echo ""
echo "  AD OGNI RIAVVIO POD:"
echo "    source /workspace/activate_trellis.sh"
echo ""
echo "  BENCHMARK:"
echo "    python /workspace/Hunyuan3D-2/run/benchmark_trellis.py \\"
echo "        --input_dir /workspace/Hunyuan3D-2/benchmark/inputs --seed 1234"
echo "============================================================"
