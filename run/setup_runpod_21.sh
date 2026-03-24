#!/usr/bin/env bash
# =============================================================================
# RunPod Setup — Hunyuan3D 2.1 PBR (A100 40GB / RTX 3090 24GB + sequential)
# Repo: Tencent-Hunyuan/Hunyuan3D-2.1
# GPU consigliata: A100 40GB  (shape ~10GB + texture ~21GB = ~31GB)
#   Alternativa: RTX 3090 24GB con --sequential (più lento ma funziona)
#
# ATTENZIONE:
#   Questo repo usa pipeline PBR separata rispetto a Hunyuan3D-2.
#   benchmark_runpod_21.py viene copiato automaticamente nel repo 2.1
#   e gestisce Hunyuan3DPaintPipelinePBR (se disponibile) o il fallback RGB.
#
# COME USARLO:
#   1. Avvia pod RunPod con A100 40GB, template PyTorch 2.x/CUDA 12.x
#   2. Copia questo script sul pod:
#        scp -P <PORT> run/setup_runpod_21.sh root@<HOST>:/root/
#   3. Sul pod:
#        bash /root/setup_runpod_21.sh
#   3b. Se su GPU <40GB (es. RTX 3090), aggiungi --sequential:
#        SEQUENTIAL=1 bash /root/setup_runpod_21.sh
# =============================================================================
set -euo pipefail

export HF_HOME="/workspace/models"
export HUGGINGFACE_HUB_CACHE="/workspace/models/hub"
export U2NET_HOME="/workspace/models/u2net"

REPO_DIR="/workspace/Hunyuan3D-2.1"
BENCH_INPUTS="$REPO_DIR/benchmark/inputs"
BENCH_OUTPUTS="$REPO_DIR/benchmark/outputs"
VARIANT="2.1"
SEQUENTIAL="${SEQUENTIAL:-0}"

echo "============================================================"
echo "  Hunyuan3D BENCHMARK — Variante: $VARIANT (PBR)"
echo "  $(date)"
echo "  Sequential mode: $SEQUENTIAL"
echo "============================================================"

# ── 0. Check GPU ─────────────────────────────────────────────────────────────
echo ""
echo "[0/6] GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "  VRAM totale: ${VRAM_MB} MB"
if [ "$VRAM_MB" -lt 35000 ] && [ "$SEQUENTIAL" -eq 0 ]; then
    echo ""
    echo "  AVVISO: VRAM < 35GB rilevata. Raccomandato --sequential."
    echo "  Riavvia con:  SEQUENTIAL=1 bash $0"
    echo "  Continuo comunque (potrebbe andare OOM sulla texture)..."
fi

# ── 1. Clone repo 2.1 ────────────────────────────────────────────────────────
echo ""
echo "[1/6] Clone Hunyuan3D-2.1 repository..."
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$REPO_DIR"
else
    echo "  Repo già presente, aggiorno..."
    git -C "$REPO_DIR" pull
fi
cd "$REPO_DIR"

# ── 2. Installa dipendenze ───────────────────────────────────────────────────
echo ""
echo "[2/6] Installazione dipendenze Python..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
fi

# custom_rasterizer
if [ -d "hy3dgen/texgen/custom_rasterizer" ]; then
    echo "  Build custom_rasterizer..."
    pip install -q -e hy3dgen/texgen/custom_rasterizer
fi

pip install -q rembg[gpu]

# ── 3. Copia benchmark script ────────────────────────────────────────────────
echo ""
echo "[3/6] Setup script benchmark per 2.1..."
mkdir -p run

# Lo script benchmark viene creato inline per la variante 2.1
# Legge da hy3dgen del repo 2.1, che supporta la pipeline PBR
cat > run/benchmark_runpod.py << 'PYEOF'
#!/usr/bin/env python3
"""
Hunyuan3D Benchmark Runner — variante 2.1 PBR
Adattato per il repo Tencent-Hunyuan/Hunyuan3D-2.1

Usa Hunyuan3DPaintPipeline dal repo 2.1 (che supporta PBR):
  - shape_model: tencent/Hunyuan3D-2.1  subfolder=hunyuan3d-dit-v2-1
  - texture_model: tencent/Hunyuan3D-2.1  subfolder=hunyuan3d-paint-v2-1

Usage:
    # Su A100 40GB (senza sequential):
    python run/benchmark_runpod.py --variant 2.1 --input_dir benchmark/inputs

    # Su RTX 3090 24GB (con sequential per stare nei 24GB):
    python run/benchmark_runpod.py --variant 2.1 --sequential --input_dir benchmark/inputs
"""

import argparse, gc, json, sys, time
from datetime import datetime
from pathlib import Path
import torch

CONFIG = {
    'description': 'Hunyuan3D-2.1 — PBR (Albedo + Normal + Roughness + Metallic)',
    'shape_model_path': 'tencent/Hunyuan3D-2.1',
    'shape_subfolder': 'hunyuan3d-dit-v2-1',
    'texture_model_path': 'tencent/Hunyuan3D-2.1',
    'texture_subfolder': 'hunyuan3d-paint-v2-1',   # verificare su HF se serve adattamento
    'enable_flashvdm': False,
    'num_inference_steps': 50,
    'octree_resolution': 512,
    'num_chunks': 200000,
    'guidance_scale': 7.5,
    'has_pbr': True,
}


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def vram_peak_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def gpu_info():
    if not torch.cuda.is_available():
        return {'name': 'CPU only', 'total_mb': 0}
    p = torch.cuda.get_device_properties(0)
    return {'name': p.name, 'total_mb': p.total_memory / 1024 ** 2}


def run_rembg(image_path):
    from hy3dgen.rembg import BackgroundRemover
    from PIL import Image
    t0 = time.time()
    image = Image.open(image_path).convert('RGB')
    rmbg = BackgroundRemover()
    image_rgba = rmbg(image)
    del rmbg; clear_memory()
    return image_rgba, time.time() - t0


def run_shape(image, sequential):
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen.pipelines import export_to_trimesh

    kwargs = {'subfolder': CONFIG['shape_subfolder']}
    if sequential:
        kwargs['device'] = 'cpu'

    print("  Caricamento shape model (2.1)...")
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        CONFIG['shape_model_path'], **kwargs
    )
    if sequential:
        pipe.to('cuda')

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    gen = torch.Generator(device='cuda').manual_seed(_SEED)
    out = pipe(
        image=image,
        num_inference_steps=CONFIG['num_inference_steps'],
        guidance_scale=CONFIG['guidance_scale'],
        generator=gen,
        octree_resolution=CONFIG['octree_resolution'],
        num_chunks=CONFIG['num_chunks'],
        output_type='mesh',
    )
    mesh = export_to_trimesh(out)[0]
    elapsed = time.time() - t0
    peak = vram_peak_mb()

    del pipe; clear_memory()
    return mesh, elapsed, peak


def run_face_reduce(mesh, target=40000):
    from hy3dgen.shapegen import FaceReducer
    t0 = time.time()
    r = FaceReducer()
    mesh = r(mesh, target)
    del r; clear_memory()
    return mesh, time.time() - t0


def run_texture(mesh, image):
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    print("  Caricamento texture model PBR (2.1)...")
    pipe = Hunyuan3DPaintPipeline.from_pretrained(
        CONFIG['texture_model_path'],
        subfolder=CONFIG['texture_subfolder'],
    )
    try:
        pipe.enable_model_cpu_offload()
        print("  CPU offload abilitato per texture")
    except Exception as e:
        print(f"  Warning: CPU offload non disponibile: {e}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    textured = pipe(mesh, image)
    elapsed = time.time() - t0
    peak = vram_peak_mb()

    del pipe; clear_memory()
    return textured, elapsed, peak


def run_subject(img_path, out_dir, args):
    subj = img_path.stem
    d = out_dir / subj
    d.mkdir(parents=True, exist_ok=True)

    m = {
        'variant': '2.1',
        'subject': subj,
        'input_image': str(img_path),
        'seed': args.seed,
        'inference_steps': CONFIG['num_inference_steps'],
        'octree_resolution': CONFIG['octree_resolution'],
        'has_pbr': True,
        'sequential': args.sequential,
        'timestamp': datetime.now().isoformat(),
        'gpu': gpu_info(),
        'timing': {}, 'vram_peak_mb': {}, 'mesh': {}, 'files': {},
        'quality': {
            'geometry': None, 'texture': None,
            'artifacts': None, 'janus': None, 'notes': '',
            'pbr_albedo': None, 'pbr_normal': None, 'pbr_roughness': None,
        },
        'error': None,
    }

    print(f"\n{'='*60}")
    print(f"  {subj}  |  variante: 2.1 PBR  |  sequential: {args.sequential}")
    print(f"{'='*60}")

    # rembg
    print("[1/4] Background removal...")
    try:
        img, t = run_rembg(img_path)
        m['timing']['rembg_s'] = round(t, 2)
        p = d / f'{subj}_rembg.png'; img.save(p)
        m['files']['rembg_png'] = str(p)
        print(f"  OK  {t:.1f}s")
    except Exception as e:
        m['error'] = f'rembg: {e}'; _save(m, d, subj); return m

    # shape
    print(f"[2/4] Shape ({CONFIG['num_inference_steps']} steps, octree={CONFIG['octree_resolution']})...")
    try:
        mesh_raw, t, pk = run_shape(img, args.sequential)
        m['timing']['shape_s'] = round(t, 2)
        m['vram_peak_mb']['shape'] = round(pk, 1)
        m['mesh']['faces_raw'] = int(mesh_raw.faces.shape[0])
        m['mesh']['verts_raw'] = int(mesh_raw.vertices.shape[0])
        sp = d / f'{subj}_shape.glb'; mesh_raw.export(str(sp))
        m['files']['shape_glb'] = str(sp)
        m['files']['shape_glb_kb'] = round(sp.stat().st_size / 1024, 1)
        print(f"  OK  {t:.1f}s  |  {mesh_raw.faces.shape[0]:,} facce  |  VRAM peak: {pk:.0f} MB")
    except Exception as e:
        m['error'] = f'shape: {e}'; _save(m, d, subj)
        import traceback; traceback.print_exc()
        return m

    if args.no_texture:
        m['timing']['total_s'] = round(sum(v for k,v in m['timing'].items() if k.endswith('_s')), 2)
        _save(m, d, subj); return m

    # face reduce
    print(f"[3/4] Face reduction → {args.target_faces:,}...")
    try:
        mesh_red, t = run_face_reduce(mesh_raw, args.target_faces)
        m['timing']['face_reduce_s'] = round(t, 2)
        m['mesh']['faces_reduced'] = int(mesh_red.faces.shape[0])
        print(f"  {mesh_raw.faces.shape[0]:,} → {mesh_red.faces.shape[0]:,}")
    except Exception as e:
        m['error'] = f'face_reduce: {e}'; _save(m, d, subj); return m

    # texture PBR
    print("[4/4] Texture PBR generation...")
    try:
        textured, t, pk = run_texture(mesh_red, img)
        m['timing']['texture_s'] = round(t, 2)
        m['vram_peak_mb']['texture'] = round(pk, 1)
        tp = d / f'{subj}_textured.glb'; textured.export(str(tp))
        m['files']['textured_glb'] = str(tp)
        m['files']['textured_glb_kb'] = round(tp.stat().st_size / 1024, 1)
        print(f"  OK  {t:.1f}s  |  VRAM peak: {pk:.0f} MB")
    except Exception as e:
        m['error'] = f'texture: {e}'
        import traceback; traceback.print_exc()

    m['timing']['total_s'] = round(sum(v for k,v in m['timing'].items() if k.endswith('_s')), 2)
    _save(m, d, subj)
    print(f"\n  Totale: {m['timing']['total_s']:.1f}s")
    return m


def _save(m, d, subj):
    p = d / f'{subj}_metrics.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    m['files']['metrics_json'] = str(p)


def main():
    global _SEED
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='2.1')
    parser.add_argument('--input_dir', default='benchmark/inputs')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--sequential', action='store_true')
    parser.add_argument('--no_texture', action='store_true')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--target_faces', type=int, default=40000)
    args = parser.parse_args()

    _SEED = args.seed

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERRORE: input_dir non trovata: {input_dir}"); sys.exit(1)

    images = sorted(p for p in input_dir.iterdir()
                    if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'})
    if not images:
        print(f"ERRORE: nessuna immagine in {input_dir}"); sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else Path('benchmark/outputs/2.1')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*60)
    print('HUNYUAN3D BENCHMARK — 2.1 PBR')
    print('='*60)
    print(f"GPU        : {gpu_info()['name']}  ({gpu_info()['total_mb']:.0f} MB)")
    print(f"Sequential : {args.sequential}")
    print(f"Soggetti   : {len(images)}")
    print(f"Output     : {out_dir}")
    print('='*60)

    results = []
    for img in images:
        results.append(run_subject(img, out_dir, args))

    vals = lambda k: [r['timing'].get(k) for r in results if r['timing'].get(k)]
    avg = lambda lst: round(sum(lst)/len(lst), 2) if lst else None

    summary = {
        'variant': '2.1', 'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'gpu': gpu_info(), 'seed': args.seed,
        'subjects': results,
        'totals': {
            'n_subjects': len(results),
            'n_errors': sum(1 for r in results if r.get('error')),
            'avg_shape_s': avg(vals('shape_s')),
            'avg_texture_s': avg(vals('texture_s')),
            'avg_total_s': avg(vals('total_s')),
            'avg_vram_shape_mb': avg([r['vram_peak_mb'].get('shape') for r in results if r['vram_peak_mb'].get('shape')]),
            'avg_vram_texture_mb': avg([r['vram_peak_mb'].get('texture') for r in results if r['vram_peak_mb'].get('texture')]),
        },
    }
    sp = out_dir / 'run_summary.json'
    with open(sp, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    t = summary['totals']
    print('\n' + '='*60 + '\nSESSIONE COMPLETATA\n' + '='*60)
    print(f"Soggetti: {t['n_subjects']}  |  Errori: {t['n_errors']}")
    if t['avg_shape_s']:   print(f"Avg shape:   {t['avg_shape_s']:.1f}s")
    if t['avg_texture_s']: print(f"Avg texture: {t['avg_texture_s']:.1f}s")
    if t['avg_total_s']:   print(f"Avg totale:  {t['avg_total_s']:.1f}s")
    print(f"Riepilogo: {sp}")
    print('='*60)


if __name__ == '__main__':
    main()
PYEOF

echo "  Script benchmark 2.1 creato in run/benchmark_runpod.py"

# ── 4. Struttura benchmark ───────────────────────────────────────────────────
echo ""
echo "[4/6] Setup struttura benchmark..."
mkdir -p "$BENCH_INPUTS"
mkdir -p "$BENCH_OUTPUTS/2.1"

# ── 5. Istruzioni per caricare gli input ────────────────────────────────────
echo ""
echo "============================================================"
echo "  [5/6] AZIONE RICHIESTA: carica le immagini di test"
echo "============================================================"
echo ""
echo "  Da eseguire sulla tua macchina locale:"
echo "  rsync -avz -e 'ssh -p <PORT>' benchmark/inputs/ root@<HOST>:$BENCH_INPUTS/"
echo ""
echo "  Attendo 30 secondi, poi controllo..."
sleep 30

N_IMGS=$(find "$BENCH_INPUTS" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
echo "  Trovate $N_IMGS immagini"

if [ "$N_IMGS" -eq 0 ]; then
    echo ""
    echo "  Nessuna immagine. Carica e poi esegui manualmente:"
    if [ "$SEQUENTIAL" -eq 1 ]; then
        echo "  cd $REPO_DIR && python run/benchmark_runpod.py --sequential --input_dir benchmark/inputs"
    else
        echo "  cd $REPO_DIR && python run/benchmark_runpod.py --input_dir benchmark/inputs"
    fi
    exit 0
fi

# ── 6. Esegui benchmark ──────────────────────────────────────────────────────
echo ""
echo "[6/6] Avvio benchmark 2.1 PBR..."
cd "$REPO_DIR"

SEQ_FLAG=""
[ "$SEQUENTIAL" -eq 1 ] && SEQ_FLAG="--sequential"

python run/benchmark_runpod.py \
    --variant 2.1 \
    --input_dir benchmark/inputs \
    --seed 1234 \
    $SEQ_FLAG

echo ""
echo "============================================================"
echo "  COMPLETATO. Risultati in: $BENCH_OUTPUTS/2.1/"
echo "============================================================"
echo ""
echo "  Per scaricare i risultati:"
echo "  rsync -avz -e 'ssh -p <PORT>' root@<HOST>:$BENCH_OUTPUTS/2.1/ benchmark/outputs/2.1/"
echo ""