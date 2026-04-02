#!/usr/bin/env python3
"""
Test comparativo Hunyuan3D 2.0 full vs 2.1 PBR — singola immagine.

Usage:
    python run/test_compare.py --image benchmark/inputs/T1_apple.png
    python run/test_compare.py --image /path/to/img.png --no_texture
    python run/test_compare.py --image /path/to/img.png --only 2.1
    python run/test_compare.py --image /path/to/img.png --seed 42

Output: benchmark/outputs/compare/<nome_immagine>/
  2.0_full/  shape.glb | textured.glb | rembg.png
  2.1/       shape.glb | textured.glb | rembg.png
  compare_summary.json
"""

import argparse, gc, json, sys, time
from pathlib import Path

# ── Repo paths ────────────────────────────────────────────────────────────────
_REPO_20  = Path(__file__).resolve().parent.parent   # /workspace/Hunyuan3D-2
_REPO_21  = Path("/workspace/Hunyuan3D-2.1")

# torchvision fix PRIMA di qualsiasi import ML
sys.path.insert(0, str(_REPO_20))
try:
    from torchvision_fix import apply_fix
    apply_fix()
    print("torchvision_fix applicato")
except Exception:
    try:
        import torchvision.transforms.functional as _F, types as _t
        _m = _t.ModuleType('torchvision.transforms.functional_tensor')
        _m.rgb_to_grayscale = _F.rgb_to_grayscale
        sys.modules['torchvision.transforms.functional_tensor'] = _m
    except Exception as e:
        print(f"Warning torchvision patch: {e}")

import torch

# ── Configurazioni modelli ────────────────────────────────────────────────────
CONFIGS = {
    '2.0 full': {
        'shape_model':     'tencent/Hunyuan3D-2',
        'shape_subfolder': None,
        'texture_model':   'tencent/Hunyuan3D-2',
        'texture_subfolder': 'hunyuan3d-paint-v2-0',
        'steps': 50, 'octree': 512, 'num_chunks': 200000,
        'guidance': 7.5, 'has_pbr': False,
    },
    '2.1': {
        'shape_model':     'tencent/Hunyuan3D-2.1',
        'shape_subfolder': 'hunyuan3d-dit-v2-1',
        'paint_cfg':       str(_REPO_21 / 'hy3dpaint/cfgs/hunyuan-paint-pbr.yaml'),
        'paint_pipeline':  str(_REPO_21 / 'hy3dpaint/hunyuanpaintpbr'),
        'paint_realesrgan':str(_REPO_21 / 'hy3dpaint/ckpt/RealESRGAN_x4plus.pth'),
        'steps': 50, 'octree': 512, 'num_chunks': 200000,
        'guidance': 7.5, 'has_pbr': True,
    },
}

_SEED = 1234


# ── Utilities ─────────────────────────────────────────────────────────────────
def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

def gpu_info():
    if not torch.cuda.is_available():
        return 'CPU only'
    p = torch.cuda.get_device_properties(0)
    return f"{p.name} ({p.total_memory/1024**2:.0f}MB)"


# ── Background removal (usa rembg direttamente) ───────────────────────────────
def run_rembg(image_path: Path):
    from rembg import remove
    from PIL import Image
    import io
    img = Image.open(image_path).convert('RGB')
    buf = io.BytesIO(); img.save(buf, format='PNG')
    result_bytes = remove(buf.getvalue())
    return Image.open(io.BytesIO(result_bytes)).convert('RGBA')


# ── Shape 2.0 full ────────────────────────────────────────────────────────────
def run_shape_20(image, cfg):
    sys.path.insert(0, str(_REPO_20))
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    kw = {}
    if cfg['shape_subfolder']:
        kw['subfolder'] = cfg['shape_subfolder']

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(cfg['shape_model'], **kw)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    gen = torch.Generator(device='cuda').manual_seed(_SEED)
    mesh = pipe(
        image=image,
        num_inference_steps=cfg['steps'],
        guidance_scale=cfg['guidance'],
        generator=gen,
        octree_resolution=cfg['octree'],
        num_chunks=cfg['num_chunks'],
    )[0]
    elapsed = time.time() - t0
    pk = peak_mb()
    del pipe; clear()
    return mesh, elapsed, pk


# ── Shape 2.1 ─────────────────────────────────────────────────────────────────
def run_shape_21(image, cfg):
    sys.path.insert(0, str(_REPO_21 / 'hy3dshape'))

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        cfg['shape_model'],
        subfolder=cfg['shape_subfolder'],
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    gen = torch.Generator(device='cuda').manual_seed(_SEED)
    mesh = pipe(
        image=image,
        num_inference_steps=cfg['steps'],
        guidance_scale=cfg['guidance'],
        generator=gen,
        octree_resolution=cfg['octree'],
        num_chunks=cfg['num_chunks'],
    )[0]
    elapsed = time.time() - t0
    pk = peak_mb()
    del pipe; clear()
    return mesh, elapsed, pk


# ── Face reduction ────────────────────────────────────────────────────────────
def face_reduce(mesh, target=40000):
    import trimesh
    if len(mesh.faces) <= target:
        return mesh
    try:
        sys.path.insert(0, str(_REPO_21 / 'hy3dshape'))
        from hy3dshape.utils.mesh import FaceReducer
        r = FaceReducer(); m = r(mesh, target); del r; clear(); return m
    except Exception as e:
        print(f"  FaceReducer fallito ({e}), uso trimesh decimation...")
        return mesh.simplify_quadric_decimation(target)


# ── Texture 2.0 full ──────────────────────────────────────────────────────────
def run_texture_20(mesh_path: str, image, cfg):
    sys.path.insert(0, str(_REPO_20))
    from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline

    kw = {}
    if cfg['texture_subfolder']:
        kw['subfolder'] = cfg['texture_subfolder']

    pipe = Hunyuan3DPaintPipeline.from_pretrained(cfg['texture_model'], **kw)
    pipe.enable_model_cpu_offload()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    import trimesh as _tr
    mesh = _tr.load(mesh_path)
    result = pipe(mesh, image)
    elapsed = time.time() - t0
    pk = peak_mb()
    del pipe; clear()
    return result, elapsed, pk


# ── Texture 2.1 PBR ───────────────────────────────────────────────────────────
def run_texture_21(mesh_path: str, image, image_path: str, cfg):
    sys.path.insert(0, str(_REPO_21 / 'hy3dpaint'))
    import trimesh as _tr
    import textureGenPipeline as _tgp
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    # Patch remesh_mesh nel namespace del modulo (pymeshlab rotto senza libOpenGL)
    def _trimesh_remesh(input_path, output_path):
        _tr.load(input_path, force='mesh').export(output_path)
        return output_path
    _tgp.remesh_mesh = _trimesh_remesh

    conf = Hunyuan3DPaintConfig(6, 512)
    conf.realesrgan_ckpt_path = cfg['paint_realesrgan']
    conf.multiview_cfg_path   = cfg['paint_cfg']
    conf.custom_pipeline      = cfg['paint_pipeline']
    pipe = Hunyuan3DPaintPipeline(conf)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        result = pipe(mesh_path, image)
    except TypeError:
        print("  pipe(PIL) fallito, riprovo con path immagine...")
        result = pipe(mesh_path, image_path)
    elapsed = time.time() - t0
    pk = peak_mb()
    del pipe; clear()

    if isinstance(result, (str, Path)):
        result = _tr.load(str(result))
    return result, elapsed, pk


# ── Runner per singola variante ───────────────────────────────────────────────
def run_variant(variant: str, img_path: Path, out_dir: Path, args):
    import trimesh
    cfg = CONFIGS[variant]
    tag = variant.replace(' ', '_').replace('.', '')
    d = out_dir / tag
    d.mkdir(parents=True, exist_ok=True)

    m = {
        'variant': variant,
        'image': str(img_path),
        'seed': _SEED,
        'timestamp': '',
        'gpu': gpu_info(),
        'timing': {}, 'vram_mb': {}, 'mesh': {}, 'files': {},
        'error': None,
    }

    print(f"\n{'='*55}")
    print(f"  {variant}  |  {'PBR' if cfg['has_pbr'] else 'RGB'}")
    print(f"{'='*55}")

    # [1] Background removal
    print("[1/4] Background removal...")
    try:
        t0 = time.time()
        img = run_rembg(img_path)
        m['timing']['rembg_s'] = round(time.time() - t0, 2)
        p = d / 'rembg.png'; img.save(p)
        m['files']['rembg'] = str(p)
        print(f"  OK  {m['timing']['rembg_s']:.1f}s")
    except Exception as e:
        m['error'] = f'rembg: {e}'; return m

    # [2] Shape
    print(f"[2/4] Shape ({cfg['steps']} steps, octree={cfg['octree']})...")
    try:
        fn = run_shape_21 if variant == '2.1' else run_shape_20
        mesh, t, pk = fn(img, cfg)
        m['timing']['shape_s'] = round(t, 2)
        m['vram_mb']['shape']   = round(pk, 1)
        m['mesh']['faces_raw']  = len(mesh.faces)
        m['mesh']['verts_raw']  = len(mesh.vertices)
        sp = d / 'shape.glb'; mesh.export(str(sp))
        m['files']['shape'] = str(sp)
        print(f"  OK  {t:.1f}s  |  {len(mesh.faces):,} facce  |  VRAM peak: {pk:.0f}MB")
    except Exception as e:
        import traceback; traceback.print_exc()
        m['error'] = f'shape: {e}'; return m

    if args.no_texture:
        m['timing']['total_s'] = round(m['timing'].get('shape_s', 0), 2)
        return m

    # [3] Face reduction
    print(f"[3/4] Face reduction → 40k...")
    try:
        mesh_red = face_reduce(mesh)
        m['mesh']['faces_reduced'] = len(mesh_red.faces)
        rp = d / 'reduced.glb'; mesh_red.export(str(rp))
        print(f"  {len(mesh.faces):,} → {len(mesh_red.faces):,}")
    except Exception as e:
        m['error'] = f'face_reduce: {e}'; return m

    # [4] Texture
    print("[4/4] Texture generation...")
    try:
        if variant == '2.1':
            textured, t, pk = run_texture_21(str(rp), img, m['files']['rembg'], cfg)
        else:
            textured, t, pk = run_texture_20(str(rp), img, cfg)
        m['timing']['texture_s'] = round(t, 2)
        m['vram_mb']['texture']   = round(pk, 1)
        tp = d / 'textured.glb'; textured.export(str(tp))
        m['files']['textured'] = str(tp)
        print(f"  OK  {t:.1f}s  |  VRAM peak: {pk:.0f}MB")
    except Exception as e:
        import traceback; traceback.print_exc()
        m['error'] = f'texture: {e}'

    m['timing']['total_s'] = round(
        sum(v for k, v in m['timing'].items() if k.endswith('_s')), 2
    )
    return m


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _SEED
    parser = argparse.ArgumentParser(
        description='Confronto Hunyuan3D 2.0 full vs 2.1 su singola immagine'
    )
    parser.add_argument('--image',      required=True,   help='Path immagine input')
    parser.add_argument('--output_dir', default='benchmark/outputs/compare')
    parser.add_argument('--no_texture', action='store_true')
    parser.add_argument('--only',       choices=['2.0 full', '2.1'], default=None,
                        help='Esegui solo una variante')
    parser.add_argument('--seed',       type=int, default=1234)
    args = parser.parse_args()
    _SEED = args.seed

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"ERRORE: immagine non trovata: {img_path}")
        sys.exit(1)

    from datetime import datetime
    out_dir = Path(args.output_dir) / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [args.only] if args.only else list(CONFIGS.keys())

    print('\n' + '='*55)
    print('HUNYUAN3D — TEST COMPARATIVO')
    print('='*55)
    print(f"Immagine  : {img_path.name}")
    print(f"Varianti  : {', '.join(variants)}")
    print(f"GPU       : {gpu_info()}")
    print(f"Output    : {out_dir}")
    print('='*55)

    results = {}
    for v in variants:
        results[v] = run_variant(v, img_path, out_dir, args)

    # Riepilogo
    print('\n' + '='*55 + '\nRIEPILOGO\n' + '='*55)
    for v, m in results.items():
        if m['error']:
            print(f"  {v:12s}  ERRORE: {m['error']}")
        else:
            s  = m['timing'].get('shape_s', 0)
            t  = m['timing'].get('texture_s', 0)
            vs = m['vram_mb'].get('shape', 0)
            vt = m['vram_mb'].get('texture', 0)
            f  = m['mesh'].get('faces_raw', 0)
            print(f"  {v:12s}  shape: {s:.0f}s ({vs:.0f}MB)  "
                  f"texture: {t:.0f}s ({vt:.0f}MB)  facce: {f:,}")

    summary_path = out_dir / 'compare_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'timestamp': datetime.now().isoformat(),
                   'image': str(img_path),
                   'results': results}, f, indent=2, ensure_ascii=False)
    print(f"\n  Riepilogo: {summary_path}")
    print('='*55)


if __name__ == '__main__':
    main()
