#!/usr/bin/env python3
"""
Hunyuan3D Benchmark Runner — variante 2.1 PBR
Da eseguire nel repo Tencent-Hunyuan/Hunyuan3D-2.1 (copiato da setup_runpod_21.sh)

  - shape:   tencent/Hunyuan3D-2.1  subfolder=hunyuan3d-dit-v2-1
  - texture: tencent/Hunyuan3D-2.1  subfolder=hunyuan3d-paint-v2-1  (PBR)

Usage:
    python run/benchmark_runpod.py --input_dir benchmark/inputs --seed 1234
    python run/benchmark_runpod.py --input_dir benchmark/inputs --seed 1234 --sequential
"""

import argparse, gc, json, sys, time
from datetime import datetime
from pathlib import Path

# ── path setup per repo 2.1 (hy3dshape / hy3dpaint) ────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'hy3dshape'))
sys.path.insert(0, str(_REPO_ROOT / 'hy3dpaint'))

import torch

# Applica fix torchvision per compatibilità 2.1
try:
    from torchvision_fix import apply_fix
    apply_fix()
except Exception:
    pass

CONFIG = {
    'description': 'Hunyuan3D-2.1 — PBR (Albedo + Normal + Roughness + Metallic)',
    'shape_model_path': 'tencent/Hunyuan3D-2.1',
    'shape_subfolder': 'hunyuan3d-dit-v2-1',
    'enable_flashvdm': False,
    'num_inference_steps': 50,
    'octree_resolution': 512,
    'num_chunks': 200000,
    'guidance_scale': 7.5,
    'has_pbr': True,
    # texture — path locali nel repo
    'paint_multiview_cfg': str(_REPO_ROOT / 'hy3dpaint/cfgs/hunyuan-paint-pbr.yaml'),
    'paint_custom_pipeline': str(_REPO_ROOT / 'hy3dpaint/hunyuanpaintpbr'),
    'paint_realesrgan_ckpt': str(_REPO_ROOT / 'hy3dpaint/ckpt/RealESRGAN_x4plus.pth'),
    'paint_max_views': 6,
    'paint_resolution': 512,
}

_SEED = 1234


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def peak_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def gpu_info():
    if not torch.cuda.is_available():
        return {'name': 'CPU only', 'total_mb': 0}
    p = torch.cuda.get_device_properties(0)
    return {'name': p.name, 'total_mb': p.total_memory / 1024 ** 2}


def run_rembg(image_path):
    from hy3dshape.rembg import BackgroundRemover
    from PIL import Image
    t0 = time.time()
    image = Image.open(image_path).convert('RGB')
    r = BackgroundRemover()
    out = r(image)
    del r; clear_memory()
    return out, time.time() - t0


def run_shape(image, sequential):
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    kw = {'subfolder': CONFIG['shape_subfolder']}
    if sequential:
        kw['device'] = 'cpu'

    print("  Caricamento shape model (2.1)...")
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        CONFIG['shape_model_path'], **kw
    )
    if sequential:
        pipe.to('cuda')

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    gen = torch.Generator(device='cuda').manual_seed(_SEED)
    mesh = pipe(
        image=image,
        num_inference_steps=CONFIG['num_inference_steps'],
        guidance_scale=CONFIG['guidance_scale'],
        generator=gen,
        octree_resolution=CONFIG['octree_resolution'],
        num_chunks=CONFIG['num_chunks'],
    )[0]
    elapsed = time.time() - t0
    pk = peak_mb()
    del pipe; clear_memory()
    return mesh, elapsed, pk


def run_face_reduce(mesh, target=40000):
    try:
        from hy3dshape.utils.mesh import FaceReducer
    except ImportError:
        try:
            from hy3dshape import FaceReducer
        except ImportError:
            print("  FaceReducer non disponibile in 2.1, skip")
            return mesh, 0.0
    t0 = time.time()
    r = FaceReducer()
    mesh = r(mesh, target)
    del r; clear_memory()
    return mesh, time.time() - t0


def run_texture(mesh, image):
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    print("  Caricamento texture model PBR (2.1)...")
    conf = Hunyuan3DPaintConfig(
        CONFIG['paint_max_views'],
        CONFIG['paint_resolution'],
    )
    conf.realesrgan_ckpt_path = CONFIG['paint_realesrgan_ckpt']
    conf.multiview_cfg_path = CONFIG['paint_multiview_cfg']
    conf.custom_pipeline = CONFIG['paint_custom_pipeline']
    pipe = Hunyuan3DPaintPipeline(conf)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    textured = pipe(mesh, image)
    elapsed = time.time() - t0
    pk = peak_mb()
    del pipe; clear_memory()
    return textured, elapsed, pk


def _save(m, d, subj):
    p = d / f'{subj}_metrics.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    m['files']['metrics_json'] = str(p)


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
    print(f"  {subj}  |  2.1 PBR  |  sequential={args.sequential}")
    print(f"{'='*60}")

    print("[1/4] Background removal...")
    try:
        img, t = run_rembg(img_path)
        m['timing']['rembg_s'] = round(t, 2)
        p = d / f'{subj}_rembg.png'; img.save(p)
        m['files']['rembg_png'] = str(p)
        print(f"  OK  {t:.1f}s")
    except Exception as e:
        m['error'] = f'rembg: {e}'; _save(m, d, subj); return m

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
        m['timing']['total_s'] = round(sum(v for k, v in m['timing'].items() if k.endswith('_s')), 2)
        _save(m, d, subj); return m

    print(f"[3/4] Face reduction → {args.target_faces:,}...")
    try:
        mesh_red, t = run_face_reduce(mesh_raw, args.target_faces)
        m['timing']['face_reduce_s'] = round(t, 2)
        m['mesh']['faces_reduced'] = int(mesh_red.faces.shape[0])
        print(f"  {mesh_raw.faces.shape[0]:,} → {mesh_red.faces.shape[0]:,}")
    except Exception as e:
        m['error'] = f'face_reduce: {e}'; _save(m, d, subj); return m

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

    m['timing']['total_s'] = round(sum(v for k, v in m['timing'].items() if k.endswith('_s')), 2)
    _save(m, d, subj)
    print(f"\n  Totale: {m['timing']['total_s']:.1f}s")
    return m


def main():
    global _SEED
    parser = argparse.ArgumentParser()
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

    g = gpu_info()
    print('\n' + '='*60)
    print('HUNYUAN3D BENCHMARK — 2.1 PBR')
    print('='*60)
    print(f"GPU        : {g['name']}  ({g['total_mb']:.0f} MB)")
    print(f"Sequential : {args.sequential}")
    print(f"Soggetti   : {len(images)}")
    print(f"Output     : {out_dir}")
    print('='*60)

    results = []
    for img in images:
        results.append(run_subject(img, out_dir, args))

    avg = lambda lst: round(sum(lst) / len(lst), 2) if lst else None
    get = lambda k: [r['timing'][k] for r in results if r['timing'].get(k)]

    summary = {
        'variant': '2.1', 'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'gpu': gpu_info(), 'seed': args.seed,
        'subjects': results,
        'totals': {
            'n_subjects': len(results),
            'n_errors': sum(1 for r in results if r.get('error')),
            'avg_shape_s': avg(get('shape_s')),
            'avg_texture_s': avg(get('texture_s')),
            'avg_total_s': avg(get('total_s')),
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
