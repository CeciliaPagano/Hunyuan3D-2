#!/usr/bin/env python3
"""
TRELLIS.2 Benchmark Runner
Da eseguire nel repo Microsoft TRELLIS.2 con l'env conda 'trellis2' attivo.

Pipeline: image → shape+texture PBR unificata → GLB

Usage:
    python benchmark_trellis.py --input_dir /workspace/Hunyuan3D-2/benchmark/inputs --seed 1234
    python benchmark_trellis.py --input_dir /path/to/imgs --pipeline_type 512 --seed 1234
    python benchmark_trellis.py --input_dir /path/to/imgs --only T1_apple --seed 1234
    python benchmark_trellis.py --input_dir /path/to/imgs --no_texture --seed 1234

pipeline_type disponibili:
    512           — risoluzione 512³, ~3s su H100 (più veloce)
    1024          — risoluzione 1024³, single-stage
    1024_cascade  — risoluzione 1024³, two-stage 512→1024 (DEFAULT, miglior qualità/velocità)
    1536_cascade  — risoluzione 1536³, massima qualità (~60s su H100)
"""

import argparse, gc, json, os, sys, time
from datetime import datetime
from pathlib import Path

# Env vars obbligatorie per TRELLIS.2
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch

MODEL_ID = 'microsoft/TRELLIS.2-4B'
TEXTURE_SIZE = 1024   # 1024 per benchmark (4096 per qualità massima)
DECIMATION_TARGET = 500000  # target poligoni per export GLB


def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0


def gpu_info():
    if not torch.cuda.is_available():
        return {'name': 'CPU only', 'total_mb': 0}
    p = torch.cuda.get_device_properties(0)
    return {'name': p.name, 'total_mb': round(p.total_memory / 1024**2)}


def load_pipeline():
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    print(f"  Caricamento modello {MODEL_ID}...")
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
    pipe.cuda()
    return pipe


def export_glb(mesh, pipeline, output_path: str):
    import o_voxel
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=DECIMATION_TARGET,
        texture_size=TEXTURE_SIZE,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
    )
    glb.export(output_path, extension_webp=True)


def run_subject(img_path: Path, out_dir: Path, pipeline, args):
    subj = img_path.stem
    d = out_dir / subj
    d.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    m = {
        'variant': 'trellis2',
        'subject': subj,
        'input_image': str(img_path),
        'seed': args.seed,
        'pipeline_type': args.pipeline_type,
        'texture_size': TEXTURE_SIZE,
        'timestamp': datetime.now().isoformat(),
        'gpu': gpu_info(),
        'timing': {}, 'vram_peak_mb': {}, 'mesh': {}, 'files': {},
        'quality': {
            'geometry': None, 'texture': None,
            'artifacts': None, 'pbr_quality': None, 'notes': '',
        },
        'error': None,
    }

    print(f"\n{'='*60}")
    print(f"  {subj}  |  TRELLIS.2  |  {args.pipeline_type}")
    print(f"{'='*60}")

    # ── Carica immagine ───────────────────────────────────────────────────────
    try:
        image = Image.open(img_path).convert('RGB')
        p = d / f'{subj}_input.png'; image.save(p)
        m['files']['input_png'] = str(p)
    except Exception as e:
        m['error'] = f'load_image: {e}'
        return m

    # ── Generazione 3D (shape + texture unificati) ───────────────────────────
    print(f"[1/2] Generazione 3D (pipeline_type={args.pipeline_type})...")
    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        meshes = pipeline.run(
            image,
            seed=args.seed,
            pipeline_type=args.pipeline_type,
            num_samples=1,
            preprocess_image=True,   # TRELLIS.2 gestisce internamente il preprocessing
        )
        mesh = meshes[0]

        elapsed = time.time() - t0
        pk = peak_mb()

        m['timing']['generate_s'] = round(elapsed, 2)
        m['vram_peak_mb']['generate'] = round(pk, 1)
        m['mesh']['faces'] = int(mesh.faces.shape[0]) if hasattr(mesh.faces, 'shape') else 0
        m['mesh']['verts'] = int(mesh.vertices.shape[0]) if hasattr(mesh.vertices, 'shape') else 0

        print(f"  OK  {elapsed:.1f}s  |  "
              f"{m['mesh']['faces']:,} facce  |  VRAM peak: {pk:.0f}MB")

    except Exception as e:
        import traceback; traceback.print_exc()
        m['error'] = f'generate: {e}'
        _save(m, d, subj)
        return m

    if args.no_texture:
        m['timing']['total_s'] = m['timing']['generate_s']
        _save(m, d, subj)
        return m

    # ── Export GLB ────────────────────────────────────────────────────────────
    print(f"[2/2] Export GLB (texture_size={TEXTURE_SIZE})...")
    try:
        t0 = time.time()
        glb_path = str(d / f'{subj}_textured.glb')
        export_glb(mesh, pipeline, glb_path)
        elapsed_export = time.time() - t0

        m['timing']['export_s'] = round(elapsed_export, 2)
        m['files']['textured_glb'] = glb_path
        size_kb = round(Path(glb_path).stat().st_size / 1024, 1)
        m['files']['textured_glb_kb'] = size_kb
        print(f"  OK  {elapsed_export:.1f}s  |  {size_kb:.0f} KB")

    except Exception as e:
        import traceback; traceback.print_exc()
        m['error'] = f'export_glb: {e}'

    m['timing']['total_s'] = round(
        sum(v for k, v in m['timing'].items() if k.endswith('_s')), 2
    )
    print(f"\n  Totale: {m['timing']['total_s']:.1f}s")
    _save(m, d, subj)
    return m


def _save(m, d, subj):
    p = d / f'{subj}_metrics.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    m['files']['metrics_json'] = str(p)


def main():
    parser = argparse.ArgumentParser(description='TRELLIS.2 Benchmark')
    parser.add_argument('--input_dir',     default='benchmark/inputs')
    parser.add_argument('--output_dir',    default=None)
    parser.add_argument('--pipeline_type', default='1024_cascade',
                        choices=['512', '1024', '1024_cascade', '1536_cascade'])
    parser.add_argument('--seed',          type=int, default=1234)
    parser.add_argument('--no_texture',    action='store_true',
                        help='Genera solo la mesh, salta export GLB con texture')
    parser.add_argument('--only',          default=None,
                        help='Esegui solo su un soggetto (es. T1_apple)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERRORE: input_dir non trovata: {input_dir}")
        sys.exit(1)

    images = sorted(p for p in input_dir.iterdir()
                    if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'})
    if args.only:
        images = [p for p in images if args.only in p.stem]
    if not images:
        print(f"ERRORE: nessuna immagine trovata in {input_dir}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else \
              Path('/workspace/Hunyuan3D-2/benchmark/outputs/trellis2')
    out_dir.mkdir(parents=True, exist_ok=True)

    g = gpu_info()
    print('\n' + '='*60)
    print('TRELLIS.2 BENCHMARK')
    print('='*60)
    print(f"Modello       : {MODEL_ID}")
    print(f"Pipeline type : {args.pipeline_type}")
    print(f"GPU           : {g['name']}  ({g['total_mb']} MB)")
    print(f"Soggetti      : {len(images)}")
    print(f"Output        : {out_dir}")
    print('='*60)

    # Carica pipeline UNA volta sola (ricarica per ogni soggetto è lenta)
    print("\nCaricamento pipeline...")
    pipeline = load_pipeline()
    clear()
    print("  Pipeline pronta.\n")

    results = []
    for img in images:
        r = run_subject(img, out_dir, pipeline, args)
        results.append(r)
        clear()

    # Summary
    ok = [r for r in results if not r.get('error')]
    avg = lambda lst: round(sum(lst)/len(lst), 2) if lst else None

    summary = {
        'variant': 'trellis2',
        'model_id': MODEL_ID,
        'pipeline_type': args.pipeline_type,
        'texture_size': TEXTURE_SIZE,
        'timestamp': datetime.now().isoformat(),
        'gpu': gpu_info(),
        'seed': args.seed,
        'subjects': results,
        'totals': {
            'n_subjects':       len(results),
            'n_errors':         sum(1 for r in results if r.get('error')),
            'avg_generate_s':   avg([r['timing'].get('generate_s') for r in ok if r['timing'].get('generate_s')]),
            'avg_export_s':     avg([r['timing'].get('export_s')   for r in ok if r['timing'].get('export_s')]),
            'avg_total_s':      avg([r['timing'].get('total_s')    for r in ok if r['timing'].get('total_s')]),
            'avg_vram_mb':      avg([r['vram_peak_mb'].get('generate') for r in ok if r['vram_peak_mb'].get('generate')]),
        },
    }
    sp = out_dir / 'run_summary.json'
    with open(sp, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    t = summary['totals']
    print('\n' + '='*60 + '\nSESSIONE COMPLETATA\n' + '='*60)
    print(f"Soggetti: {t['n_subjects']}  |  Errori: {t['n_errors']}")
    if t['avg_generate_s']: print(f"Avg generate: {t['avg_generate_s']:.1f}s")
    if t['avg_export_s']:   print(f"Avg export:   {t['avg_export_s']:.1f}s")
    if t['avg_total_s']:    print(f"Avg totale:   {t['avg_total_s']:.1f}s")
    if t['avg_vram_mb']:    print(f"Avg VRAM:     {t['avg_vram_mb']:.0f}MB")
    print(f"Riepilogo: {sp}")
    print('='*60)


if __name__ == '__main__':
    main()
