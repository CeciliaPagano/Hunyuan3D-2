#!/usr/bin/env python3
"""
Hunyuan3D Benchmark Runner — confronto modelli mini / 2.0 / 2.1 / 2.5

Raccoglie per ogni soggetto:
  - timing separati: rembg | shape | texture
  - peak VRAM per fase
  - statistiche mesh: face/vertex count, dimensione file
  - slot qualità manuale (da compilare dopo l'ispezione visiva)

Output per soggetto: *_rembg.png | *_shape.glb | *_textured.glb | *_metrics.json
Output di sessione: run_summary.json

──────────────────────────────────────────────────────────────────────────────
QUALE REPO USARE SU RUNPOD:

  mini / 2.0  →  questo stesso repo (Tencent-Hunyuan/Hunyuan3D-2)
  2.1         →  clonare Tencent-Hunyuan/Hunyuan3D-2.1, poi usare questo script
  2.5         →  clonare il repo 2.5 (verificare su HuggingFace/GitHub), poi usare questo script

Dopo aver clonato il repo corretto su RunPod, copiare questo script nella
cartella run/ del repo clonato e lanciarlo da lì.
──────────────────────────────────────────────────────────────────────────────

Usage:
    # 1. Baseline locale (mini)
    python run/benchmark_runpod.py --variant mini --input_dir benchmark/inputs

    # 2. RunPod — Hunyuan3D 2.0 (RTX 3090 24GB, stesso repo)
    python run/benchmark_runpod.py --variant 2.0 --input_dir benchmark/inputs

    # 3. RunPod — Hunyuan3D 2.1 (da repo 2.1, A100 40GB o --sequential su 24GB)
    python run/benchmark_runpod.py --variant 2.1 --sequential --input_dir benchmark/inputs

    # 4. RunPod — Hunyuan3D 2.5 (da repo 2.5, RTX 4090 24GB)
    python run/benchmark_runpod.py --variant 2.5 --input_dir benchmark/inputs

    # Confronto risultati (legge i JSON di output_dir e stampa tabella)
    python run/benchmark_runpod.py --compare benchmark/outputs
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Configurazioni per variante
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    'mini': {
        'description': 'Hunyuan3D-2 mini 0.6B Turbo — baseline locale 8GB VRAM',
        'shape_model_path': 'tencent/Hunyuan3D-2mini',
        'shape_subfolder': 'hunyuan3d-dit-v2-mini-turbo',
        'texture_model_path': 'tencent/Hunyuan3D-2',
        'texture_subfolder': 'hunyuan3d-paint-v2-0-turbo',
        'enable_flashvdm': True,
        'flashvdm_topk_mode': 'mean',
        'num_inference_steps': 5,
        'octree_resolution': 380,
        'num_chunks': 20000,
        'has_pbr': False,
        'sequential_default': True,
    },
    '2.0': {
        'description': 'Hunyuan3D-2.0 full 2B — texture RGB, no PBR',
        'shape_model_path': 'tencent/Hunyuan3D-2',
        'shape_subfolder': None,
        'texture_model_path': 'tencent/Hunyuan3D-2',
        'texture_subfolder': 'hunyuan3d-paint-v2-0-turbo',
        'enable_flashvdm': False,
        'flashvdm_topk_mode': None,
        'num_inference_steps': 50,
        'octree_resolution': 512,
        'num_chunks': 200000,
        'has_pbr': False,
        'sequential_default': False,   # 6+16=22GB, ok su RTX 3090 24GB
    },
    '2.1': {
        'description': 'Hunyuan3D-2.1 — PBR completo (Albedo + Normal + Roughness + Metallic)',
        # ⚠️  Eseguire dal repo Tencent-Hunyuan/Hunyuan3D-2.1
        # ⚠️  Verificare texture_subfolder su HuggingFace: tencent/Hunyuan3D-2.1
        'shape_model_path': 'tencent/Hunyuan3D-2.1',
        'shape_subfolder': 'hunyuan3d-dit-v2-1',
        'texture_model_path': 'tencent/Hunyuan3D-2.1',
        'texture_subfolder': 'hunyuan3d-paint-v2-1',   # TODO: verificare su HF
        'enable_flashvdm': False,
        'flashvdm_topk_mode': None,
        'num_inference_steps': 50,
        'octree_resolution': 512,
        'num_chunks': 200000,
        'has_pbr': True,
        'sequential_default': True,    # 10+21=31GB, serve sequential su GPU <40GB
    },
    '2.5': {
        'description': 'Hunyuan3D-2.5 LATTICE — risoluzione geometrica 1024',
        # ⚠️  Verificare i path su HuggingFace prima di avviare
        # ⚠️  Eseguire dal repo corretto (verificare su GitHub/HuggingFace)
        'shape_model_path': 'tencent/Hunyuan3D-2.5',
        'shape_subfolder': None,       # TODO: verificare su HuggingFace
        'texture_model_path': 'tencent/Hunyuan3D-2.5',
        'texture_subfolder': None,     # TODO: verificare su HuggingFace
        'enable_flashvdm': False,
        'flashvdm_topk_mode': None,
        'num_inference_steps': 50,
        'octree_resolution': 512,
        'num_chunks': 200000,
        'has_pbr': True,
        'sequential_default': False,   # 6+16=22GB, ok su RTX 4090 24GB
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024 ** 2


def reset_peak_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {'name': 'CPU only', 'total_mb': 0}
    props = torch.cuda.get_device_properties(0)
    return {
        'name': props.name,
        'total_mb': props.total_memory / 1024 ** 2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage: rimozione sfondo
# ─────────────────────────────────────────────────────────────────────────────

def run_rembg(image_path: Path):
    from hy3dgen.rembg import BackgroundRemover
    from PIL import Image

    t0 = time.time()
    image = Image.open(image_path).convert('RGB')
    rmbg = BackgroundRemover()
    image_rgba = rmbg(image)
    del rmbg
    clear_memory()
    elapsed = time.time() - t0

    return image_rgba, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Stage: generazione shape
# ─────────────────────────────────────────────────────────────────────────────

def run_shape_gen(image, config: dict, sequential: bool):
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen.pipelines import export_to_trimesh

    from_pretrained_kwargs = {}
    if config['shape_subfolder']:
        from_pretrained_kwargs['subfolder'] = config['shape_subfolder']
    if sequential:
        from_pretrained_kwargs['device'] = 'cpu'

    print("  Caricamento shape model...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        config['shape_model_path'],
        **from_pretrained_kwargs,
    )

    if config['enable_flashvdm']:
        pipeline.enable_flashvdm(topk_mode=config['flashvdm_topk_mode'])

    if sequential:
        pipeline.to(device='cuda')

    reset_peak_vram()
    t0 = time.time()

    generator = torch.Generator(device='cuda').manual_seed(
        _CURRENT_SEED if '_CURRENT_SEED' in dir() else 1234
    )
    outputs = pipeline(
        image=image,
        num_inference_steps=config['num_inference_steps'],
        guidance_scale=7.5,
        generator=generator,
        octree_resolution=config['octree_resolution'],
        num_chunks=config['num_chunks'],
        output_type='mesh',
    )

    mesh = export_to_trimesh(outputs)[0]
    elapsed = time.time() - t0
    peak = peak_vram_mb()

    del pipeline
    clear_memory()

    return mesh, elapsed, peak


# ─────────────────────────────────────────────────────────────────────────────
# Stage: riduzione facce (necessario prima di texture)
# ─────────────────────────────────────────────────────────────────────────────

def run_face_reduce(mesh, target_faces: int = 40000):
    from hy3dgen.shapegen import FaceReducer

    t0 = time.time()
    reducer = FaceReducer()
    mesh = reducer(mesh, target_faces)
    del reducer
    clear_memory()

    return mesh, time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Stage: generazione texture
# ─────────────────────────────────────────────────────────────────────────────

def run_texture_gen(mesh, image, config: dict):
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    from_pretrained_kwargs = {}
    if config['texture_subfolder']:
        from_pretrained_kwargs['subfolder'] = config['texture_subfolder']

    print("  Caricamento texture model...")
    pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        config['texture_model_path'],
        **from_pretrained_kwargs,
    )

    try:
        pipeline.enable_model_cpu_offload()
        print("  CPU offloading abilitato per texture model")
    except Exception as e:
        print(f"  Warning: CPU offload non disponibile: {e}")

    reset_peak_vram()
    t0 = time.time()
    textured_mesh = pipeline(mesh, image)
    elapsed = time.time() - t0
    peak = peak_vram_mb()

    del pipeline
    clear_memory()

    return textured_mesh, elapsed, peak


# ─────────────────────────────────────────────────────────────────────────────
# Raccolta statistiche mesh
# ─────────────────────────────────────────────────────────────────────────────

def mesh_stats(mesh) -> dict:
    stats = {
        'faces': int(mesh.faces.shape[0]),
        'vertices': int(mesh.vertices.shape[0]),
    }
    try:
        bb = mesh.bounds
        stats['bbox_x'] = float(round(bb[1][0] - bb[0][0], 4))
        stats['bbox_y'] = float(round(bb[1][1] - bb[0][1], 4))
        stats['bbox_z'] = float(round(bb[1][2] - bb[0][2], 4))
    except Exception:
        pass
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Runner per singolo soggetto
# ─────────────────────────────────────────────────────────────────────────────

def run_subject(image_path: Path, config: dict, output_dir: Path, args) -> dict:
    global _CURRENT_SEED
    _CURRENT_SEED = args.seed

    subject = image_path.stem
    subj_dir = output_dir / subject
    subj_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'variant': args.variant,
        'subject': subject,
        'input_image': str(image_path),
        'seed': args.seed,
        'inference_steps': config['num_inference_steps'],
        'octree_resolution': config['octree_resolution'],
        'has_pbr': config['has_pbr'],
        'sequential': args.sequential,
        'timestamp': datetime.now().isoformat(),
        'gpu': gpu_info(),
        'timing': {},
        'vram_peak_mb': {},
        'mesh': {},
        'files': {},
        'quality': {
            'geometry': None,   # 1-5 — da compilare manualmente dopo ispezione visiva
            'texture': None,    # 1-5
            'artifacts': None,  # 1-5 (5 = nessun artefatto)
            'janus': None,      # true/false (true = problema presente)
            'notes': '',
        },
        'error': None,
    }

    print(f"\n{'='*60}")
    print(f"Soggetto: {subject}  |  Variante: {args.variant}")
    print(f"{'='*60}")

    # ── Stage 0: rembg ──────────────────────────────────────────────────────
    print("[1/4] Background removal...")
    try:
        image, t_rembg = run_rembg(image_path)
        metrics['timing']['rembg_s'] = round(t_rembg, 2)
        rembg_path = subj_dir / f'{subject}_rembg.png'
        image.save(rembg_path)
        metrics['files']['rembg_png'] = str(rembg_path)
        print(f"  Completato in {t_rembg:.1f}s")
    except Exception as e:
        metrics['error'] = f'rembg: {e}'
        print(f"  ERRORE: {e}")
        return metrics

    # ── Stage 1: shape ───────────────────────────────────────────────────────
    print(f"[2/4] Shape generation ({config['num_inference_steps']} steps, "
          f"octree={config['octree_resolution']})...")
    try:
        mesh_raw, t_shape, peak_shape = run_shape_gen(image, config, args.sequential)
        metrics['timing']['shape_s'] = round(t_shape, 2)
        metrics['vram_peak_mb']['shape'] = round(peak_shape, 1)
        raw_stats = mesh_stats(mesh_raw)
        metrics['mesh']['faces_raw'] = raw_stats['faces']
        metrics['mesh']['verts_raw'] = raw_stats['vertices']

        shape_path = subj_dir / f'{subject}_shape.glb'
        mesh_raw.export(str(shape_path))
        metrics['files']['shape_glb'] = str(shape_path)
        metrics['files']['shape_glb_kb'] = round(shape_path.stat().st_size / 1024, 1)
        print(f"  Shape completata in {t_shape:.1f}s  |  "
              f"{raw_stats['faces']:,} facce  |  "
              f"VRAM peak: {peak_shape:.0f} MB")
    except Exception as e:
        metrics['error'] = f'shape: {e}'
        print(f"  ERRORE shape: {e}")
        import traceback; traceback.print_exc()
        return metrics

    if args.no_texture:
        metrics['timing']['total_s'] = round(
            metrics['timing'].get('rembg_s', 0) + metrics['timing'].get('shape_s', 0), 2
        )
        _save_metrics(metrics, subj_dir, subject)
        print("  (texture saltata — --no_texture)")
        return metrics

    # ── Stage 2: riduzione facce ─────────────────────────────────────────────
    print(f"[3/4] Face reduction → {args.target_faces:,} facce...")
    try:
        mesh_reduced, t_reduce = run_face_reduce(mesh_raw, args.target_faces)
        metrics['timing']['face_reduce_s'] = round(t_reduce, 2)
        metrics['mesh']['faces_reduced'] = int(mesh_reduced.faces.shape[0])
        print(f"  {raw_stats['faces']:,} → {mesh_reduced.faces.shape[0]:,} facce")
    except Exception as e:
        metrics['error'] = f'face_reduce: {e}'
        print(f"  ERRORE face reduction: {e}")
        return metrics

    # ── Stage 3: texture ─────────────────────────────────────────────────────
    print("[4/4] Texture generation...")
    try:
        textured_mesh, t_tex, peak_tex = run_texture_gen(mesh_reduced, image, config)
        metrics['timing']['texture_s'] = round(t_tex, 2)
        metrics['vram_peak_mb']['texture'] = round(peak_tex, 1)

        tex_path = subj_dir / f'{subject}_textured.glb'
        textured_mesh.export(str(tex_path))
        metrics['files']['textured_glb'] = str(tex_path)
        metrics['files']['textured_glb_kb'] = round(tex_path.stat().st_size / 1024, 1)
        print(f"  Texture completata in {t_tex:.1f}s  |  "
              f"VRAM peak: {peak_tex:.0f} MB")
    except Exception as e:
        metrics['error'] = f'texture: {e}'
        print(f"  ERRORE texture: {e}")
        import traceback; traceback.print_exc()

    # ── Totale ───────────────────────────────────────────────────────────────
    metrics['timing']['total_s'] = round(
        sum(v for k, v in metrics['timing'].items() if k.endswith('_s')), 2
    )

    _save_metrics(metrics, subj_dir, subject)
    print(f"\n  Totale: {metrics['timing']['total_s']:.1f}s")
    return metrics


def _save_metrics(metrics: dict, subj_dir: Path, subject: str):
    json_path = subj_dir / f'{subject}_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    metrics['files']['metrics_json'] = str(json_path)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregatore risultati (--compare)
# ─────────────────────────────────────────────────────────────────────────────

def compare_results(results_root: Path):
    """
    Legge tutti i *_metrics.json sotto results_root e stampa una tabella
    comparativa. Struttura attesa:
        results_root/{variant}/{subject}/{subject}_metrics.json
    """
    import glob

    all_jsons = sorted(results_root.rglob('*_metrics.json'))
    if not all_jsons:
        print(f"Nessun file *_metrics.json trovato in: {results_root}")
        return

    records = []
    for path in all_jsons:
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            records.append(data)
        except Exception as e:
            print(f"Errore leggendo {path}: {e}")

    if not records:
        print("Nessun record valido trovato.")
        return

    # Intestazione
    col_w = [10, 20, 8, 8, 10, 10, 10, 10, 8, 8]
    headers = [
        'variante', 'soggetto', 't_shape', 't_tex', 'tot(s)',
        'vram_shp', 'vram_tex', 'facce_raw', 'q_geo', 'q_tex'
    ]
    sep = '+' + '+'.join('-' * (w + 2) for w in col_w) + '+'
    fmt = '| ' + ' | '.join(f'{{:<{w}}}' for w in col_w) + ' |'

    print('\n' + sep)
    print(fmt.format(*headers))
    print(sep)

    for r in records:
        t = r.get('timing', {})
        v = r.get('vram_peak_mb', {})
        m = r.get('mesh', {})
        q = r.get('quality', {})
        row = [
            r.get('variant', '?'),
            r.get('subject', '?')[:20],
            f"{t.get('shape_s', '?')}s",
            f"{t.get('texture_s', '?')}s",
            f"{t.get('total_s', '?')}s",
            f"{v.get('shape', '?')} MB",
            f"{v.get('texture', '?')} MB",
            str(m.get('faces_raw', '?')),
            str(q.get('geometry', '-')),
            str(q.get('texture', '-')),
        ]
        row = [str(x)[:w] for x, w in zip(row, col_w)]
        print(fmt.format(*row))

    print(sep)
    print(f"\n{len(records)} record totali. "
          "Colonne q_geo/q_tex = NULL finché non compilate manualmente nei JSON.\n")

    # Esporta CSV
    csv_path = results_root / 'benchmark_comparison.csv'
    csv_headers = [
        'variant', 'subject', 'seed', 'inference_steps', 'octree_resolution',
        't_rembg_s', 't_shape_s', 't_facereduce_s', 't_texture_s', 't_total_s',
        'vram_peak_shape_mb', 'vram_peak_texture_mb',
        'faces_raw', 'verts_raw', 'faces_reduced',
        'shape_glb_kb', 'textured_glb_kb',
        'has_pbr', 'sequential', 'gpu_name',
        'q_geometry', 'q_texture', 'q_artifacts', 'q_janus', 'q_notes',
        'error',
    ]
    rows = []
    for r in records:
        t = r.get('timing', {})
        v = r.get('vram_peak_mb', {})
        m = r.get('mesh', {})
        f = r.get('files', {})
        q = r.get('quality', {})
        rows.append({
            'variant': r.get('variant'),
            'subject': r.get('subject'),
            'seed': r.get('seed'),
            'inference_steps': r.get('inference_steps'),
            'octree_resolution': r.get('octree_resolution'),
            't_rembg_s': t.get('rembg_s'),
            't_shape_s': t.get('shape_s'),
            't_facereduce_s': t.get('face_reduce_s'),
            't_texture_s': t.get('texture_s'),
            't_total_s': t.get('total_s'),
            'vram_peak_shape_mb': v.get('shape'),
            'vram_peak_texture_mb': v.get('texture'),
            'faces_raw': m.get('faces_raw'),
            'verts_raw': m.get('verts_raw'),
            'faces_reduced': m.get('faces_reduced'),
            'shape_glb_kb': f.get('shape_glb_kb'),
            'textured_glb_kb': f.get('textured_glb_kb'),
            'has_pbr': r.get('has_pbr'),
            'sequential': r.get('sequential'),
            'gpu_name': r.get('gpu', {}).get('name'),
            'q_geometry': q.get('geometry'),
            'q_texture': q.get('texture'),
            'q_artifacts': q.get('artifacts'),
            'q_janus': q.get('janus'),
            'q_notes': q.get('notes'),
            'error': r.get('error'),
        })

    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV esportato in: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Hunyuan3D benchmark runner — mini / 2.0 / 2.1 / 2.5',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--variant', choices=list(MODEL_CONFIGS.keys()),
        help='Variante del modello da testare.',
    )
    parser.add_argument(
        '--input_dir', type=str, default='benchmark/inputs',
        help='Cartella con le immagini di test (PNG/JPG). Default: benchmark/inputs',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Cartella output. Default: benchmark/outputs/{variant}',
    )
    parser.add_argument(
        '--sequential', action='store_true',
        help='Carica i modelli in sequenza (CPU→GPU→del). Obbligatorio per 2.1 su GPU <40GB.',
    )
    parser.add_argument(
        '--no_texture', action='store_true',
        help='Salta la generazione texture (solo shape). Più veloce, meno VRAM.',
    )
    parser.add_argument(
        '--seed', type=int, default=1234,
        help='Seed per la generazione (default: 1234). Usare lo stesso su tutti i modelli.',
    )
    parser.add_argument(
        '--target_faces', type=int, default=40000,
        help='Numero facce target per FaceReducer prima della texture (default: 40000).',
    )
    parser.add_argument(
        '--steps', type=int, default=None,
        help='Override inference steps (default: usa quello della config variante).',
    )
    parser.add_argument(
        '--compare', type=str, default=None, metavar='DIR',
        help='Legge tutti i JSON in DIR e stampa la tabella comparativa. Non genera nulla.',
    )

    args = parser.parse_args()

    # ── Modalità confronto ────────────────────────────────────────────────────
    if args.compare:
        compare_results(Path(args.compare))
        return

    # ── Modalità generazione ──────────────────────────────────────────────────
    if not args.variant:
        parser.error("--variant è obbligatorio in modalità generazione.")

    config = dict(MODEL_CONFIGS[args.variant])
    if args.steps:
        config['num_inference_steps'] = args.steps
    if config['sequential_default'] and not args.sequential:
        print(f"INFO: --sequential consigliato per {args.variant} su GPU <40GB. "
              f"Aggiungilo se ottieni OOM.")

    out_dir = Path(args.output_dir) if args.output_dir else Path('benchmark/outputs') / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERRORE: cartella input non trovata: {input_dir}")
        sys.exit(1)

    images = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}
    ])
    if not images:
        print(f"ERRORE: nessuna immagine trovata in {input_dir}")
        sys.exit(1)

    # Stampa intestazione sessione
    print('\n' + '=' * 60)
    print(f"HUNYUAN3D BENCHMARK — {args.variant.upper()}")
    print('=' * 60)
    print(f"Modello    : {config['description']}")
    print(f"GPU        : {gpu_info()['name']}  ({gpu_info()['total_mb']:.0f} MB)")
    print(f"Sequential : {args.sequential}")
    print(f"Soggetti   : {len(images)}")
    print(f"Output     : {out_dir}")
    print('=' * 60)

    all_results = []
    for img_path in images:
        result = run_subject(img_path, config, out_dir, args)
        all_results.append(result)

    # Salva riepilogo sessione
    summary = {
        'variant': args.variant,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'gpu': gpu_info(),
        'seed': args.seed,
        'subjects': all_results,
        'totals': {
            'n_subjects': len(all_results),
            'n_errors': sum(1 for r in all_results if r.get('error')),
            'avg_shape_s': _safe_avg([r['timing'].get('shape_s') for r in all_results]),
            'avg_texture_s': _safe_avg([r['timing'].get('texture_s') for r in all_results]),
            'avg_total_s': _safe_avg([r['timing'].get('total_s') for r in all_results]),
            'avg_vram_shape_mb': _safe_avg([r['vram_peak_mb'].get('shape') for r in all_results]),
            'avg_vram_texture_mb': _safe_avg([r['vram_peak_mb'].get('texture') for r in all_results]),
        },
    }

    summary_path = out_dir / 'run_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 60)
    print('SESSIONE COMPLETATA')
    print('=' * 60)
    t = summary['totals']
    print(f"Soggetti: {t['n_subjects']}  |  Errori: {t['n_errors']}")
    if t['avg_shape_s']:
        print(f"Avg shape:   {t['avg_shape_s']:.1f}s")
    if t['avg_texture_s']:
        print(f"Avg texture: {t['avg_texture_s']:.1f}s")
    if t['avg_total_s']:
        print(f"Avg totale:  {t['avg_total_s']:.1f}s")
    print(f"Riepilogo salvato: {summary_path}")
    print()
    print("PROSSIMI PASSI:")
    print(f"  1. Scarica la cartella {out_dir}/ (rsync/SCP da RunPod)")
    print(f"  2. Apri i GLB in Blender per la valutazione visiva")
    print(f"  3. Compila i campi 'quality' nei *_metrics.json")
    print(f"  4. Confronta tutti i modelli:")
    print(f"     python run/benchmark_runpod.py --compare benchmark/outputs/")
    print('=' * 60)


def _safe_avg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


if __name__ == '__main__':
    main()