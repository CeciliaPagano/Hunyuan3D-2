#!/usr/bin/env python3
"""
Text → Immagine → 3D PBR con TRELLIS.2

Flusso:
  1. HunyuanDiT-v1.1-Distilled  →  immagine 768x768 su sfondo bianco
  2. Trellis2ImageTo3DPipeline   →  mesh + texture PBR unificati → GLB

Il preprocessing immagine (rimozione sfondo, normalizzazione) è gestito
internamente da TRELLIS.2 con preprocess_image=True.

Prerequisiti:
  source /workspace/activate_trellis.sh   # conda base + PYTHONPATH + ATTN_BACKEND
  cd /workspace/TRELLIS.2

Usage:
  python /workspace/Hunyuan3D-2/run/text_to_3d_trellis.py \\
      --prompt "a ceramic vase with floral patterns" \\
      --style realistic --seed 42

  python /workspace/Hunyuan3D-2/run/text_to_3d_trellis.py \\
      --prompt "a golden dragon statue" \\
      --style fantasy --seed 1234 \\
      --pipeline_type 1024_cascade

pipeline_type disponibili:
  512           — risoluzione 512³, più veloce
  1024          — risoluzione 1024³, single-stage
  1024_cascade  — risoluzione 1024³, two-stage 512→1024 (DEFAULT, miglior qualità)
  1536_cascade  — risoluzione 1536³, massima qualità (~60s su H100)
"""

import argparse
import gc
import os
import time
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR',    '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',     'expandable_segments:True')
os.environ.setdefault('HF_HOME',                     '/workspace/models')
os.environ.setdefault('HUGGINGFACE_HUB_CACHE',       '/workspace/models/hub')

import torch


# ── Stili ─────────────────────────────────────────────────────────────────────
STYLE_SUFFIXES = {
    'realistic': (
        'photorealistic, 8k ultra-detailed, sharp focus, physically accurate materials, '
        'DSLR photography, ray-traced lighting'
    ),
    'fantasy': (
        'fantasy concept art, painterly, magical atmosphere, intricate details, '
        'ArtStation trending, ethereal glow, matte painting'
    ),
    'scifi': (
        'science fiction, futuristic design, hard-surface modeling, metallic surfaces, '
        'neon accents, cyberpunk aesthetic, CGI render'
    ),
    'cartoon': (
        'cartoon style, vibrant flat colors, cel-shaded, clean outlines, stylized proportions'
    ),
    'anime': (
        'anime style, Japanese animation, clean line art, soft cel shading, vibrant colors'
    ),
    'generic': '',
}

ISOLATION_SUFFIX = (
    'isolated on pure white background, clean studio shot, soft diffused lighting, '
    'no shadows on background, centered composition, single object, no clutter'
)

UNIVERSAL_NEGATIVE = (
    'blurry, low quality, watermark, signature, text overlay, deformed, distorted, '
    'ugly, bad proportions, out of frame, cropped, multiple objects, crowd, person, hands'
)

TEXT2IMG_MODEL  = 'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled'
TRELLIS_MODEL   = 'microsoft/TRELLIS.2-4B'
TEXTURE_SIZE    = 1024
DECIMATION_TARGET = 500000


# ── Utilities ─────────────────────────────────────────────────────────────────
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def peak_mb():
    return torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0


# ── Step 1: Text → Image ──────────────────────────────────────────────────────
def generate_image(prompt: str, style: str, seed: int, out_path: Path, steps: int = 25):
    from diffusers import AutoPipelineForText2Image

    style_suffix = STYLE_SUFFIXES.get(style, '')
    parts = [prompt.strip()]
    if style_suffix:
        parts.append(style_suffix)
    parts.append(ISOLATION_SUFFIX)
    final_prompt = ', '.join(p.strip() for p in parts if p.strip())

    print(f"  Prompt: {final_prompt[:120]}...")
    print(f"  Modello: {TEXT2IMG_MODEL}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        TEXT2IMG_MODEL,
        torch_dtype=dtype,
        enable_pag=True,
        pag_applied_layers=['blocks.(16|17|18|19)'],
    )
    pipe.enable_model_cpu_offload()

    generator = torch.Generator(device='cpu').manual_seed(seed)
    image = pipe(
        prompt=final_prompt,
        negative_prompt=UNIVERSAL_NEGATIVE,
        width=768, height=768,
        num_inference_steps=steps,
        guidance_scale=4.0,
        pag_scale=1.4,
        generator=generator,
    ).images[0]

    del pipe
    clear_memory()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return image


# ── Step 2: TRELLIS.2 Image → 3D ─────────────────────────────────────────────
def load_trellis_pipeline():
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    print(f"  Caricamento modello {TRELLIS_MODEL}...")
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(TRELLIS_MODEL)
    pipe.cuda()
    return pipe


def generate_3d(pipeline, image, seed: int, pipeline_type: str):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    meshes = pipeline.run(
        image,
        seed=seed,
        pipeline_type=pipeline_type,
        num_samples=1,
        preprocess_image=True,
    )
    elapsed = time.time() - t0
    pk = peak_mb()
    mesh = meshes[0]
    faces = int(mesh.faces.shape[0]) if hasattr(mesh.faces, 'shape') else 0
    print(f"  {faces:,} facce  |  {elapsed:.1f}s  |  VRAM peak: {pk:.0f} MB")
    return mesh, elapsed, pk


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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Text → Image → 3D PBR con TRELLIS.2'
    )
    parser.add_argument('--prompt',        required=True, help='Descrizione testuale dell\'oggetto')
    parser.add_argument('--style',         default='realistic', choices=list(STYLE_SUFFIXES.keys()))
    parser.add_argument('--seed',          type=int, default=1234)
    parser.add_argument('--out_dir',       default='/workspace/outputs/text_to_3d_trellis')
    parser.add_argument('--name',          default=None, help='Nome base per i file output (default: usa seed)')
    parser.add_argument('--pipeline_type', default='1024_cascade',
                        choices=['512', '1024', '1024_cascade', '1536_cascade'])
    parser.add_argument('--no_texture',    action='store_true',
                        help='Genera solo la mesh grezza, salta export GLB con texture')
    parser.add_argument('--img_steps',     type=int, default=25,
                        help='Inference steps per text2image')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.name or f'seed{args.seed}'

    t_total = time.time()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    print('\n' + '=' * 60)
    print('TEXT → 3D PBR  —  TRELLIS.2')
    print('=' * 60)
    print(f"Prompt        : {args.prompt}")
    print(f"Stile         : {args.style}")
    print(f"Seed          : {args.seed}")
    print(f"Pipeline type : {args.pipeline_type}")
    print(f"GPU           : {gpu_name}")
    print(f"Output        : {out_dir}/{base}_*")
    print('=' * 60)

    # ── 1. Text → Image ───────────────────────────────────────────────────────
    img_path = out_dir / f'{base}_2D.png'
    print(f"\n[1/3] Text → Image...")
    t0 = time.time()
    image = generate_image(args.prompt, args.style, args.seed, img_path, steps=args.img_steps)
    print(f"  Salvata: {img_path}  ({time.time()-t0:.1f}s)")

    # ── 2. Image → 3D ─────────────────────────────────────────────────────────
    print(f"\n[2/3] Caricamento TRELLIS.2 pipeline...")
    pipeline = load_trellis_pipeline()
    clear_memory()

    print(f"\n[2/3] Generazione 3D (pipeline_type={args.pipeline_type})...")
    mesh, t_gen, vram_gen = generate_3d(pipeline, image, args.seed, args.pipeline_type)

    if args.no_texture:
        print(f"\n  --no_texture impostato, pipeline completata.")
        print(f"  Totale: {time.time()-t_total:.1f}s")
        return

    # ── 3. Export GLB con texture PBR ─────────────────────────────────────────
    glb_path = out_dir / f'{base}_textured.glb'
    print(f"\n[3/3] Export GLB (texture_size={TEXTURE_SIZE})...")
    t0 = time.time()
    export_glb(mesh, pipeline, str(glb_path))
    t_export = time.time() - t0
    size_kb = glb_path.stat().st_size // 1024
    print(f"  {t_export:.1f}s  |  {size_kb} KB")

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('COMPLETATO')
    print('=' * 60)
    print(f"  2D image   : {img_path}")
    print(f"  3D textured: {glb_path}  ({size_kb} KB)")
    print(f"  Generate   : {t_gen:.1f}s  (VRAM peak: {vram_gen:.0f} MB)")
    print(f"  Export GLB : {t_export:.1f}s")
    print(f"  Totale     : {time.time()-t_total:.1f}s")
    print('=' * 60)


if __name__ == '__main__':
    main()
