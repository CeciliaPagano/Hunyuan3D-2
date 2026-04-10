#!/usr/bin/env python3
"""
Text → Immagine → 3D PBR con Hunyuan3D-2.1

Flusso:
  1. HunyuanDiT-v1.1-Distilled  →  immagine 768x768 su sfondo bianco
  2. rembg  →  rimozione sfondo
  3. hy3dshape (2.1)  →  mesh grezza
  4. FaceReducer  →  decimazione a --target_faces
  5. hy3dpaint PBR  →  texture Albedo + Normal + Roughness + Metallic

Prerequisiti (A100 / RTX 4090 con SEQUENTIAL=1):
  source /workspace/venv-21/bin/activate
  cd /workspace/Hunyuan3D-2.1

Usage:
  python /workspace/Hunyuan3D-2/run/text_to_3d_21.py \\
      --prompt "a ceramic vase with floral patterns" \\
      --style realistic --seed 42

  python /workspace/Hunyuan3D-2/run/text_to_3d_21.py \\
      --prompt "a golden dragon statue" \\
      --style fantasy --seed 1234 --sequential --no_texture

  # Con out_dir e nome personalizzati
  python /workspace/Hunyuan3D-2/run/text_to_3d_21.py \\
      --prompt "a sci-fi helmet" --style scifi \\
      --out_dir /workspace/outputs/my_run --name sci_helmet
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

# ── Variabili d'ambiente obbligatorie ────────────────────────────────────────
os.environ.setdefault('HF_HOME',               '/workspace/models')
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', '/workspace/models/hub')
os.environ.setdefault('U2NET_HOME',            '/workspace/models/u2net')

import torch


# ── Stili (subset di text2image_asset.py) ────────────────────────────────────
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

TEXT2IMG_MODEL = 'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled'

# ── CONFIG 2.1 ────────────────────────────────────────────────────────────────
CONFIG_21 = {
    'shape_model_path':   'tencent/Hunyuan3D-2.1',
    'shape_subfolder':    'hunyuan3d-dit-v2-1',
    'num_inference_steps': 50,
    'octree_resolution':  512,
    'num_chunks':         200000,
    'guidance_scale':     7.5,
    'paint_max_views':    6,
    'paint_resolution':   512,
}


# ── Utilities ─────────────────────────────────────────────────────────────────
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def peak_mb():
    return torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0


def setup_repo_paths(repo21: str):
    """Aggiunge al sys.path il repo 2.1 e i suoi sottomoduli."""
    root = Path(repo21).resolve()
    for p in [root, root / 'hy3dshape', root / 'hy3dpaint']:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


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


# ── Step 2: rembg ─────────────────────────────────────────────────────────────
def remove_background(image, out_path: Path):
    from hy3dshape.rembg import BackgroundRemover
    r = BackgroundRemover()
    result = r(image)
    del r
    clear_memory()
    result.save(out_path)
    return result


# ── Step 3: Shape generation ──────────────────────────────────────────────────
def generate_shape(image, seed: int, sequential: bool):
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    kw = {'subfolder': CONFIG_21['shape_subfolder']}
    if sequential:
        kw['device'] = 'cpu'

    print(f"  Caricamento shape model (subfolder={CONFIG_21['shape_subfolder']})...")
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        CONFIG_21['shape_model_path'], **kw
    )
    if sequential:
        pipe.to('cuda')

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    gen = torch.Generator(device='cuda').manual_seed(seed)
    mesh = pipe(
        image=image,
        num_inference_steps=CONFIG_21['num_inference_steps'],
        guidance_scale=CONFIG_21['guidance_scale'],
        generator=gen,
        octree_resolution=CONFIG_21['octree_resolution'],
        num_chunks=CONFIG_21['num_chunks'],
    )[0]
    elapsed = time.time() - t0
    pk = peak_mb()
    del pipe
    clear_memory()
    print(f"  {mesh.faces.shape[0]:,} facce  |  {elapsed:.1f}s  |  VRAM peak: {pk:.0f} MB")
    return mesh, elapsed, pk


# ── Step 4: Face reduction ────────────────────────────────────────────────────
def reduce_faces(mesh, target: int):
    import trimesh
    n = len(mesh.faces)
    if n <= target:
        print(f"  {n:,} facce già sotto il target {target:,}, skip")
        return mesh
    try:
        try:
            from hy3dshape.utils.mesh import FaceReducer
        except ImportError:
            from hy3dshape import FaceReducer
        r = FaceReducer()
        mesh = r(mesh, target)
        del r
        clear_memory()
    except Exception as e:
        print(f"  FaceReducer fallito ({e}), uso trimesh decimation...")
        mesh = mesh.simplify_quadratic_decimation(target)
    print(f"  {n:,} → {len(mesh.faces):,} facce")
    return mesh


# ── Step 5: Texture PBR ───────────────────────────────────────────────────────
def generate_texture(mesh_path: str, image, image_path: str, repo_root: Path):
    import trimesh as _trimesh
    import textureGenPipeline as _tgp
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    # Patch remesh_mesh nel namespace di textureGenPipeline
    def _trimesh_remesh(input_path, output_path):
        m = _trimesh.load(input_path, force='mesh')
        m.export(output_path)
        return output_path
    _tgp.remesh_mesh = _trimesh_remesh

    conf = Hunyuan3DPaintConfig(CONFIG_21['paint_max_views'], CONFIG_21['paint_resolution'])
    conf.realesrgan_ckpt_path = str(repo_root / 'hy3dpaint/ckpt/RealESRGAN_x4plus.pth')
    conf.multiview_cfg_path   = str(repo_root / 'hy3dpaint/cfgs/hunyuan-paint-pbr.yaml')
    conf.custom_pipeline      = str(repo_root / 'hy3dpaint/hunyuanpaintpbr')

    pipe = Hunyuan3DPaintPipeline(conf)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        result = pipe(mesh_path, image)
    except TypeError as e:
        print(f"  pipe(PIL) fallito ({e}), riprovo con path...")
        result = pipe(mesh_path, image_path)
    elapsed = time.time() - t0
    pk = peak_mb()
    del pipe
    clear_memory()

    if isinstance(result, (str, Path)):
        result = _trimesh.load(str(result))
    print(f"  {elapsed:.1f}s  |  VRAM peak: {pk:.0f} MB")
    return result, elapsed, pk


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Text → Image → 3D PBR con Hunyuan3D-2.1'
    )
    parser.add_argument('--prompt',       required=True, help='Descrizione testuale dell\'oggetto')
    parser.add_argument('--style',        default='realistic', choices=list(STYLE_SUFFIXES.keys()))
    parser.add_argument('--seed',         type=int, default=1234)
    parser.add_argument('--out_dir',      default='/workspace/outputs/text_to_3d_21')
    parser.add_argument('--name',         default=None, help='Nome base per i file output (default: usa seed)')
    parser.add_argument('--sequential',   action='store_true', help='CPU offload shape model (per GPU <35GB)')
    parser.add_argument('--no_texture',   action='store_true', help='Genera solo geometria, salta texture')
    parser.add_argument('--target_faces', type=int, default=40000)
    parser.add_argument('--img_steps',    type=int, default=25, help='Inference steps per text2image')
    parser.add_argument('--repo21',       default='/workspace/Hunyuan3D-2.1',
                        help='Path del repo Hunyuan3D-2.1')
    args = parser.parse_args()

    repo_root = setup_repo_paths(args.repo21)

    # Path base per i file output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.name or f'seed{args.seed}'

    t_total = time.time()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    print('\n' + '=' * 60)
    print('TEXT → 3D PBR  —  Hunyuan3D-2.1')
    print('=' * 60)
    print(f"Prompt   : {args.prompt}")
    print(f"Stile    : {args.style}")
    print(f"Seed     : {args.seed}")
    print(f"GPU      : {gpu_name}")
    print(f"Output   : {out_dir}/{base}_*")
    print('=' * 60)

    # ── 1. Text → Image ───────────────────────────────────────────────────────
    img_path = out_dir / f'{base}_2D.png'
    print(f"\n[1/5] Text → Image...")
    t0 = time.time()
    image = generate_image(args.prompt, args.style, args.seed, img_path, steps=args.img_steps)
    print(f"  Salvata: {img_path}  ({time.time()-t0:.1f}s)")

    # ── 2. rembg ──────────────────────────────────────────────────────────────
    rembg_path = out_dir / f'{base}_rembg.png'
    print(f"\n[2/5] Background removal...")
    t0 = time.time()
    image_nobg = remove_background(image, rembg_path)
    print(f"  Salvata: {rembg_path}  ({time.time()-t0:.1f}s)")

    # ── 3. Shape ──────────────────────────────────────────────────────────────
    shape_path = out_dir / f'{base}_shape.glb'
    print(f"\n[3/5] Shape generation (steps={CONFIG_21['num_inference_steps']}, octree={CONFIG_21['octree_resolution']})...")
    mesh_raw, t_shape, vram_shape = generate_shape(image_nobg, args.seed, args.sequential)
    mesh_raw.export(str(shape_path))
    print(f"  Salvata: {shape_path}")

    if args.no_texture:
        print(f"\n  --no_texture impostato, pipeline completata.")
        print(f"  Mesh: {shape_path}")
        print(f"  Totale: {time.time()-t_total:.1f}s")
        return

    # ── 4. Face reduction ─────────────────────────────────────────────────────
    reduced_path = out_dir / f'{base}_reduced.glb'
    print(f"\n[4/5] Face reduction → {args.target_faces:,}...")
    mesh_red = reduce_faces(mesh_raw, args.target_faces)
    mesh_red.export(str(reduced_path))

    # ── 5. Texture PBR ────────────────────────────────────────────────────────
    textured_path = out_dir / f'{base}_textured.glb'
    print(f"\n[5/5] Texture PBR generation...")
    mesh_tex, t_tex, vram_tex = generate_texture(
        str(reduced_path), image_nobg, str(rembg_path), repo_root
    )
    mesh_tex.export(str(textured_path))
    print(f"  Salvata: {textured_path}")

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('COMPLETATO')
    print('=' * 60)
    print(f"  2D image   : {img_path}")
    print(f"  Shape      : {shape_path}")
    print(f"  3D textured: {textured_path}  ({textured_path.stat().st_size // 1024} KB)")
    print(f"  Shape time : {t_shape:.1f}s  (VRAM peak: {vram_shape:.0f} MB)")
    print(f"  Texture    : {t_tex:.1f}s  (VRAM peak: {vram_tex:.0f} MB)")
    print(f"  Totale     : {time.time()-t_total:.1f}s")
    print('=' * 60)


if __name__ == '__main__':
    main()
