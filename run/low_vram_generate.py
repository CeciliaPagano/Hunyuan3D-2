"""
Low VRAM (8GB) script for Hunyuan3D-2 with texture generation.
Loads models sequentially and clears memory between stages.
"""

import torch
import gc
import re
import argparse
from pathlib import Path


def get_next_filename(output_dir, prefix="image_3D", extension=".glb"):
    """Find the next available numbered filename."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find existing files matching pattern
    pattern = re.compile(rf"{prefix}_(\d+){re.escape(extension)}$")
    max_num = 0

    for f in output_dir.iterdir():
        match = pattern.match(f.name)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)

    next_num = max_num + 1
    return output_dir / f"{prefix}_{next_num}{extension}"


def clear_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def generate_shape(image_path, seed=1234, steps=5, guidance_scale=5.0, octree_resolution=256):
    """Generate 3D shape from image."""
    print("Loading shape generation model...")

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    from hy3dgen.rembg import BackgroundRemover
    from PIL import Image

    # Load and process image
    image = Image.open(image_path)

    # Remove background
    print("Removing background...")
    rmbg = BackgroundRemover()
    image = rmbg(image.convert('RGB'))
    del rmbg
    clear_memory()

    # Load shape model on CPU first, then move to GPU
    print("Loading shape model (this may take a moment)...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mini',
        subfolder='hunyuan3d-dit-v2-2.0 mini-turbo',
        device='cpu'  # Load on CPU first
    )
    pipeline.enable_flashvdm(topk_mode='mean')

    # Move to GPU
    pipeline.to(device='cuda')

    # Generate
    print("Generating 3D shape...")
    generator = torch.Generator(device='cuda').manual_seed(seed)
    outputs = pipeline(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution,
        num_chunks=8000,
        output_type='mesh'
    )

    mesh = export_to_trimesh(outputs)[0]

    # Clean up shape model
    print("Cleaning up shape model...")
    del pipeline
    clear_memory()

    return mesh, image


def reduce_faces(mesh, target_faces=40000):
    """Reduce mesh faces for texture generation."""
    print(f"Reducing faces to {target_faces}...")

    from hy3dgen.shapegen import FaceReducer


    reducer = FaceReducer()
    mesh = reducer(mesh, target_faces)

    del reducer
    clear_memory()

    return mesh


def generate_texture(mesh, image, output_path):
    """Generate texture for the mesh."""
    print("Loading texture generation model...")

    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    # Load texture model with CPU offloading
    pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')

    # Try to enable CPU offloading (may help with VRAM)
    try:
        pipeline.enable_model_cpu_offload()
        print("CPU offloading enabled for texture model")
    except Exception as e:
        print(f"Warning: Could not enable CPU offload: {e}")

    # Generate texture
    print("Generating texture (this takes a while)...")
    textured_mesh = pipeline(mesh, image)

    # Save
    print(f"Saving to {output_path}...")
    textured_mesh.export(output_path)

    # Clean up
    del pipeline
    clear_memory()

    return textured_mesh


def find_latest_2d_image(output_dir="outputs/2D"):
    """Find the most recent image_2D_*.png file."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None

    pattern = re.compile(r"image_2D_(\d+)\.png$")
    candidates = []

    for f in output_dir.iterdir():
        match = pattern.match(f.name)
        if match:
            candidates.append((int(match.group(1)), f))

    if not candidates:
        return None

    # Return the one with highest number
    candidates.sort(reverse=True)
    return str(candidates[0][1])


def main():
    parser = argparse.ArgumentParser(description='Low VRAM 3D generation with texture')
    parser.add_argument('--image', type=str, default=None, help='Input image path (uses latest 2D image if not specified)')
    parser.add_argument('--output', type=str, default=None, help='Output file path (auto-numbered if not specified)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--steps', type=int, default=5, help='Inference steps')
    parser.add_argument('--no-texture', action='store_true', help='Skip texture generation')
    parser.add_argument('--target-faces', type=int, default=40000, help='Target face count for texture')
    args = parser.parse_args()

    # Auto-find latest 2D image if not specified
    if args.image is None:
        args.image = find_latest_2d_image()
        if args.image is None:
            print("Error: No image specified and no image_2D_*.png found in outputs/2D/")
            return

    # Auto-generate output filename if not specified
    if args.output is None:
        args.output = str(get_next_filename("outputs/3D", "image_3D", ".glb"))

    # Check input exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Input: {args.image}")
    print(f"Output: {args.output}")
    print("-" * 40)

    # Clear any existing GPU memory
    clear_memory()

    # Stage 1: Generate shape
    mesh, image = generate_shape(
        args.image,
        seed=args.seed,
        steps=args.steps
    )
    print(f"Generated mesh with {mesh.faces.shape[0]} faces")

    if args.no_texture:
        # Save without texture
        print(f"Saving untextured mesh to {args.output}...")
        mesh.export(args.output)
        print("Done!")
        return

    # Stage 2: Reduce faces for texture
    mesh = reduce_faces(mesh, args.target_faces)

    # Stage 3: Generate texture
    generate_texture(mesh, image, args.output)

    print("-" * 40)
    print(f"Done! Output saved to: {args.output}")


if __name__ == '__main__':
    main()
