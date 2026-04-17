"""
Text-to-image with FLUX.1-dev for isolated 3D-ready asset generation.

FLUX.1-dev (gated — accetta licenza su HuggingFace + HF_TOKEN richiesto).
20 inference steps, bfloat16, guidance_scale 3.5, no negative prompt.

Usage:
    export HF_TOKEN="hf_..."
    python flux_text2image.py --prompt "a wooden treasure chest, closed, dusty"
    python flux_text2image.py --prompt "a red sports car" --style realistic --seeds 1 2 3 4
    python flux_text2image.py --prompt "a medieval sword" --size 1024 --output outputs/sword.png
"""

import argparse
import gc
from pathlib import Path

import torch
from diffusers import FluxPipeline

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "black-forest-labs/FLUX.1-dev"

# ---------------------------------------------------------------------------
# Style suffixes
# ---------------------------------------------------------------------------
STYLE_SUFFIXES: dict[str, str] = {
    "realistic": (
        "photorealistic, 8k ultra-detailed, sharp focus, physically accurate materials, "
        "studio photography, ray-traced lighting"
    ),
    "cartoon": (
        "cartoon style, vibrant flat colors, cel-shaded, clean outlines, stylized proportions"
    ),
    "fantasy": (
        "fantasy concept art, painterly, magical atmosphere, intricate details, "
        "ArtStation trending, ethereal glow"
    ),
    "scifi": (
        "science fiction, futuristic design, hard-surface modeling, metallic surfaces, "
        "neon accents, cyberpunk aesthetic, CGI render"
    ),
    "lowpoly": (
        "low-poly 3D art, geometric facets, minimal polygons, clean stylized look, game asset style"
    ),
    "painterly": (
        "oil painting, impressionist brushstrokes, textured canvas, artistic, classical art style"
    ),
    "anime": (
        "anime style, Japanese animation, clean line art, soft cel shading, vibrant colors"
    ),
    "generic": "",
}

# Isolation suffix for clean product-style shots (ideal for 3D input)
ISOLATION_SUFFIX = (
    "isolated on pure white background, clean studio shot, soft diffused lighting, "
    "no shadows on background, centered composition, single object, no clutter"
)


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_next_path(output_dir: str, prefix: str = "flux_asset", ext: str = ".png") -> Path:
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    existing = [int(p.stem.split("_")[-1]) for p in d.glob(f"{prefix}_*.png") if p.stem.split("_")[-1].isdigit()]
    return d / f"{prefix}_{max(existing, default=0) + 1}{ext}"


def build_prompt(base: str, style: str, isolate: bool) -> str:
    parts = [base.strip()]
    suffix = STYLE_SUFFIXES.get(style, "")
    if suffix:
        parts.append(suffix)
    if isolate:
        parts.append(ISOLATION_SUFFIX)
    return ", ".join(p.strip(", ") for p in parts if p.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate isolated 2D assets with FLUX.1-schnell",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--prompt", required=True, help="Prompt testuale.")
    parser.add_argument(
        "--style",
        choices=list(STYLE_SUFFIXES.keys()),
        default="realistic",
        help="Stile visivo (default: realistic).",
    )
    parser.add_argument(
        "--no-isolate", action="store_true",
        help="Non forzare sfondo bianco isolato.",
    )
    parser.add_argument("--output", default=None, help="Percorso output. Se omesso, auto-numerazione in --output-dir.")
    parser.add_argument("--output-dir", default="run/outputs/flux", help="Cartella output.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Lista seed (es: --seeds 1 2 3 4). Genera una immagine per seed.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed singolo (default: 1234).")
    parser.add_argument("--size", type=int, default=1024, help="Dimensione immagine quadrata (default: 1024).")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps (default: 20, ottimale per dev).")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale (default: 3.5 per dev).")
    parser.add_argument("--show-prompt", action="store_true", help="Stampa il prompt finale prima di generare.")
    parser.add_argument("--model", default=MODEL_ID, help="Model ID HuggingFace (default: FLUX.1-schnell).")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    isolate = not args.no_isolate
    final_prompt = build_prompt(args.prompt, args.style, isolate)

    print("=" * 60)
    print("FLUX.1-dev — ASSET 2D GENERATION")
    print("=" * 60)
    print(f"Stile   : {args.style}")
    print(f"Isolato : {'sì' if isolate else 'no'}")
    print(f"Seeds   : {seeds}")
    print(f"Size    : {args.size}x{args.size}")
    print(f"Steps   : {args.steps}")
    if args.show_prompt:
        print(f"\nPrompt  : {final_prompt}")
    print("-" * 60)

    print("Caricamento modello FLUX.1-schnell...")
    clear_memory()
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    print("  Modello pronto.\n")

    generated = []
    for seed in seeds:
        if args.output and len(seeds) == 1:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = get_next_path(args.output_dir)

        print(f"Generando seed={seed} → {out_path}")
        generator = torch.Generator("cpu").manual_seed(seed)
        image = pipe(
            prompt=final_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.size,
            height=args.size,
            generator=generator,
        ).images[0]
        image.save(out_path)
        generated.append(out_path)
        print(f"  Salvato: {out_path}")

    del pipe
    clear_memory()

    print("\n" + "=" * 60)
    print(f"Completato — {len(generated)} immagine/i generate")
    for p in generated:
        print(f"  {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
