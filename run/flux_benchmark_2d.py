"""
FLUX.1-dev — Benchmark 2D per input TRELLIS.2
Carica il modello una volta sola e genera tutti i soggetti in sequenza.

Usage:
    python flux_benchmark_2d.py
    python flux_benchmark_2d.py --seed 42
    python flux_benchmark_2d.py --only A1 B2 E1
    python flux_benchmark_2d.py --steps 28 --size 1024
    python flux_benchmark_2d.py --output-dir /workspace/benchmark/inputs
"""

import argparse
import gc
from pathlib import Path

import torch
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-dev"

ISOLATION = (
    "3D CGI render, product visualization, not a photograph, "
    "pure white background, perfectly uniform ambient lighting, "
    "no shadows, centered composition, single object, no clutter, "
    "razor sharp, every part in focus, zero depth of field, no bokeh whatsoever"
)

SUBJECTS = [
    # ── Categoria A — Oggetti solidi standard ─────────────────────────────────
    {
        "id": "A1",
        "name": "anello_rubino",
        "prompt": (
            "A polished silver ring with a single red hexagonal ruby at center "
            "flanked by two small round diamonds, "
            f"product photography, photorealistic, sharp focus, physically accurate materials, {ISOLATION}"
        ),
    },
    {
        "id": "A2",
        "name": "chiave_inglese",
        "prompt": (
            "A steel adjustable wrench, slightly angled view showing depth, "
            f"studio lighting with soft shadows, sharp metallic edges, photorealistic, {ISOLATION}"
        ),
    },
    {
        "id": "A3",
        "name": "melograno",
        "prompt": (
            "A ripe pomegranate cut in half showing seeds inside, "
            "diffuse studio lighting, soft natural shadows, food photography style, "
            f"high detail on texture, photorealistic, {ISOLATION}"
        ),
    },

    # ── Categoria B — Topologie complesse ─────────────────────────────────────
    {
        "id": "B1",
        "name": "cestino_vimini",
        "prompt": (
            "A woven wicker basket with visible gaps between weaves, "
            "three-quarter view from slightly above, even studio lighting, "
            f"clear depth and structure visible through the gaps, photorealistic, {ISOLATION}"
        ),
    },
    {
        "id": "B2",
        "name": "bicicletta",
        "prompt": (
            "A classic road bicycle, side view slightly angled, thin spokes clearly visible, "
            "studio product photography, sharp focus on all mechanical parts, no motion blur, "
            f"photorealistic, {ISOLATION}"
        ),
    },
    {
        "id": "B3",
        "name": "nautilus",
        "prompt": (
            "A nautilus shell cut in half showing the internal spiral chambers, "
            "dramatic but even studio lighting, visible depth inside each chamber, "
            f"macro photography style, photorealistic, {ISOLATION}"
        ),
    },

    # ── Categoria C — Trasparenze ─────────────────────────────────────────────
    {
        "id": "C1",
        "name": "calice_vetro",
        "prompt": (
            "A clear glass wine glass, empty, three-quarter view, "
            "studio lighting showing transparent glass with subtle reflections and refractions, "
            f"product photography, sharp edges visible, photorealistic, {ISOLATION}"
        ),
    },
    {
        "id": "C2",
        "name": "sfera_ambra",
        "prompt": (
            "A smooth amber resin sphere with a small insect trapped inside, "
            "visible through the translucent material, "
            "backlit studio lighting showing light passing through, "
            f"photorealistic, {ISOLATION}"
        ),
    },

    # ── Categoria D — Soggetti simmetrici ─────────────────────────────────────
    {
        "id": "D1",
        "name": "maschera_veneziana",
        "prompt": (
            "A Venetian carnival mask, ornate gold and white, perfectly symmetrical, "
            "front-facing view, even studio lighting from both sides, "
            f"high detail on surface decorations, photorealistic, {ISOLATION}"
        ),
    },
    {
        "id": "D2",
        "name": "gufo_ceramica",
        "prompt": (
            "A stylized cartoon owl figurine, front-facing symmetrical pose, "
            "big round eyes, smooth ceramic material, "
            f"soft even studio lighting, no harsh shadows, photorealistic, {ISOLATION}"
        ),
    },

    # ── Categoria F — Figura umana stilizzata ─────────────────────────────────
    {
        "id": "F1",
        "name": "omino_stilizzato",
        "prompt": (
            "A simple stylized human figure standing upright, flat matte gray color, "
            "smooth featureless surface, no texture details, like a store mannequin or scale reference figure, "
            f"full body visible head to toe, neutral A-pose, {ISOLATION}"
        ),
    },

    # ── Categoria E — Figura umana realistica ─────────────────────────────────
    {
        "id": "E1",
        "name": "donna_figura_intera",
        "prompt": (
            "A realistic young woman standing in a neutral A-pose, "
            "full body visible from head to toe, casual modern clothing, "
            "even soft studio lighting from both sides, no harsh shadows, "
            f"product photography style, sharp focus, photorealistic, {ISOLATION}"
        ),
    },
    {
        "id": "E2",
        "name": "uomo_figura_intera",
        "prompt": (
            "A realistic adult man standing upright, slight three-quarter turn, "
            "arms relaxed at sides, full body shot, "
            "contemporary casual clothes, diffuse studio lighting, "
            f"photorealistic, sharp focus, no motion blur, {ISOLATION}"
        ),
    },
    {
        "id": "E3",
        "name": "volto_realistico",
        "prompt": (
            "A realistic woman, upper body portrait, three-quarter view, "
            "neutral expression, even soft studio lighting, "
            f"photorealistic skin texture, contemporary clothing, sharp focus on facial features, {ISOLATION}"
        ),
    },
]


def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="FLUX.1-dev benchmark 2D batch generation")
    parser.add_argument("--seed",       type=int,   default=1234)
    parser.add_argument("--steps",      type=int,   default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--size",       type=int,   default=1024)
    parser.add_argument("--output-dir", default="run/outputs/flux_benchmark")
    parser.add_argument("--only",       nargs="+",  default=None,
                        help="Genera solo i soggetti specificati, es: --only A1 B2 E1")
    parser.add_argument("--model",      default=MODEL_ID)
    args = parser.parse_args()

    subjects = SUBJECTS
    if args.only:
        subjects = [s for s in SUBJECTS if s["id"] in args.only]
        if not subjects:
            print(f"ERRORE: nessun soggetto trovato tra {args.only}")
            return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FLUX.1-dev — BENCHMARK 2D")
    print("=" * 60)
    print(f"Soggetti : {len(subjects)}")
    print(f"Seed     : {args.seed}")
    print(f"Steps    : {args.steps}")
    print(f"Size     : {args.size}x{args.size}")
    print(f"Output   : {out_dir}")
    print("=" * 60)

    print("\nCaricamento modello...")
    clear()
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    print("  Modello pronto.\n")

    results = []
    for s in subjects:
        out_path = out_dir / f"{s['id']}_{s['name']}.png"
        print(f"[{s['id']}] {s['name']}")

        generator = torch.Generator("cpu").manual_seed(args.seed)
        image = pipe(
            prompt=s["prompt"],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.size,
            height=args.size,
            generator=generator,
        ).images[0]
        image.save(out_path)
        results.append(out_path)
        print(f"  → {out_path}\n")

    del pipe
    clear()

    print("=" * 60)
    print(f"Completato — {len(results)} immagini generate in {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
