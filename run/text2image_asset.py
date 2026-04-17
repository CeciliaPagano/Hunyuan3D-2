"""
Text-to-image for generic scene asset generation.

Generates isolated 2D assets (characters, props, environments, vehicles,
furniture, creatures, food, ...) intended for later assembly into complete 3D scenes.

Supports style presets, category-specific negative prompts, scene context,
and the same YAML/JSON template system as test_text2image.py.

Usage examples:
    # Minimal — free prompt
    python text2image_asset.py --prompt "a wooden treasure chest, closed, dusty"

    # With category and style presets
    python text2image_asset.py \\
        --prompt "a medieval tavern table" \\
        --category furniture --style fantasy

    # From YAML config
    python text2image_asset.py --prompt-config configs/my_asset.yml

    # Batch: generate the same asset at multiple seeds
    python text2image_asset.py --prompt "a red sports car" --category vehicle \\
        --style realistic --seeds 1 2 3 4

    # In-scene (no isolation, describe environment)
    python text2image_asset.py \\
        --prompt "a rusted iron lamp post" \\
        --category prop --style realistic \\
        --scene-context "rainy Victorian London street at night" \\
        --no-isolate
"""

import argparse
import gc
import inspect
import json
import re
from pathlib import Path
from string import Formatter
from typing import Any

import torch
from diffusers import AutoPipelineForText2Image


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"

# ---------------------------------------------------------------------------
# Default template (used when variables are provided but no template is set)
# ---------------------------------------------------------------------------
DEFAULT_PROMPT_TEMPLATE = (
    "A {style} {asset_type}, {description}, {lighting}, {background}"
)

# ---------------------------------------------------------------------------
# Style suffixes — added to the positive prompt
# ---------------------------------------------------------------------------
STYLE_SUFFIXES: dict[str, str] = {
    "realistic": (
        "photorealistic, 8k ultra-detailed, sharp focus, physically accurate materials, "
        "DSLR photography, ray-traced lighting"
    ),
    "cartoon": (
        "cartoon style, vibrant flat colors, cel-shaded, clean outlines, "
        "stylized proportions, animated series aesthetic"
    ),
    "fantasy": (
        "fantasy concept art, painterly, magical atmosphere, intricate details, "
        "ArtStation trending, ethereal glow, matte painting"
    ),
    "scifi": (
        "science fiction, futuristic design, hard-surface modeling, metallic surfaces, "
        "neon accents, cyberpunk aesthetic, CGI render"
    ),
    "lowpoly": (
        "low-poly 3D art, geometric facets, minimal polygons, clean stylized look, "
        "isometric-friendly, game asset style"
    ),
    "painterly": (
        "oil painting, impressionist brushstrokes, textured canvas, artistic, "
        "museum quality, classical art style"
    ),
    "anime": (
        "anime style, Japanese animation, clean line art, soft cel shading, "
        "vibrant colors, manga-inspired"
    ),
    "generic": "",
}

# ---------------------------------------------------------------------------
# Category-specific negative prompts
# ---------------------------------------------------------------------------
CATEGORY_NEGATIVES: dict[str, str] = {
    "character": (
        "multiple characters, crowd, extra limbs, missing limbs, bad hands, fused fingers, "
        "mutation, deformed face, bad anatomy, extra heads, cloned face, disfigured"
    ),
    "prop": (
        "person, human, hands, crowd, multiple objects, cluttered scene, busy background, "
        "floating parts, broken geometry, missing parts"
    ),
    "environment": (
        "person, human, crowd, floating objects, text, watermark, inconsistent lighting, "
        "cut-off edges, missing ground, incomplete scene"
    ),
    "vehicle": (
        "person inside, floating, deformed chassis, broken wheels, wrong perspective, "
        "multiple vehicles, unrecognizable make, cartoon when realistic requested"
    ),
    "furniture": (
        "person, human, broken, floating, multiple items, cluttered, misaligned parts, "
        "structurally impossible, disconnected legs"
    ),
    "creature": (
        "multiple creatures, mutation, extra limbs, human face on animal, "
        "deformed anatomy, unrecognizable species"
    ),
    "food": (
        "rotten, mold, disgusting, unappetizing, hands, person, insects, "
        "blurry, unrecognizable dish"
    ),
    "generic": "",
}

# ---------------------------------------------------------------------------
# Shared universal negative prompt (applied to all categories)
# ---------------------------------------------------------------------------
UNIVERSAL_NEGATIVE = (
    "blurry, low quality, low resolution, jpeg artifacts, watermark, signature, "
    "text overlay, deformed, distorted, ugly, bad proportions, out of frame, cropped"
)

# ---------------------------------------------------------------------------
# Isolation suffix (for clean product-style shots)
# ---------------------------------------------------------------------------
ISOLATION_SUFFIX = (
    "isolated on pure white background, clean studio shot, soft diffused lighting, "
    "no shadows on background, centered composition, no clutter"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def select_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA non disponibile: uso CPU (più lento).")
        return "cpu"
    return device


def get_next_filename(output_dir: str, prefix: str = "asset_2D", extension: str = ".png") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"{prefix}_(\d+){re.escape(extension)}$")
    max_num = 0
    for item in output_path.iterdir():
        match = pattern.match(item.name)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return output_path / f"{prefix}_{max_num + 1}{extension}"


def parse_key_value_arg(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"Argomento key=value non valido: {text}")
    key, value = text.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"Chiave mancante in: {text}")
    return key, value


def stringify_prompt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [stringify_prompt_value(item) for item in value]
        return ", ".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = []
        for key, nested_value in value.items():
            normalized = stringify_prompt_value(nested_value)
            if normalized:
                parts.append(f"{key}: {normalized}")
        return "; ".join(parts)
    return str(value).strip()


def load_prompt_config(path: Path) -> tuple[str | None, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt config non trovato: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None, {}
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON non valido: {path}") from exc
    elif suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyYAML non installato: pip install pyyaml") from exc
        try:
            data = yaml.safe_load(raw)
        except Exception as exc:
            raise ValueError(f"YAML non valido: {path}") from exc
    else:
        raise ValueError("Prompt config supportato solo per .json/.yml/.yaml")
    if data is None:
        return None, {}
    if not isinstance(data, dict):
        raise ValueError(f"Prompt config non valido (atteso dict): {path}")
    template_raw = data.get("template")
    template = str(template_raw).strip() if template_raw else None
    variables: dict[str, str] = {}
    for key, value in data.items():
        if str(key).strip() == "template":
            continue
        normalized = stringify_prompt_value(value)
        if normalized:
            variables[str(key).strip()] = normalized
    return template, variables


def extract_template_fields(template: str) -> list[str]:
    fields: list[str] = []
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        base_name = field_name.split(".", 1)[0].split("[", 1)[0].strip()
        if base_name and base_name not in fields:
            fields.append(base_name)
    return fields


def render_prompt_template(template: str, variables: dict[str, str]) -> str:
    missing = [f for f in extract_template_fields(template) if not variables.get(f, "").strip()]
    if missing:
        raise ValueError(
            f"Valori mancanti per il template: {', '.join(missing)}. "
            "Passali via --prompt-config o --prompt-var key=value."
        )
    try:
        return template.format_map(variables).strip()
    except Exception as exc:
        raise ValueError(f"Template non valido: {exc}") from exc


def resolve_input_prompt(
    prompt: str,
    prompt_template: str | None,
    prompt_config_path: str | None,
    prompt_var_args: list[str],
) -> str:
    variables: dict[str, str] = {}
    config_template: str | None = None
    if prompt_config_path:
        config_template, config_variables = load_prompt_config(Path(prompt_config_path))
        variables.update(config_variables)
    for item in prompt_var_args:
        key, value = parse_key_value_arg(item)
        variables[key] = value
    selected_template = (prompt_template or "").strip() or (config_template or "").strip()
    if not selected_template and variables:
        selected_template = DEFAULT_PROMPT_TEMPLATE
    rendered_template = render_prompt_template(selected_template, variables) if selected_template else ""
    free_prompt = prompt.strip()
    if rendered_template and free_prompt:
        return join_prompt_parts(rendered_template, free_prompt)
    return rendered_template or free_prompt


def join_prompt_parts(*parts: str) -> str:
    cleaned = [p.strip().strip(",") for p in parts if p and p.strip()]
    return ", ".join(cleaned)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_positive_prompt(
    base_prompt: str,
    style: str,
    scene_context: str,
    isolate: bool,
) -> str:
    parts = [base_prompt]

    style_suffix = STYLE_SUFFIXES.get(style, "")
    if style_suffix:
        parts.append(style_suffix)

    if isolate:
        parts.append(ISOLATION_SUFFIX)
    elif scene_context.strip():
        parts.append(f"set in {scene_context.strip()}")

    return join_prompt_parts(*parts)


def build_negative_prompt(
    category: str,
    extra_negative: str,
    isolate: bool,
) -> str:
    parts = [UNIVERSAL_NEGATIVE]

    cat_neg = CATEGORY_NEGATIVES.get(category, "")
    if cat_neg:
        parts.append(cat_neg)

    if isolate:
        parts.append(
            "dark background, colored background, gradient background, "
            "floor reflection, cast shadow on background, environment, scene"
        )

    if extra_negative.strip():
        parts.append(extra_negative.strip())

    return ", ".join(parts)


def count_tokens(tokenizer, text: str) -> int:
    tokens = tokenizer(text, return_tensors="pt", truncation=False)
    return int(tokens.input_ids.shape[-1])


def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    tokenized = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2D asset images from text for scene assembly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Prompt input ---
    parser.add_argument("--prompt", default="", help="Prompt testuale libero.")
    parser.add_argument(
        "--prompt-config", default=None,
        help="File config prompt (.json/.yaml/.yml) con 'template' e variabili.",
    )
    parser.add_argument(
        "--prompt-template", default=None,
        help="Template con placeholder, es: '{style} {asset_type} in {setting}'.",
    )
    parser.add_argument(
        "--prompt-var", action="append", default=[],
        help="Variabile per il template come key=value (ripetibile).",
    )

    # --- Asset description ---
    parser.add_argument(
        "--category",
        choices=list(CATEGORY_NEGATIVES.keys()),
        default="generic",
        help=(
            "Categoria dell'asset: character, prop, environment, vehicle, "
            "furniture, creature, food, generic."
        ),
    )
    parser.add_argument(
        "--style",
        choices=list(STYLE_SUFFIXES.keys()),
        default="generic",
        help=(
            "Stile visivo: realistic, cartoon, fantasy, scifi, lowpoly, "
            "painterly, anime, generic."
        ),
    )
    parser.add_argument(
        "--scene-context", default="",
        help="Contesto della scena, es: 'medieval forest at dusk'. Usato solo senza --isolate.",
    )
    parser.add_argument(
        "--no-isolate", action="store_true",
        help="Non forzare sfondo bianco isolato. Usa --scene-context per descrivere l'ambiente.",
    )

    # --- Negative prompt ---
    parser.add_argument("--extra-negative", default="", help="Negative prompt aggiuntivo.")

    # --- Generation ---
    parser.add_argument("--output", default=None, help="Percorso output immagine.")
    parser.add_argument("--output-dir", default="outputs/2D_assets", help="Cartella output (auto-numerazione).")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Lista di seed per generare più varianti (es: --seeds 1 2 3 4). Ignora --seed.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed singolo (default: 1234).")
    parser.add_argument("--size", type=int, default=512, help="Dimensione immagine quadrata.")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps.")
    parser.add_argument("--pag-scale", type=float, default=1.4, help="PAG scale.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG guidance scale.")

    # --- Model ---
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--model", default=MODEL_ID, help="Diffusers model id o percorso locale.")

    # --- Token management ---
    parser.add_argument("--max-tokens", type=int, default=77, help="Token cap.")
    parser.add_argument("--truncate-prompt", action="store_true", help="Tronca il prompt ai max token.")
    parser.add_argument("--show-prompt", action="store_true", help="Stampa i prompt finali prima della generazione.")

    args = parser.parse_args()

    device = select_device(args.device)
    isolate = not args.no_isolate
    seeds = args.seeds if args.seeds else [args.seed]

    # Resolve prompt
    try:
        input_prompt = resolve_input_prompt(
            prompt=args.prompt,
            prompt_template=args.prompt_template,
            prompt_config_path=args.prompt_config,
            prompt_var_args=args.prompt_var,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    if not input_prompt:
        parser.error("Fornisci --prompt oppure un template via --prompt-template/--prompt-config.")

    final_prompt = build_positive_prompt(
        base_prompt=input_prompt,
        style=args.style,
        scene_context=args.scene_context,
        isolate=isolate,
    )
    final_negative = build_negative_prompt(
        category=args.category,
        extra_negative=args.extra_negative,
        isolate=isolate,
    )

    print("=" * 60)
    print("ASSET 2D GENERATION")
    print("=" * 60)
    print(f"Categoria : {args.category}")
    print(f"Stile     : {args.style}")
    print(f"Isolato   : {'sì' if isolate else 'no'}")
    if args.scene_context:
        print(f"Scena     : {args.scene_context}")
    print(f"Seeds     : {seeds}")
    if args.show_prompt:
        print(f"\nPrompt    : {final_prompt}")
        print(f"Negative  : {final_negative}")
    print("-" * 60)

    # Load model once, generate all seeds
    print("Caricamento modello...")
    clear_memory()
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model,
        torch_dtype=dtype,
        enable_pag=True,
        pag_applied_layers=["blocks.(16|17|18|19)"],
    )
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")

    # Token check
    if args.max_tokens > 0 and hasattr(pipe, "tokenizer") and pipe.tokenizer is not None:
        try:
            token_count = count_tokens(pipe.tokenizer, final_prompt)
            if token_count > args.max_tokens:
                if args.truncate_prompt:
                    final_prompt = truncate_to_tokens(pipe.tokenizer, final_prompt, args.max_tokens)
                    print(f"Prompt troncato a {args.max_tokens} token.")
                else:
                    print(f"Warning: prompt da {token_count} token (>{args.max_tokens}), CLIP potrebbe tagliare.")
        except Exception as exc:
            print(f"Warning: controllo token saltato: {exc}")

    signature = inspect.signature(pipe.__call__)
    generated_paths = []

    for seed in seeds:
        # Determine output path
        if args.output and len(seeds) == 1:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = get_next_filename(args.output_dir, prefix="asset_2D")

        print(f"\nGenerando seed={seed} → {out_path}")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        call_kwargs = {
            "prompt": final_prompt,
            "negative_prompt": final_negative,
            "num_inference_steps": args.steps,
            "pag_scale": args.pag_scale,
            "guidance_scale": args.guidance_scale,
            "width": args.size,
            "height": args.size,
            "generator": generator,
        }
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in signature.parameters}
        image = pipe(**call_kwargs).images[0]
        image.save(out_path)
        generated_paths.append(out_path)
        print(f"  Salvato: {out_path}")

    del pipe
    clear_memory()

    print("\n" + "=" * 60)
    print(f"Completato — {len(generated_paths)} immagine/i generate")
    print("=" * 60)
    for p in generated_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()