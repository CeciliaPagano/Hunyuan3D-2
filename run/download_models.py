#!/usr/bin/env python3
"""
Scarica i modelli Hunyuan3D sul Network Volume.
Uso: python3 download_models.py --variant 2.0
"""
import argparse, os
from huggingface_hub import snapshot_download

MODELS_DIR = "/workspace/models/hub"
os.makedirs(MODELS_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["2.0", "2.1", "2.5", "mini"], required=True)
args = parser.parse_args()

kw = dict(cache_dir=MODELS_DIR, local_dir_use_symlinks=False)

if args.variant == "mini":
    print("Scarico mini shape...")
    snapshot_download("tencent/Hunyuan3D-2mini", allow_patterns="hunyuan3d-dit-v2-mini-turbo/**", **kw)
    print("Scarico mini texture...")
    snapshot_download("tencent/Hunyuan3D-2", allow_patterns="hunyuan3d-paint-v2-0-turbo/**", **kw)

elif args.variant == "2.0":
    # Scarica solo i file necessari — esclude gli altri subfolder (turbo, mini, 2.1, ecc.)
    print("Scarico 2.0 shape (solo file root, esclusi altri subfolder)...")
    snapshot_download(
        "tencent/Hunyuan3D-2",
        ignore_patterns=[
            "hunyuan3d-paint-v2-0/**",
            "hunyuan3d-paint-v2-0-turbo/**",
            "hunyuan3d-dit-v2-mini*/**",
        ],
        **kw,
    )
    print("Scarico 2.0 texture RGB...")
    snapshot_download("tencent/Hunyuan3D-2", allow_patterns=["hunyuan3d-paint-v2-0/**"], **kw)

elif args.variant == "2.1":
    print("Scarico 2.1 shape...")
    snapshot_download("tencent/Hunyuan3D-2.1", allow_patterns="hunyuan3d-dit-v2-1/**", **kw)
    print("Scarico 2.1 texture PBR...")
    snapshot_download("tencent/Hunyuan3D-2.1", allow_patterns="hunyuan3d-paint-v2-1/**", **kw)

elif args.variant == "2.5":
    print("Scarico 2.5...")
    snapshot_download("tencent/Hunyuan3D-2.5", **kw)

print(f"\nDONE — modelli {args.variant} in {MODELS_DIR}")