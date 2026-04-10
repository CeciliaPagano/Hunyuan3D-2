#!/usr/bin/env python3
"""
Scarica i modelli Hunyuan3D sul Network Volume.
Uso: python3 download_models.py --variant 2.0 full
"""
import argparse, os
from huggingface_hub import snapshot_download

MODELS_DIR = "/workspace/models/hub"
os.makedirs(MODELS_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["2.0 full", "2.1", "2.5", "2.0 mini"], required=True)
args = parser.parse_args()

kw = dict(cache_dir=MODELS_DIR, local_dir_use_symlinks=False)

if args.variant == "2.0 mini":
    print("Scarico 2.0 mini shape...")
    snapshot_download("tencent/Hunyuan3D-2mini", allow_patterns="hunyuan3d-dit-v2-2.0 mini-turbo/**", **kw)
    print("Scarico 2.0 mini texture...")
    snapshot_download("tencent/Hunyuan3D-2", allow_patterns="hunyuan3d-paint-v2-0-turbo/**", **kw)

elif args.variant == "2.0 full":
    # Scarica solo i file necessari — esclude gli altri subfolder (turbo, 2.0 mini, 2.1, ecc.)
    print("Scarico 2.0 full shape (solo file root, esclusi altri subfolder)...")
    snapshot_download(
        "tencent/Hunyuan3D-2",
        ignore_patterns=[
            "hunyuan3d-paint-v2-0/**",
            "hunyuan3d-paint-v2-0-turbo/**",
            "hunyuan3d-dit-v2-2.0 mini*/**",
        ],
        **kw,
    )
    print("Scarico 2.0 full texture RGB...")
    snapshot_download("tencent/Hunyuan3D-2", allow_patterns=["hunyuan3d-paint-v2-0/**"], **kw)

elif args.variant == "2.1":
    print("Scarico 2.1 shape...")
    snapshot_download("tencent/Hunyuan3D-2.1", allow_patterns="hunyuan3d-dit-v2-1/**", **kw)
    print("Scarico 2.1 texture PBR...")
    snapshot_download("tencent/Hunyuan3D-2.1", allow_patterns="hunyuan3d-paintpbr-v2-1/**", **kw)

elif args.variant == "2.5":
    print("Scarico 2.5...")
    snapshot_download("tencent/Hunyuan3D-2.5", **kw)

print(f"\nDONE — modelli {args.variant} in {MODELS_DIR}")