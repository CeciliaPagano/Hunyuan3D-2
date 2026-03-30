#!/usr/bin/env python3
"""
Crea un modulo stub per bpy (Blender Python API) nel repo Hunyuan3D-2.1.
Necessario perché bpy non ha wheel per Python 3.12 su PyPI.
Lo stub soddisfa l'import senza richiedere Blender installato.

Usage:
    python /workspace/Hunyuan3D-2/run/create_bpy_stub.py
"""
from pathlib import Path

STUB = '''\
# bpy stub — soddisfa l'import senza Blender installato
# Generato da create_bpy_stub.py

class _Stub:
    def __getattr__(self, name):
        return _Stub()
    def __call__(self, *args, **kwargs):
        return _Stub()
    def __iter__(self):
        return iter([])
    def __bool__(self):
        return False
    def __len__(self):
        return 0

import sys as _sys
_s = _Stub()
for _m in ['bpy', 'bpy.ops', 'bpy.context', 'bpy.data', 'bpy.types',
           'bpy.props', 'bpy.utils', 'bpy.app']:
    _sys.modules[_m] = _s
'''

dest = Path('/workspace/Hunyuan3D-2.1/hy3dpaint/bpy.py')
dest.write_text(STUB)
print(f'Stub creato: {dest}')
