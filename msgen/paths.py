"""Locations, and the shims that let us drive a PRISTINE TraceGen checkout.

The clone under TRACEGEN_DIR is never edited. TraceGen has three loading quirks
that earlier work solved by dropping launcher scripts inside the repo; here they
are applied from the parent process instead.
"""
from __future__ import annotations

import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# default: the sibling checkout used by the research repo; the public release keeps it under third_party/
_TG_DEFAULT = f"{WORKSPACE}/TraceGen" if os.path.isdir(f"{WORKSPACE}/TraceGen") else f"{WORKSPACE}/third_party/TraceGen"
TRACEGEN_DIR = os.environ.get("TRACEGEN_DIR", _TG_DEFAULT)
_GEN_DEFAULT = f"{WORKSPACE}/assets_ckpt/Generalist/tracegen_model.pth"
if not os.path.exists(_GEN_DEFAULT):
    _GEN_DEFAULT = f"{TRACEGEN_DIR}/assets_ckpt/Generalist/tracegen_model.pth"
GENERALIST = os.environ.get("TRACEGEN_GENERALIST", _GEN_DEFAULT)


def add_tracegen_to_path() -> str:
    if TRACEGEN_DIR not in sys.path:
        sys.path.insert(0, TRACEGEN_DIR)
    return TRACEGEN_DIR


class DictToNamespace:
    """Stand-in for the class the Generalist checkpoint pickles from its training
    `__main__`. Attribute access is what the unpickled config needs."""

    def __init__(self, d=None, **kw):
        for k, v in (d or {}).items():
            setattr(self, k, v)
        for k, v in kw.items():
            setattr(self, k, v)

    def __getattr__(self, name):          # tolerate absent keys
        raise AttributeError(name)


def patch_torch_load():
    """torch 2.11 defaults weights_only=True, which cannot unpickle the
    checkpoint's config object; and the pickled DictToNamespace must be injected
    into whatever module is currently `__main__` at each call, because the
    launcher runpy-swaps sys.modules['__main__'].
    """
    import torch

    if getattr(torch.load, "_msgen_patched", False):
        return
    original = torch.load

    def loader(*a, **kw):
        kw.setdefault("weights_only", False)
        main = sys.modules.get("__main__")
        if main is not None and not hasattr(main, "DictToNamespace"):
            main.DictToNamespace = DictToNamespace
        return original(*a, **kw)

    loader._msgen_patched = True
    torch.load = loader
