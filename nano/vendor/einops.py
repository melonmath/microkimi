"""Minimal `einops` compatibility shim: only rearrange(), and only the
patterns the vendored reference model files actually use:

  "b s ... -> (b s) ..."      merge the first two axes
  "... (h d) -> ... h d"      split the last axis in two (d given)
  "b t h d -> b t (h d)"      merge the last two axes

Anything else raises, loudly - a silent wrong reshape would corrupt the model.
If the real package IS installed it is loaded and re-exported instead
(NANO_VENDOR_FORCE_SHIM=1 forces the shim, used to validate it).
"""
import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))  # vendor/


def _load_real():
    if os.environ.get("NANO_VENDOR_FORCE_SHIM") == "1":
        return None
    search = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]
    spec = importlib.machinery.PathFinder.find_spec(__name__, search)
    if spec is None or spec.origin is None:
        return None
    if os.path.abspath(spec.origin).startswith(_HERE):
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[__name__] = module
    spec.loader.exec_module(module)
    return module


if _load_real() is None:

    def rearrange(t, pattern, **axes):
        src, _, dst = pattern.partition("->")
        src, dst = src.strip(), dst.strip()
        if src == "b s ..." and dst == "(b s) ...":
            return t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])
        if src == "... (h d)" and dst == "... h d":
            d = axes["d"]
            return t.reshape(*t.shape[:-1], t.shape[-1] // d, d)
        if src == "b t h d" and dst == "b t (h d)":
            return t.reshape(*t.shape[:-2], t.shape[-2] * t.shape[-1])
        raise NotImplementedError(f"einops shim: unsupported pattern {pattern!r}")
