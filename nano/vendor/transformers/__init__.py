"""Minimal `transformers` compatibility shim.

The vendored reference model files (vendor/moonshot/, downloaded at runtime)
import a handful of names from the transformers package at module level. Hosts
that run the nano training stack do not otherwise need the package, so this
shim provides exactly that surface with plain Python classes:

  configuration_utils.PretrainedConfig     (kwargs -> attributes)
  activations.ACT2FN                       (a dict; the model registers "situ")
  cache_utils.Cache                        (type only, never instantiated here)
  generation.GenerationMixin               (base class, never exercised)
  masking_utils.create_causal_mask         (unused: NanoModel builds its mask)
  modeling_flash_attention_utils.FlashAttentionKwargs   (typing only)
  modeling_outputs.{BaseModelOutputWithPast, CausalLMOutputWithPast}
  modeling_utils.{ALL_ATTENTION_FUNCTIONS, PreTrainedModel}
  processing_utils.Unpack                  (typing.Unpack)
  pytorch_utils.ALL_LAYERNORM_LAYERS       (a list, appended at import)
  utils.{TransformersKwargs, auto_docstring, can_return_tuple, logging}
  utils.generic.{OutputRecorder, check_model_inputs}

If the real package IS installed, this shim loads and re-exports it instead
(zero behavior change on a fully provisioned host). Set NANO_VENDOR_FORCE_SHIM=1
to skip the real package even when present (used to validate the shim itself).
"""
import importlib.machinery
import importlib.util
import os
import sys
import types
import typing

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # vendor/


def _load_real():
    """Loads the real package from outside this vendor directory, if present."""
    if os.environ.get("NANO_VENDOR_FORCE_SHIM") == "1":
        return None
    search = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]
    spec = importlib.machinery.PathFinder.find_spec(__name__, search)
    if spec is None or spec.origin is None:
        return None
    if os.path.abspath(spec.origin).startswith(_HERE):
        return None  # that is this shim again
    module = importlib.util.module_from_spec(spec)
    sys.modules[__name__] = module
    spec.loader.exec_module(module)
    return module


if _load_real() is None:
    __version__ = "4.56.0"  # the vendored files assert >= 4.56.0 at import

    class PretrainedConfig:
        model_type = ""

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class PreTrainedModel(__import__("torch").nn.Module):
        config_class = None

        def __init__(self, config=None, *args, **kwargs):
            super().__init__()
            self.config = config

    class Cache:
        pass

    class GenerationMixin:
        pass

    class FlashAttentionKwargs(typing.TypedDict, total=False):
        pass

    class TransformersKwargs(typing.TypedDict, total=False):
        pass

    class _Output:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class OutputRecorder:
        def __init__(self, *args, **kwargs):
            pass

    def _identity_decorator(fn=None, **kwargs):
        return fn if callable(fn) else (lambda f: f)

    auto_docstring = _identity_decorator
    can_return_tuple = _identity_decorator
    check_model_inputs = _identity_decorator
    Unpack = typing.Unpack
    ALL_ATTENTION_FUNCTIONS = {}
    ALL_LAYERNORM_LAYERS = []

    def create_causal_mask(*args, **kwargs):
        raise NotImplementedError("transformers shim: create_causal_mask")

    class _Logger:
        def __init__(self, name):
            import logging as _logging
            self._log = _logging.getLogger(name)

        def __getattr__(self, attr):
            # warning_once and friends degrade to the plain stdlib methods
            return getattr(self._log, attr.replace("_once", ""), self._log.warning)

    class _LoggingModule:
        @staticmethod
        def get_logger(name):
            return _Logger(name)

    _logging_ns = _LoggingModule()

    _SUBMODULES = {
        "configuration_utils": {"PretrainedConfig": PretrainedConfig},
        "activations": {"ACT2FN": {}},
        "cache_utils": {"Cache": Cache},
        "generation": {"GenerationMixin": GenerationMixin},
        "masking_utils": {"create_causal_mask": create_causal_mask},
        "modeling_flash_attention_utils": {"FlashAttentionKwargs": FlashAttentionKwargs},
        "modeling_outputs": {
            "BaseModelOutputWithPast": _Output,
            "CausalLMOutputWithPast": _Output,
        },
        "modeling_utils": {
            "ALL_ATTENTION_FUNCTIONS": ALL_ATTENTION_FUNCTIONS,
            "PreTrainedModel": PreTrainedModel,
        },
        "processing_utils": {"Unpack": Unpack},
        "pytorch_utils": {"ALL_LAYERNORM_LAYERS": ALL_LAYERNORM_LAYERS},
        "utils": {
            "TransformersKwargs": TransformersKwargs,
            "auto_docstring": auto_docstring,
            "can_return_tuple": can_return_tuple,
            "logging": _logging_ns,
        },
        "utils.generic": {
            "OutputRecorder": OutputRecorder,
            "check_model_inputs": check_model_inputs,
        },
    }
    for _name, _attrs in _SUBMODULES.items():
        _mod = types.ModuleType(f"{__name__}.{_name}")
        _mod.__dict__.update(_attrs)
        sys.modules[f"{__name__}.{_name}"] = _mod
        # attribute access on the parent package (import transformers; transformers.utils)
        if "." not in _name:
            globals()[_name] = _mod
    ACT2FN = _SUBMODULES["activations"]["ACT2FN"]
