#!/usr/bin/env python3
"""Build and inspect external model adapter packs for microkimi.

An MKADAPT1 pack contains standard low-rank updates for named fp32 matrices:

    W <- W + multiplier * (lora_alpha / rank) * B @ A

The pack records the SHA-256 of the exact base .bin. The Rust engine refuses a
different base, verifies every factor, and folds one or more packs into private
copy-on-write pages at load. The base file is never modified and inference has
no live adapter branch.

Only elementary PEFT LoRA adapters are accepted. DoRA, rsLoRA, QALoRA, saved
biases, modules_to_save, layer replication, rank patterns, alpha patterns, and
non-fp32 base targets are rejected instead of being approximated.

Examples:
  python3 nano/adapter_pack.py create --base model.bin \
      --adapter /path/to/peft_adapter --name arithmetic --out arithmetic.mkap
  python3 nano/adapter_pack.py create --base model.bin \
      --adapter /path/to/peft_adapter --multiplier 0.5 \
      --name arithmetic-half --out arithmetic-half.mkap
  python3 nano/adapter_pack.py create --base model.bin \
      --adapter-config adapter_config.json \
      --adapter-model adapter_model.safetensors \
      --name arithmetic --out arithmetic.mkap
  python3 nano/adapter_pack.py inspect arithmetic.mkap
  python3 nano/adapter_pack.py --selftest
"""

import argparse
import hashlib
import json
import math
import os
import re
import struct
import tempfile
import unicodedata
from pathlib import Path

import numpy as np


MAGIC = b"MKADAPT1"
MODEL_MAGIC = {b"MKIM0001", b"MKIM0002"}
DTYPE_F32 = 0
ADAPTER_KEY = re.compile(
    r"^base_model\.model\.(?P<module>.+)\.lora_(?P<factor>[AB])\.weight$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Unknown fields are rejected so a future PEFT option cannot silently change
# the meaning of a fold. Fields in this set are either consumed below, checked
# to be disabled, or metadata that cannot affect the serialized A/B factors.
SUPPORTED_CONFIG_FIELDS = frozenset(
    {
        "alora_invocation_tokens",
        "alpha_pattern",
        "arrow_config",
        "auto_mapping",
        "base_model_name_or_path",
        "bias",
        "corda_config",
        "ensure_weight_tying",
        "eva_config",
        "exclude_modules",
        "fan_in_fan_out",
        "inference_mode",
        "init_lora_weights",
        "layer_replication",
        "layers_pattern",
        "layers_to_transform",
        "loftq_config",
        "lora_alpha",
        "lora_bias",
        "lora_dropout",
        "lora_ga_config",
        "megatron_config",
        "megatron_core",
        "modules_to_save",
        "peft_type",
        "peft_version",
        "qalora_group_size",
        "r",
        "rank_pattern",
        "revision",
        "runtime_config",
        "target_modules",
        "target_parameters",
        "task_type",
        "trainable_token_indices",
        "use_bdlora",
        "use_dora",
        "use_qalora",
        "use_rslora",
        "velora_config",
    }
)


def file_sha256(path, chunk_bytes=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact(handle, size, label):
    payload = handle.read(size)
    if len(payload) != size:
        raise ValueError(f"truncated {label}")
    return payload


def _contains_control(value):
    return any(unicodedata.category(character) == "Cc" for character in value)


def _reject_json_constant(token):
    raise ValueError(f"non-finite JSON number {token}")


def validated_multiplier(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value == 0.0
        or abs(value) > 1e6
    ):
        raise ValueError(
            "LoRA multiplier must be finite, non-zero, and at most 1e6 in magnitude"
        )
    return float(value)


def read_model_directory(path):
    """Return {tensor: {dtype, dims, offset, size}} for MKIM0001/2."""
    with open(path, "rb") as handle:
        magic = _read_exact(handle, 8, "model magic")
        if magic not in MODEL_MAGIC:
            raise ValueError(f"{path}: expected an MKIM0001 or MKIM0002 model")
        if magic == b"MKIM0002":
            config_size = struct.unpack("<I", _read_exact(handle, 4, "config size"))[0]
            _read_exact(handle, config_size, "model config")
        count = struct.unpack("<I", _read_exact(handle, 4, "tensor count"))[0]
        if count > 10_000_000:
            raise ValueError("unreasonable model tensor count")
        entries = {}
        for _ in range(count):
            name_size = struct.unpack("<H", _read_exact(handle, 2, "name size"))[0]
            name = _read_exact(handle, name_size, "tensor name").decode("utf-8")
            dtype, ndim = struct.unpack("<BB", _read_exact(handle, 2, "tensor type"))
            dims = list(struct.unpack(f"<{ndim}I", _read_exact(handle, ndim * 4, "tensor dims")))
            offset, size = struct.unpack("<QQ", _read_exact(handle, 16, "tensor range"))
            if name in entries:
                raise ValueError(f"duplicate model tensor {name!r}")
            entries[name] = {
                "dtype": dtype,
                "dims": dims,
                "offset": offset,
                "size": size,
            }
    file_size = os.path.getsize(path)
    for name, entry in entries.items():
        if entry["offset"] + entry["size"] > file_size:
            raise ValueError(f"model tensor {name!r} extends beyond the file")
    return entries


class Safetensors:
    """Minimal read-only safetensors reader for LoRA factors."""

    DTYPES = {
        "F32": np.dtype("<f4"),
        "F16": np.dtype("<f2"),
        "BF16": np.dtype("<u2"),
    }

    def __init__(self, path):
        self.path = str(path)
        with open(path, "rb") as handle:
            header_size = struct.unpack("<Q", _read_exact(handle, 8, "safetensors header size"))[0]
            if header_size == 0 or header_size > (64 << 20):
                raise ValueError("invalid safetensors header size")
            pairs = json.loads(
                _read_exact(handle, header_size, "safetensors header"),
                object_pairs_hook=_pairs_without_duplicates,
                parse_constant=_reject_json_constant,
            )
        if not isinstance(pairs, dict):
            raise ValueError("safetensors header must be an object")
        self.data_start = 8 + header_size
        self.entries = {}
        ranges = []
        for name, entry in pairs.items():
            if name == "__metadata__":
                continue
            if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
                raise ValueError(f"invalid safetensors entry {name!r}")
            dtype = entry["dtype"]
            shape = entry["shape"]
            offsets = entry["data_offsets"]
            if dtype not in self.DTYPES:
                raise ValueError(f"{name}: unsupported factor dtype {dtype}")
            if (
                not isinstance(shape, list)
                or not shape
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
            ):
                raise ValueError(f"{name}: invalid shape")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in offsets)
                or offsets[1] < offsets[0]
            ):
                raise ValueError(f"{name}: invalid data_offsets")
            expected = math.prod(shape) * self.DTYPES[dtype].itemsize
            if offsets[1] - offsets[0] != expected:
                raise ValueError(f"{name}: payload size does not match shape")
            self.entries[name] = (dtype, shape, offsets)
            ranges.append((offsets[0], offsets[1], name))
        cursor = 0
        for start, end, name in sorted(ranges):
            if start != cursor:
                raise ValueError(f"{name}: safetensors payload is not canonical and contiguous")
            cursor = end
        if self.data_start + cursor != os.path.getsize(path):
            raise ValueError("safetensors file contains trailing payload bytes")
        self.mapping = np.memmap(path, dtype=np.uint8, mode="r")

    def tensor(self, name):
        dtype, shape, (start, end) = self.entries[name]
        absolute_start = self.data_start + start
        absolute_end = self.data_start + end
        if absolute_end > self.mapping.size:
            raise ValueError(f"{name}: factor extends beyond the safetensors file")
        raw = self.mapping[absolute_start:absolute_end]
        values = np.frombuffer(raw, dtype=self.DTYPES[dtype]).reshape(shape)
        if dtype == "BF16":
            values = (values.astype(np.uint32) << 16).view(np.float32)
        else:
            values = values.astype(np.float32, copy=False)
        values = np.ascontiguousarray(values, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: factor contains a non-finite value")
        return values


def _pairs_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def load_standard_lora_config(path):
    with open(path, "rb") as handle:
        config = json.loads(
            handle.read(),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    if not isinstance(config, dict):
        raise ValueError("adapter_config must be an object")
    unknown = sorted(set(config) - SUPPORTED_CONFIG_FIELDS)
    if unknown:
        raise ValueError("unsupported adapter_config fields: " + ", ".join(unknown))
    if config.get("peft_type") != "LORA":
        raise ValueError("adapter_config peft_type must be LORA")
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("adapter_config r must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise ValueError("adapter_config lora_alpha must be finite")
    disabled = (
        "ensure_weight_tying",
        "use_bdlora",
        "use_dora",
        "use_qalora",
        "use_rslora",
        "lora_bias",
    )
    for field in disabled:
        if config.get(field, False) not in (None, False):
            raise ValueError(f"unsupported adapter feature {field}")
    if config.get("fan_in_fan_out", False) is not False:
        raise ValueError("fan_in_fan_out adapters are not supported")
    if config.get("bias", "none") not in (None, "none"):
        raise ValueError("saved LoRA biases are not supported")
    for field in (
        "modules_to_save",
        "rank_pattern",
        "alpha_pattern",
        "layer_replication",
        "target_parameters",
        "trainable_token_indices",
        "alora_invocation_tokens",
        "arrow_config",
        "corda_config",
        "eva_config",
        "loftq_config",
        "lora_ga_config",
        "megatron_config",
        "runtime_config",
        "velora_config",
    ):
        if config.get(field) not in (None, {}, []):
            raise ValueError(f"unsupported adapter feature {field}")
    dropout = config.get("lora_dropout", 0.0)
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(dropout)
        or not 0.0 <= dropout <= 1.0
    ):
        raise ValueError("adapter_config lora_dropout must be between zero and one")
    target_modules = config.get("target_modules")
    if target_modules is not None and not (
        isinstance(target_modules, str)
        or (
            isinstance(target_modules, list)
            and target_modules
            and all(isinstance(value, str) and value for value in target_modules)
        )
    ):
        raise ValueError("adapter_config target_modules must be a string or non-empty string list")
    return rank, float(alpha)


def target_candidates(module):
    candidates = []

    def add(value):
        name = value + ".weight"
        if name not in candidates:
            candidates.append(name)

    add(module)
    value = module
    for prefix in ("model.", "language_model.", "model.language_model."):
        if value.startswith(prefix):
            add(value[len(prefix):])
    marker = value.find("layers.")
    if marker >= 0:
        tail = value[marker:]
        for prefix in ("", "model.", "language_model.", "model.language_model."):
            add(prefix + tail)
    for root in ("embed_tokens", "lm_head", "norm"):
        marker = value.rfind(root)
        if marker >= 0:
            add(value[marker:])
    return candidates


def pair_adapter_factors(safetensors):
    pairs = {}
    unexpected = []
    for key in safetensors.entries:
        match = ADAPTER_KEY.fullmatch(key)
        if match is None:
            unexpected.append(key)
            continue
        module = match.group("module")
        factor = match.group("factor")
        if factor in pairs.setdefault(module, {}):
            raise ValueError(f"duplicate LoRA {factor} factor for {module}")
        pairs[module][factor] = key
    if unexpected:
        raise ValueError("unsupported adapter tensor keys: " + ", ".join(sorted(unexpected)))
    if not pairs:
        raise ValueError("adapter contains no LoRA factor pairs")
    for module, factors in pairs.items():
        if set(factors) != {"A", "B"}:
            raise ValueError(f"unpaired LoRA factors for {module}")
    return pairs


def build_pack(base_path, config_path, adapter_path, name, out_path, multiplier=1.0):
    if not name or len(name.encode("utf-8")) > 128 or _contains_control(name):
        raise ValueError("pack name must contain no control characters and fit in 128 UTF-8 bytes")
    multiplier = validated_multiplier(multiplier)
    model_entries = read_model_directory(base_path)
    rank, alpha = load_standard_lora_config(config_path)
    adapter = Safetensors(adapter_path)
    pairs = pair_adapter_factors(adapter)
    targets = []
    for module, factors in pairs.items():
        matches = [candidate for candidate in target_candidates(module) if candidate in model_entries]
        if len(matches) != 1:
            raise ValueError(
                f"{module}: expected exactly one base tensor match, found {matches or 'none'}"
            )
        tensor = matches[0]
        entry = model_entries[tensor]
        if entry["dtype"] != DTYPE_F32:
            raise ValueError(f"{tensor}: adapter packs require an fp32 base target")
        if len(entry["dims"]) != 2:
            raise ValueError(f"{tensor}: adapter target must be a matrix")
        out_features, in_features = entry["dims"]
        a = adapter.tensor(factors["A"])
        b = adapter.tensor(factors["B"])
        if a.shape != (rank, in_features):
            raise ValueError(
                f"{module}: LoRA A shape {a.shape}, expected {(rank, in_features)}"
            )
        if b.shape != (out_features, rank):
            raise ValueError(
                f"{module}: LoRA B shape {b.shape}, expected {(out_features, rank)}"
            )
        targets.append((tensor, out_features, in_features, a, b))
    targets.sort(key=lambda item: item[0])

    payload = bytearray()
    manifest_targets = []
    scale = multiplier * alpha / rank
    if not math.isfinite(scale) or scale == 0.0 or abs(scale) > 1e6:
        raise ValueError("LoRA alpha/rank scale is outside the supported range")
    for tensor, out_features, in_features, a, b in targets:
        a_bytes = a.tobytes(order="C")
        b_bytes = b.tobytes(order="C")
        a_offset = len(payload)
        payload.extend(a_bytes)
        b_offset = len(payload)
        payload.extend(b_bytes)
        manifest_targets.append(
            {
                "tensor": tensor,
                "out": out_features,
                "in": in_features,
                "rank": rank,
                "scale": scale,
                "a_offset": a_offset,
                "a_bytes": len(a_bytes),
                "a_sha256": hashlib.sha256(a_bytes).hexdigest(),
                "b_offset": b_offset,
                "b_bytes": len(b_bytes),
                "b_sha256": hashlib.sha256(b_bytes).hexdigest(),
            }
        )
    print(f"hashing base model: {base_path}", flush=True)
    manifest = {
        "format": 1,
        "name": name,
        "base_sha256": file_sha256(base_path),
        "fold": "f32_ba_v1",
        "targets": manifest_targets,
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(manifest_bytes) > (16 << 20):
        raise ValueError("adapter-pack manifest exceeds 16 MiB")
    body = MAGIC + struct.pack("<I", len(manifest_bytes)) + manifest_bytes + payload
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_path.parent, prefix=out_path.name + ".", delete=False) as tmp:
        tmp.write(body)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, out_path)
    summary = {
        "path": str(out_path),
        "name": name,
        "sha256": hashlib.sha256(body).hexdigest(),
        "base_sha256": manifest["base_sha256"],
        "targets": len(manifest_targets),
        "factor_bytes": len(payload),
        "multiplier": multiplier,
    }
    print(json.dumps(summary, indent=2))
    return summary


def inspect_pack(path, verify=True):
    body = Path(path).read_bytes()
    if len(body) < 12 or body[:8] != MAGIC:
        raise ValueError(f"{path}: expected MKADAPT1")
    manifest_size = struct.unpack("<I", body[8:12])[0]
    if manifest_size == 0 or manifest_size > (16 << 20) or 12 + manifest_size > len(body):
        raise ValueError("invalid adapter-pack manifest length")
    raw = body[12:12 + manifest_size]
    if raw.rstrip() != raw:
        raise ValueError("adapter-pack manifest must not contain trailing whitespace")
    manifest = json.loads(
        raw,
        object_pairs_hook=_pairs_without_duplicates,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(manifest, dict) or set(manifest) != {
        "format", "name", "base_sha256", "fold", "targets"
    }:
        raise ValueError("invalid adapter-pack manifest fields")
    if manifest["format"] != 1 or manifest["fold"] != "f32_ba_v1":
        raise ValueError("unsupported adapter-pack format")
    if (
        not isinstance(manifest["name"], str)
        or not manifest["name"]
        or len(manifest["name"].encode("utf-8")) > 128
        or _contains_control(manifest["name"])
    ):
        raise ValueError("invalid adapter-pack name")
    if not isinstance(manifest["base_sha256"], str) or SHA256.fullmatch(manifest["base_sha256"]) is None:
        raise ValueError("invalid base SHA-256")
    if not isinstance(manifest["targets"], list) or not manifest["targets"]:
        raise ValueError("adapter pack has no targets")
    payload = body[12 + manifest_size:]
    cursor = 0
    tensors = []
    for index, target in enumerate(manifest["targets"]):
        required = {
            "tensor", "out", "in", "rank", "scale",
            "a_offset", "a_bytes", "a_sha256",
            "b_offset", "b_bytes", "b_sha256",
        }
        if not isinstance(target, dict) or set(target) != required:
            raise ValueError(f"invalid target fields at index {index}")
        tensor = target["tensor"]
        if (
            not isinstance(tensor, str)
            or not tensor
            or len(tensor.encode("utf-8")) > 1024
            or _contains_control(tensor)
        ):
            raise ValueError(f"invalid tensor name at target {index}")
        for field in ("out", "in", "rank", "a_offset", "a_bytes", "b_offset", "b_bytes"):
            value = target[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"target {index} {field} must be a non-negative integer")
        if target["out"] == 0 or target["in"] == 0 or target["rank"] == 0:
            raise ValueError(f"target {index} dimensions must be positive")
        scale = target["scale"]
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or scale == 0.0
            or abs(scale) > 1e6
        ):
            raise ValueError(f"invalid scale at target {index}")
        expected_a = target["rank"] * target["in"] * 4
        expected_b = target["out"] * target["rank"] * 4
        if target["a_bytes"] != expected_a or target["b_bytes"] != expected_b:
            raise ValueError(f"factor byte count does not match dimensions at target {index}")
        if any(
            not isinstance(target[field], str) or SHA256.fullmatch(target[field]) is None
            for field in ("a_sha256", "b_sha256")
        ):
            raise ValueError(f"invalid factor SHA-256 at target {index}")
        if target["a_offset"] != cursor:
            raise ValueError(f"non-canonical A offset at target {index}")
        cursor += target["a_bytes"]
        if target["b_offset"] != cursor:
            raise ValueError(f"non-canonical B offset at target {index}")
        cursor += target["b_bytes"]
        if cursor > len(payload):
            raise ValueError(f"target {index} extends beyond the payload")
        if verify:
            a = payload[target["a_offset"]:target["a_offset"] + target["a_bytes"]]
            b = payload[target["b_offset"]:target["b_offset"] + target["b_bytes"]]
            if hashlib.sha256(a).hexdigest() != target["a_sha256"]:
                raise ValueError(f"target {index} A SHA-256 mismatch")
            if hashlib.sha256(b).hexdigest() != target["b_sha256"]:
                raise ValueError(f"target {index} B SHA-256 mismatch")
            if not np.isfinite(np.frombuffer(a, dtype="<f4")).all() or not np.isfinite(
                np.frombuffer(b, dtype="<f4")
            ).all():
                raise ValueError(f"target {index} contains a non-finite factor")
        tensors.append(tensor)
    if cursor != len(payload):
        raise ValueError("trailing adapter-pack payload bytes")
    if tensors != sorted(set(tensors)):
        raise ValueError("adapter targets must be unique and sorted")
    summary = {
        "path": str(path),
        "name": manifest["name"],
        "sha256": hashlib.sha256(body).hexdigest(),
        "base_sha256": manifest["base_sha256"],
        "fold": manifest["fold"],
        "targets": tensors,
        "factor_bytes": len(payload),
        "verified": bool(verify),
    }
    return summary


def _write_safetensors(path, tensors):
    header = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        tensor = np.ascontiguousarray(tensor, dtype="<f4")
        start = len(payload)
        raw = tensor.tobytes()
        payload.extend(raw)
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [start, len(payload)],
        }
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    Path(path).write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def _write_tiny_model(path):
    name = b"layers.0.proj.weight"
    config = b"{}"
    entry_size = 2 + len(name) + 2 + 2 * 4 + 16
    offset = 8 + 4 + len(config) + 4 + entry_size
    weight = np.arange(6, dtype="<f4").reshape(2, 3).tobytes()
    body = bytearray(b"MKIM0002" + struct.pack("<I", len(config)) + config)
    body.extend(struct.pack("<I", 1))
    body.extend(struct.pack("<H", len(name)) + name)
    body.extend(struct.pack("<BBIIQQ", DTYPE_F32, 2, 2, 3, offset, len(weight)))
    body.extend(weight)
    Path(path).write_bytes(body)


def selftest():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base = root / "base.bin"
        config = root / "adapter_config.json"
        adapter = root / "adapter_model.safetensors"
        pack = root / "skill.mkap"
        _write_tiny_model(base)
        config.write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "r": 1,
                    "lora_alpha": 2,
                    "fan_in_fan_out": False,
                    "bias": "none",
                    "loftq_config": {},
                    "lora_ga_config": None,
                    "peft_version": "0.19.1",
                }
            ),
            encoding="utf-8",
        )
        unsupported = root / "unsupported.json"
        unsupported.write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "r": 1,
                    "lora_alpha": 2,
                    "future_semantics": True,
                }
            ),
            encoding="utf-8",
        )
        try:
            load_standard_lora_config(unsupported)
        except ValueError as error:
            assert "unsupported adapter_config fields" in str(error)
        else:
            raise AssertionError("unknown adapter semantics were accepted")
        semantic = root / "semantic.json"
        semantic.write_text(
            '{"peft_type":"LORA","r":1,"lora_alpha":2,"loftq_config":{"x":NaN}}',
            encoding="utf-8",
        )
        try:
            load_standard_lora_config(semantic)
        except ValueError as error:
            assert "non-finite JSON number" in str(error)
        else:
            raise AssertionError("non-finite JSON was accepted")
        _write_safetensors(
            adapter,
            {
                "base_model.model.model.layers.0.proj.lora_A.weight": np.array([[1, 2, 3]], np.float32),
                "base_model.model.model.layers.0.proj.lora_B.weight": np.array([[4], [5]], np.float32),
            },
        )
        summary = build_pack(base, config, adapter, "selftest", pack, multiplier=0.25)
        checked = inspect_pack(pack)
        assert checked["sha256"] == summary["sha256"]
        assert checked["base_sha256"] == file_sha256(base)
        assert checked["targets"] == ["layers.0.proj.weight"]
        manifest_size = struct.unpack("<I", pack.read_bytes()[8:12])[0]
        manifest = json.loads(pack.read_bytes()[12:12 + manifest_size])
        assert manifest["targets"][0]["scale"] == 0.5
        assert summary["multiplier"] == 0.25
        for invalid in (False, 0, float("inf"), 1e7):
            try:
                validated_multiplier(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid multiplier {invalid!r} was accepted")
        assert "model.language_model.layers.0.proj.weight" in target_candidates(
            "model.layers.0.proj"
        )
        corrupted = bytearray(pack.read_bytes())
        corrupted[-1] ^= 1
        bad = root / "bad.mkap"
        bad.write_bytes(corrupted)
        try:
            inspect_pack(bad)
        except ValueError as error:
            assert "SHA-256 mismatch" in str(error)
        else:
            raise AssertionError("corrupted factor was accepted")
    print("adapter_pack selftest OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    create = subparsers.add_parser("create", help="build MKADAPT1 from a standard PEFT LoRA")
    create.add_argument("--base", required=True, help="exact microkimi .bin base")
    create.add_argument("--adapter", help="PEFT adapter directory")
    create.add_argument("--adapter-config")
    create.add_argument("--adapter-model")
    create.add_argument("--name", required=True)
    create.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
        help="multiply the adapter's declared alpha/rank scale (default: 1)",
    )
    create.add_argument("--out", required=True)
    inspect = subparsers.add_parser("inspect", help="validate and describe an MKADAPT1 pack")
    inspect.add_argument("pack")
    inspect.add_argument("--no-verify", action="store_true", help="skip factor hashes")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.command == "create":
        if args.adapter is not None:
            if args.adapter_config is not None or args.adapter_model is not None:
                create.error("--adapter cannot be combined with explicit adapter paths")
            adapter_root = Path(args.adapter)
            adapter_config = adapter_root / "adapter_config.json"
            adapter_model = adapter_root / "adapter_model.safetensors"
        else:
            if args.adapter_config is None or args.adapter_model is None:
                create.error("pass --adapter DIR or both --adapter-config and --adapter-model")
            adapter_config = Path(args.adapter_config)
            adapter_model = Path(args.adapter_model)
        build_pack(
            args.base,
            adapter_config,
            adapter_model,
            args.name,
            args.out,
            args.multiplier,
        )
    elif args.command == "inspect":
        print(json.dumps(inspect_pack(args.pack, not args.no_verify), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
