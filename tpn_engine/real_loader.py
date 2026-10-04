"""
Real checkpoint loader for the TPN engine.

Unlike ``safetensors_loader`` (which targets a single-file MLX export) this
module handles what real, current checkpoints actually look like:

- **Sharded** safetensors: a model split across many files, described by a
  ``model.safetensors.index.json`` weight map.
- **bfloat16**: the dtype the Ling/Laguna checkpoints ship in. numpy has
  no native bf16, so it is decoded by widening to float32 via the
  high-word trick (bf16 is the top 16 bits of an fp32).
- **Standard HuggingFace config**: nested under ``text_config`` for many
  models, flat at the top level for others.
- **Architectural mismatch**: a checkpoint may be a hybrid/MoE model whose
  blocks do not line up one-to-one with TPNModel's dense block. Anything
  that cannot be mapped is reported rather than silently dropped.

Design rule: never silently destroy information. If a tensor cannot be
encoded faithfully, or a block does not fit the target architecture, the
loader says so in its report instead of quietly producing a wrong model.
"""

import json
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .inference import TPNConfig, TPNModel

# safetensors dtype tag -> numpy dtype
_SAFETENSORS_DTYPES = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "BF16": None,        # handled specially: widened to f32 on read
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U8": "u1",
    "BOOL": "u1",
    "U16": "<u2",
    "U32": "<u4",
    "U64": "<u8",
}


def _decode_bf16(raw: bytes) -> np.ndarray:
    """
    Decode bfloat16 bytes into float32.

    bf16 shares its exponent/sign layout with the high 16 bits of float32,
    so widening is a left-shift by 16 bits into a uint32, viewed as f32.
    Exact -- no precision is lost relative to the stored value.
    """
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


@dataclass
class TensorSpec:
    """Where a tensor lives and how it is stored."""
    name: str
    dtype: str
    shape: Tuple[int, ...]
    shard: str
    data_offsets: Tuple[int, int]

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class LoadReport:
    """What a load actually did, including everything it could not do."""
    loaded: Dict[str, str] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)
    config: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.skipped and not self.warnings

    def summary(self) -> str:
        lines = [
            f"loaded:   {len(self.loaded)} tensors",
            f"skipped:  {len(self.skipped)} tensors",
            f"warnings: {len(self.warnings)}",
        ]
        for k, v in self.skipped.items():
            lines.append(f"  SKIP {k}: {v}")
        for w in self.warnings:
            lines.append(f"  WARN {w}")
        return "\n".join(lines)


class ShardedSafetensors:
    """
    Reader for one or more safetensors shards, without loading the model.

    Only headers are parsed up front (tens of KB across all shards); tensor
    data is read on demand, so a 15 GB / 32-shard checkpoint can be
    inspected and sampled on a machine with far less memory.
    """

    def __init__(self, model_dir: str):
        self.model_dir = os.path.abspath(model_dir)
        if not os.path.isdir(self.model_dir):
            raise NotADirectoryError(f"model dir not found: {model_dir}")

        self.weight_map: Dict[str, str] = {}
        self._specs: Dict[str, TensorSpec] = {}
        self._shard_paths: Dict[str, str] = {}
        self._shard_base: Dict[str, int] = {}

        self._load_index()

    # -- discovery ------------------------------------------------------ #

    def _load_index(self) -> None:
        index_path = os.path.join(self.model_dir, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f).get("weight_map", {})

        if not self.weight_map:
            # Single-file checkpoint: no index, one shard on disk.
            single = os.path.join(self.model_dir, "model.safetensors")
            if not os.path.isfile(single):
                raise FileNotFoundError(
                    f"no model.safetensors or index found in {self.model_dir}"
                )
            for name, spec in self._read_header(single).items():
                if name == "__metadata__":
                    continue
                self.weight_map[name] = "model.safetensors"
        else:
            for shard in set(self.weight_map.values()):
                self._read_header(os.path.join(self.model_dir, shard))

        for name, shard in self.weight_map.items():
            spec = self._specs.get(name)
            if spec is None:
                continue
            self._specs[name] = TensorSpec(
                name=spec.name,
                dtype=spec.dtype,
                shape=spec.shape,
                shard=shard,
                data_offsets=spec.data_offsets,
            )

    def _read_header(self, path: str) -> Dict:
        """Parse one shard header and cache every tensor spec it declares."""
        shard = os.path.basename(path)
        if shard in self._shard_base:
            return {}

        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode("utf-8"))

        self._shard_base[shard] = 8 + header_size
        self._shard_paths[shard] = path

        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in self._specs:
                continue
            start, end = info["data_offsets"]
            self._specs[name] = TensorSpec(
                name=name,
                dtype=info["dtype"],
                shape=tuple(info["shape"]),
                shard=shard,
                data_offsets=(start, end),
            )
        return header

    # -- introspection --------------------------------------------------- #

    def tensor_names(self) -> List[str]:
        """Every tensor name in the checkpoint."""
        return list(self.weight_map.keys())

    def spec(self, name: str) -> Optional[TensorSpec]:
        """Storage details for one tensor, or None if absent."""
        return self._specs.get(name)

    def has(self, name: str) -> bool:
        return name in self._specs

    def shards(self) -> List[str]:
        return sorted(set(self.weight_map.values()))

    def num_shards(self) -> int:
        return len(set(self.weight_map.values()))

    def total_bytes(self) -> int:
        """Total on-disk size of all tensor payloads."""
        return sum(e - s for s, e in
                   (sp.data_offsets for sp in self._specs.values()))

    def dtypes(self) -> Dict[str, int]:
        """Histogram of dtypes across all tensors."""
        hist: Dict[str, int] = {}
        for sp in self._specs.values():
            hist[sp.dtype] = hist.get(sp.dtype, 0) + 1
        return hist

    def unique_name_patterns(self) -> List[str]:
        """
        Distinct tensor-name shapes with layer indices collapsed.

        ``model.layers.7.mlp.experts.12.gate_proj.weight`` becomes
        ``model.layers.N.mlp.experts.N.gate_proj.weight`` so an unfamiliar
        architecture can be read at a glance.
        """
        pats = set()
        for name in self._specs:
            pats.add(re.sub(r"\.\d+\.", ".N.", name))
        return sorted(pats)

    def layer_indices(self) -> List[int]:
        """Sorted, distinct transformer layer indices present."""
        found = set()
        for name in self._specs:
            m = re.search(r"\.layers\.(\d+)\.", name)
            if m:
                found.add(int(m.group(1)))
        return sorted(found)

    # -- data access ----------------------------------------------------- #

    def load(self, name: str) -> Optional[np.ndarray]:
        """
        Read one tensor fully into memory as float32 (or its native int type).

        Returns None if the tensor is not present.
        """
        spec = self._specs.get(name)
        if spec is None:
            return None
        return self._read(spec)

    def iter_tensors(self, names: Optional[List[str]] = None) -> Iterator[Tuple[str, np.ndarray]]:
        """Stream (name, array) pairs, reading one tensor at a time."""
        for name in (names if names is not None else self.tensor_names()):
            arr = self.load(name)
            if arr is not None:
                yield name, arr

    def _read(self, spec: TensorSpec) -> np.ndarray:
        path = self._shard_paths[spec.shard]
        start, end = spec.data_offsets
        nbytes = end - start

        with open(path, "rb") as f:
            f.seek(self._shard_base[spec.shard] + start)
            raw = f.read(nbytes)

        if spec.dtype == "BF16":
            data = _decode_bf16(raw)
        else:
            dt = _SAFETENSORS_DTYPES.get(spec.dtype)
            if dt is None:
                raise ValueError(f"unsupported dtype {spec.dtype} for {spec.name}")
            data = np.frombuffer(raw, dtype=np.dtype(dt))

        return data.reshape(spec.shape)

    def stats(self, name: str) -> Dict[str, float]:
        """min/max/std/absmean of one tensor (loads it, so use sparingly)."""
        arr = self.load(name)
        if arr is None or arr.size == 0:
            return {}
        f = arr.astype(np.float64, copy=False)
        return {
            "min": float(f.min()),
            "max": float(f.max()),
            "std": float(f.std()),
            "absmean": float(np.abs(f).mean()),
            "absmax": float(np.abs(f).max()),
            "out_of_unit": int((np.abs(f) > 1.0).sum()),
        }


# --------------------------------------------------------------------- #
# Config handling
# --------------------------------------------------------------------- #

def read_hf_config(model_dir: str) -> Dict:
    """
    Read config.json and flatten it into one lookup namespace.

    Checkpoints vary: some nest everything under ``text_config`` (the
    common HF multimodal/multiplexer layout), some keep it flat at the
    top level, and some split it across ``text_config`` plus siblings.
    Rather than assume one shape, merge the nested block *underneath* the
    top level so explicit top-level keys always win, and keep every
    original key reachable.
    """
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        raw = json.load(f)

    merged: Dict = {}

    # Merge known nested config blocks first, so flat top-level values
    # take precedence.
    for block_name in ("text_config", "llm_config", "decoder_config"):
        block = raw.get(block_name)
        if isinstance(block, dict):
            for k, v in block.items():
                merged.setdefault(k, v)

    # Then everything at the top level overrides.
    for k, v in raw.items():
        if isinstance(v, dict) and k in ("text_config", "llm_config",
                                         "decoder_config"):
            continue
        merged[k] = v

    return merged


def tpn_config_from_hf(hf: Dict) -> Tuple[TPNConfig, List[str]]:
    """
    Build a TPNConfig from a HuggingFace config dict.

    Returns (config, warnings). TPN has a dense SwiGLU block, so any
    architecture-specific extras (MoE experts, MLA lora ranks, conv
    layers) are reported here rather than being pretended away.
    """
    warnings: List[str] = []

    def pick(*names, default=None):
        for n in names:
            if n in hf and hf[n] is not None:
                return hf[n]
        return default

    hidden = int(pick("hidden_size", "n_embd", default=0) or 0)
    layers = int(pick("num_hidden_layers", "n_layer", default=0) or 0)
    heads = int(pick("num_attention_heads", "n_head", default=0) or 0)
    intermediate = int(pick("intermediate_size", "n_inner", "ffn_dim",
                             default=4 * hidden if hidden else 0) or 0)
    vocab = int(pick("vocab_size", default=0) or 0)
    max_pos = int(pick("max_position_embeddings", default=2048) or 2048)

    if not hidden or not layers:
        raise ValueError(
            "config.json is missing hidden_size/num_hidden_layers; "
            f"keys present: {sorted(hf)[:20]}"
        )

    head_dim = int(pick("head_dim", default=0) or 0) or max(1, hidden // max(1, heads))

    if pick("num_experts"):
        warnings.append(
            f"checkpoint is MoE ({pick('num_experts')} experts, "
            f"{pick('num_experts_per_tok')} active); TPNModel is dense, so "
            "expert weights are not mapped -- only shared/dense MLPs are."
        )
    for flag, label in [
        ("q_lora_rank", "MLA"),
        ("kv_lora_rank", "MLA"),
        ("mqa", "multi-query attention"),
        ("linear_attn", "linear/short-conv attention layers"),
    ]:
        if pick(flag):
            warnings.append(f"checkpoint uses {label} ({flag}); mapped as dense.")

    return (
        TPNConfig(
            hidden_size=hidden,
            num_layers=layers,
            num_heads=heads or max(1, hidden // head_dim),
            head_dim=head_dim,
            intermediate_size=intermediate,
            max_position_embeddings=max_pos,
            vocab_size=vocab,
        ),
        warnings,
    )


# --------------------------------------------------------------------- #
# Layer-name mapping
# --------------------------------------------------------------------- #

# Checkpoint suffix -> TPNModel block weight name. Covers both the plain
# ("q_proj") and the BailingMoE/Ling ("attention.q_proj") spellings.
_TENSOR_TO_BLOCK_WEIGHT = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "o_proj",
    "gate_proj": "gate_proj",
    "up_proj": "up_proj",
    "down_proj": "down_proj",
}


def _classify(name: str) -> Tuple[Optional[int], str]:
    """
    Parse a checkpoint tensor name into (layer_idx, canonical_key).

    Returns (None, "root") for tensors that are not part of a transformer
    block, and (idx, key) for ones that are.

    The key is the tensor's role with its grouping prefixes removed, e.g.
    ``attention.q_proj`` -> ``q_proj``.

    Important: per-expert MoE weights (``mlp.experts.12.gate_proj``) are
    keyed as ``experts.gate_proj`` rather than ``gate_proj``. Collapsing
    them to the same key as the dense block would let whichever expert
    sorted first be silently installed as the block FFN -- a wrong model
    that looks fine. Keeping the ``experts.`` marker makes them
    deliberately unmappable, and ``map_block_weights`` will skip them.
    """
    m = re.search(r"\.layers\.(\d+)\.(.+)", name)
    if not m:
        return None, "root"

    idx = int(m.group(1))
    tail = m.group(2)

    # Strip the attention/mlp grouping prefix and normalize aliases.
    tail = re.sub(r"^(attention|attn|self_attn)\.", "", tail)
    tail = re.sub(r"^(mlp|ffn|feed_forward)\.", "", tail)

    # A shared expert is dense-shaped and can legitimately stand in for the
    # block FFN, so it is reduced to the plain key. Individually routed
    # experts are NOT equivalent: they are one of many parallel branches,
    # and collapsing them would mislabel one branch as the whole FFN.
    tail = re.sub(r"^shared_experts\.", "", tail)
    tail = re.sub(r"^shared_?\d+\.", "", tail)

    # Everything still under `experts.N.` is a routed branch: keep the
    # marker so it never aliases a dense weight.
    tail = re.sub(r"^experts\.\d+\.", "experts.", tail)
    tail = re.sub(r"^experts\.", "experts.", tail)

    tail = tail.replace(".weight", "")
    return idx, tail


def map_block_weights(store: ShardedSafetensors, layer_idx: int,
                      prefix: str = "model.layers.") -> Dict[str, str]:
    """
    Find the seven TPN block weights for one layer.

    Returns a mapping of canonical weight name -> actual checkpoint tensor
    name. Only tensors whose shape fits the block are considered; anything
    else is left out so the caller can report it.
    """
    found: Dict[str, str] = {}
    target = f"{prefix}{layer_idx}."

    for name in store.tensor_names():
        if not name.startswith(target):
            continue
        idx, key = _classify(name)
        if idx != layer_idx:
            continue
        canonical = _TENSOR_TO_BLOCK_WEIGHT.get(key)
        if canonical and canonical not in found:
            found[canonical] = name
    return found


# --------------------------------------------------------------------- #
# Top-level loader
# --------------------------------------------------------------------- #

def load_checkpoint(model_dir: str, max_layers: Optional[int] = None,
                    create_model: bool = True,
                    dtype=np.float32) -> Tuple[Optional[TPNModel], LoadReport]:
    """
    Load a real sharded checkpoint into a TPNModel.

    This is the entry point that closes the gap the older loaders left:
    weights actually land in the model's blocks, so ``model.forward``
    uses them, and everything that could not be mapped is reported.

    Args:
        model_dir: directory containing config.json + safetensors shards.
        max_layers: load only the first N layers (rest stay random-initialized).
            Useful for a quick check on a large model.
        create_model: if False, only inspect and report; no model is built.
        dtype: numpy dtype for the stored weights.

    Returns:
        (model, report). ``model`` is None when ``create_model`` is False
        or when no layers could be mapped at all.
    """
    report = LoadReport()
    store = ShardedSafetensors(model_dir)

    hf = read_hf_config(model_dir)
    report.config = hf
    if not hf:
        report.warnings.append("config.json missing or unreadable")

    try:
        config, cfg_warnings = tpn_config_from_hf(hf)
        report.warnings.extend(cfg_warnings)
    except Exception as exc:
        report.warnings.append(f"could not derive TPNConfig: {exc}")
        return None, report

    available = store.layer_indices()
    n_load = len(available) if max_layers is None else min(max_layers, len(available))

    if not available:
        report.warnings.append(
            "no transformer blocks found; this does not look like a "
            "decoder-only checkpoint"
        )
        return None, report

    if max_layers is not None and max_layers < len(available):
        report.warnings.append(
            f"loaded {n_load} of {len(available)} layers (max_layers="
            f"{max_layers}); the rest stay randomly initialized"
        )

    if not create_model:
        report.warnings.append("create_model=False: no model built")
        return None, report

    model = TPNConfig_and_model(config)

    loaded_any = False
    for layer_idx in available[:n_load]:
        mapping = map_block_weights(store, layer_idx)
        if not mapping:
            report.skipped[f"layer {layer_idx}"] = "no mappable block weights"
            continue

        payload = {}
        for canonical, tensor_name in mapping.items():
            arr = store.load(tensor_name)
            if arr is None or arr.ndim != 2:
                report.skipped[tensor_name] = (
                    "not a 2-D matrix" if arr is not None else "unreadable"
                )
                continue
            payload[canonical] = np.ascontiguousarray(arr, dtype=dtype)

        outcome = model.load_block(layer_idx, payload, strict=False)
        for name, status in outcome.items():
            source = payload.get(name)
            origin = mapping.get(name, "<unknown>")
            if status == "loaded":
                report.loaded[f"layer {layer_idx}.{name}"] = origin
                loaded_any = True
            else:
                report.skipped[f"layer {layer_idx}.{name}"] = status
            if source is not None:
                report.config.setdefault("_mapped_shapes", {})[
                    f"{layer_idx}.{name}"] = tuple(source.shape)

    if not loaded_any:
        report.warnings.append("no weights could be loaded into any block")
        return None, report

    return model, report


def TPNConfig_and_model(config: TPNConfig) -> TPNModel:
    """Construct a TPNModel on the numpy backend for a given config."""
    return TPNModel(config, backend="numpy")
