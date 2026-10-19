"""Convert the published Qwen3.8-Flash-Next v2 artifact to a v3 file collection.

Why this exists
---------------
`tools/upgrade_ninfer_v2_to_v3.py` rebuilds a v3 directory from per-model knowledge: its
`make_directory` is hand-written for the qwen3.6/3.8-27b and qwen3.6-35b-a3b families (a
`KNOWN_COUNTS` allowlist, a `FORMATS`/`LAYOUTS` table covering only those families, and a
hand-written parameter/binding map). It does not know Qwen3.8-Flash-Next (`architectures ==
["Qwen4ExpForCausalLM"]`, 1560 tensors), and every copy of it on this machine is identical.

This converter takes the other route: it does not re-derive the layout. The v2 payload is already
quantized, and the engine's own geometry checker reconciles all 1560 of its tensor objects
byte-for-byte (`ninfer-artifact-geometry-check`, 1560/1560, exit 0), so the bytes are copied
verbatim and only the directory is rebuilt.

What keeps it small: **in v2 the object name IS the logical parameter name** --
`text/layers/<i>/gdn/a_log`, `text/layers/<i>/mlp/experts/gate_up`,
`text/layers/<i>/ple/embedding/shards/<s>`, `vision/layers/<i>/attention/qkv`,
`frontend/tokenizer.json` -- and the ported qwen3_8_flash_next target binds exactly those names
(`bind_gdn(binder, prefix + "gdn/")` and friends). So `bindings[name] = {"object": name}` is an
identity map, not an invention; it is the shape `upgrade_ninfer_v2_to_v3.py` writes too
(`binding = {"object": obj["id"]}`).

Everything about the container comes from the checked-in v3 tooling under `tools/artifact/`:
`ArtifactWriter` plans the objects, resolves the JSON reserve and the 32 GB part split, streams the
payload and refuses to publish unless every object's coverage is exactly its declared length.
Nothing here re-implements the container.

The engine refuses a v2 entry by design (`src/artifact/reader.cpp`: "NInfer v2 artifact is not
supported"); this is the conversion that message points at.

Block-5 (--expert-gguf)
-----------------------
With `--expert-gguf …-00001-of-00002.gguf` the 96 routed-expert objects
(`text/layers/<i>/mlp/experts/{gate_up,down}`) are replaced by the raw ggml block rows of the
community IQ3_S release (gate and up concatenated per expert -- this container's row order -- and
down copied exactly as stored), while every other object stays a byte copy of the v2 payload. The
expert blocks are NEVER re-encoded: the artifact keeps the community release's own values, which is
what same-source quality anchors require. The gguf reader below is deliberately self-contained
(stdlib only, header + directory) so this tool acquires no new dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from artifact.formats import GGUF_FORMATS_BY_TYPE  # noqa: E402
from artifact.framing import PAYLOAD_ALIGNMENT  # noqa: E402
from artifact.schema import ResourceSpec, TensorSpec  # noqa: E402
from artifact.writer import ArtifactWriter  # noqa: E402

V2_MAGIC = b"NINFER\0\2"
CHUNK = 32 * 1024 * 1024

# v2 spelling -> v3 name. The four tensor representations this artifact uses were measured by the
# geometry checker (1237 BF16, 128 U4Z8G16_F16S, 96 NVFP4, 96 FP8_E4M3FN_ROW_F32S, 3 I64). The
# table is explicit so an unknown spelling fails loudly instead of being guessed.
FORMATS = {
    "BF16": "bf16",
    "FP32": "fp32",
    "I32": "int32",
    "I64": "int64",
    "NVFP4": "nvfp4",
    "U4Z8G16_F16S": "u4z8g16_f16",
    "FP8_E4M3FN_ROW_BF16S": "fp8_e4m3fn_row_bf16",
    "FP8_E4M3FN_ROW_F32S": "fp8_e4m3fn_row_f32",
}
# v2 spelling -> the v3 name this tree's own tooling accepts (tools/artifact/layouts.py). The v2
# container uses hyphens; the v3 constant names use underscores for the same layouts. Every entry is
# verified against the container's declared byte count (9/9 cases), so a name that does not size the
# object exactly as stored cannot get in.
LAYOUTS = {
    "contiguous-le-v1": "contiguous_le_v1",
    "row-split-k128-v1": "row_split_k128_v1",
    "blockscale-k16-m128x4-v1": "block_scale_k16_m128x4_v1",
    "expert-blockscale-k16-m128x4-v1": "expert_blockscale_k16_m128x4_v1",
    "row-scale-v1": "row_scale_v1",
    "row-scale-f32-v1": "row_scale_f32_v1",
    "packed-u4-g16-v1": "packed_u4_g16_v1",
}

RESOURCE_ROLES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def read_v2(path: Path):
    """Parse the v2 framing; return (identity, objects, payload_start)."""
    size = os.path.getsize(path)
    with path.open("rb") as stream:
        header = stream.read(16)
        if len(header) != 16 or header[:8] != V2_MAGIC:
            raise ValueError(f"{path}: expected a NInfer v2 entry")
        dirlen = struct.unpack_from("<Q", header, 8)[0]
        if dirlen == 0 or dirlen > size - 16:
            raise ValueError(f"{path}: invalid v2 directory length")
        directory = json.loads(stream.read(dirlen))
    if set(directory) != {"identity", "objects"}:
        raise ValueError(f"{path}: unexpected v2 directory keys {sorted(directory)}")
    payload_start = (16 + dirlen + PAYLOAD_ALIGNMENT - 1) // PAYLOAD_ALIGNMENT * PAYLOAD_ALIGNMENT
    return directory["identity"], directory["objects"], payload_start


# --------------------------------------------------------------- block-5: gguf expert swap

_GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q",
                 12: "d"}
_GGUF_TYPE_STRING, _GGUF_TYPE_ARRAY = 8, 9
_EXPERT_LAYERS = 48
_EXPERTS = 512
_EXPERT_ROWS = 640  # gate rows (and up rows) per expert; down has 2,560
_GGUF_HEAD_READ = 64 * 1024 * 1024  # enough for the header + directory of the block-5 release


def read_gguf_directory(path: Path):
    """(tensors, data_offset) of a GGUF v3 file; tensors = {name: (row-major shape, type id, off)}."""
    with path.open("rb") as stream:
        head = stream.read(_GGUF_HEAD_READ)
    if head[:4] != b"GGUF":
        raise ValueError(f"{path}: not a GGUF file")
    pos = [4]

    def take(fmt):
        values = struct.unpack_from("<" + fmt, head, pos[0])
        pos[0] += struct.calcsize(fmt)
        return values

    (version,) = take("I")
    if version != 3:
        raise ValueError(f"{path}: unsupported GGUF version {version}")
    (tensor_count,) = take("Q")
    (kv_count,) = take("Q")

    def skip_value(value_type):
        if value_type == _GGUF_TYPE_STRING:
            (length,) = take("Q")
            pos[0] += length
        elif value_type == _GGUF_TYPE_ARRAY:
            (element_type,) = take("I")
            (count,) = take("Q")
            if element_type in _GGUF_SCALARS:
                take(_GGUF_SCALARS[element_type] * count)
            else:
                for _ in range(count):
                    skip_value(element_type)
        elif value_type in _GGUF_SCALARS:
            take(_GGUF_SCALARS[value_type] * 1)
        else:
            raise ValueError(f"{path}: unsupported GGUF value type {value_type}")

    alignment = 32
    for _ in range(kv_count):
        (length,) = take("Q")
        key = head[pos[0]:pos[0] + length].decode("utf-8")
        pos[0] += length
        (value_type,) = take("I")
        if key == "general.alignment" and value_type in _GGUF_SCALARS:
            alignment = int(take(_GGUF_SCALARS[value_type])[0])
        else:
            skip_value(value_type)
    tensors = {}
    for _ in range(tensor_count):
        (length,) = take("Q")
        name = head[pos[0]:pos[0] + length].decode("utf-8")
        pos[0] += length
        (rank,) = take("I")
        dims = take("Q" * rank)
        (type_id,) = take("I")
        (offset,) = take("Q")
        # GGUF lists the fastest dimension first; this tool's shapes are row-major.
        tensors[name] = (tuple(reversed(dims)), type_id, offset)
    data_offset = (pos[0] + alignment - 1) // alignment * alignment
    return tensors, data_offset


def plan_expert_swap(tensors, data_offset):
    """{object name: (format, runs, total bytes)} for the 96 expert objects, raw blocks only.

    `runs` are (absolute file offset, length) pairs in the object's own write order: gate rows then
    up rows per expert for gate_up (this container stores gate [0:I) and up [I:2I) of each expert's
    1,280 rows), and the down tensor copied as one contiguous run.
    """
    plan = {}
    for layer in range(_EXPERT_LAYERS):
        gate, up, down = (tensors.get(f"blk.{layer}.ffn_{role}_exps.weight")
                          for role in ("gate", "up", "down"))
        if gate is None or up is None or down is None:
            raise ValueError(f"blk.{layer}: the gguf is missing one of ffn_gate/up/down_exps")
        if gate[0] != (_EXPERTS, _EXPERT_ROWS, 2_560) or up[0] != (_EXPERTS, _EXPERT_ROWS, 2_560):
            raise ValueError(f"blk.{layer}: expert gate/up shapes are not (512, 640, 2560)")
        if down[0] != (_EXPERTS, 2_560, _EXPERT_ROWS):
            raise ValueError(f"blk.{layer}: expert down shape is not (512, 2560, 640)")
        if gate[1] != up[1]:
            raise ValueError(f"blk.{layer}: gate and up are stored in different block types")
        gu_format, dn_format = GGUF_FORMATS_BY_TYPE.get(gate[1]), GGUF_FORMATS_BY_TYPE.get(down[1])
        if gu_format is None or dn_format is None:
            raise ValueError(f"blk.{layer}: ggml types {gate[1]}/{down[1]} are not storable")
        if 2_560 % gu_format.block_elements or _EXPERT_ROWS % dn_format.block_elements:
            raise ValueError(f"blk.{layer}: expert columns are not whole blocks of their types")
        gu_row = (2_560 // gu_format.block_elements) * gu_format.block_bytes
        dn_row = (_EXPERT_ROWS // dn_format.block_elements) * dn_format.block_bytes
        gu_runs = []
        for expert in range(_EXPERTS):
            gu_runs.append((data_offset + gate[2] + expert * _EXPERT_ROWS * gu_row,
                            _EXPERT_ROWS * gu_row))
            gu_runs.append((data_offset + up[2] + expert * _EXPERT_ROWS * gu_row,
                            _EXPERT_ROWS * gu_row))
        plan[f"text/layers/{layer}/mlp/experts/gate_up"] = (
            gu_format.name, gu_runs, _EXPERTS * 2 * _EXPERT_ROWS * gu_row)
        plan[f"text/layers/{layer}/mlp/experts/down"] = (
            dn_format.name, [(data_offset + down[2], _EXPERTS * 2_560 * dn_row)],
            _EXPERTS * 2_560 * dn_row)
    return plan


def parts() -> dict:
    """The directory blocks that are not the object list.

    The v2 container stores `{identity, objects}` and nothing else -- it has no component config --
    so the text config here is rebuilt from two sources, both of which were checked against this
    artifact's own geometry rather than taken on faith:

      * the shape of the published Qwen4Exp text config (its required and optional member names, as
        `tools/convert/qwen4_exp.py::text_config` normalises them); and
      * the values below, every one of which is confirmed by a tensor shape in the artifact:
        num_hidden_layers 48 (text/layers/0..47), layer_types alternating 3 linear + 1 full
        (36 gdn blocks at 0,1,2,4.. and 12 attention blocks at 3,7,11..), hidden_size 2560
        (token_embedding [248320,2560]), vocab_size 248320, num_attention_heads 24 x head_dim 256
        = 6144 (query_gate_key_value [16384,2560] / output [2560,6144]), num_key_value_heads 2,
        linear_num_key_heads 16 x linear_key_head_dim 128 = 2048 and linear_num_value_heads 48 x
        linear_value_head_dim 128 = 6144 (convolution [4,10240] = 2*2048+6144 gated by kernel 4),
        num_experts 512, num_experts_per_tok 10, moe_intermediate_size 640 and
        shared_expert_intermediate_size 640 (experts/gate_up [512,1280,2560], experts/down
        [512,2560,640], shared_expert/gate [640,2560]), hc_count 4 with hc_lowrank 320
        (hyper_connection/input_mix/down [320,...]), ple_embed_dim 2560, ple_layer_ids [2]
        one-indexed -> block 1 zero-indexed (the only block carrying ple/ tensors),
        ngram_size 3 with heads_per_ngram 8 and ngram_vocab_size_base 20000000
        (ple/embedding/shards/0 [2500012,160]), and max_position_embeddings 262144.

    A wrong value here does not corrupt any weight -- the payload is copied verbatim and the engine
    re-derives every tensor's size from the object table -- but it would make startup reject the
    model, which is exactly how this was found: the first run stopped with
    "Qwen3.5 config: text config: missing model_type".
    """
    layers = ["linear_attention"] * 48
    for index in range(3, 48, 4):
        layers[index] = "full_attention"
    # The member set below is copied from the engine's own validator, NOT from the upstream
    # converter: `src/models/qwen3_5/config.cpp::text()` calls require_members with a whitelist, so any
    # key outside it is rejected ("unknown member ..."). Verified by reading that call:
    #   required: architectures, model_type, hidden_size, vocab_size, num_hidden_layers,
    #             max_position_embeddings, tie_word_embeddings, rms_norm_eps, layer_types
    #   optional: num_attention_heads, num_key_value_heads, head_dim, rope_parameters,
    #             linear_num_key_heads, linear_key_head_dim, linear_num_value_heads,
    #             linear_value_head_dim, linear_conv_kernel_dim, intermediate_size, num_experts,
    #             num_experts_per_tok, moe_intermediate_size, shared_expert_intermediate_size
    # The upstream config carries more (hc_count, indexer_*, ple_*, ngram_*), and none of it is here:
    # the engine's TextConfig has no such members, because the ported target keeps those numbers as
    # C++ constants. Sending them is what produced "unknown member hc_count" on the first start.
    #
    # The architecture/model_type pair is likewise the engine's, from src/models/registry.cpp:18 —
    # `Qwen4ExpForConditionalGeneration` / `qwen4_exp`. The upstream text_config spells them
    # `Qwen4ExpForCausalLM` / `qwen4_exp_text`, which resolve_architecture does not accept.
    text_config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "hidden_size": 2560,
        "vocab_size": 248320,
        "num_hidden_layers": 48,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": False,
        "rms_norm_eps": 1e-06,
        "layer_types": layers,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "rope_parameters": {
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
        },
    }
    resources = {role: f"frontend/{role}" for role in RESOURCE_ROLES}
    return {
        "components": {
            "text": {
                "config": text_config,
                "resources": resources,
                # The proposal head's row count is the one component field the loader reads.
                "proposal": {"domain": "indexed", "rows": 248320},
            },
            "vision": {
                "config": {
                    "model_type": "qwen4_exp",
                    "depth": 27,
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "num_heads": 16,
                    "num_position_embeddings": 2304,
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                },
                "target": "text",
                "resources": {},
            },
            "mtp": {
                "config": {
                    "architectures": ["Qwen4ExpMTP"],
                    "layer_types": ["full_attention"],
                },
                "target": "text",
            },
        },
        "uses": [],
    }


def convert(
    input_path: Path,
    output_path: Path,
    *,
    max_file_bytes: int | None = None,
    limit: int | None = None,
    expert_gguf: Path | None = None,
    weights_id: str | None = None,
) -> dict:
    identity, old_objects, payload_start = read_v2(input_path)
    if set(identity) != {"model_id", "weights_id"}:
        raise ValueError(f"unexpected identity keys {sorted(identity)}")
    identity = dict(identity)
    if weights_id is not None:
        identity["weights_id"] = weights_id
    # `limit` exists so the whole path (directory, bindings, framing, publish) can be exercised on a
    # handful of objects before committing to the full 113 GB copy.
    if limit is not None:
        old_objects = old_objects[:limit]

    expert_swap = {}
    if expert_gguf is not None:
        expert_swap = plan_expert_swap(*read_gguf_directory(expert_gguf))

    specs, bindings, order = [], {}, []
    for value in old_objects:
        name = value["name"]
        if name in bindings:
            raise ValueError(f"duplicate object name {name!r}")
        if value["kind"] == "tensor":
            if name in expert_swap:
                # Block-5: raw ggml block rows instead of the v2 payload; the writer derives the
                # object's declared length from (format, layout, shape) and refuses to publish
                # unless the copied runs cover it exactly.
                fmt = expert_swap[name][0]
                specs.append(TensorSpec(name, tuple(value["shape"]), fmt, "gguf_blocks_v1"))
            else:
                try:
                    fmt, lay = FORMATS[value["format"]], LAYOUTS[value["layout"]]
                except KeyError as missing:
                    raise ValueError(f"{name}: unmapped v2 spelling {missing}") from None
                specs.append(TensorSpec(name, tuple(value["shape"]), fmt, lay))
        elif value["kind"] == "resource":
            if value["encoding"] != "raw-bytes-v1":
                raise ValueError(f"{name}: unsupported resource encoding {value['encoding']!r}")
            specs.append(ResourceSpec(name, value["bytes"]))
        else:
            raise ValueError(f"{name}: unsupported kind {value['kind']!r}")
        # Identity map: the v2 name is the parameter name the target binds. Only tensors appear
        # here -- schema.binding_parts rejects anything that is not a TensorObject -- while the six
        # frontend resources are declared through components.text.resources instead.
        if value["kind"] == "tensor":
            bindings[name] = {"object": name}
        order.append((name, payload_start + value["offset"], value["bytes"]))

    blocks = parts()
    kwargs = {} if max_file_bytes is None else {"max_file_bytes": max_file_bytes}
    with ArtifactWriter(
        output_path,
        specs,
        components=blocks["components"],
        bindings=bindings,
        uses=blocks["uses"],
        metadata={"name": identity["model_id"]},
        identity=dict(identity),
        **kwargs,
    ) as writer:
        # plan_objects assigns each object the offset the layout gives it, in this order. The v2
        # payload is copied object by object because the two layouts align differently, so the gaps
        # between objects are not the same run of bytes in both files.
        # os.pread is POSIX-only, so this seeks instead; it is the only reader on the descriptor.
        # Block-5: objects in `expert_swap` stream from the gguf instead, run by run (gate/up rows
        # per expert for gate_up, one contiguous run for down) -- still raw bytes, never decoded.
        source = open(input_path, "rb")
        gguf_source = open(expert_gguf, "rb") if expert_swap else None
        try:
            for name, begin, length in order:
                if name in expert_swap:
                    _, runs, _ = expert_swap[name]
                    written = 0
                    for at, run_length in runs:
                        done = 0
                        while done < run_length:
                            take = min(CHUNK, run_length - done)
                            gguf_source.seek(at + done)
                            data = gguf_source.read(take)
                            if len(data) != take:
                                raise IOError(f"{name}: short gguf read at {at + done}")
                            writer.write_region(name, written, data)
                            written += take
                            done += take
                    continue
                source.seek(begin)
                written = 0
                while written < length:
                    take = min(CHUNK, length - written)
                    data = source.read(take)
                    if len(data) != take:
                        raise IOError(f"{name}: short read at {begin + written}")
                    writer.write_region(name, written, data)
                    written += take
        finally:
            source.close()
            if gguf_source is not None:
                gguf_source.close()

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "objects": len(specs),
        "tensors": sum(1 for s in specs if isinstance(s, TensorSpec)),
        "resources": sum(1 for s in specs if isinstance(s, ResourceSpec)),
        "bindings": len(bindings),
        "files": len(writer.directory.files),
        "payload_bytes": sum(f.payload_bytes for f in writer.directory.files),
        "identity": identity,
    }
    if expert_swap:
        report["expert_swap"] = {
            "gguf": str(expert_gguf),
            "objects": len(expert_swap),
            "expert_bytes": sum(entry[2] for entry in expert_swap.values()),
            "formats": sorted({entry[0] for entry in expert_swap.values()}),
        }
    report_path = Path(str(output_path) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=None,
        help="part size limit; defaults to the writer's 32 GB",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="convert only the first N objects; for exercising the path on a small copy",
    )
    parser.add_argument(
        "--expert-gguf",
        type=Path,
        default=None,
        help="block-5: take the 96 routed-expert objects from this gguf (part 1 of the split "
        "release), raw blocks, no re-encode",
    )
    parser.add_argument(
        "--weights-id",
        default=None,
        help="override the weights_id the conversion carries (default: the v2 value)",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            convert(
                args.input,
                args.output,
                max_file_bytes=args.max_file_bytes,
                limit=args.limit,
                expert_gguf=args.expert_gguf,
                weights_id=args.weights_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
