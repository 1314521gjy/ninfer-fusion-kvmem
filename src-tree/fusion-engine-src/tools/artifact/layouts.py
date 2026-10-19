"""Persistent tensor layouts shared by NInfer converters and reference models.

This module maps already-selected numeric words to their registered byte layout.
It deliberately does not quantize floating-point source weights or assign model
roles to tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import operator
from types import MappingProxyType
from typing import Sequence


from .formats import (
    GGUF_FORMATS,
    DirectFormat,
    Fp8RowFormat,
    GgufFormat,
    Nvfp4Format,
    NumericFormat,
    QuantFormat,
    get_format,
)

PLANE_ALIGNMENT = 256
K_ALIGNMENT = 128


@dataclass(frozen=True, slots=True)
class Layout:
    name: str
    alignment: int
    formats: frozenset[str]


@dataclass(frozen=True, slots=True)
class RowSplitGeometry:
    n: int
    k: int
    k_pad: int
    groups_per_row: int
    base_bytes_per_group: int
    high_bytes_per_group: int
    base_row_bytes: int
    high_row_bytes: int
    scale_row_bytes: int
    base_offset: int
    base_bytes: int
    high_offset: int
    high_bytes: int
    scale_offset: int
    scale_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class BlockScaleGeometry:
    n: int
    k: int
    groups_per_row: int
    k_tiles: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    weight_divisor_offset: int
    weight_divisor_count: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class GgufBlocksGeometry:
    n: int
    k: int
    blocks_per_row: int
    row_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class RowScaleGeometry:
    n: int
    k: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class PackedPlaneGeometry:
    """Two planes: a code plane, then a scale plane starting at a 256-byte boundary.

    Used by `packed_u4_g16_v1` (the Qwen3.8-Flash-Next PLE shard). Same shape as RowScaleGeometry --
    one code plane and one scale plane -- but with a different scale word, so it is its own type
    rather than a row-scale with different numbers.
    """

    n: int
    k: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    payload_bytes: int


CONTIGUOUS_LE_V1 = Layout(
    "contiguous_le_v1", 256, frozenset(("bf16", "fp32", "int32", "int64"))
)
ROW_SPLIT_K128_V1 = Layout(
    "row_split_k128_v1",
    256,
    frozenset(
        ("q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16", "t2_g128_fp16",
         "ptq1_g128_fp16")
    ),
)
BLOCK_SCALE_K16_M128X4_V1 = Layout(
    "block_scale_k16_m128x4_v1",
    256,
    frozenset(("nvfp4",)),
)
ROW_SCALE_V1 = Layout(
    "row_scale_v1",
    256,
    frozenset(("fp8_e4m3fn_row_bf16",)),
)
GGUF_BLOCKS_V1 = Layout("gguf_blocks_v1", 256, frozenset(GGUF_FORMATS))

# The three layouts the Qwen3.8-Flash-Next container declares. Their geometry is copied from the
# engine's own `weight_geometry` (src/core/weight_view.cpp), which is what actually reads these
# bytes; both sides must agree or `describe_tensor` rejects the object.
#
# `packed_u4_g16_v1`: one code plane (K/2 bytes per row, two 4-bit codes per byte) then a scale plane
# of (K/16)*2 bytes per row -- one binary16 per 16-column group. Appended rather than folded into
# row_scale_v1 because the scale word is per GROUP here, not per row.
PACKED_U4_G16_V1 = Layout(
    "packed_u4_g16_v1",
    256,
    frozenset(("u4z8g16_f16",)),
)
# `expert_blockscale_k16_m128x4_v1`: the same bytes as BLOCK_SCALE_K16_M128X4_V1 under the name this
# container uses. The engine parses it to the identical QuantLayout, so one geometry serves both.
EXPERT_BLOCK_SCALE_K16_M128X4_V1 = Layout(
    "expert_blockscale_k16_m128x4_v1",
    256,
    frozenset(("nvfp4",)),
)
# `row_scale_f32_v1`: the same two planes as ROW_SCALE_V1 with a 4-byte scale word. The engine keeps
# one `row_scale_geometry` and picks the word width from the FORMAT, so this is a second layout
# constant for one code path rather than a second formula.
ROW_SCALE_F32_V1 = Layout(
    "row_scale_f32_v1",
    256,
    frozenset(("fp8_e4m3fn_row_f32",)),
)

LAYOUTS = MappingProxyType(
    {
        layout.name: layout
        for layout in (
            CONTIGUOUS_LE_V1,
            ROW_SPLIT_K128_V1,
            BLOCK_SCALE_K16_M128X4_V1,
            EXPERT_BLOCK_SCALE_K16_M128X4_V1,
            ROW_SCALE_V1,
            ROW_SCALE_F32_V1,
            PACKED_U4_G16_V1,
            GGUF_BLOCKS_V1,
        )
    }
)


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("align_up requires a nonnegative value and positive alignment")
    return (value + alignment - 1) // alignment * alignment


def get_layout(name: str) -> Layout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise ValueError(f"unknown tensor layout: {name!r}") from None


def _format(value: str | NumericFormat) -> NumericFormat:
    if isinstance(value, str):
        return get_format(value)
    registered = get_format(value.name)
    if value != registered:
        raise ValueError(f"numeric format does not match registered {value.name!r}")
    return registered


def _layout(value: str | Layout) -> Layout:
    if isinstance(value, str):
        return get_layout(value)
    registered = get_layout(value.name)
    if value != registered:
        raise ValueError(f"layout does not match registered {value.name!r}")
    return registered


def _shape(value: Sequence[int], *, rank: int | None = None) -> tuple[int, ...]:
    dims = []
    for dim in value:
        if isinstance(dim, bool):
            raise ValueError("shape dimensions must be positive integers")
        try:
            item = operator.index(dim)
        except TypeError:
            raise ValueError("shape dimensions must be positive integers") from None
        if item <= 0:
            raise ValueError("shape dimensions must be positive integers")
        dims.append(item)
    result = tuple(dims)
    if rank is not None and len(result) != rank:
        raise ValueError(f"layout requires rank {rank}, got rank {len(result)}")
    return result


def row_split_geometry(
    format: str | QuantFormat, shape: Sequence[int]
) -> RowSplitGeometry:
    spec = _format(format)
    if not isinstance(spec, QuantFormat):
        raise ValueError("row_split_k128_v1 requires a grouped quantized format")
    n, k = _shape(shape, rank=2)
    k_pad = align_up(k, K_ALIGNMENT)
    groups_per_row = k_pad // spec.group_size
    if spec.name == "ptq1_g128_fp16":
        # PTQ1_0 is a third packing and fits NEITHER formula below: base-3 digits staged five per
        # byte (24 bytes) plus a 2-byte tail, which is 1.75 bits per weight and so not expressible
        # as `bits` at all. Named explicitly, exactly as the engine's row_split_format names it, so
        # the container and the engine cannot end up describing the same artifact differently.
        # A partial trailing group cannot be staged at all, so K must be whole 128-groups; T2 gets
        # away with padding because its codes pack per byte, this one does not.
        if k % spec.group_size:
            raise ValueError("row_split_k128_v1 ptq1_g128_fp16 requires K to be whole 128-groups")
        base_bytes_per_group = 24
        high_bytes_per_group = 2
    else:
        # 2- and 8-bit codes pack whole; 4/5/6-bit codes keep a 4-bit base plane plus a high plane.
        base_bytes_per_group = (
            spec.group_size * spec.bits // 8 if spec.bits in (2, 8) else spec.group_size // 2
        )
        high_bytes_per_group = (
            0 if spec.bits in (2, 4, 8) else spec.group_size * (spec.bits - 4) // 8
        )
    base_row_bytes = groups_per_row * base_bytes_per_group
    high_row_bytes = groups_per_row * high_bytes_per_group
    scale_row_bytes = groups_per_row * 2
    base_bytes = n * base_row_bytes
    high_bytes = n * high_row_bytes
    scale_bytes = n * scale_row_bytes
    high_offset = align_up(base_bytes, PLANE_ALIGNMENT)
    scale_offset = high_offset + align_up(high_bytes, PLANE_ALIGNMENT)
    return RowSplitGeometry(
        n=n,
        k=k,
        k_pad=k_pad,
        groups_per_row=groups_per_row,
        base_bytes_per_group=base_bytes_per_group,
        high_bytes_per_group=high_bytes_per_group,
        base_row_bytes=base_row_bytes,
        high_row_bytes=high_row_bytes,
        scale_row_bytes=scale_row_bytes,
        base_offset=0,
        base_bytes=base_bytes,
        high_offset=high_offset,
        high_bytes=high_bytes,
        scale_offset=scale_offset,
        scale_bytes=scale_bytes,
        payload_bytes=scale_offset + scale_bytes,
    )


def expert_blockscale_bank_geometry(
    format: str | Nvfp4Format, shape: Sequence[int]
) -> BlockScaleGeometry:
    """A rank-three NVFP4 expert bank, sized exactly as the engine sizes it.

    Copied from `block_scale_bank_geometry` (src/artifact/bank_geometry.cpp), which the engine uses
    for this layout name: one code plane over every expert's matrix, a 256-aligned scale plane, then
    **four divisor bytes per expert** -- that last term is what a flattened rank-two view gets wrong
    (it charges one divisor for the whole bank). Checked against the container: (512,1280,2560) is
    943,720,448 bytes = 838,860,800 code + 104,857,600 scale + 512*4 divisors, after a 4-byte close.
    """
    spec = _format(format)
    if not isinstance(spec, Nvfp4Format):
        raise ValueError("expert_blockscale_k16_m128x4_v1 requires NVFP4")
    dims = _shape(shape)
    if len(dims) != 3:
        raise ValueError(
            f"expert_blockscale_k16_m128x4_v1 requires a rank-three shape, got rank {len(dims)}"
        )
    experts, rows, columns = dims
    if rows % 128 or columns % 64:
        raise ValueError(
            "expert_blockscale_k16_m128x4_v1 requires N divisible by 128 "
            "and K divisible by 64"
        )
    matrices = experts * rows
    elements = matrices * columns
    code_plane_bytes = elements // 2
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = elements // spec.group_size
    divisor_offset = scale_plane_offset + scale_plane_bytes
    divisor_bytes = experts * 4
    return BlockScaleGeometry(
        n=matrices,
        k=columns,
        groups_per_row=columns // spec.group_size,
        k_tiles=columns // 64,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        weight_divisor_offset=divisor_offset,
        weight_divisor_count=experts,
        payload_bytes=divisor_offset + divisor_bytes,
    )


def block_scale_geometry(
    format: str | Nvfp4Format, shape: Sequence[int], divisors: int = 1
) -> BlockScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Nvfp4Format):
        raise ValueError("block_scale_k16_m128x4_v1 requires NVFP4")
    # A rank-three shape is an expert bank and has its own rule (four divisor bytes per expert),
    # exactly as the engine dispatches it. Only the bank layout name accepts rank three.
    if len(_shape(shape)) == 3:
        raise ValueError(
            "block_scale_k16_m128x4_v1 is rank two; a rank-three bank is "
            "expert_blockscale_k16_m128x4_v1"
        )
    n, k = _shape(shape, rank=2)
    if n % 128 != 0 or k % 64 != 0:
        raise ValueError(
            "block_scale_k16_m128x4_v1 requires N divisible by 128 "
            "and K divisible by 64"
        )
    if divisors < 1 or n % divisors:
        raise ValueError("NVFP4 divisor count must divide the plane's rows")
    # Each divisor covers one source matrix, and a 128-row scale tile may not span two of them:
    # the swizzled scale plane is addressed in whole tiles and the engine takes the divisor from
    # the row.
    if divisors > 1 and (n // divisors) % 128:
        raise ValueError(
            "each NVFP4 divisor must cover a whole number of 128-row scale tiles"
        )
    code_plane_bytes = n * k // 2
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * k // spec.group_size
    weight_divisor_offset = scale_plane_offset + scale_plane_bytes
    return BlockScaleGeometry(
        n=n,
        k=k,
        groups_per_row=k // spec.group_size,
        k_tiles=k // 64,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        weight_divisor_offset=weight_divisor_offset,
        weight_divisor_count=divisors,
        payload_bytes=weight_divisor_offset + 4 * divisors,
    )


def packed_u4_g16_geometry(
    format: str | NumericFormat, shape: Sequence[int]
) -> PackedPlaneGeometry:
    """The PLE shard: a code plane then a per-16-column binary16 scale plane.

    Copied from the engine's `QuantLayout::PackedU4G16` branch (src/core/weight_view.cpp): two 4-bit
    codes per byte, one scale word per 16 columns, scale plane aligned to 256.
    """
    spec = _format(format)
    if spec.name != "u4z8g16_f16":
        raise ValueError("packed_u4_g16_v1 requires u4z8g16_f16")
    n, k = _shape(shape, rank=2)
    if k % 16:
        raise ValueError(f"packed_u4_g16_v1 requires K divisible by 16, got {k}")
    code_plane_bytes = n * k // 2
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * (k // 16) * 2
    return PackedPlaneGeometry(
        n=n,
        k=k,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        payload_bytes=scale_plane_offset + scale_plane_bytes,
    )


def row_scale_geometry(
    format: str | Fp8RowFormat, shape: Sequence[int]
) -> RowScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Fp8RowFormat):
        raise ValueError("row_scale_v1 requires a row-scaled FP8 format")
    n, k = _shape(shape, rank=2)
    # The scale word width is a property of the FORMAT, exactly as the engine's row-scale branch
    # reads it: the BF16-scale spelling gets 2 bytes per row, the FP32-scale one gets 4.
    scale_word_bytes = 4 if spec.name == "fp8_e4m3fn_row_f32" else 2
    code_plane_bytes = n * k
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * scale_word_bytes
    return RowScaleGeometry(
        n=n,
        k=k,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        payload_bytes=scale_plane_offset + scale_plane_bytes,
    )


def gguf_blocks_geometry(
    format: str | GgufFormat, shape: Sequence[int]
) -> GgufBlocksGeometry:
    spec = _format(format)
    if not isinstance(spec, GgufFormat):
        raise ValueError("gguf_blocks_v1 requires a ggml block format")
    # Block-5: a rank-three shape is an expert bank -- (experts, rows, k), one contiguous run of
    # whole-block rows with no planes and no padding -- so it sizes exactly as a rank-two matrix of
    # experts * rows rows, which is how the engine's GgufBlocks branch sizes it too.
    dims = _shape(shape)
    if len(dims) == 3:
        n, k = dims[0] * dims[1], dims[2]
    elif len(dims) == 2:
        n, k = dims
    else:
        raise ValueError(f"gguf_blocks_v1 requires rank 2 or 3, got rank {len(dims)}")
    if k % spec.block_elements:
        raise ValueError(
            f"gguf_blocks_v1 requires K divisible by {spec.block_elements}, got {k}"
        )
    blocks_per_row = k // spec.block_elements
    row_bytes = blocks_per_row * spec.block_bytes
    return GgufBlocksGeometry(
        n=n,
        k=k,
        blocks_per_row=blocks_per_row,
        row_bytes=row_bytes,
        payload_bytes=n * row_bytes,
    )


def encoded_size(
    layout: str | Layout,
    format: str | NumericFormat,
    shape: Sequence[int],
    divisors: int = 1,
) -> int:
    layout_spec = _layout(layout)
    numeric_spec = _format(format)
    if divisors != 1 and layout_spec is not BLOCK_SCALE_K16_M128X4_V1:
        raise ValueError(
            f"layout {layout_spec.name!r} stores one divisor, not {divisors}"
        )
    if numeric_spec.name not in layout_spec.formats:
        raise ValueError(
            f"layout {layout_spec.name!r} does not accept format {numeric_spec.name!r}"
        )
    if layout_spec is CONTIGUOUS_LE_V1:
        if not isinstance(numeric_spec, DirectFormat):
            raise ValueError("contiguous_le_v1 requires a direct format")
        dims = _shape(shape)
        if len(dims) > 16:
            raise ValueError("contiguous_le_v1 supports rank 0 through 16")
        return prod(dims) * numeric_spec.word_bytes
    if layout_spec is ROW_SPLIT_K128_V1:
        if not isinstance(numeric_spec, QuantFormat):
            raise ValueError("row_split_k128_v1 requires a grouped quantized format")
        return row_split_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is EXPERT_BLOCK_SCALE_K16_M128X4_V1:
        return expert_blockscale_bank_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is BLOCK_SCALE_K16_M128X4_V1:
        if not isinstance(numeric_spec, Nvfp4Format):
            raise ValueError("block_scale_k16_m128x4_v1 requires NVFP4")
        return block_scale_geometry(numeric_spec, shape, divisors).payload_bytes
    if layout_spec is ROW_SCALE_V1 or layout_spec is ROW_SCALE_F32_V1:
        if not isinstance(numeric_spec, Fp8RowFormat):
            raise ValueError("row_scale_v1 requires a row-scaled FP8 format")
        return row_scale_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is PACKED_U4_G16_V1:
        return packed_u4_g16_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is GGUF_BLOCKS_V1:
        if not isinstance(numeric_spec, GgufFormat):
            raise ValueError("gguf_blocks_v1 requires a ggml block format")
        return gguf_blocks_geometry(numeric_spec, shape).payload_bytes
    raise ValueError(f"unsupported tensor layout: {layout_spec.name!r}")
