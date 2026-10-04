"""component_bytes.py -- how many bytes does each component of a .ninfer artifact take?

Why: the beta kit question "does dflash cost size, and what do we lose by keeping only MTP"
cannot be answered by counting tensor-name occurrences in the header. This reads the artifact's
own conversion report (per-object shape + format) and computes real payload bytes per component.

Byte model (from this project's notes on the ternary block layouts):
    t2_g128_fp16 : 128 weights per block = 32 B codes + 2 B fp16 scale = 34 B
    bf16 2 B/elt | fp16 2 B/elt | fp32 4 B/elt | int32 4 B/elt | int8 1 B/elt
Anything else is reported as UNKNOWN and listed, never silently counted.

ASCII-only source. Usage: python component_bytes.py <conversion.json> [more.json ...]
"""

import json
import os
import sys

T2_GROUP = 128
T2_BYTES_PER_GROUP = 34  # 32 B codes + 2 B fp16 scale

PER_ELEMENT = {
    "bf16": 2.0,
    "f16": 2.0,
    "fp16": 2.0,
    "fp32": 4.0,
    "f32": 4.0,
    "int32": 4.0,
    "i32": 4.0,
    "int8": 1.0,
    "i8": 1.0,
    "u8": 1.0,
}

# Block-quantised formats: (weights per block, bytes per block), scale included.
# The table is VALIDATED by the total: the computed sum must equal the report's payload_bytes,
# otherwise the run prints the delta and the script is not trusted (no silent guessing).
BLOCK = {
    "q8_g32_fp16": (32, 34),     # 32 x int8 + fp16 scale
    "q4_g64_fp16": (64, 34),     # 64 x 4-bit + fp16 scale
    "q5_g64_fp16": (64, 42),     # 64 x 5-bit + fp16 scale
    "q6_g64_fp16": (64, 50),     # 64 x 6-bit + fp16 scale
    "gguf_iq1_m": (256, 56),
    "gguf_iq2_xxs": (256, 66),
    "gguf_iq2_xs": (256, 74),
    "gguf_iq2_s": (256, 82),
    "gguf_iq3_xxs": (256, 98),
    "gguf_iq3_s": (256, 110),
    "gguf_iq4_xs": (256, 136),
    "gguf_q2_k": (256, 84),
    "gguf_q4_k": (256, 144),
    "gguf_q6_k": (256, 210),
}


def numel(shape):
    n = 1
    for d in shape or []:
        n *= int(d)
    return n


def object_bytes(obj):
    """(bytes, kind) for one object; kind is 'known' or the format name when unknown."""
    fmt = str(obj.get("format", "")).lower()
    shape = obj.get("shape") or []
    n = numel(shape)
    if "t2_g128" in fmt:
        groups = (n + T2_GROUP - 1) // T2_GROUP
        return groups * T2_BYTES_PER_GROUP, "known"
    if fmt in BLOCK:
        per, size = BLOCK[fmt]
        return ((n + per - 1) // per) * size, "known"
    if fmt in PER_ELEMENT:
        return int(n * PER_ELEMENT[fmt]), "known"
    return 0, obj.get("format", "?")


def component_of(obj):
    params = obj.get("parameters") or []
    if not params:
        return "(no-parameter)"
    head = str(params[0])
    return head.split("/")[0] if "/" in head else head


def main(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        # NOTE: in these reports `objects` is a COUNT (int); the per-object list lives in `methods`
        # (each entry: {object, parameters:[path], sources:[...], method, format, layout, shape}).
        objects = doc.get("methods") or []
        totals = {}
        counts = {}
        unknown = {}
        for obj in objects:
            comp = component_of(obj)
            size, kind = object_bytes(obj)
            totals[comp] = totals.get(comp, 0) + size
            counts[comp] = counts.get(comp, 0) + 1
            if kind != "known":
                unknown[kind] = unknown.get(kind, 0) + 1
        grand = sum(totals.values())
        payload = doc.get("payload_bytes")
        print("=" * 92)
        print("artifact   : %s" % doc.get("name"))
        print("report     : %s" % os.path.basename(path))
        print("objects    : %d   payload_bytes(report) = %s" % (len(objects), payload))
        print("computed   : %d bytes (%.2f GiB)   delta vs report = %s" % (
            grand, grand / (1024 ** 3),
            ("%+d" % (grand - int(payload))) if payload else "n/a"))
        print("")
        print("  %-18s %9s %14s %12s" % ("component", "objects", "bytes", "GiB"))
        for comp in sorted(totals, key=lambda c: -totals[c]):
            print("  %-18s %9d %14d %12.3f" % (comp, counts[comp], totals[comp], totals[comp] / (1024 ** 3)))
        if unknown:
            print("")
            print("  UNKNOWN formats (not counted): %s" % json.dumps(unknown, ensure_ascii=False))
        print("")
        dfl = totals.get("dflash2", 0) + totals.get("dflash", 0)
        mtp = totals.get("mtp", 0)
        print("  dflash2 = %.1f MiB (%.1f%% of payload)" % (dfl / (1024 ** 2), 100.0 * dfl / grand if grand else 0))
        print("  mtp     = %.2f MiB (%.2f%% of payload)" % (mtp / (1024 ** 2), 100.0 * mtp / grand if grand else 0))
        print("  text+vision+proposal = %.1f MiB" % ((grand - dfl - mtp) / (1024 ** 2)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1:])
