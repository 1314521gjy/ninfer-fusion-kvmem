"""peek_container.py -- show a .ninfer container's layout so a component-stripping repack can be
written correctly (offsets, lengths, hashes, artifact id).

Format observed on disk (2026-09-30):
    [0..6]   magic "NINFER\\0"
    [7]      format version byte (03)
    [8..15]  manifest length, uint64 little-endian
    [16..31] artifact id, 16 bytes
    [32..]   manifest JSON (manifest length bytes), then the payload

ASCII-only source. Usage: python peek_container.py <artifact.ninfer> [--objects]
"""

import json
import struct
import sys


def load(path):
    with open(path, "rb") as fh:
        head = fh.read(32)
        magic = head[0:7]
        version = head[7]
        (mlen,) = struct.unpack("<Q", head[8:16])
        aid = head[16:32].hex()
        manifest = json.loads(fh.read(mlen).decode("utf-8"))
        payload_offset = 32 + mlen
    return {"magic": magic, "version": version, "manifest_len": mlen, "artifact_id": aid,
            "manifest": manifest, "payload_offset": payload_offset}


def main(argv):
    path = argv[0]
    want_objects = "--objects" in argv
    info = load(path)
    m = info["manifest"]
    print("file          : %s" % path)
    print("magic         : %r  version=%d" % (info["magic"], info["version"]))
    print("manifest_len  : %d B" % info["manifest_len"])
    print("artifact_id   : %s" % info["artifact_id"])
    print("payload start : %d" % info["payload_offset"])
    print("manifest keys : %s" % ", ".join(sorted(m.keys())))
    for key in ("artifact_id", "name", "model_id", "format_version", "producer", "payload_bytes",
                "total_bytes", "created", "recipe", "converter"):
        if key in m:
            print("  m[%-14s] = %s" % (key, str(m[key])[:160]))
    comps = m.get("components") or {}
    print("")
    print("components    : %s" % ", ".join(sorted(comps.keys())))
    for name in sorted(comps):
        comp = comps[name]
        keys = sorted(comp.keys()) if isinstance(comp, dict) else []
        objs = comp.get("objects") if isinstance(comp, dict) else None
        print("  %-10s keys=%-40s objects=%s" % (name, ",".join(keys), len(objs) if objs else 0))
        if want_objects and objs:
            for obj in objs[:2]:
                if isinstance(obj, dict):
                    shown = {k: (str(v)[:60]) for k, v in obj.items()}
                    print("      %s" % json.dumps(shown, ensure_ascii=False))
    # global object table (some layouts keep one flat list with an offset per object)
    for key in ("objects", "entries", "payload"):
        if key in m:
            val = m[key]
            print("")
            print("m[%s] type=%s len=%s" % (key, type(val).__name__, len(val) if hasattr(val, "__len__") else "?"))
            if isinstance(val, list) and val:
                print("  first entry: %s" % json.dumps({k: str(v)[:60] for k, v in val[0].items()}
                                                      if isinstance(val[0], dict) else str(val[0])[:200],
                                                      ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1:])
