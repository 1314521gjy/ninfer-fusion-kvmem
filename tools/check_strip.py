"""check_strip.py -- engine-free gate on a stripped artifact.

The engine's load path is MODEL-DRIVEN (fusion-master/src/src/models/qwen3_5/load/prepare.cpp:47-121):
for every parameter the model declares it calls binder.use(name, input), which throws
"<name>: unresolved Use <input>" when that (parameter,input) pair is missing from the
directory's `uses` map (artifact/binder.cpp:70). So stripping is only safe when every use
we removed belonged to a binding we also removed -- a kept parameter must keep ALL its uses.

Also reports the components' sub-keys, because `proposal` lived as a sub-key rather than a
top-level component in these artifacts.

    python check_strip.py <src.ninfer> <dst.ninfer> [--drop dflash2,vision,proposal]
"""

import argparse
import collections
import json
import os
import struct
import sys

HEADER = struct.Struct("<8sQ16s")
PAYLOAD_ALIGNMENT = 4096
MAGIC = b"NINFER\x00\x03"


def read_directory(path):
    with open(path, "rb") as f:
        head = f.read(HEADER.size)
        magic, json_len, _artifact_id = HEADER.unpack(head)
        if magic != MAGIC:
            raise SystemExit("not a NInfer v3 container: %s" % magic)
        blob = f.read(json_len)
    return json.loads(blob.decode("utf-8")), len(blob)


def use_key(u):
    return (u.get("parameter"), u.get("input"))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--drop", default="dflash2,vision,proposal")
    args = ap.parse_args(argv)

    prefixes = tuple(x.strip() + "/" for x in args.drop.split(",") if x.strip())
    src, _ = read_directory(args.src)
    dst, _ = read_directory(args.dst)

    src_bind = src["bindings"]
    dst_bind = dst["bindings"]
    dropped_names = sorted(n for n in src_bind if n.startswith(prefixes))
    kept_names = sorted(n for n in src_bind if not n.startswith(prefixes))
    print("bindings : src %d -> dst %d   (dropped %d, prefixes %s)"
          % (len(src_bind), len(dst_bind), len(dropped_names), ",".join(p.rstrip("/") for p in prefixes)))

    # --- 1. the dropped-name set must match exactly (no accidental survivors/losses) ---
    missing = [n for n in kept_names if n not in dst_bind]
    extra = [n for n in dst_bind if n not in src_bind]
    print("kept binding names all present in dst : %s" % ("PASS" if not missing else "FAIL %s" % missing[:5]))
    print("no unexpected new binding names        : %s" % ("PASS" if not extra else "FAIL %s" % extra[:5]))

    # --- 2. every removed Use must belong to a dropped binding (the actual safety gate) ---
    src_uses = {use_key(u) for u in src["uses"]}
    dst_uses = {use_key(u) for u in dst["uses"]}
    removed = src_uses - dst_uses
    added = dst_uses - src_uses
    offenders = sorted({p for p, _ in removed if not (p or "").startswith(prefixes)})
    print("uses     : src %d -> dst %d   removed %d, added %d"
          % (len(src["uses"]), len(dst["uses"]), len(removed), len(added)))
    print("every removed Use belonged to a dropped binding : %s"
          % ("PASS" if not offenders else "FAIL kept parameters lost uses: %s" % offenders[:8]))

    # --- 3. per kept parameter, the use COUNT must be identical (no starvation) ---
    src_count = collections.Counter(p for p, _ in src_uses if p in set(kept_names))
    dst_count = collections.Counter(p for p, _ in dst_uses)
    starved = sorted(p for p in src_count if dst_count.get(p, 0) != src_count[p])
    print("kept parameters keep all their uses       : %s"
          % ("PASS (%d parameters)" % len(src_count) if not starved else "FAIL %s" % starved[:8]))

    # --- 4. same check over the object/binding graph the loader walks (parts) ---
    def part_ids(bindings):
        out = set()
        for ref in bindings.values():
            if isinstance(ref, dict):
                for part in ref.get("parts") or []:
                    if isinstance(part, dict) and isinstance(part.get("object"), str):
                        out.add(part["object"])
                if isinstance(ref.get("object"), str):
                    out.add(ref["object"])
        return out

    dst_objects = {o["id"] for o in dst["objects"]}
    dangling = sorted(part_ids(dst_bind) - dst_objects)
    print("every part referenced by a kept binding exists : %s"
          % ("PASS" if not dangling else "FAIL %s" % dangling[:5]))

    # --- 5. components + their sub-keys (proposal lived as a sub-key, not a component) ---
    for label, doc in (("src", src), ("dst", dst)):
        comps = doc["components"]
        shape = {name: sorted(v.keys()) if isinstance(v, dict) else type(v).__name__
                 for name, v in sorted(comps.items())}
        print("components[%s] : %s" % (label, shape))

    ok = not (missing or extra or offenders or starved or dangling)
    print("VERDICT  : %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
