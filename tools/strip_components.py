"""strip_components.py -- repack a NInfer v3 artifact with some components removed.

Why: the beta kit only needs the MTP speculative head. dflash2 + proposal + vision are
optional components that cost real bytes (PQ2: dflash2 1.296 GiB + vision 0.274 GiB +
proposal 0.167 GiB = 1.74 GiB of an 8.87 GiB file). This drops them and keeps text + mtp.

How (no re-implementation of the container):
  * read the source with the upstream reader (validates header/length/directory),
  * keep the objects the surviving bindings name (plus resource/text/* and auxiliary/*),
  * hand the surviving specs to the upstream ArtifactWriter, which recomputes the aligned
    object layout, writes the header + directory, and refuses to publish unless every
    declared object got its full coverage,
  * stream each kept object's bytes across -- bounded memory, no full-file buffering.

Never writes over the source. Usage:
    python strip_components.py <src.ninfer> <dst.ninfer> [--drop dflash2,vision,proposal] [--dry-run]
"""

import argparse
import json
import os
import sys
from datetime import datetime

UPSTREAM = r"E:\infer-build\refs\infer-all-full\ninfer-all-master"
if UPSTREAM not in sys.path:
    sys.path.insert(0, UPSTREAM)

from tools.artifact.reader import Artifact  # noqa: E402
from tools.artifact.writer import ArtifactWriter  # noqa: E402
from tools.artifact.schema import ResourceObject, TensorSpec, ResourceSpec  # noqa: E402


def referenced_ids(bindings):
    """Every object id a binding names (values are {'object': id} and/or {'parts': [...]})."""
    ids = set()
    for ref in bindings.values():
        if not isinstance(ref, dict):
            continue
        obj = ref.get("object")
        if isinstance(obj, str):
            ids.add(obj)
        for part in ref.get("parts") or []:
            if isinstance(part, dict) and isinstance(part.get("object"), str):
                ids.add(part["object"])
    return ids


def uses_ids(entry):
    """Collect object-id-looking strings from one `uses` record (kept for diagnostics)."""
    found = set()

    def walk(node):
        if isinstance(node, str):
            if node.startswith(("weight/", "resource/", "auxiliary/")):
                found.add(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(entry)
    return found


def keep_use(entry, kept_binding_names):
    """A Use is valid only while its `parameter` (and every auxiliary) still has a binding.

    schema.parse_directory enforces exactly this: a Use whose `parameter` is absent from
    `bindings` is rejected with "<name>@<source>: missing parameter or duplicate Use".
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("parameter") not in kept_binding_names:
        return False
    auxiliaries = entry.get("auxiliaries") or {}
    if isinstance(auxiliaries, dict):
        for binding in auxiliaries.values():
            if isinstance(binding, dict):
                name = binding.get("object")
                if isinstance(name, str) and name.startswith("__binding__"):
                    pass
            # auxiliary entries are binding names in this schema; check the dict keys instead
        for role in auxiliaries:
            if isinstance(auxiliaries[role], str) and auxiliaries[role] not in kept_binding_names:
                return False
    return True


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--drop", default="dflash2,vision,proposal")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    drop = tuple(x.strip() + "/" for x in args.drop.split(",") if x.strip())
    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        print("REFUSE: source and destination are the same file")
        return 2
    if os.path.exists(args.dst):
        print("REFUSE: destination exists: %s" % args.dst)
        return 2

    art = Artifact(args.src)
    d = art.directory
    print("source        : %s" % os.path.basename(args.src))
    print("  file bytes  : %d" % os.path.getsize(args.src))
    print("  artifact_id : %s" % art.artifact_id.hex())
    print("  payload     : %d" % d.payload_bytes)
    print("  components  : %s" % ", ".join(sorted(d.components.keys())))
    print("  bindings    : %d   objects: %d   uses: %d" % (len(d.bindings), len(d.objects), len(d.uses)))

    kept_bindings = {k: v for k, v in d.bindings.items() if not k.startswith(drop)}
    dropped_binding_names = [k for k in d.bindings if k.startswith(drop)]
    kept_components = {}
    for name, comp in d.components.items():
        if any(name == p.rstrip("/") for p in drop):
            continue
        comp = dict(comp)
        for sub in ("proposal",):
            comp.pop(sub, None)
        kept_components[name] = comp

    keep_ids = referenced_ids(kept_bindings)
    for obj in d.objects:
        if isinstance(obj, ResourceObject):
            if obj.id.startswith("resource/text"):
                keep_ids.add(obj.id)
        elif obj.id.startswith("auxiliary/"):
            keep_ids.add(obj.id)          # tiny constants the engine may name by convention
    kept_uses = [u for u in d.uses if keep_use(u, set(kept_bindings.keys()))]

    kept = [o for o in d.objects if o.id in keep_ids]
    dropped = [o for o in d.objects if o.id not in keep_ids]
    kept_bytes = sum(o.bytes for o in kept)
    dropped_bytes = sum(o.bytes for o in dropped)
    print("keep          : %d objects / %.3f GiB" % (len(kept), kept_bytes / 1024 ** 3))
    print("drop          : %d objects / %.3f GiB  (%d binding names)" % (
        len(dropped), dropped_bytes / 1024 ** 3, len(dropped_binding_names)))
    print("components 保留: %s" % ", ".join(sorted(kept_components.keys())))
    print("uses 保留      : %d / %d" % (len(kept_uses), len(d.uses)))
    print("预计新文件 ≈ %d bytes (+目录 0.4 MB 级) = %.3f GiB" % (
        kept_bytes, kept_bytes / 1024 ** 3))
    if args.dry_run:
        art.close()
        print("[dry-run] 未写任何文件")
        return 0

    specs = []
    for obj in kept:
        if isinstance(obj, ResourceObject):
            # ResourceSpec(id, bytes, encoding=RAW_BYTES_V1) -- keyword args on purpose: passing
            # these positionally is how the first attempt fed encoding into bytes (upstream then
            # rejected it as "expected integer").
            specs.append(ResourceSpec(id=obj.id, bytes=obj.bytes, encoding=obj.encoding))
        else:
            specs.append(TensorSpec(id=obj.id, shape=obj.shape, format=obj.format,
                                    layout=obj.layout, divisors=obj.divisors))

    provenance = dict(d.provenance)
    provenance["stripped"] = {
        "by": "strip_components.py",
        "when": datetime.now().isoformat(timespec="seconds"),
        "dropped_prefixes": [p.rstrip("/") for p in drop],
        "dropped_objects": len(dropped),
        "dropped_bytes": dropped_bytes,
        "source_file": os.path.basename(args.src),
        "source_artifact_id": art.artifact_id.hex(),
    }

    writer = ArtifactWriter(
        args.dst,
        specs,
        components=kept_components,
        bindings=kept_bindings,
        uses=kept_uses,
        metadata=d.metadata,
        provenance=provenance,
    )
    written = 0
    try:
        for obj in kept:
            writer.write_object(obj.id, art.iter_object(obj.id))
            written += obj.bytes
        directory = writer.finish()
    except BaseException:
        writer.abort()
        raise
    finally:
        art.close()

    print("written       : %d bytes" % written)
    print("destination   : %s" % args.dst)
    print("  file bytes  : %d" % os.path.getsize(args.dst))

    check = Artifact(args.dst)          # re-open: validates header, length, directory, layout
    print("re-open OK    : artifact_id=%s payload=%d objects=%d components=%s" % (
        check.artifact_id.hex(), check.payload_bytes, len(check.directory.objects),
        ",".join(sorted(check.directory.components.keys()))))
    check.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
