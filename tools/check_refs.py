"""check_refs.py -- referential integrity over the WHOLE artifact directory JSON.

strip_components.py rebuilds `objects` / `bindings` / `uses` from the surviving graph, but the
component tables (components.<name>.resources, .config, .target) are copied verbatim. Those
tables hold object ids too, and nothing in the upstream reader validates them -- a stale id there
only explodes later, inside the engine, as a failed host_object()/config lookup.

So: walk every string in the directory (outside the `objects` array itself) that looks like an
object id, and require it to resolve. Also checks the four text resources the frontend binds
unconditionally (load/resources.cpp:12-15).

    python check_refs.py <artifact.ninfer> [...]
"""

import json
import struct
import sys

HEADER = struct.Struct("<8sQ16s")
MAGIC = b"NINFER\x00\x03"
ID_PREFIXES = ("weight/", "resource/", "auxiliary/")
REQUIRED_TEXT_RESOURCES = (
    "resource/text/tokenizer.json",
    "resource/text/tokenizer_config.json",
    "resource/text/chat_template.jinja",
    "resource/text/generation_config.json",
)


def read_directory(path):
    with open(path, "rb") as f:
        magic, json_len, _ = HEADER.unpack(f.read(HEADER.size))
        if magic != MAGIC:
            raise SystemExit("%s: not a NInfer v3 container" % path)
        return json.loads(f.read(json_len).decode("utf-8"))


def walk(node, path, out):
    if isinstance(node, str):
        if node.startswith(ID_PREFIXES):
            out.append((path, node))
    elif isinstance(node, dict):
        for k, v in node.items():
            walk(v, "%s.%s" % (path, k), out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, "%s[%d]" % (path, i), out)


def main(argv):
    bad = 0
    for path in argv:
        doc = read_directory(path)
        print("=" * 88)
        print(path)
        present = {o["id"] for o in doc["objects"]}
        print("  objects: %d   components: %s"
              % (len(present), ",".join(sorted(doc["components"]))))

        refs = []
        for key in ("components", "bindings", "uses", "metadata", "provenance"):
            if key in doc:
                walk(doc[key], key, refs)
        # `objects[].id` is the definition, not a reference -- excluded by walking only the rest.

        dangling = sorted({(p, i) for p, i in refs if i not in present})
        print("  id references outside objects[]: %d distinct %d"
              % (len(refs), len({i for _, i in refs})))
        if dangling:
            bad += 1
            print("  DANGLING: FAIL")
            for p, i in dangling[:10]:
                print("     %s -> %s" % (p, i))
        else:
            print("  all id references resolve: PASS")

        missing = [r for r in REQUIRED_TEXT_RESOURCES if r not in present]
        print("  four text resources the frontend always binds: %s"
              % ("PASS" if not missing else "FAIL %s" % missing))

        weights = len([o for o in present if o.startswith("weight/")])
        print("  weight objects: %d   resources: %d   auxiliary: %d"
              % (weights,
                 len([o for o in present if o.startswith("resource/")]),
                 len([o for o in present if o.startswith("auxiliary/")])))
        if missing:
            bad += 1
    print("=" * 88)
    print("VERDICT: %s" % ("PASS" if not bad else "FAIL (%d file(s))" % bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
