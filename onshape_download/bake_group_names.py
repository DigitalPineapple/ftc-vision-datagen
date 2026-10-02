"""Onshape's OBJ export orders things as: `g <PartName>` -> vertex block -> `o meshN`
-> face block. Blender's importer names objects from the `o` line only, discarding
the `g` part name entirely - so "FTC Field Side Glass 11in" etc. would be lost as
just "mesh116". This rewrites each `o` line to embed the most recent `g` name so
Blender's imported object names retain real part identity (e.g. to detect glass
panels and give them a transparent material).
"""
import re
import sys

in_path, out_path = sys.argv[1], sys.argv[2]


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())[:60]


current_group = None
out_lines = []
with open(in_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("g "):
            current_group = sanitize(line[2:])
            out_lines.append(line)
        elif line.startswith("o ") and current_group:
            orig_name = line[2:].strip()
            out_lines.append(f"o {current_group}__{orig_name}\n")
        else:
            out_lines.append(line)

with open(out_path, "w", encoding="utf-8") as f:
    f.writelines(out_lines)

print(f"Wrote {out_path} ({len(out_lines)} lines)")
