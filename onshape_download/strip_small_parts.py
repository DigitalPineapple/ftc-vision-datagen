"""Strip small/negligible hardware and non-visible reference geometry out of
a large Onshape OBJ export, keeping only the visually-significant parts.
Generalized from strip_hive_hardware.py with a configurable drop-keyword
list (passed as extra CLI args) since each CAD assembly's fastener/part
names differ. OBJ vertex/normal indices are global across the whole file,
so a naive line-deletion would corrupt face references for every part
after a dropped one - this does a proper two-pass parse + renumber instead.
"""
import re
import sys


def should_drop(group_name: str, drop_keywords) -> bool:
    low = group_name.lower()
    return any(kw in low for kw in drop_keywords)


def main(in_path, out_path, drop_keywords):
    verts = []  # 1-indexed via list (verts[0] unused placeholder)
    norms = []
    verts.append(None)
    norms.append(None)

    current_group = ""
    drop_current = False
    kept_face_lines = []
    header_lines = []
    dropped_groups = set()
    kept_groups = set()

    face_re = re.compile(r"(\d+)(?:/(\d*)/?(\d*))?")

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("vn "):
                parts = line.split()
                norms.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("g "):
                current_group = line[2:].strip()
                drop_current = should_drop(current_group, drop_keywords)
                (dropped_groups if drop_current else kept_groups).add(current_group)
                kept_face_lines.append(("g", line))
            elif line.startswith("mtllib") or line.startswith("#"):
                header_lines.append(line)
            elif line.startswith("o ") or line.startswith("usemtl") or line.startswith("f "):
                if not drop_current:
                    kept_face_lines.append(("body", line))

    print(f"Dropped groups ({len(dropped_groups)}):")
    for g in sorted(dropped_groups):
        print("  -", g)
    print(f"Kept groups ({len(kept_groups)}):")
    for g in sorted(kept_groups):
        print("  -", g)

    used_v = {}
    used_vn = {}

    def remap_face_token(tok):
        m = face_re.match(tok)
        vi = int(m.group(1))
        vni = m.group(3)
        new_vi = used_v.setdefault(vi, len(used_v) + 1)
        if vni:
            vni = int(vni)
            new_vni = used_vn.setdefault(vni, len(used_vn) + 1)
            return f"{new_vi}//{new_vni}"
        return f"{new_vi}"

    out_body = []
    for kind, line in kept_face_lines:
        if kind == "g":
            out_body.append(line)
        elif line.startswith("f "):
            toks = line.split()[1:]
            new_toks = [remap_face_token(t) for t in toks]
            out_body.append("f " + " ".join(new_toks) + "\n")
        else:
            out_body.append(line)

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(header_lines)
        for old_vi in sorted(used_v, key=lambda k: used_v[k]):
            x, y, z = verts[old_vi]
            f.write(f"v {x} {y} {z}\n")
        for old_vni in sorted(used_vn, key=lambda k: used_vn[k]):
            x, y, z = norms[old_vni]
            f.write(f"vn {x} {y} {z}\n")
        f.writelines(out_body)

    print(f"Wrote {out_path}: {len(used_v)} verts (was {len(verts) - 1}), "
          f"{len(used_vn)} normals (was {len(norms) - 1})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [kw.lower() for kw in sys.argv[3:]])
