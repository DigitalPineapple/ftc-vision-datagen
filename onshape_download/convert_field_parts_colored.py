"""Re-convert the perimeter/tiles STEP files to OBJ, this time preserving
per-face color info (STEP STYLED_ITEM/COLOUR_RGB) as OBJ material groups
instead of collapsing everything into one uncolored compound.

Must be run with FreeCAD's bundled Python (has FreeCAD/Part/MeshPart/ImportGui
on its path), e.g.:
    "C:\\Program Files\\FreeCAD 1.1\\bin\\freecad.exe" convert_field_parts_colored.py \
        <in.step> <out_basename> [<in.step> <out_basename> ...]

Each (in_step, out_basename) pair produces out_basename.obj + out_basename.mtl.
"""
import os
import sys
import FreeCAD
import Part
import MeshPart
import ImportGui

args = sys.argv[1:]
if len(args) < 2 or len(args) % 2 != 0:
    sys.exit(
        "Usage: freecad.exe convert_field_parts_colored.py "
        "<in1.step> <out1_basename> [<in2.step> <out2_basename> ...]"
    )
JOBS = list(zip(args[0::2], args[1::2]))

LINEAR_DEFLECTION = 1.5
ANGULAR_DEFLECTION = 0.5


def rgb_key(c):
    return (round(c[0], 3), round(c[1], 3), round(c[2], 3))


for in_path, out_base in JOBS:
    doc = FreeCAD.newDocument("d")
    ImportGui.insert(in_path, doc.Name)
    doc.recompute()

    # Bucket every face across every object by its resolved color.
    buckets = {}  # rgb_key -> list of Part.Face
    default_color_objs = 0
    for o in doc.Objects:
        if not hasattr(o, "Shape") or o.Shape.isNull():
            continue
        vp = o.ViewObject
        shape_color = tuple(vp.ShapeColor[:3]) if vp else (0.8, 0.8, 0.8)
        diffuse = list(vp.DiffuseColor) if vp and vp.DiffuseColor else []
        faces = o.Shape.Faces
        if len(diffuse) == len(faces) and len(faces) > 0:
            per_face_colors = [tuple(c[:3]) for c in diffuse]
        else:
            per_face_colors = [shape_color] * len(faces)
            default_color_objs += 1
        for face, color in zip(faces, per_face_colors):
            buckets.setdefault(rgb_key(color), []).append(face)

    print(f"{os.path.basename(in_path)}: {len(doc.Objects)} objects, "
          f"{len(buckets)} distinct colors, {default_color_objs} objs used shape-level color")
    for k, v in buckets.items():
        print(f"   color {k}: {len(v)} faces")

    obj_path = out_base + ".obj"
    mtl_path = out_base + ".mtl"
    mtl_name = os.path.basename(mtl_path)

    with open(mtl_path, "w") as mtl_f:
        for i, color in enumerate(buckets.keys()):
            mtl_f.write(f"newmtl mat_{i}\n")
            mtl_f.write(f"Kd {color[0]:.4f} {color[1]:.4f} {color[2]:.4f}\n")
            mtl_f.write("Ka 0 0 0\nKs 0.05 0.05 0.05\nNs 10\n\n")

    vertex_offset = 1  # OBJ indices are 1-based
    with open(obj_path, "w") as obj_f:
        obj_f.write(f"mtllib {mtl_name}\n")
        for i, (color, faces) in enumerate(buckets.items()):
            compound = Part.makeCompound(faces)
            mesh = MeshPart.meshFromShape(
                Shape=compound, LinearDeflection=LINEAR_DEFLECTION,
                AngularDeflection=ANGULAR_DEFLECTION, Relative=False,
            )
            obj_f.write(f"g color_{i}\nusemtl mat_{i}\n")
            for p in mesh.Points:
                obj_f.write(f"v {p.x} {p.y} {p.z}\n")
            for f in mesh.Facets:
                a, b, c = f.PointIndices
                obj_f.write(f"f {a + vertex_offset} {b + vertex_offset} {c + vertex_offset}\n")
            vertex_offset += len(mesh.Points)
            print(f"   group {i}: {len(mesh.Points)} verts, {mesh.CountFacets} facets")

    print(f"   -> {obj_path} / {mtl_path}")
    FreeCAD.closeDocument(doc.Name)

import os as _os
_os._exit(0)
