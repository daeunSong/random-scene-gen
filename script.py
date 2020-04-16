import bpy
import bmesh
import sys

from bpy import context

argv = sys.argv
argv = argv[argv.index("--") + 1]  # get all args after "--"

bpy.ops.import_mesh.stl(filepath="stl/"+argv+".stl")
bm = bmesh.new()
meshes = [o.data for o in context.selected_objects
            if o.type == 'MESH']
for mesh in meshes:
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.clear()
    mesh.update()

bm.free()
bpy.ops.export_mesh.stl(filepath="stl/"+argv+".stl", use_selection=True)
