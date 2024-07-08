import sys
sys.path.append(".")
import deep_sdf.mesh
import deep_sdf.utils
import os
import json
import torch
import deep_sdf.workspace as ws
import pathlib
import time
import datetime
import splinepy as sp
import gustaf as gus
import numpy as np
import skimage
import meshio
import tetgenpy
import igl

accuracy_deep_sdf_reconstruction = 32
number_of_final_elements = 1e3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_directory = "./experiments/snappy3D"
checkpoint = "1000"

decoder = ws.load_trained_model(experiment_directory, checkpoint)
latent = ws.load_latent_vectors(experiment_directory, checkpoint)

box = sp.helpme.create.box(10,10,10).bspline
small_box = sp.helpme.create.box(1,1,1).bspline
box.insert_knots(0, [0.5])
box.insert_knots(1, [0.5])
box.insert_knots(2, [0.5])

lat_vec1 = latent[1]
lat_vec2 = latent[15]
lat_vec3 = latent[39]

latent_vec_interpolation = sp.BSpline(
    degrees=[1, 1, 1],
    knot_vectors=[[-1, -1, 0, 1, 1], 
                  [-1, -1, 0, 1, 1], 
                  [-1, -1, 1, 1]],
    control_points=[lat_vec2]*18,
)

index = sp.helpme.multi_index.MultiIndex(latent_vec_interpolation.control_mesh_resolutions)
# center thicker
latent_vec_interpolation.control_points[index[1,1,0]] = lat_vec3
latent_vec_interpolation.control_points[index[1,1,1]] = lat_vec3
# sides smaller (I don't know why it is not [0,1,0] instead of [1,0,0])
latent_vec_interpolation.control_points[index[1,0,0]] = lat_vec1
latent_vec_interpolation.control_points[index[1,0,1]] = lat_vec1
latent_vec_interpolation.control_points[index[1,2,0]] = lat_vec1
latent_vec_interpolation.control_points[index[1,2,1]] = lat_vec1

microstructure = sp.Microstructure()
# set outer spline and a (micro) tile
microstructure.deformation_function = box
microstructure.microtile = small_box
# tiling determines tile resolutions within each bezier patch
microstructure.tiling = [2, 2, 2]
ms = microstructure.create().patches
# ms[0].evaluate()
# ms[0].control_points[0] = [0.2,0.2,0.2]
tiling = [6, 6, 1]
N_base = accuracy_deep_sdf_reconstruction
N = [N_base * t for t in tiling]

cap_border_dict = {
    "x0": {"cap": 1, "measure": 0.1},
    "x1": {"cap": 1, "measure": 0.1},
    "y0": {"cap": 1, "measure": 0.1},
    "y1": {"cap": 1, "measure": 0.1},
}

verts, faces = deep_sdf.mesh.create_mesh_microstructure(tiling, decoder, latent_vec_interpolation, "none", cap_border_dict=cap_border_dict, N=N)
# Free Form Deformation
# geometric parameters
vert_deformation = 0.15
width = 10
length = 10
scaling = 5
depth = 0.2*scaling
pts = []

control_points=np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0.5, -vert_deformation, 0],
        [0.5, (1-vert_deformation), 0],
        [1, 0, 0],
        [1, 1, 0]
    ])

deformation_surf = sp.BSpline(
    degrees=[1,2],
    control_points=control_points*scaling,
    knot_vectors=[[0, 0, 1, 1],[0, 0, 0, 1, 1, 1]],
)

deformation_volume = deformation_surf.create.extruded(extrusion_vector=[0,0,depth])

# bring slightly outside vertices back 
verts[verts>1] = 1
verts[verts<0] = 0

verts_FFD_transformed = deformation_volume.evaluate(verts)
mesh = gus.faces.Faces(verts_FFD_transformed, faces)
fname_surf = f"data/meshs/facade_snappy_{'_'.join([str(l) for l in N])}_surf.inp"
gus.io.meshio.export(fname_surf, mesh)

r = igl.decimate(mesh.vertices, mesh.faces, int(3e5))
dmesh = gus.Faces(r[1], r[2])

t_in = tetgenpy.TetgenIO()
t_in.setup_plc(dmesh.vertices, dmesh.faces.tolist())
t_out = tetgenpy.tetrahedralize("pqa", t_in)

fname_volume = f"data/meshs/facade_snappy_{'_'.join([str(l) for l in N])}_volume.inp"
tets = np.vstack(t_out.tetrahedra())
verts = t_out.points()

mesh = gus.Volumes(verts, tets)

faces = mesh.to_faces(False)
boundary_faces = faces.single_faces()

BC = {1: [], 2: [], 3: []} 
for i in boundary_faces:
    # mark boundaries at x = 0 with 1
    if np.max(verts[faces.const_faces[i], 0]) < 3e-2:
        BC[1].append(i)
    # mark boundaries at x = 1 with 2
    elif np.min(verts[faces.const_faces[i], 0]) > 4.999:
        BC[2].append(i)
    # mark rest of the boundaries with 3
    else:
        BC[3].append(i)
gus.io.mfem.export(fname_volume.replace(".inp", ".mesh"), mesh, BC)
gus.io.meshio.export(fname_volume, mesh)