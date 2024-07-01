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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_directory = "./experiments/simple_geom"
checkpoint = "1000"

decoder = ws.load_trained_model(experiment_directory, checkpoint)
latent = ws.load_latent_vectors(experiment_directory, checkpoint).to(device)

box = sp.helpme.create.box(10,10,10).bspline
small_box = sp.helpme.create.box(1,1,1).bspline
box.insert_knots(0, [0.5])
box.insert_knots(1, [0.5])
box.insert_knots(2, [0.5])

lat_vec1 = latent[20]
lat_vec2 = latent[30]
lat_vec3 = latent[39]

latent_vec_interpolation = sp.BSpline(
    degrees=[1, 1, 1],
    knot_vectors=[[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]],
    control_points=[lat_vec1, lat_vec1, lat_vec1, lat_vec2, lat_vec2, lat_vec2, lat_vec3, lat_vec3],
)

microstructure = sp.Microstructure()
# set outer spline and a (micro) tile
microstructure.deformation_function = box
microstructure.microtile = small_box
# tiling determines tile resolutions within each bezier patch
microstructure.tiling = [2, 2, 2]
ms = microstructure.create().patches
# ms[0].evaluate()
# ms[0].control_points[0] = [0.2,0.2,0.2]
tiling = [2, 2, 1]
N = [64, 64, 32]
verts, faces = deep_sdf.mesh.create_mesh_microstructure(tiling, decoder, latent_vec_interpolation, "test_20_30_39_capped", cap_borders=True, N=N)
# deep_sdf.mesh.create_mesh_microstructure(1, decoder, latent_vec_interpolation, "test_20_30_39")

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


verts_FFD_transformed = deformation_volume.evaluate(verts)
mesh = gus.faces.Faces(verts_FFD_transformed, faces)
gus.io.meshio.export("my_mesh.inp", mesh)
print(verts)