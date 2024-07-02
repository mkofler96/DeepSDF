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
tiling = [4, 4, 8]
N = [256, 256, 256]
verts, faces = deep_sdf.mesh.create_mesh_microstructure(tiling, decoder, latent_vec_interpolation, "test_20_30_39_capped", cap_borders=True, N=N)
# deep_sdf.mesh.create_mesh_microstructure(1, decoder, latent_vec_interpolation, "test_20_30_39")

# Free Form Deformation
# geometric parameters
width = 10
length = 20

deformation_volume = sp.helpme.create.box(width, width, length).bspline
deformation_volume.elevate_degrees([2,2])

deformation_volume.control_points[8,1] = 2
deformation_volume.control_points[9,1] = 2

deformation_volume.control_points[12,1] = 5
deformation_volume.control_points[13,1] = 5

verts_FFD_transformed = deformation_volume.evaluate(verts)
mesh = gus.faces.Faces(verts_FFD_transformed, faces)
gus.io.meshio.export("beam.inp", mesh)
print(verts)