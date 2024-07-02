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
import pygalmesh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_directory = "./experiments/snappy3D"
checkpoint = "1000"

decoder = ws.load_trained_model(experiment_directory, checkpoint)
latent = ws.load_latent_vectors(experiment_directory, checkpoint).to(device)

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
N_base = 32
N = [N_base * t for t in tiling]
verts, faces = deep_sdf.mesh.create_mesh_microstructure(tiling, decoder, latent_vec_interpolation, "none", cap_borders=True, N=N)
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
fname_surf = f"meshs/facade_snappy_{'_'.join([str(l) for l in N])}_surf.inp"
gus.io.meshio.export(fname_surf, mesh)

out_mesh = pygalmesh.generate_volume_mesh_from_surface_mesh(
    fname_surf,
    min_facet_angle=25.0,
    max_radius_surface_delaunay_ball=0.15,
    max_facet_distance=0.008,
    max_circumradius_edge_ratio=3.0,
    verbose=False,
)
fname_volume = f"meshs/facade_snappy_{'_'.join([str(l) for l in N])}_volume.inp"

out_mesh.cells.pop(0)

node_sets = {
    "left": np.argwhere(out_mesh.points[:,0] < 0.5)[:,0],
    "right": np.argwhere(out_mesh.points[:,0] > 4.5)[:,0],
}
out_mesh.point_sets = node_sets
out_mesh.write(fname_volume)
print("done")