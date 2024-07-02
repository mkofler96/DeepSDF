
import numpy as np
import gustaf as gus
import splinepy as sp
import vedo
import trimesh
from sdf_sampler.plotting import scatter_contour_at_z_level
from sdf_sampler import sdf_sampler
from sdf_sampler.microstructures import CrossMsSDF, CornerSpheresSDF

from sdf_sampler.snappy_3d import Snappy3D

geometry_dir = "data/geometry/snappy_tile"
outdir = "data/SdfSamples"
splitdir = "data/splits"
SDF_sampler = sdf_sampler.SDFSampler(outdir, splitdir)

sdfs = []

for i, t in enumerate(np.linspace(0, 0.7, 40)):
    tile_creator = Snappy3D()
    tile = tile_creator.create_tile(parameters=np.array([[t]]))
    mesh = sp.helpme.extract.faces(sp.multipatch.Multipatch(tile[0]), 20)
    # scale from [0,1] to [-1,1]
    mesh.vertices = mesh.vertices*2 - np.array([1, 1, 1])
    t_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    t_mesh.export(geometry_dir + f"/snappy_{i}.ply")
    sdfs.append(sdf_sampler.SDFfromMesh(t_mesh))


dataset_info = {
    "dataset_name": "microstructure",
    "class_name": "snappy3D"}
training_split_info = SDF_sampler.sample_sdfs(sdfs, dataset_info, n_samples=1e5, sampling_strategy="uniform", show=False)
SDF_sampler.write_json("snappy3D.json", dataset_info, training_split_info)