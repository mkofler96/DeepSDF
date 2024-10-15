
import numpy as np
import gustaf as gus
import splinepy as sp
import vedo
import trimesh
import pathlib
import os
from sdf_sampler.plotting import scatter_contour_at_z_level
from sdf_sampler import sdf_sampler
from sdf_sampler.microstructures import CrossMsSDF, CornerSpheresSDF

from sdf_sampler.double_lattice_extruded import DoubleLatticeExtruded

geometry_dir = pathlib.Path("data/geometry/double_lattice")
if not geometry_dir.is_dir():
    os.makedirs(geometry_dir)
outdir = pathlib.Path("data/SdfSamples")
splitdir = pathlib.Path("data/splits")
SDF_sampler = sdf_sampler.SDFSampler(str(outdir), str(splitdir))

sdfs = []
index = 0
# old was from 0.01 to 0.2, now 0.05 to 0.2, to avoid topological changes
for i, t1 in enumerate(np.linspace(0.05, 0.2, 21)):
    for j, t2 in enumerate(np.linspace(0.05, 0.2, 21)):
        tile_creator = DoubleLatticeExtruded()
        tile = tile_creator.create_tile(parameters=np.array([[t1, t2]]))
        mesh = sp.helpme.extract.faces(sp.multipatch.Multipatch(tile[0]), 20)
        # scale from [0,1] to [-1,1]
        mesh.vertices = mesh.vertices*2 - np.array([1, 1, 1])
        t_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        filename = f"{index}_double_lattice_{t1:.2f}_{t2:.2f}".replace(".", "_") + ".ply"
        t_mesh.export(geometry_dir/filename)
        sdfs.append(sdf_sampler.SDFfromMesh(t_mesh))
        index = index + 1

dataset_info = {
    "dataset_name": "microstructure",
    "class_name": "double_lattice_no_topo"}
training_split_info = SDF_sampler.sample_sdfs(sdfs, dataset_info, n_samples=1e5, sampling_strategy="uniform", show=False)
SDF_sampler.write_json("double_lattice_no_topo_3D.json", dataset_info, training_split_info)