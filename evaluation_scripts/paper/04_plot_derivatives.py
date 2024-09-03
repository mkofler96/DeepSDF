import kaolin
import torch
import vedo.mesh
import gustaf as gus
import numpy as np
import vedo
import matplotlib.pyplot as plt

import numpy as np
import gustaf as gus
import splinepy as sp

from sdf_sampler.plotting import scatter_contour_at_z_level
import matplotlib.pyplot as plt

import torch
from deep_sdf import workspace as ws
import deep_sdf.utils
from deep_sdf.mesh import CapBorderDict, location_lookup
import pathlib

this_folder = pathlib.Path(__file__).parent

import igl
params = {'text.usetex': False, 'mathtext.fontset': 'cm', 'axes.labelsize': 12}
plt.rcParams.update(params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

experiment_directory = "./experiments/snappy3D_latent_2D"
checkpoint = "1000"

graded = True

latent = ws.load_latent_vectors(experiment_directory, checkpoint).to(device)
decoder = ws.load_trained_model(experiment_directory, checkpoint).to(device)
decoder.eval()

N = 21

cap_border_dict = {
    "x0": {"cap": -1, "measure": 0.01},
    "x1": {"cap": -1, "measure": 0.01},
    "y0": {"cap": -1, "measure": 0.01},
    "y1": {"cap": -1, "measure": 0.01},
    "z0": {"cap": -1, "measure": 0.01},
    "z1": {"cap": -1, "measure": 0.01},
}

def deep_sdf_function(samples, parameter):
    samples_orig = deep_sdf.utils.decode_sdf(decoder, parameter, samples)
    sdf_values = samples_orig[:,-1]
    for loc, cap_dict in cap_border_dict.items():
        cap, measure = cap_dict["cap"], cap_dict["measure"]
        dim, multiplier = location_lookup[loc]
        border_sdf = (samples[:, dim] - multiplier*(1-measure))*-multiplier
        if cap == -1:
            sdf_values = torch.maximum(sdf_values, -border_sdf)
        elif cap == 1:
            sdf_values = torch.minimum(sdf_values, border_sdf)
        else:
            raise ValueError("Cap must be -1 or 1")

    return -sdf_values

reconstructor = kaolin.non_commercial.FlexiCubes(device='cuda')

samples, cube_idx = reconstructor.construct_voxel_grid(resolution=N)
samples = samples*2

parameter = torch.tensor([0, -0.4], device='cuda', requires_grad = True)
parameter = latent[25]
parameter.requires_grad = True

# sdf_values = sphere_sdf(samples, parameter)
sdf_values = deep_sdf_function(samples, parameter)
# sdf_values.requires_grad = True
output_tetmesh = False

verts, faces, loss = reconstructor(voxelgrid_vertices=samples,
                            scalar_field=sdf_values.view(-1), 
                            cube_idx=cube_idx,
                            resolution=N,
                            output_tetmesh=output_tetmesh)

faces_np = faces.cpu().numpy()
verts_np = verts.detach().cpu().numpy()
# mesh = vedo.mesh.Mesh([verts_np, faces_np])
mesh = gus.Faces(verts_np, faces_np)

def verts_from_param(param):
    sdf_values = deep_sdf_function(samples, param)
    verts, faces, loss = reconstructor(voxelgrid_vertices=samples,
                            scalar_field=sdf_values.view(-1), 
                            cube_idx=cube_idx,
                            resolution=N,
                            output_tetmesh=output_tetmesh)
    return verts

faces = gus.Faces(verts_np, faces_np)


jac = torch.autograd.functional.jacobian(verts_from_param, parameter, strict=True)
print(jac)
directions = jac.detach().cpu().numpy()
positions = verts.cpu().detach().numpy()
faces.vertex_data["directions"] = directions[:,:,1]
faces.show_options["arrow_data"] = "directions"

gus.show(faces, axes=1)

# arrs = []
# arrs.append(mesh)

# fig, ax = plt.subplots()
# for pt, grad in zip(positions, directions):
#         # if np.abs(pt[2]) < 0.001: 
#         arrs.append(vedo.Arrow(pt, pt+grad[:,0]*0.1, s=0.0005))

# vedo.show(arrs)
