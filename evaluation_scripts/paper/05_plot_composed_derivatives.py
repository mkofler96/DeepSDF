
import numpy as np
import gustaf as gus
import splinepy as sp
import vedo

import matplotlib.pyplot as plt

import torch
from deep_sdf import workspace as ws
import deep_sdf.utils
import pathlib
import argparse

import igl

vedo.settings.default_backend = 'k3d'
params = {'text.usetex': False, 'mathtext.fontset': 'cm', 'axes.labelsize': 12}
plt.rcParams.update(params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
deep_sdf.add_common_args(arg_parser)
args = arg_parser.parse_args()
args.debug = True
deep_sdf.configure_logging(args)

this_file_path = pathlib.Path(__file__).parent
experiment_directory = "experiments/round_cross_big_network"
checkpoint = "1000"

graded = True

latent = ws.load_latent_vectors(experiment_directory, checkpoint).to("cpu").numpy()
decoder = ws.load_trained_model(experiment_directory, checkpoint).to(device)
decoder.eval()
latent_base = np.array([0])
latent_base = latent[8]


control_points_ungraded = np.array([latent_base]*6)
control_points_graded = control_points_ungraded
# control_points_graded[3] += 0.2

tiling = [6, 3, 3]
N_base = 30

control_points_for_min_max = np.vstack([control_points_graded, control_points_ungraded])

if graded:
    graded_string = "_single_graded_derivative"
    control_points = np.vstack([control_points_graded, control_points_graded])
else:
    graded_string = "_single"
    control_points = np.vstack([control_points_ungraded, control_points_ungraded])

latent_vec_interpolation = sp.BSpline(
    degrees=[2, 1, 1],
    knot_vectors=[[-1,-1, -1, 1, 1, 1], 
                [-1, -1, 1, 1], 
                [-1, -1, 1, 1]],
    control_points=control_points,
)



def transform(x, t):
    p = 2/t
    return (2/p)*torch.abs((x-t%2) % (p*2) - p) -1 

def sdf_struct(queries):
    queries = torch.tensor(queries, dtype=torch.float32).to(device)
    tx, ty, tz = tiling


    samples = torch.zeros(queries.shape[0], 3)
    samples[:, 0] = transform(queries[:, 0], tx)
    samples[:, 1] = transform(queries[:, 1], ty)
    samples[:, 2] = transform(queries[:, 2], tz)
    lat_vec_red = torch.tensor(latent_vec_interpolation.evaluate(queries.cpu().numpy()), dtype=torch.float32)
    queries = torch.hstack([torch.tensor(lat_vec_red).to(torch.float32).to(device), samples])

    return deep_sdf.utils.decode_sdf(decoder, None, queries).squeeze(1).detach().cpu().numpy()



cap_border_dict = {
    "x0": {"cap": 1, "measure": 0.1},
    "x1": {"cap": 1, "measure": 0.1},
    "y0": {"cap": 1, "measure": 0.1},
    "y1": {"cap": 1, "measure": 0.1},
}

N = [N_base * t+1 for t in tiling]
verts, faces, jac = deep_sdf.mesh.create_mesh_microstructure_diff(tiling, decoder, latent_vec_interpolation, cap_border_dict=cap_border_dict, N=N, device=device, compute_derivatives=True)



# geometric parameters
width = 5
height = 5
depth = 1

control_points=np.array([
        [0, 0, 0],
        [0, height, 0],
        [width, 0, 0],
        [width, height, 0]
    ])

deformation_surf = sp.BSpline(
    degrees=[1,1],
    control_points=control_points,
    knot_vectors=[[0, 0, 1, 1],[0, 0, 1, 1]],
)

deformation_volume = deformation_surf.create.extruded(extrusion_vector=[0,0,depth])

# bring slightly outside vertices back 
verts[verts>1] = 1
verts[verts<0] = 0

verts_FFD_transformed = deformation_volume.evaluate(verts)

surf_mesh = gus.faces.Faces(verts_FFD_transformed, faces)

r = igl.decimate(surf_mesh.vertices, surf_mesh.faces, int(1e5))
dmesh = gus.Faces(r[1], r[2])


