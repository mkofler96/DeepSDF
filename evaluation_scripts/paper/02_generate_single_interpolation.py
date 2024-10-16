
import pathlib

import gustaf as gus
import matplotlib.pyplot as plt
import numpy as np
import splinepy as sp
import torch

import deep_sdf.utils
from deep_sdf import workspace as ws
from sdf_sampler.plotting import scatter_contour_at_z_level

this_folder = pathlib.Path(__file__).parent

import igl

import gustaf as gus
import numpy as np
import splinepy as sp
import vedo



import igl
import matplotlib.pyplot as plt
import torch

import deep_sdf.utils
from deep_sdf import workspace as ws
from sdf_sampler.plotting import scatter_contour_at_origin

params = {'text.usetex': False, 'mathtext.fontset': 'cm', 'axes.labelsize': 12}
plt.rcParams.update(params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


experiment_directory = "experiments/double_lattice_3D_no_topo"
checkpoint = "1000"

graded = True

latent = ws.load_latent_vectors(experiment_directory, checkpoint).to("cpu").numpy()
decoder = ws.load_trained_model(experiment_directory, checkpoint).to(device)
decoder.eval()
latent_base = np.array([0, -0.4])


control_points_ungraded = np.array([latent_base]*4)
control_points_graded = control_points_ungraded
control_points_graded[3] += 0.2

tiling = [1, 1, 1]
N_base = 50


control_points_for_min_max = np.vstack([control_points_graded, control_points_ungraded])

r_min = control_points_for_min_max[:,0].min()
r_max = control_points_for_min_max[:,0].max()
g_min = control_points_for_min_max[:,1].min()
g_max = control_points_for_min_max[:,1].max()


if graded:
    graded_string = "_single_graded"
    control_points = np.vstack([control_points_graded, control_points_graded])
else:
    graded_string = "_single"
    control_points = np.vstack([control_points_ungraded, control_points_ungraded])

latent_vec_interpolation = sp.BSpline(
    degrees=[1, 1, 1],
    knot_vectors=[[-1, -1, 1, 1],
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

scatter_contour_at_origin(sdf_struct, normal=(0,1,0), res=1000)


