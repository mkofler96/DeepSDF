import matplotlib.pyplot as plt
import numpy as np
from sdf_sampler.plotting import scatter_contour_at_origin
from deep_sdf.mesh import location_lookup, CapBorderDict
import torch
import gustaf as gus
from deep_sdf.utils import _TUWIEN_COLOR_SCHEME
import tetgenpy 

from flexicubes.flexicubes import FlexiCubes

import torch
import numpy as np
import splinepy as sp
import matplotlib.pyplot as plt

import deep_sdf.workspace as ws
import deep_sdf.utils
from sdf_sampler.plotting import scatter_contour_at_origin
import os
os.chdir("/usr2/mkofler/DeepSDF_microstructures")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

experiment_directory = "./experiments/round_cross_big_network"
checkpoint = "1000"

params = {'text.usetex': True}
plt.rcParams.update(params)

# latent = ws.load_latent_vectors(experiment_directory, checkpoint).to("cpu").numpy()
decoder = ws.load_trained_model(experiment_directory, checkpoint).to(device)
decoder.eval()
graded = True

control_points = [[0.8], [0.5], 
                  [0.8], [0.25]]

control_points = control_points+control_points

continuos_interpolation = sp.BSpline(
    degrees=[1, 1, 1],
    knot_vectors=[[-1,-1, 1, 1],
                  [-1,-1, 1, 1],
                  [-1,-1, 1, 1]],
    control_points=control_points,
)

tiling = (3,5,3)

def transform(x, t):
    p = 2/t
    return (2/p)*torch.abs((x-t%2) % (p*2) - p) -1 

def sdf_struct(queries, latent_vec_interpolation, tiling=[5, 1, 1]):
    queries = torch.tensor(queries, dtype=torch.float32).to(device)
    tx, ty, tz = tiling[0], tiling[1], tiling[2]
    samples = torch.zeros(queries.shape[0], 3, device=device)
    samples[:, 0] = transform(queries[:, 0], tx)
    samples[:, 1] = transform(queries[:, 1], ty)
    samples[:, 2] = transform(queries[:, 2], tz)
    lat_vec_red = torch.tensor(latent_vec_interpolation.evaluate(queries.cpu().numpy()), dtype=torch.float32, device=device)
    queries = torch.hstack([torch.tensor(lat_vec_red).to(torch.float32).to(device), samples])
    sdf = deep_sdf.utils.decode_sdf(decoder, None, queries)
    return sdf


flexi_cubes_constructor = FlexiCubes(device=device)

N = (20*tiling[0], 40*tiling[1], 20*tiling[2])

cap_border_dict = {
    "x0": {"cap": -1, "measure": 0.1},
    "x1": {"cap": -1, "measure": 0.1},
    "y0": {"cap": -1, "measure": 0.1},
    "y1": {"cap": -1, "measure": 0.1},
    "z0": {"cap": -1, "measure": 0.1},
    "z1": {"cap": -1, "measure": 0.1},
}

samples_orig, cube_idx = flexi_cubes_constructor.construct_voxel_grid(res=tuple(N))
samples_orig = samples_orig*2
sdf_values = torch.tensor(sdf_struct(samples_orig, continuos_interpolation,tiling=tiling))
# logger.debug("sampling takes: %f" % (sample_time - start_time))
for loc, cap_dict in cap_border_dict.items():
    cap, measure = cap_dict["cap"], cap_dict["measure"]
    dim, multiplier = location_lookup[loc]
    border_sdf = (samples_orig[:, dim] - multiplier*(1-measure))*-multiplier
    if cap == -1:
        sdf_values = torch.maximum(sdf_values, -border_sdf.view(-1,1))
    elif cap == 1:
        sdf_values = torch.minimum(sdf_values, border_sdf.view(-1,1))
    else:
        raise ValueError("Cap must be -1 or 1")

#cap everything outside the unit cube

for (dim, measure) in zip([0, 0, 1, 1, 2, 2], [-1, 1, -1, 1, -1, 1]):
    border_sdf = (samples_orig[:, dim] - measure)*-measure
    sdf_values = torch.maximum(sdf_values, -border_sdf.view(-1,1))


verts, faces, loss = flexi_cubes_constructor(x_nx3=samples_orig[:, :3].to(torch.float),
                            s_n=sdf_values.to(torch.float), 
                            cube_fx8=cube_idx,
                            res=tuple(N),
                            output_tetmesh=False)

faces = gus.Faces(verts.detach().cpu().numpy(), faces.detach().cpu().numpy())

# print(cube_idx)                                                     #0 1 2 3 4 5 6 7
cube_idx_swapped = torch.index_select(cube_idx, 1, torch.LongTensor([1,5,4,0,3,7,6,2]))

lines = []

lines.append(gus.Edges(samples_orig, torch.index_select(cube_idx, 1, torch.LongTensor([1,5]))))
lines.append(gus.Edges(samples_orig, torch.index_select(cube_idx, 1, torch.LongTensor([5,4]))))
lines.append(gus.Edges(samples_orig, torch.index_select(cube_idx, 1, torch.LongTensor([4,0]))))
lines.append(gus.Edges(samples_orig, torch.index_select(cube_idx, 1, torch.LongTensor([0,1]))))
lines.append(gus.Edges(samples_orig, torch.index_select(cube_idx, 1, torch.LongTensor([3,7]))))
lines.append(gus.Edges(samples_orig, torch.index_select(cube_idx, 1, torch.LongTensor([7,6]))))
lines.append(gus.Edges(samples_orig, torch.index_select(cube_idx, 1, torch.LongTensor([6,2]))))
lines.append(gus.Edges(samples_orig, torch.index_select(cube_idx, 1, torch.LongTensor([2,3]))))
lines.append(gus.Edges(samples_orig, torch.index_select(cube_idx, 1, torch.LongTensor([1,3]))))
lines.append(gus.Edges(samples_orig, torch.index_select(cube_idx, 1, torch.LongTensor([5,7]))))
lines.append(gus.Edges(samples_orig, torch.index_select(cube_idx, 1, torch.LongTensor([4,6]))))
lines.append(gus.Edges(samples_orig, torch.index_select(cube_idx, 1, torch.LongTensor([0,2]))))
# edges = gus.Edges(samples_orig, cube_idx_swapped)
# gus.show(lines)
# Volumes = gus.Volumes(samples_orig, cube_idx)
# Volumes.show_options["alpha"] = 0.00001
# Volumes.show_options["lw"] = 10
# Volumes.show_options["lc"] = "black"
cam = dict(
    position=(-3.15523, 4.82828, 6.00027),
    focal_point=(0.0322782, 0.130553, -0.0801401),
    viewup=(-0.923127, -0.206849, -0.324115),
    roll=-104.515,
    distance=8.31867,
    clipping_range=(4.22003, 13.5044),
)
# faces.show_options["lw"] = 1
faces.vertices[:,1] = faces.vertices[:,1]*2
faces.show_options["c"] = _TUWIEN_COLOR_SCHEME["grey_2"]
print(f"N faces: {len(faces.faces)}")
showable = gus.show(faces, cam=cam, interactive=False)
showable.screenshot("evaluation_scripts/AICOMAS2024/plots/mesh_extraction.png")
t_in = tetgenpy.TetgenIO()
t_in.setup_plc(faces.vertices, faces.faces.tolist())
t_out = tetgenpy.tetrahedralize("pYq", t_in) #pqa
print(f"N volumes: {len(t_out.tetrahedra())}")