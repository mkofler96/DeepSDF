import matplotlib.pyplot as plt
import numpy as np
from sdf_sampler.plotting import scatter_contour_at_origin
from deep_sdf.mesh import location_lookup, CapBorderDict
import torch
import gustaf as gus
from deep_sdf.utils import _TUWIEN_COLOR_SCHEME
import tetgenpy 

plt.style.use("gmod.mplstyle")
from flexicubes.flexicubes import FlexiCubes
def hor_beam_sdf(xyz, d1, d2, d3=0.3):
    output = np.inf * np.ones(xyz.shape[0])
    # add x cylinder
    cylinder = np.sqrt(xyz[:,1]**2 + xyz[:,2]**2) - d1
    # cylinder = np.abs(xyz[:,1]) - d1
    output = np.minimum(output, cylinder)
    # add y cylinder
    cylinder = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2) - d2
    output = np.minimum(output, cylinder)
    #
    cylinder = np.sqrt(xyz[:,2]**2 + xyz[:,0]**2) - d3
    output = np.minimum(output, cylinder)
    return output

n_cells = 2
tiling = (1,n_cells,1)

def transform(x, t):
    p = 2/t
    return (2/p)*np.abs((x-t%2) % (p*2) - p) -1

def sdf_struct(queries):
    tx, ty, tz = tiling


    samples = np.zeros_like(queries)
    samples[:, 0] = transform(queries[:, 0], tx)
    samples[:, 1] = transform(queries[:, 1], ty)
    samples[:, 2] = transform(queries[:, 2], tz)
    return hor_beam_sdf(samples, np.array(0.3), np.array(0.3))

plt.rcParams.update({'axes.titlesize': "small"})
fig, axs = plt.subplots(1,2, width_ratios=[1, n_cells])
scatter_contour_at_origin(sdf_struct, custom_axis=axs[0])
scatter_contour_at_origin(sdf_struct, custom_axis=axs[1], scale=1/n_cells)
axs[0].set_xlabel(r"$x$")
axs[0].set_ylabel(r"$y$")
axs[1].set_xlabel(r"$\bar{x}$")
axs[1].set_ylabel(r"$\bar{y}$")
# axs[0].set_title(r"Untransformed")
# axs[1].set_title(r"FFD-Transformed")
# plt.show()
# plt.savefig("screenshots/ffd_transformation.png", bbox_inches="tight", dpi=1000)


device = "cpu"
flexi_cubes_constructor = FlexiCubes(device=device)

N = (20, 40, 20)

cap_border_dict = {
    "x0": {"cap": -1, "measure": 0.1},
    "x1": {"cap": -1, "measure": 0.1},
    "y0": {"cap": -1, "measure": 0.1},
    "y1": {"cap": -1, "measure": 0.1},
    "z0": {"cap": -1, "measure": 0.1},
    "z1": {"cap": -1, "measure": 0.1},
}

samples_orig, cube_idx = flexi_cubes_constructor.construct_voxel_grid(res=tuple(N))
samples_orig = samples_orig*2.1
sdf_values = torch.tensor(sdf_struct(samples_orig))
# logger.debug("sampling takes: %f" % (sample_time - start_time))
for loc, cap_dict in cap_border_dict.items():
    cap, measure = cap_dict["cap"], cap_dict["measure"]
    dim, multiplier = location_lookup[loc]
    border_sdf = (samples_orig[:, dim] - multiplier*(1-measure))*-multiplier
    if cap == -1:
        sdf_values = torch.maximum(sdf_values, -border_sdf)
    elif cap == 1:
        sdf_values = torch.minimum(sdf_values, border_sdf)
    else:
        raise ValueError("Cap must be -1 or 1")

#cap everything outside the unit cube

for (dim, measure) in zip([0, 0, 1, 1, 2, 2], [-1, 1, -1, 1, -1, 1]):
    border_sdf = (samples_orig[:, dim] - measure)*-measure
    sdf_values = torch.maximum(sdf_values, -border_sdf)


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
    position=(-2.13990, 3.92425, 5.22591),
    focal_point=(0.0322765, 0.130556, -0.0801359),
    viewup=(-0.948551, -0.166111, -0.269551),
    distance=6.87493,
    clipping_range=(2.88140, 11.9185),
)
faces.show_options["lw"] = 1
faces.vertices[:,1] = faces.vertices[:,1]*2
faces.show_options["c"] = _TUWIEN_COLOR_SCHEME["grey_2"]
print(f"N faces: {len(faces.faces)}")
showable = gus.show(faces, cam=cam, interactive=False)
showable.screenshot("screenshots/mesh_extraction_with_lines.png")
t_in = tetgenpy.TetgenIO()
t_in.setup_plc(faces.vertices, faces.faces.tolist())
t_out = tetgenpy.tetrahedralize("pYq", t_in) #pqa
print(f"N volumes: {len(t_out.tetrahedra())}")