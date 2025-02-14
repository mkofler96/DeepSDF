from analysis.geometry import DeepSDFMesh
import gustaf as gus
import numpy as np
from deep_sdf.utils import _TUWIEN_COLOR_SCHEME
mesh_setup = {
        "N_base_reconstruction": 30,
        "decimate_mesh": True,
        "tiling": [5, 3, 3],
        "degrees": [1, 1, 1],
        "refinement": [],
        "experiment_directory": "experiments/round_cross_big_network",
        "checkpoint": "1000",
        "cap_border_dict": {
            "x0": {"cap": -1, "measure": 0.05},
            "x1": {"cap": -1, "measure": 0.05},
            "y0": {"cap": -1, "measure": 0.05},
            "y1": {"cap": -1, "measure": 0.05},
            "z0": {"cap": -1, "measure": 0.05},
            "z1": {"cap": -1, "measure": 0.05}
        },
        "remove_orphans": False
    }

mesh = DeepSDFMesh(mesh_options=mesh_setup)


control_points = [[0.5], [0.25], 
                  [0.8], [0.8]]

control_points = control_points+control_points
mesh.generate_surface_mesh(control_points=control_points, normalize_jac=False)
jacobian = mesh.jacobian
surf = mesh.surface_mesh


cam = dict(
    position=(4.01840, 2.18423, 3.40005),
    focal_point=(0.234872, -9.31578e-3, -0.227031),
    viewup=(-0.264191, 0.922270, -0.282173),
    roll=1.19567,
    distance=5.68176,
    clipping_range=(1.98179, 7.72853),
)

gus_surf_faces = gus.Faces(surf.vertices, surf.faces)
gus_surf_faces.show_options["c"] = _TUWIEN_COLOR_SCHEME["grey_2"]
showable = gus.show(gus_surf_faces, cam=cam, interactive=False)
showable.screenshot("evaluation_scripts/AICOMAS2024/plots/mesh_extraction_2.png")

# normalize theta


def dot_prod(A, B) -> np.ndarray:
    dot_ai_bi = (A * B).sum(axis=-1, keepdims=True)
    dot_bi_bi = (B * B).sum(axis=-1, keepdims=True)  # or square `norm`
    zero_normals = np.all(B==0, axis=1)
    n_zero_normals = len(np.nonzero(zero_normals))
    if n_zero_normals > 0:
        dot_bi_bi[zero_normals] = np.Inf
    C = dot_ai_bi / dot_bi_bi * B
    return C

normals = surf.vertex_normals
# gus_faces = gus.Faces(faces.vertices, faces.faces)
# normals = gus.create.faces.vertex_normals(gus_faces, angle_weighting=True, area_weighting=True)
# # .vertex_data["normals"]    
zero_normals = np.all(normals==0, axis=1)
n_zero_normals = len(np.nonzero(zero_normals))
if n_zero_normals > 0:
    print(f"{n_zero_normals} 0-Normal vectors detected")
dVertices_normal = np.zeros_like(jacobian)

delete_above = 0.5
for i in range(jacobian.shape[2]):
    dVertices_normal[:,:,i] = dot_prod(np.float64(jacobian[:,:,i]),normals)
    mean_norm = np.linalg.norm(dVertices_normal[:, :, i], axis=1).mean()
    mask = np.linalg.norm(dVertices_normal[:, :, i], axis=1) > delete_above
    dVertices_normal[mask, :, :] = 1e-12


gus_faces = gus.Faces(surf.vertices, surf.faces)

gus_faces.vertex_data["directions"] = dVertices_normal[:,:,4]
gus_faces.show_options["arrow_data"] = "directions"

cam = dict(
    position=(4.01840, 2.18423, 3.40005),
    focal_point=(0.234872, -9.31578e-3, -0.227031),
    viewup=(-0.264191, 0.922270, -0.282173),
    roll=1.19567,
    distance=5.68176,
    clipping_range=(1.98179, 7.72853),
)

showable = gus.show(gus_faces, cam=cam, interactive=False)
showable.screenshot("evaluation_scripts/AICOMAS2024/plots/mesh_extraction_with_gradient.png")