import gustaf as gus
import napf
import numpy as np
import pathlib
import splinepy as sp
import tetgenpy
import torch


import deep_sdf.utils
from deep_sdf import workspace as ws

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_microstructure_from_experiment(experiment_directory: str, tiling=None, N_base=30):
    """
    generates microstructure from DeepSDF experiment
    """
    # experiment_directory = "experiments/round_cross_big_network"
    if tiling is None:
        tiling = [2, 1, 1]
    checkpoint = "1000"

    latent = ws.load_latent_vectors(experiment_directory, checkpoint).to("cpu").numpy()
    decoder = ws.load_trained_model(experiment_directory, checkpoint).to(device)
    decoder.eval()
    latent_base = np.array([0])
    latent_base = latent[8]


    control_points_ungraded = np.array([latent_base]*6)
    control_points_graded = control_points_ungraded

    # note: tiling [4,4,1] N_base 50 produces the error: shape '[204, 204, 54]' is invalid for input of size 2269350

    control_points = np.vstack([control_points_graded, control_points_graded])


    latent_vec_interpolation = sp.BSpline(
        degrees=[2, 1, 1],
        knot_vectors=[[-1, -1, -1, 1, 1, 1],
                    [-1, -1, 1, 1],
                    [-1, -1, 1, 1]],
        control_points=control_points,
    )





    cap_border_dict = {
        "x0": {"cap": 1, "measure": 0.1},
        "x1": {"cap": 1, "measure": 0.1},
        "y0": {"cap": -1, "measure": 0.1},
        "y1": {"cap": -1, "measure": 0.1},
        "z0": {"cap": 1, "measure": 0.1},
        "z1": {"cap": 1, "measure": 0.1},
    }

    N = [N_base * t+1 for t in tiling]

    verts, faces, jac = deep_sdf.mesh.create_mesh_microstructure_diff(tiling, decoder, latent_vec_interpolation, cap_border_dict=cap_border_dict, N=N, device=device, compute_derivatives=True)
    jac = jac.reshape((jac.shape[0], jac.shape[1], -1))
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()


    # "freeform deformation" of the mesh
    verts_np[:,0] = verts_np[:,0]*2
    jac[:,0,:] = jac[:,0,:]*2


    def dot_prod(A, B) -> np.ndarray:
        dot_ai_bi = (A * B).sum(axis=-1, keepdims=True)
        dot_bi_bi = (B * B).sum(axis=-1, keepdims=True)  # or square `norm`
        C = dot_ai_bi / dot_bi_bi * B
        return C


    faces = []
    jac[np.where(jac>1)] = 0
    jac[np.where(jac<-1)] = 0
    return gus.Faces(verts_np, faces_np), jac

def transform(x, t):
    p = 2/t
    return (2/p)*torch.abs((x-t%2) % (p*2) - p) -1

def sdf_struct(decoder, queries, tiling, latent_vec_interpolation):
    queries = torch.tensor(queries, dtype=torch.float32).to(device)
    tx, ty, tz = tiling


    samples = torch.zeros(queries.shape[0], 3)
    samples[:, 0] = transform(queries[:, 0], tx)
    samples[:, 1] = transform(queries[:, 1], ty)
    samples[:, 2] = transform(queries[:, 2], tz)
    lat_vec_red = torch.tensor(latent_vec_interpolation.evaluate(queries.cpu().numpy()), dtype=torch.float32)
    queries = torch.hstack([torch.tensor(lat_vec_red).to(torch.float32).to(device), samples])

    return deep_sdf.utils.decode_sdf(decoder, None, queries).squeeze(1).detach().cpu().numpy()

def tetrahedralize_surface(surface_mesh):
    t_in = tetgenpy.TetgenIO()
    t_in.setup_plc(surface_mesh.vertices, surface_mesh.faces.tolist())
    # gus.show(dmesh)
    t_out = tetgenpy.tetrahedralize("pYq", t_in) #pqa

    tets = np.vstack(t_out.tetrahedra())
    verts = t_out.points()


    kdt = napf.KDT(tree_data=verts, metric=1)

    distances, face_indices = kdt.knn_search(
        queries=surface_mesh.vertices,
        kneighbors=1,
        nthread=4,
    )
    tol = 1e-6
    if distances.max() > tol:
        Warning("Not all surface nodes as included in the volumetric mesh.")
    return gus.Volumes(verts, tets), face_indices

def export_mesh(volumes: gus.Volumes, filename: str, show_mesh=False, export_abaqus=False):
    filepath = pathlib.Path(filename)
    faces = volumes.to_faces(False)
    boundary_faces = faces.single_faces()
    verts = volumes.vertices

    BC = {1: [], 2: [], 3: []}

    tolerance = 3e-2
    width = verts[:,0].max()
    for i in boundary_faces:
        # mark boundaries at x = 0 with 1
        if np.max(verts[faces.const_faces[i], 0]) < tolerance:
            BC[1].append(i)
        # mark boundaries at x = width with 2
        elif np.max(verts[faces.const_faces[i], 0]) > (width - tolerance):
            BC[2].append(i)
        # mark rest of the boundaries with 3
        else:
            BC[3].append(i)
    volumes.BC = BC
    if show_mesh:
        gus.show(volumes)
    print(f"Exporting mesh with {len(volumes.volumes)} elements, {len(volumes.vertices)} vertices, {len(BC[1])} boundaries with marker 1, {len(BC[2])} boundaries with marker 2, and {len(BC[3])} boundaries with marker 3.")
    gus.io.mfem.export(str(filepath), volumes)
    if export_abaqus:
        gus.io.meshio.export(str(filepath.with_suffix(".inp")))
