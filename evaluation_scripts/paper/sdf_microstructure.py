import numpy as np
import gustaf as gus
import splinepy as sp
import torch
import tetgenpy
from deep_sdf import workspace as ws
import deep_sdf.utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_microstructure_from_experiment(experiment_directory: str, tiling=[2,1,1], N_base=30):
    """
    generates microstructure from DeepSDF experiment
    """
    # experiment_directory = "experiments/round_cross_big_network"
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
        "y0": {"cap": 1, "measure": 0.1},
        "y1": {"cap": 1, "measure": 0.1},
        "z0": {"cap": -1, "measure": 0.1},
        "z1": {"cap": -1, "measure": 0.1},
    }

    N = [N_base * t+1 for t in tiling]

    verts, faces, jac = deep_sdf.mesh.create_mesh_microstructure_diff(tiling, decoder, latent_vec_interpolation, cap_border_dict=cap_border_dict, N=N, device=device, compute_derivatives=True)
    jac = jac.reshape((jac.shape[0], jac.shape[1], -1))
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    print(jac.shape)


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
    # max_plots = 8
    # for i in range(min(max_plots,jac.shape[2])):
    #     faces_der1 = gus.Faces(verts_np, faces_np)
    #     normals = gus.create.faces.vertex_normals(faces_der1, angle_weighting=True, area_weighting=True)
    #     directions = jac
    #     positions = verts_np
    #     switch_signs = -(2*(i%2)-1)
    #     switch_signs = -1
    #     dSdC = dot_prod(jac[:,:,i],normals.vertex_data["normals"])
    #     faces_der1.vertex_data["directions"] = jac[:,:,i]*switch_signs
    #     faces_der1.vertex_data["directions_normalized"] = switch_signs*dSdC
    #     faces_der1.vertex_data["directions_magnitude"] = np.linalg.norm(switch_signs*dSdC, axis=1)
    #     faces_der1.show_options["arrow_data"] = "directions_normalized"
    #     faces_der1.show_options["data"] = "directions_magnitude"
    #     faces.append(faces_der1)
    # gus.show(*faces)

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
    t_out = tetgenpy.tetrahedralize("p", t_in) #pqa

    tets = np.vstack(t_out.tetrahedra())
    verts = t_out.points()

    return gus.Volumes(verts, tets)

def export_mesh(volumes: gus.Volumes, filename: str, show_mesh=False):
    faces = volumes.to_faces(False)
    boundary_faces = faces.single_faces()
    verts = volumes.vertices

    BC = {1: [], 2: [], 3: []}
    width = volumes.vertices[:,0].max()
    height = volumes.vertices[:,1].max()
    for i in boundary_faces:
        # mark boundaries at x = 0 with 1
        if np.max(verts[faces.const_faces[i], 0]) < 3e-2:
            BC[1].append(i)
        # mark boundaries at x = 1 with 2
        elif np.logical_and(np.min(verts[faces.const_faces[i], 0]) > 0.49*width,
                            np.min(verts[faces.const_faces[i], 1]) > 0.999*height):
            BC[2].append(i)
        # mark rest of the boundaries with 3
        else:
            BC[3].append(i)
    volumes.BC = BC
    if show_mesh:
        gus.show(volumes)
    gus.io.mfem.export(filename, volumes)