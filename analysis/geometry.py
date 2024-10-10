import gustaf as gus
import napf
import numpy as np
import pathlib
import splinepy as sp
import tetgenpy
import torch
import logging
import trimesh
import os

import deep_sdf.utils
from deep_sdf import workspace as ws

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepSDFMesh():
    """
    generates microstructure from DeepSDF experiment
    """
    def __init__(self, mesh_options):
        # check if required keys are present
        if not ("experiment_directory" in mesh_options):
            raise KeyError("Key experiment_directory not found in general settings")
        if not ("checkpoint" in mesh_options):
            raise KeyError("Key checkpoint not found in general settings")
        if not os.path.exists(mesh_options["experiment_directory"]):
            raise FileNotFoundError(f"Experiment directory {mesh_options['experiment_directory']} not found")
        self.options = mesh_options

        self.exp_dir = self.options["experiment_directory"]
        checkpoint = self.options["checkpoint"]
        self.latent = ws.load_latent_vectors(self.exp_dir, checkpoint).to("cpu").numpy()
        self.decoder = ws.load_trained_model(self.exp_dir, checkpoint).to(device)
    
        self.decoder.eval()
        latent_base = np.zeros_like(self.latent[8])

        n = np.array(self.options["degrees"])
        knot_vectors = [[-1]*(n[0]+1)+[1]*(n[0]+1),
                        [-1]*(n[1]+1)+[1]*(n[1]+1),
                        [-1]*(n[2]+1)+[1]*(n[2]+1)]
        # n_initial control points = order + 1
        n_initial_control_points = np.prod(n+1)
        control_points = np.array([latent_base]*n_initial_control_points)
        
        latent_vec_interpolation = sp.BSpline(
            degrees=n,
            knot_vectors=knot_vectors,
            control_points=control_points,
        )
        latent_vec_interpolation.uniform_refine(self.options["refinement"])
        self.latent_vec_interpolation = latent_vec_interpolation

        self.logger = logging.getLogger(__name__)

    def get_latent_shape(self) -> int:
        return self.latent.shape[1]

    def get_n_control_points(self) -> int:
        return self.latent_vec_interpolation.control_points.shape[0]

    def generate_surface_mesh(self, control_points):
        """
        generates mesh from control points
        """

        cap_border_dict = self.options["cap_border_dict"]
        N_base = self.options["N_base_reconstruction"]
        tiling = self.options["tiling"]
        N = [N_base * t+1 for t in tiling]
        self.latent_vec_interpolation.control_points = control_points
        latent_vec_interpolation = self.latent_vec_interpolation
        decoder = self.decoder

        verts, faces, jac = deep_sdf.mesh.create_mesh_microstructure_diff(tiling, decoder, latent_vec_interpolation, cap_border_dict=cap_border_dict, N=N, device=device, compute_derivatives=True)

        jac = jac.reshape((jac.shape[0], jac.shape[1], -1))
        verts_np = verts.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()

        # "freeform deformation" of the mesh
        verts_np[:,0] = verts_np[:,0]*2
        jac[:,0,:] = jac[:,0,:]*2

        faces = []
        jac[np.where(jac>1)] = 0
        jac[np.where(jac<-1)] = 0
        # check watertightness of mesh
        tri_m = trimesh.Trimesh(verts_np, faces_np, vertex_attributes={"jac": jac})
        if self.options["remove_orphans"]:
            self.logger.debug("Removing orphan meshs")
            cc = trimesh.graph.connected_components(tri_m.face_adjacency)
            mask = np.zeros(len(tri_m.faces), dtype=np.bool_)
            n_faces = np.array([c.shape[0] for c in cc], dtype=np.int64)
            mask[cc[np.argmax(n_faces)]] = True
            tri_m.update_faces(mask)
        if not tri_m.is_watertight:
            self.logger.debug("Mesh is not watertight - trying to fix by eliminating degenerate faces")
            tri_m.update_faces(tri_m.nondegenerate_faces())
            if not tri_m.is_watertight:
                self.logger.warning("Mesh is still not watertight after eliminating degenerate faces")
            else:
                self.logger.debug("Successfully fixed mesh.")

        self.surface_mesh = tri_m
        # self.surface_mesh = tri_m
        self.jacobian = tri_m.vertex_attributes["jac"]

    def tetrahedralize_surface(self):
        self.logger.debug("Tetrahedralizing surface mesh")
        t_in = tetgenpy.TetgenIO()
        t_in.setup_plc(self.surface_mesh.vertices, self.surface_mesh.faces.tolist())
        # gus.show(dmesh)
        switch_command = "pYq"
        if logging.DEBUG <= logging.root.level:
            switch_command += "Q"
        t_out = tetgenpy.tetrahedralize(switch_command, t_in) #pqa

        tets = np.vstack(t_out.tetrahedra())
        verts = t_out.points()


        kdt = napf.KDT(tree_data=verts, metric=1)

        distances, face_indices = kdt.knn_search(
            queries=self.surface_mesh.vertices,
            kneighbors=1,
            nthread=4,
        )
        tol = 1e-6
        if distances.max() > tol:
            Warning("Not all surface nodes as included in the volumetric mesh.")
        self.volumes = gus.Volumes(verts, tets)
        self.surface_mesh_indices = face_indices

    def export_volume_mesh(self, filename: str, show_mesh=False, export_abaqus=False):
        """
        export a mesh and adds corresponding boundary conditions
        """
        volumes = self.volumes
        filepath = pathlib.Path(filename)
        faces = volumes.to_faces(False)
        boundary_faces = faces.single_faces()
        verts = volumes.vertices

        BC = {1: [], 2: [], 3: []}

        tolerance = 3e-2
        width = verts[:,0].max()
        height = verts[:,2].max()
        for i in boundary_faces:
            # mark boundaries at x = 0 with 1
            if np.max(verts[faces.const_faces[i], 0]) < tolerance:
                BC[1].append(i)
            # mark boundaries at x = width with 2
            elif np.max(verts[faces.const_faces[i], 2]) > (height - tolerance):
                BC[2].append(i)
            # mark rest of the boundaries with 3
            else:
                BC[3].append(i)
        volumes.BC = BC
        if show_mesh:
            gus.show(volumes)
        self.logger.debug(f"Exporting mesh with {len(volumes.volumes)} elements, {len(volumes.vertices)} vertices, {len(BC[1])} boundaries with marker 1, {len(BC[2])} boundaries with marker 2, and {len(BC[3])} boundaries with marker 3.")
        gus.io.mfem.export(str(filepath), volumes)
        if export_abaqus:
            gus.io.meshio.export(str(filepath.with_suffix(".inp")))

    def get_dTheta(self):
        volumes = self.volumes
        faces = self.surface_mesh
        jacobian = self.jacobian
        dVertices = None
        dVertices = np.zeros((volumes.vertices.shape[0], volumes.vertices.shape[1], jacobian.shape[2]))
        
        if np.any(np.isnan(jacobian)):
            logging.warning("Nan values in jacobian detected")
        normals = self.surface_mesh.vertex_normals
        # gus_faces = gus.Faces(faces.vertices, faces.faces)
        # normals = gus.create.faces.vertex_normals(gus_faces, angle_weighting=True, area_weighting=True)
        # # .vertex_data["normals"]    
        zero_normals = np.all(normals==0, axis=1)
        n_zero_normals = len(np.nonzero(zero_normals))
        if n_zero_normals > 0:
            self.logger.debug(f"{n_zero_normals} 0-Normal vectors detected")
        dVertices_normal = np.zeros_like(jacobian)
        for i in range(jacobian.shape[2]):
            dVertices_normal[:,:,i] = dot_prod(np.float64(jacobian[:,:,i]),normals)
            dVertices[self.surface_mesh_indices[:,0],:,i] = dVertices_normal[:,:,i]
        return dVertices
        


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

def dot_prod(A, B) -> np.ndarray:
    dot_ai_bi = (A * B).sum(axis=-1, keepdims=True)
    dot_bi_bi = (B * B).sum(axis=-1, keepdims=True)  # or square `norm`
    zero_normals = np.all(B==0, axis=1)
    n_zero_normals = len(np.nonzero(zero_normals))
    if n_zero_normals > 0:
        dot_bi_bi[zero_normals] = np.Inf
    C = dot_ai_bi / dot_bi_bi * B
    return C