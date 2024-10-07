
import splinepy
import gustaf as gus
import numpy as np
import vedo
from sdf_microstructure import (
    create_microstructure_from_experiment,
    export_mesh,
    tetrahedralize_surface,
)

from deep_sdf.analysis import linear_elasticity as le

vedo.settings.default_backend = "k3d"


faces, jacobian = create_microstructure_from_experiment("/home/michael.kofler/DeepSDF/experiments/round_cross_big_network", tiling=[4,2,2], N_base=20)


volumes, surface_indices = tetrahedralize_surface(faces)


export_mesh(volumes, "test_mesh.mesh", show_mesh=False)
le_problem = le.LinearElasticityProblem()
le_problem.read_mesh("test_mesh.mesh")


def dot_prod(A, B) -> np.ndarray:
    dot_ai_bi = (A * B).sum(axis=-1, keepdims=True)
    dot_bi_bi = (B * B).sum(axis=-1, keepdims=True)  # or square `norm`
    C = dot_ai_bi / dot_bi_bi * B
    return C


le_problem = le.LinearElasticityProblem()
le_problem.read_mesh("test_mesh.mesh")
# before solve, we should add a problem setup and set material properties
le_problem.set_up()
dVertices = None
dVertices = np.zeros((volumes.vertices.shape[0], volumes.vertices.shape[1], jacobian.shape[2]))
normals = gus.create.faces.vertex_normals(faces, angle_weighting=True, area_weighting=True)
dVertices_normal = np.zeros_like(jacobian)
for i in range(jacobian.shape[2]):
    dVertices_normal[:,:,i] = dot_prod(np.float64(jacobian[:,:,i]),normals.vertex_data["normals"])
    dVertices[surface_indices[:,0],:,i] = dVertices_normal[:,:,i]


vol, der_vol = le_problem.compute_volume(dTheta=dVertices)
if der_vol is None:
    der_vol = 0
print(f"Vol: {vol:.5g}, dVol: {der_vol}")
le_problem.solve()
compliance, der_compliance = le_problem.compute_compliance(dTheta=dVertices)
if der_compliance is None:
    der_compliance = 0
print(f"Compliance: {compliance:.5g}, dCompliance: {der_compliance}")
expected_vol = vol+der_vol
expected_compl = compliance + der_compliance


le_problem = le.LinearElasticityProblem()
volumes_stretched = volumes.copy()
volumes_stretched.vertices = volumes.vertices + dVertices
export_mesh(volumes_stretched, "test_mesh_stretched.mesh")
le_problem.read_mesh("test_mesh_stretched.mesh")
# before solve, we should add a problem setup and set material properties
le_problem.set_up(ref_levels=0)
vol, _ = le_problem.compute_volume()
print(f"Volume of deformed mesh {vol:.5g} ({expected_vol:.5g} expected)")
le_problem.solve()
compliance, der_compliance = le_problem.compute_compliance()
print(f"Compliance of deformed mesh: {compliance:.5g} ({expected_compl:.5g} expected)")


