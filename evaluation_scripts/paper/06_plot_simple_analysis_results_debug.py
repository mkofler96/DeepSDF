import gustaf as gus
import splinepy
from sdf_microstructure import (
    create_microstructure_from_experiment,
    export_mesh,
    tetrahedralize_surface,
)

from deep_sdf.analysis import linear_elasticity as le

test_mesh = True

if test_mesh:
    rect = splinepy.helpme.create.box(2, 1, 1)
    volumes = rect.extract.volumes(resolution=11)
else:
    faces, jacobian = create_microstructure_from_experiment("/home/michael.kofler/DeepSDF/experiments/round_cross_big_network", tiling=[4,2,2], N_base=20)
    volumes = tetrahedralize_surface(faces)
    pass

# base volume
export_mesh(volumes, "test_mesh.mesh")
print(len(volumes.vertices))
# stretched volume in y
volumes_stretched = volumes.copy()
volumes_stretched.vertices[:,1] = volumes.vertices[:,1]*3
export_mesh(volumes_stretched, "test_mesh_stretched.mesh")

# dVertices of stretched - base
dVertices = volumes_stretched.vertices - volumes.vertices

le_problem = le.LinearElasticityProblem()
le_problem.read_mesh("test_mesh.mesh")
# before solve, we should add a problem setup and set material properties
le_problem.set_up()
vol, der_vol = le_problem.compute_volume(dTheta=dVertices)
print(f"Vol: {vol:.5g}, dVol: {der_vol:.5g}")
le_problem.solve()
compliance, der_compliance = le_problem.compute_compliance(dTheta=dVertices)
print(f"Compliance: {compliance:.5g}, dCompliance: {der_compliance:.5g}")
# print(le_problem.compute_compliance())
# le_problem.compute_shape_derivative()


le_problem = le.LinearElasticityProblem()
le_problem.read_mesh("test_mesh_stretched.mesh")
# before solve, we should add a problem setup and set material properties
le_problem.set_up(ref_levels=1)
vol, _ = le_problem.compute_volume()
print(f"Volume of deformed mesh {vol:.5g}")
le_problem.solve()
compliance, der_compliance = le_problem.compute_compliance()
print(f"Compliance of deformed mesh: {compliance:.5g}")