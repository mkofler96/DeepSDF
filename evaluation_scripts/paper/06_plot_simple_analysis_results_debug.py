
import splinepy
import vedo
import numpy as np
from sdf_microstructure import (
    create_microstructure_from_experiment,
    export_mesh,
    tetrahedralize_surface,
)

from analysis.problems.CantileverBeam import CantileverBeam as le

resolution=10


rect = splinepy.helpme.create.box(2, 1, 1)
volumes = rect.extract.volumes(resolution=resolution)
rect_dist = splinepy.helpme.create.box(2, 1, 1)
print(rect_dist.control_points)
rect_dist.control_points[1][2] = 0.5
rect_dist.control_points[3][2] = 0.5
volumes = rect.extract.volumes(resolution=resolution)
volumes_dist = rect_dist.extract.volumes(resolution=resolution)


export_mesh(volumes, "test_mesh.mesh", show_mesh=False)
export_mesh(volumes_dist, "test_mesh_dist.mesh", show_mesh=False)


# dVertices of stretched - base
dVertices = volumes_dist.vertices - volumes.vertices
dVertices = np.dstack([dVertices])


le_problem = le("simulation/temp_sim")
le_problem.read_mesh("test_mesh.mesh")
# before solve, we should add a problem setup and set material properties
le_problem.set_up()
vol, der_vol = le_problem.compute_volume(dTheta=dVertices)
print(f"Vol: {vol:.5g}, dVol: {der_vol}")
le_problem.solve()
compliance, der_compliance = le_problem.compute_compliance(dTheta=dVertices)
print(f"Compliance: {compliance:.5g}, dCompliance: {der_compliance}")


le_problem = le("simulation/temp_sim")
le_problem.read_mesh("test_mesh_dist.mesh")
# before solve, we should add a problem setup and set material properties
le_problem.set_up()
vol, der_vol = le_problem.compute_volume(dTheta=dVertices)
print(f"Vol: {vol:.5g}, dVol: {der_vol}")
le_problem.solve()
compliance, der_compliance = le_problem.compute_compliance(dTheta=dVertices)
print(f"Compliance: {compliance:.5g}, dCompliance: {der_compliance}")
le_problem.show_solution(output=["u_vec","strain_energy_density"])


# le_problem = le.LinearElasticityProblem()
# le_problem.read_mesh("test_mesh_stretched.mesh")
# # before solve, we should add a problem setup and set material properties
# le_problem.set_up(ref_levels=0)
# vol, _ = le_problem.compute_volume()
# print(f"Volume of deformed mesh {vol:.5g}")
# le_problem.solve()
# compliance, der_compliance = le_problem.compute_compliance()
# print(f"Compliance of deformed mesh: {compliance:.5g}")
# le_problem.show_solution(output=["u_vec","strain_energy_density"])


