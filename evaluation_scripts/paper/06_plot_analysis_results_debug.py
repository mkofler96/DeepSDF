import gustaf as gus
from deep_sdf.analysis import linear_elasticity as le 

from sdf_microstructure import create_microstructure_from_experiment, tetrahedralize_surface, export_mesh

test_mesh = True

if test_mesh:
    volumes = gus.Volumes(
        vertices=[
            [0.0, 0.0, 0.0], #0
            [1.0, 0.0, 0.0], #1
            [0.0, 1.0, 0.0], #2
            [1.0, 1.0, 0.0], #3
            [0.0, 0.0, 1.0], #4
            [1.0, 0.0, 1.0], #5
            [0.0, 1.0, 1.0], #6
            [1.0, 1.0, 1.0], #7
            [2.0, 0.0, 0.0], #8
            [2.0, 1.0, 0.0], #9
            [2.0, 0.0, 1.0], #10
            [2.0, 1.0, 1.0], #11
        ],
        volumes=[
            [0, 1, 3, 2, 4, 5, 7, 6],
            [1, 8, 9, 3, 5, 10, 11, 7]
        ],
    )
else:
    faces, jacobian = create_microstructure_from_experiment("/home/michael.kofler/DeepSDF/experiments/round_cross_big_network", tiling=[2,1,1])

    volumes = tetrahedralize_surface(faces)


export_mesh(volumes, "test_mesh.mesh")
le_problem = le.LinearElasticityProblem()
le_problem.read_mesh("test_mesh.mesh")


le_problem.get_volume_and_compliance()