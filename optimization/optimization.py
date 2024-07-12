import sys
sys.path.append(".")
import deep_sdf.mesh
import deep_sdf.utils
import os
import json
import torch
import deep_sdf.workspace as ws
import pathlib
import time
import datetime
import splinepy as sp
import gustaf as gus
import numpy as np
import skimage
import meshio
import tetgenpy
import igl
import mimi
import gustaf as gus
import trimesh

import subprocess

import pathlib
import os
from typing import Union
import json

import scipy
import numpy as np
import splinepy as sp
import os
import gustaf as gus
import config
import socket
import os
import vedo
import meshio

from typing import TypedDict

import numpy as np
import splinepy as sp
import os
import shutil
import re

import pandas as pd
from dataclasses import dataclass
import dataclasses

import logging

@dataclass
class OptimizationResults:
    compliance: list[float]
    volume: list[float]

    def append_result(self, design_vector, result):
        self.volume.append(result[0])
        self.compliance.append(result[1])



class struct_optimization():
    optimization_folder: pathlib.Path
    optimization_results = OptimizationResults([], [])
    iteration = 0

    design_vectors = []

    @property
    def settings_filename(self):
        return self.optimization_folder / f"config.json"
    
    @property
    def current_simulation_folder(self) -> pathlib.Path:
        sim_f = self.optimization_folder / f"simulation_{self.iteration}"
        if not os.path.exists(sim_f):
            os.makedirs(sim_f)
        return sim_f

    @property
    def log_filename(self):
        return self.optimization_folder/"optimization_logs.log"

    def __init__(self, optimization_folder: Union[str, bytes, os.PathLike], experiment_directory, checkpoint):
        self.experiment_directory = experiment_directory
        self.checkpoint = checkpoint
        self.optimization_folder = pathlib.Path(optimization_folder)
        self.optimization_results = OptimizationResults([], [])
        if self.settings_filename.exists():
            self.load_settings()
        else:
            raise FileNotFoundError(f"No config.json in {self.optimization_folder}")

        self.cache = {}
        self.logging = logging.getLogger("optimization")
        self.logging.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.log_filename, mode='w')
        self.logging.addHandler(fh)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logging.log(logging.INFO, f"Starting optimization in {self.optimization_folder}")

    def _in_cache(self, x):
        # search for x as key in self.cache
        return str(x) in self.cache

    def objective(self, x):
        if not self._in_cache(x):
             self._compute_solution(x) # this fills the cache with objective and constraint values for x
        return self.cache[str(x)]["objective"]

    def constraint(self, x):
        if not self._in_cache(x):
             self._compute_solution(x) # same idea
        return self.cache[str(x)]["constraint"]

    def set_x0(self, x0):
        self.start_values = x0
        self.dv_names = [f"x{i}" for i in range(len(x0))]
        self.bounds = [(-1,1)]*len(x0)

    def load_settings(self):
        self.options = config.Config.load_json(self.settings_filename)

        # with open(self.settings_filename, 'r') as file:
        #     self.options = json.load(file)
        option_keys = ["mesh", "optimization"]
        for key in option_keys:
            if key not in self.options:
                raise KeyError(f"Key {key} not found in config.json")
        available_optimizer_methods = ["BFGS", "COBYLA"]
        method = self.options["optimization"]["method"]
        if not (method in available_optimizer_methods):
            raise ValueError(f"Optimizer {method} method not available. Available methods are {available_optimizer_methods}")
        

    def run_optimization(self):
        scipy_optimizers = ["BFGS", "COBYLA"]
        if self.options["optimization"]["method"] == "MOOP":
            self.run_PSO_optimization()
        elif self.options["optimization"]["method"] in scipy_optimizers:
            self.run_scipy_optimization(options=self.options["optimization"])
        elif self.options["optimization"]["method"] == "NSGA":
            self.run_NSGA_optimization()
        else:
            raise ValueError("Optimizer method not available")
        
        with open(self.optimization_folder/"results.json", "w") as f:
            writer = json.dump(dataclasses.asdict(self.optimization_results), f)



    def _compute_solution(self, control_point_values):  
        self.iteration += 1
        self.logging.log(logging.DEBUG, f"Design vector difference to start: \n {control_point_values-self.start_values}")
        accuracy_deep_sdf_reconstruction = 50
        number_of_final_elements = 1e5

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        decoder = ws.load_trained_model(self.experiment_directory, self.checkpoint)
        decoder.eval()

        box = sp.helpme.create.box(10,10,10).bspline
        small_box = sp.helpme.create.box(1,1,1).bspline
        box.insert_knots(0, [0.5])
        box.insert_knots(1, [0.5])
        box.insert_knots(2, [0.5])
        #todo replace hard coded 18 and 16, 9 = number of control points, 16 = latent vector dimension
        # in total we have 18 control points, 9 in front and 9 in back, but z direction is constant
        control_points = np.array(control_point_values).reshape((-1, decoder.lin0.in_features-3))

        latent_vec_interpolation = sp.BSpline(
            degrees=[1, 1, 1],
            knot_vectors=[[-1, -1, 0, 1, 1], 
                        [-1, -1, 1, 1], 
                        [-1, -1, 1, 1]],
            control_points=np.vstack([control_points, control_points]),
        )

        microstructure = sp.Microstructure()
        # set outer spline and a (micro) tile
        microstructure.deformation_function = box
        microstructure.microtile = small_box
        # tiling determines tile resolutions within each bezier patch
        microstructure.tiling = [2, 2, 2]
        ms = microstructure.create().patches

        tiling = self.options["mesh"]["tiling"]
        N_base = accuracy_deep_sdf_reconstruction
        N = [N_base * t+1 for t in tiling]

        cap_border_dict = {
            "x0": {"cap": 1, "measure": 0.1},
            "x1": {"cap": 1, "measure": 0.1},
            "y0": {"cap": 1, "measure": 0.1},
            "y1": {"cap": 1, "measure": 0.1},
        }
        self.logging.log(logging.INFO, f"Start Querying {np.prod(N)} DeepSDF points")

        verts, faces = deep_sdf.mesh.create_mesh_microstructure(tiling, decoder, latent_vec_interpolation, "none", cap_border_dict=cap_border_dict, N=N, use_flexicubes=self.options["mesh"]["use_flexicubes"])
        

        self.logging.log(logging.INFO, f"Finished Querying DeepSDF with {len(verts)} vertices and {len(faces)} faces")
        # Free Form Deformation
        # geometric parameters
        width = 5
        height = 2
        depth = 1

        control_points=np.array([
                [0, 0, 0],
                [0, height, 0],
                [width, 0, 0],
                [width, height, 0]
            ])

        deformation_surf = sp.BSpline(
            degrees=[1,1],
            control_points=control_points,
            knot_vectors=[[0, 0, 1, 1],[0, 0, 1, 1]],
        )

        deformation_volume = deformation_surf.create.extruded(extrusion_vector=[0,0,depth])

        # bring slightly outside vertices back 
        verts[verts>1] = 1
        verts[verts<0] = 0


        if self.options["mesh"]["use_flexicubes"]:
            faces = faces.cpu().numpy()
            verts = verts.cpu().numpy()
        self.logging.log(logging.INFO, f"Applying Free Form Deformation to {len(verts)} vertices")
        verts_FFD_transformed = deformation_volume.evaluate(verts)

        surf_mesh = gus.faces.Faces(verts_FFD_transformed, faces)
        fname_surf = self.current_simulation_folder/f"surf{self.iteration}.inp"
        self.logging.log(logging.INFO, f"Writing surface mesh to {fname_surf}")
        gus.io.meshio.export(fname_surf, surf_mesh)
        # gus.show(surf_mesh)
        if self.options["mesh"]["decimate_mesh"]:
            self.logging.log(logging.INFO, f"Decimating surface mesh to {number_of_final_elements} elements")
            r = igl.decimate(surf_mesh.vertices, surf_mesh.faces, int(number_of_final_elements))
            dmesh = gus.Faces(r[1], r[2])
            # gus.show(dmesh)
        else:
            dmesh = surf_mesh

        self.logging.log(logging.INFO, f"Tetrahedralizing decimated surface mesh with TetGen")
        t_in = tetgenpy.TetgenIO()
        t_in.setup_plc(dmesh.vertices, dmesh.faces.tolist())
        t_out = tetgenpy.tetrahedralize("pqa", t_in)

        tets = np.vstack(t_out.tetrahedra())
        verts = t_out.points()

        mesh = gus.Volumes(verts, tets)
        # gus.show(mesh, interactive=False)
        faces = mesh.to_faces(False)
        boundary_faces = faces.single_faces()

        BC = {1: [], 2: [], 3: []} 
        for i in boundary_faces:
            # mark boundaries at x = 0 with 1
            if np.max(verts[faces.const_faces[i], 0]) < 3e-2:
                BC[1].append(i)
            # mark boundaries at x = 1 with 2
            elif np.min(verts[faces.const_faces[i], 0]) > 4.999:
                BC[2].append(i)
            # mark rest of the boundaries with 3
            else:
                BC[3].append(i)
        fname_volume = self.current_simulation_folder/f"volume{self.iteration}.inp"
        gus.io.mfem.export(fname_volume.with_suffix(".mesh"), mesh, BC)
        gus.io.meshio.export(fname_volume, mesh)
        output_dir = self.optimization_folder
        simulation_name = self.current_simulation_folder
        self.logging.log(logging.INFO, f"Running simulation with mesh {fname_volume}")
        cl_beam = mimi.LECantileverBeam(str(fname_volume.with_suffix('.mesh')), str(output_dir), str(simulation_name))
        cl_beam.solve()
        compliance = cl_beam.compliance

        use_trimesh_for_volume_calculation = True

        if use_trimesh_for_volume_calculation:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces.const_faces)
            volume = mesh.volume
            self.logging.log(logging.INFO, f"Volume calculated with trimesh: {volume} | with MFEM: {cl_beam.volume}")
        else:
            volume = cl_beam.volume
        use_old_mfem = False
        if use_old_mfem:
            solver_path = "/usr2/mkofler/MFEM/mfem-4.7/examples/ex2"
            simulation_command = f"{solver_path} -m {fname_volume.with_suffix('.mesh')} -o {output_dir} -sn {simulation_name}"
            result = subprocess.run(simulation_command, shell=True, 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)
            # write output to file
            with open(self.current_simulation_folder/"mfem_output.log", "w") as f:
                f.write(result.stdout)

            # regex pattern to get volume and compliance
            pattern = r"Volume:\s*([\d\.]+)\s*Compliance:\s*([\d\.]+)"

            # Search for the pattern in the text
            match = re.search(pattern, result.stdout)
            if match:
                volume = float(match.group(1))
                compliance = float(match.group(2))
            else:
                raise ValueError("No volume or compliance found in output")

        self.cache[str(control_point_values)] = {"objective": compliance, "constraint": 6-volume}  # f, g are scalars
        self.logging.log(logging.INFO, f"Finished iteration {self.iteration} with compliance {compliance} and volume {volume}")

    def run_scipy_optimization(self, options):
        # do all your calculations only once here
        # in the end fill the cache
        
        def obj_fun(x):
            res = self.objective(x)
            return res

        def constraint(x):
            res = self.constraint(x)
            return res
        # Equality constraint means that the constraint function result is to 
        # be zero whereas inequality means that it is to be non-negative. 
        # Note that COBYLA only supports inequality constraints.
        opti_options_without_method = options.copy()
        opti_options_without_method.pop("method")
        cons = {'type': 'ineq', 'fun': constraint},
        result = scipy.optimize.minimize(obj_fun, self.start_values, 
                                bounds=self.bounds,
                                method=options["method"],
                                constraints=cons,
                                options=opti_options_without_method)
        return result


    def load_results(self, as_np_array=False):
        with open(self.optimization_folder/"results.json", "r") as f:
            self.optimization_results = json.load(f)
        if as_np_array:
            return np.array([self.optimization_results["moment"],
                             self.optimization_results["force"],
                             self.optimization_results["objective"]]).T

def create_default_simulation(simulation_path: Union[str, bytes, os.PathLike]):
    sim_path = pathlib.Path(simulation_path)
    if not os.path.exists(sim_path):
        os.makedirs(sim_path)
    
    example_config_file = pathlib.Path(__file__).parent/"example_config.json"
    shutil.copyfile(example_config_file, sim_path/"config.json")

def copy_all_simulations(source_dir, destination_dir):
    for folder in os.listdir(source_dir):
        if not os.path.isdir(destination_dir/folder):
            os.makedirs(destination_dir/folder)
        config_file = source_dir/folder/"config.json"
        parMOO_file = source_dir/folder/"PasMOO_results.csv"
        results_file = source_dir/folder/"results.json"
        files_to_copy = [config_file, parMOO_file, results_file]

        for file in files_to_copy:
            if os.path.isfile(file):
                shutil.copy(file, destination_dir/folder)
                print(f"Copying {file} to {destination_dir/folder/file.name}")
    # for pathlib.Path(source_dir).glob("*/"):
    #     shutil.copytree(source_dir, destination_dir)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_directory = "./experiments/snappy3D_latent_2D"
    checkpoint = "1000"

    latent = ws.load_latent_vectors(experiment_directory, checkpoint).to("cpu").numpy()

    lat_vec1 = latent[1]
    lat_vec2 = latent[15]
    lat_vec3 = latent[39]
    control_points = [lat_vec2]*6
    index = sp.helpme.multi_index.MultiIndex((3,3))
    # # center thicker
    # control_points[index[1,1][0]] = lat_vec3
    # # sides smaller (I don't know why it is not [0,1,0] instead of [1,0,0])
    # control_points[index[1,0][0]] = lat_vec1
    # control_points[index[1,2][0]] = lat_vec1
    control_points = np.array(control_points)
    x0 = control_points.reshape(-1)
    optimization = struct_optimization("simulations/optimization_mimi", experiment_directory, checkpoint)
    optimization.set_x0(x0)
    optimization.run_optimization()