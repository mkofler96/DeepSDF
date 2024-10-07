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
import tempfile
import matplotlib.pyplot as plt
import subprocess
import mmapy

import pathlib
import os
from typing import Union
import json

import scipy
import numpy as np
import splinepy as sp
import os
import gustaf as gus
from . import config
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
    design_vector: list[np.ndarray]

    def append_result(self, design_vector, volume, compliance):
        self.volume.append(volume)
        self.compliance.append(compliance)
        self.design_vector.append(design_vector.tolist())



class struct_optimization():
    optimization_folder: pathlib.Path
    optimization_results = OptimizationResults([], [], [])
    iteration = 0

    design_vectors = []

    @property
    def settings_filename(self):
        return self.optimization_folder / f"config.json"
    
    @property
    def current_simulation_folder(self) -> pathlib.Path:
        sim_f = self.optimization_folder / f"simulation_{self.iteration}"

        return sim_f

    def create_temp_current_simulation_folder(self) -> pathlib.Path:
        temp_dir = pathlib.Path(self.options["general"]["temp_dir"])
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        dirpath =pathlib.Path(tempfile.mkdtemp(dir=temp_dir))/f"simulation_{self.iteration}"
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        return dirpath
    
    def move_older_sims_to_temp_dir(self):
        old_sim_dir = self.optimization_folder/"old_sims"
        i_old_sim = 0
        while os.path.exists(old_sim_dir):
            i_old_sim += 1
            old_sim_dir = self.optimization_folder/f"old_sims_{i_old_sim}"
        if any (["simulation" in folder for folder in os.listdir(self.optimization_folder)]):
            os.makedirs(old_sim_dir)
        for folder in os.listdir(self.optimization_folder):
            if "simulation" in folder:
                shutil.move(self.optimization_folder/folder, old_sim_dir/folder)
                self.logging.log(logging.INFO, "Older simulation files detected.")
                self.logging.log(logging.INFO, f"Moving {folder} to {old_sim_dir}")

    @property
    def log_filename(self):
        return self.optimization_folder/"optimization_logs.log"

    def __init__(self, optimization_folder: Union[str, bytes, os.PathLike]):

        self.optimization_folder = pathlib.Path(optimization_folder)
        self.optimization_results = OptimizationResults([], [], [])
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
        self.move_older_sims_to_temp_dir()

    def _in_cache(self, x):
        # search for x as key in self.cache
        return str(x) in self.cache

    def objective(self, x):
        if not self._in_cache(x):
             self._compute_solution(x) # this fills the cache with objective and constraint values for x
        return self.cache[str(x.round(8))]["objective"]

    def constraint(self, x):
        if not self._in_cache(x):
             self._compute_solution(x) # same idea
        return self.cache[str(x.round(8))]["constraint"]

    def set_x0(self, x0):
            
        control_points = np.array([x0]*6)

        x0 = control_points.reshape(-1)
        self.start_values = x0
        self.dv_names = [f"x{i}" for i in range(len(x0))]
        self.bounds = [(-1,1)]*len(x0)

    def load_settings(self):
        self.options = config.Config.load_json(self.settings_filename)

        # with open(self.settings_filename, 'r') as file:
        #     self.options = json.load(file)
        option_keys = ["mesh", "optimization", "general"]
        for key in option_keys:
            if key not in self.options:
                raise KeyError(f"Key {key} not found in config.json")
        available_optimizer_methods = ["BFGS", "COBYLA", "MMA"]
        method = self.options["optimization"]["method"]
        if not (method in available_optimizer_methods):
            raise ValueError(f"Optimizer {method} method not available. Available methods are {available_optimizer_methods}")
        
        if not ("experiment_directory" in self.options["general"]):
            raise KeyError("Key experiment_directory not found in general settings")
        if not ("checkpoint" in self.options["general"]):
            raise KeyError("Key checkpoint not found in general settings")

        if not os.path.exists(self.options["general"]["experiment_directory"]):
            raise FileNotFoundError(f"Experiment directory {self.options['general']['experiment_directory']} not found")
        
        experiment_directory = self.options["general"]["experiment_directory"]
        checkpoint = str(self.options["general"]["checkpoint"])
        self.experiment_directory = experiment_directory
        self.checkpoint = checkpoint
        

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
        temp_current_simulation_folder = self.create_temp_current_simulation_folder()
        self.logging.log(logging.DEBUG, f"Design vector difference to start: \n {control_point_values-self.start_values}")
        accuracy_deep_sdf_reconstruction = self.options["mesh"]["accuracy_deep_sdf_reconstruction"]
        number_of_final_elements = self.options["mesh"]["number_of_final_elements"]

        decoder = ws.load_trained_model(self.experiment_directory, self.checkpoint)
        decoder.eval()

        # box = sp.helpme.create.box(10,10,10).bspline
        # small_box = sp.helpme.create.box(1,1,1).bspline
        # box.insert_knots(0, [0.5])
        # box.insert_knots(1, [0.5])
        # box.insert_knots(2, [0.5])
        #todo replace hard coded 18 and 16, 9 = number of control points, 16 = latent vector dimension
        # in total we have 18 control points, 9 in front and 9 in back, but z direction is constant
        control_points = np.array(control_point_values).reshape((-1, decoder.lin0.in_features-3))

        latent_vec_interpolation = sp.BSpline(
            degrees=[1, 2, 1],
            knot_vectors=[[-1, -1, 1, 1], 
                        [-1, -1, -1, 1, 1, 1], 
                        [-1, -1, 1, 1]],
            control_points=np.vstack([control_points, control_points]),
        )

        # microstructure = sp.Microstructure()
        # # set outer spline and a (micro) tile
        # microstructure.deformation_function = box
        # microstructure.microtile = small_box
        # # tiling determines tile resolutions within each bezier patch
        # microstructure.tiling = [2, 2, 2]
        # ms = microstructure.create().patches

        tiling = self.options["mesh"]["tiling"]
        N_base = accuracy_deep_sdf_reconstruction
        N = [N_base * t+1 for t in tiling]

        cap_border_dict = {
            "x0": {"cap": 1, "measure": 0.05},
            # "x1": {"cap": 1, "measure": 0.1},
            # "y0": {"cap": 1, "measure": 0.1},
            "y1": {"cap": 1, "measure": 0.1},
        }
        self.logging.log(logging.INFO, f"Start Querying {np.prod(N)} DeepSDF points")

        if self.options["mesh"]["tetmesh_from_flexicubes"]:
            verts, volumes = deep_sdf.mesh.create_mesh_microstructure(tiling, decoder, latent_vec_interpolation, "none", cap_border_dict=cap_border_dict, N=N, use_flexicubes=self.options["mesh"]["use_flexicubes"], output_tetmesh=True)
            mesh = gus.Volumes(verts.cpu().numpy(), volumes.cpu().numpy())
            gus.show(mesh)
        else:
            verts, faces = deep_sdf.mesh.create_mesh_microstructure(tiling, decoder, latent_vec_interpolation, "none", cap_border_dict=cap_border_dict, N=N, use_flexicubes=self.options["mesh"]["use_flexicubes"])
            
            # if flexicubes is used, mesh is exported as torch tensor, therefore we need to convert it to numpy
            if self.options["mesh"]["use_flexicubes"]:
                faces = faces.cpu().numpy()
                verts = verts.cpu().numpy()

            self.logging.log(logging.INFO, f"Finished Querying DeepSDF with {len(verts)} vertices and {len(faces)} faces")

            # step 1: apply Free Form Deformation
            # bring slightly outside vertices back
            n_verts_outside = np.sum(verts > 1) + np.sum(verts < 0)
            self.logging.log(logging.INFO, f"Maximum deviation is {np.max(verts)}")
            self.logging.log(logging.INFO, f"Minimum deviation is {np.min(verts)}")
            self.logging.log(logging.INFO, f"Moving {n_verts_outside} vertices to [0,1]")
            verts[verts>1] = 1
            verts[verts<0] = 0
            
            # Free Form Deformation
            # geometric parameters
            width, height, depth = self.options["mesh"]["macro_dimensions"]

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
            self.logging.log(logging.INFO, f"Applying Free Form Deformation to {len(verts)} vertices")
            verts = deformation_volume.evaluate(verts)

            # step 2: generate surface mesh
            surf_mesh = gus.faces.Faces(verts, faces)
            # step 3: decimate the surface mesh
            
            if self.options["mesh"]["decimate_mesh"]:
                self.logging.log(logging.INFO, f"Decimating surface mesh to {number_of_final_elements} elements")
                r = igl.decimate(surf_mesh.vertices, surf_mesh.faces, int(number_of_final_elements))
                dmesh = gus.Faces(r[1], r[2])
                # gus.show(dmesh)
            else:
                dmesh = surf_mesh

            fname_surf = temp_current_simulation_folder/f"surf{self.iteration}.inp"
            self.logging.log(logging.INFO, f"Writing surface mesh to {fname_surf}")
            gus.io.meshio.export(fname_surf, dmesh)

            self.logging.log(logging.INFO, f"Tetrahedralizing decimated surface mesh with TetGen")
            use_pygalmesh = False
            if use_pygalmesh:
                import pygalmesh
                mesh_pg = pygalmesh.generate_volume_mesh_from_surface_mesh(
                    fname_surf,

                )
                mesh = gus.Volumes(mesh_pg.points, mesh_pg.cells[1].data)
            else:
                t_in = tetgenpy.TetgenIO()
                t_in.setup_plc(dmesh.vertices, dmesh.faces.tolist())
                # gus.show(dmesh)
                t_out = tetgenpy.tetrahedralize("p", t_in) #pqa

                tets = np.vstack(t_out.tetrahedra())
                verts = t_out.points()

                mesh = gus.Volumes(verts, tets)

        faces = mesh.to_faces(False)
        boundary_faces = faces.single_faces()

        BC = {1: [], 2: [], 3: []} 
        for i in boundary_faces:
            # mark boundaries at x = 0 with 1
            if np.max(verts[faces.const_faces[i], 0]) < 3e-2*height:
                BC[1].append(i)
            # mark boundaries at x = 1 with 2
            elif np.logical_and(np.min(verts[faces.const_faces[i], 0]) > 0.9*width,
                                np.min(verts[faces.const_faces[i], 1]) > 0.999*height):
                BC[2].append(i)
            # mark rest of the boundaries with 3
            else:
                BC[3].append(i)
        fname_volume = temp_current_simulation_folder/f"volume{self.iteration}.inp"
        mesh.BC = BC
        gus.io.mfem.export(fname_volume.with_suffix(".mesh"), mesh)
        gus.io.meshio.export(fname_volume, mesh)
        simulation_name = temp_current_simulation_folder
        output_dir = temp_current_simulation_folder.parent
        self.logging.log(logging.INFO, f"Running simulation with mesh {fname_volume}")
        use_direct_solver = False
        cl_beam = mimi.LECantileverBeam(str(fname_volume.with_suffix('.mesh')), str(output_dir), str(simulation_name),use_direct_solver)
        cl_beam.solve()
        compliance = cl_beam.compliance

        use_trimesh_for_volume_calculation = True

        if use_trimesh_for_volume_calculation:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces.const_faces)
            volume = mesh.volume
            self.logging.log(logging.INFO, f"Volume calculated with trimesh: {volume} | with MFEM: {cl_beam.volume}")
        else:
            volume = cl_beam.volume
        vol_constraint = self.options["general"]["volume_constraint"]
        self.cache[str(control_point_values.round(8))] = {"objective": compliance, "constraint": vol_constraint-volume}  # f, g are scalars
        self.logging.log(logging.INFO, f"Finished iteration {self.iteration} with compliance {compliance} and volume {volume}")
        self.optimization_results.append_result(control_point_values, volume, compliance)
        self.save_and_clear(temp_current_simulation_folder)

    def save_and_clear(self, temp_current_simulation_folder):
        with open(self.optimization_folder/"results.json", "w") as f:
            json.dump(dataclasses.asdict(self.optimization_results), f)
        save_every = self.iteration % self.options["general"]["save_every"] == 0
        first_iteration = self.iteration == 1
        if save_every or first_iteration:
            self.logging.log(logging.INFO, f"Saving simulation results to {self.current_simulation_folder}")
            shutil.copytree(temp_current_simulation_folder, self.current_simulation_folder)
            
        shutil.rmtree(temp_current_simulation_folder)
        shutil.rmtree(temp_current_simulation_folder.parent)


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