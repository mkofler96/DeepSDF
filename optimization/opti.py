import os
import json
import pathlib
import numpy as np

import tempfile

import pathlib
from typing import Union

import scipy
import numpy as np
import gustaf as gus
import socket

import shutil

from dataclasses import dataclass
import dataclasses

from analysis.geometry import DeepSDFMesh
from optimization import config
from optimization import MMA
from analysis.problems import CantileverBeam

import logging
import logging.handlers
from logging.config import dictConfig
logger = logging.getLogger(__name__)
logging.getLogger("gustaf").setLevel(logging.WARNING)


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
                self.logger.info("Older simulation files detected.")
                self.logger.info(f"Moving {folder} to {old_sim_dir}")

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
        self.logger = logging.getLogger(__name__)

        self.geometry = DeepSDFMesh(self.options["mesh"])

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

    def set_x0(self, x0=None):
        if x0 is None:
            n_control_points = self.geometry.get_n_control_points()
            n_latent = self.geometry.get_latent_shape()
            control_points = np.zeros((n_control_points, n_latent))
        else:
            control_points = x0

        self.start_values = control_points.reshape(-1)
        self.dv_names = [f"x{i}" for i in range(len(self.start_values))]
        self.bounds = [(-1,1)]*len(self.start_values)

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
    
        

    def run_optimization(self):
        self.logger.info(f"Starting optimization in {self.optimization_folder} on {socket.gethostname()}")
        self.move_older_sims_to_temp_dir()
        scipy_optimizers = ["BFGS", "COBYLA"]
        if self.options["optimization"]["method"] == "MOOP":
            self.run_PSO_optimization()
        elif self.options["optimization"]["method"] in scipy_optimizers:
            self.run_scipy_optimization(options=self.options["optimization"])
        elif self.options["optimization"]["method"] == "NSGA":
            self.run_NSGA_optimization()
        elif self.options["optimization"]["method"] == "MMA":
            self.run_MMA_optimization(self.options["optimization"])
        else:
            raise ValueError("Optimizer method not available")
        
        with open(self.optimization_folder/"results.json", "w") as f:
            writer = json.dump(dataclasses.asdict(self.optimization_results), f)



    def _compute_solution(self, control_point_values):  
        self.logger.debug(f"Computing Solution")
        self.iteration += 1
        temp_current_simulation_folder = self.create_temp_current_simulation_folder()
        latent_shape = self.geometry.get_latent_shape()
        control_points = np.array(control_point_values).reshape((-1, latent_shape))
        self.logger.debug(f"Generating Geometry")
        self.geometry.generate_surface_mesh(control_points)
        fname_surf = temp_current_simulation_folder/f"surf{self.iteration}.inp"
        self.logger.debug(f"Writing surface mesh to {fname_surf}")
        surface_mesh = gus.Faces(self.geometry.surface_mesh.vertices,
                                self.geometry.surface_mesh.faces)
        gus.io.meshio.export(fname_surf, surface_mesh)

        self.geometry.tetrahedralize_surface()
        fname_volume_abq = temp_current_simulation_folder/f"volume{self.iteration}.inp"
        gus.io.meshio.export(fname_volume_abq, self.geometry.volumes)
        fname_volume_mfem = str(fname_volume_abq.with_suffix(".mesh"))
        self.geometry.export_volume_mesh(fname_volume_mfem, show_mesh=False)
        cl_beam = CantileverBeam.CantileverBeam(temp_current_simulation_folder)
        cl_beam.read_mesh(fname_volume_mfem)

        # before solve, we should add a problem setup and set material properties
        cl_beam.set_up()
        dTheta = self.geometry.get_dTheta()


        volume, der_vol = cl_beam.compute_volume(dTheta=dTheta)
        if der_vol is None:
            der_vol = 0
        der_vol_shape = der_vol.shape
        der_vol_mean = np.mean(der_vol)
        self.logger.debug(f"Volume: {volume:.5g}, "
                          f"dVolume: {der_vol_shape} array "
                          f"with mean {der_vol_mean:.5g}")
        if np.any(np.isnan(der_vol)):
            self.logger.warning("Nan detected in volume derivative.")
        cl_beam.solve()
        compliance, der_compliance = cl_beam.compute_compliance(dTheta=dTheta)
        if der_compliance is None:
            der_compliance = 0
        der_compl_shape = der_compliance.shape
        der_compl_mean = np.mean(der_compliance)
        self.logger.debug(f"Compliance: {compliance:.5g}, "
                          f"dCompliance: {der_compl_shape} array "
                          f"with mean {der_compl_mean:.5g}")
        vol_constraint = self.options["general"]["volume_constraint"]
        self.cache[str(control_point_values.round(8))] = {
            "objective": (compliance, der_compliance), 
            "constraint": (volume - vol_constraint, der_vol)}  # f, g are scalars
        self.logger.debug(f"Finished iteration {self.iteration} "
                          f" with compliance {compliance} and volume {volume}")
        self.optimization_results.append_result(control_point_values, volume, compliance)
        self.save_and_clear(temp_current_simulation_folder)

    def save_and_clear(self, temp_current_simulation_folder):
        with open(self.optimization_folder/"results.json", "w") as f:
            json.dump(dataclasses.asdict(self.optimization_results), f)
        save_every = self.iteration % self.options["general"]["save_every"] == 0
        first_iteration = self.iteration == 1
        if save_every or first_iteration:
            self.logger.debug(f"Saving simulation results to {self.current_simulation_folder}")
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

    def run_MMA_optimization(self, options):
               # do all your calculations only once here
        # in the end fill the cache
        
        def obj_fun(x):
            res = self.objective(x)
            return res

        def constraint(x):
            res = self.constraint(x)
            return res
        x0 = self.start_values
        mma_opti = MMA.MMA()
        result = mma_opti.minimize(x0, obj_fun, constraint, self.bounds, options)
        return result
    
    # def create_animation(self):
    #     self

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

def configure_logging(args):
    """
    Initialize logging defaults for Project.

    :param logfile_path: logfile used to the logfile
    :type logfile_path: string

    This function does:

    - Assign INFO and DEBUG level to logger file handler and console handler

    """
    DEFAULT_LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
    }
    dictConfig(DEFAULT_LOGGING)
    if args.logfile is not None:
        logfile = args.logfile
    else:
        logfile = "optimization.log"
    default_formatter = logging.Formatter("%(asctime)s %(module)s - %(levelname)s - %(message)s", datefmt='%H:%M:%S')

    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler.setFormatter(default_formatter)
    console_handler.setFormatter(default_formatter)

    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    logger_blocklist = [
        "numba",
    ]

    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)