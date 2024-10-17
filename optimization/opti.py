import os
import json
import pathlib
import numpy as np

import tempfile

import pathlib
from typing import Union

import scipy
import matplotlib.pyplot as plt
import numpy as np
import gustaf as gus
import splinepy as sp
import imageio.v3 as imageio
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

    def __init__(self, optimization_folder: Union[str, bytes, os.PathLike], experiment_location=None):

        self.optimization_folder = pathlib.Path(optimization_folder)
        self.optimization_results = OptimizationResults([], [], [])
        if self.settings_filename.exists():
            self.load_settings()
        else:
            raise FileNotFoundError(f"No config.json in {self.optimization_folder}")

        self.cache = {}
        self.logger = logging.getLogger(__name__)

        self.geometry = DeepSDFMesh(self.options["mesh"], experiment_location=experiment_location)

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

    def set_x0(self):
        n_control_points = self.geometry.get_n_control_points()
        n_latent = self.geometry.get_latent_shape()
        control_points = np.zeros((n_control_points, n_latent))
        if "x0" in self.options["optimization"]:
            control_points += self.options["optimization"]["x0"]
        self.logger.debug(f"Setting x0 to: {control_points}")
        self.start_values = control_points.reshape(-1)
        self.dv_names = [f"x{i}" for i in range(len(self.start_values))]
        if "bounds" in self.options["optimization"]:
            lb = self.options["optimization"]["bounds"][0]
            ub = self.options["optimization"]["bounds"][1]
        else:
            lb = -1
            ub = 1
        self.logger.debug(f"Settings bounds to ({lb}, {ub})")
        self.bounds = [(lb,ub)]*len(self.start_values)

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
        self.set_x0()
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
    
    def create_animation(self, add_boundary_conditions=False):
        mesh_files = []
        for directory in self.optimization_folder.iterdir():
            if "simulation" in str(directory):
                print(f"Sim dir: {directory}")
                sim_index = directory.stem.split("_")[1]
                surf_filename = directory/("surf"+sim_index + ".inp")
                print(f"Surface filename: {surf_filename}")
                mesh_files.append((int(sim_index), surf_filename))
                # mesh_files.append(directory.glob("surf*")[0])
                # mesh_files.extend(list(directory.glob("surf*")))
            else:
                print(f"Non dir: {directory}")
        _TUWIEN_COLOR_SCHEME = {
            "blue": (0, 102, 153),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "blue_1": (84, 133, 171),
            "blue_2": (114, 173, 213),
            "blue_3": (166, 213, 236),
            "blue_4": (223, 242, 253),
            "grey": (100, 99, 99),
            "grey_1": (157, 157, 156),
            "grey_2": (208, 208, 208),
            "grey_3": (237, 237, 237),
            "green": (0, 126, 113),
            "green_1": (106, 170, 165),
            "green_2": (162, 198, 194),
            "green_3": (233, 241, 240),
            "magenta": (186, 70, 130),
            "magenta_1": (205, 129, 168),
            "magenta_2": (223, 175, 202),
            "magenta_3": (245, 229, 239),
            "yellow": (225, 137, 34),
            "yellow_1": (238, 180, 115),
            "yellow_2": (245, 208, 168),
            "yellow_3": (153, 239, 225),
        }

        if add_boundary_conditions:
            fix = sp.helpme.create.box(0, 1.5, 1.5)
            fix.control_points -= np.array([0.001, 0.25, 0.25])
            fix.show_options["control_points"] = False
            fix.show_options["c"] = _TUWIEN_COLOR_SCHEME["black"]

            n_arrows_x = 6
            n_arrows_y = n_arrows_x/2
            l_arrows = 0.3
            area_of_application = 48/24

            start_arrow = np.array([[2-area_of_application,0,1],[2,1,1]])
            end_arrow = start_arrow + np.array([[0, 0, l_arrows]])
            resolutions = np.array([n_arrows_x,n_arrows_x, n_arrows_y])
            verts_start = gus.create.vertices.raster(bounds=start_arrow, resolutions=resolutions)
            verts_end = gus.create.vertices.raster(bounds=end_arrow, resolutions=resolutions)

            a_edges = []
            for vr, vl in zip(verts_start.vertices, verts_end.vertices):
                e = gus.Edges([vl, vr], [[0,1]])
                a_edges.append(e)

            d_F = gus.Edges.concat(a_edges)
            d_F.show_options["as_arrows"] = True
            d_F.show_options["c"] = _TUWIEN_COLOR_SCHEME["blue_1"]
            # d_F.show_options["lw"] = 30
            cam = dict(
                position=(3.73103, -4.35002, 1.65212),
                focal_point=(0.999999, 0.500001, 0.525011),
                viewup=(-0.0983954, 0.172364, 0.980107),
                distance=5.67905,
                clipping_range=(3.08500, 8.96935),
            )

           
        images = []
        for index, mesh_file in sorted(mesh_files, 
                                       key=lambda surf_ind_tuple: surf_ind_tuple[0]):
            logger.info(f"Creating sreenshot of {mesh_file}")
            mesh = gus.io.meshio.load(str(mesh_file))
            cam = dict(
                position=(3.25705, -3.50828, 1.45651),
                focal_point=(1.00000, 0.500000, 0.525011),
                viewup=(-0.0983954, 0.172364, 0.980107),
                roll=-70.6513,
                distance=4.69343,
                clipping_range=(2.65216, 7.27428),
            )
            if add_boundary_conditions:
                shown_geom = [mesh, fix, d_F]
                name = "animtation_with_fix"
            else:
                shown_geom = [mesh]
                name = "animtation"
            showable = gus.show([f"Iteration: {index:<3}", shown_geom], cam=cam, interactive=False, offscreen=True)


            showable.screenshot(mesh_file.with_suffix(".png").as_posix())
            image = imageio.imread(str(mesh_file.with_suffix(".png")))
            images.append(image)

        imageio.imwrite(self.optimization_folder/f'{name}.gif', images, duration=300)       

    # def plot_convergence(self, custom_axis=None):
    #     if custom_axis is None:
    #         fig, ax = plt.subplots()
    #     with open(self.optimization_folder / "results.json", "r") as f:
    #         self.optimization_results = dataclasses.from_dict(dataclasses.make_dataclass('OptimizationResults', []), json.load(f))
    #     ax.plot(self.optimization_results)

    def plot_convergence(self, custom_axis=None, normalize=True):
        if custom_axis is None:
            fig, ax = plt.subplots()
        # optimization_results = load_results(pathlib.Path(self.optimization_folder), as_np_array=False)
        with open(self.optimization_folder / "results.json", "r") as f:
            res_dict = json.load(f)
            self.optimization_results = OptimizationResults(**res_dict)
        print(self.optimization_results)
        compliance = np.array(self.optimization_results.compliance)
        volume = np.array(self.optimization_results.volume)
        if normalize:
            compliance = compliance/compliance[0]
            volume = volume/self.options["general"]["volume_constraint"]

        # ax.plot(np.array(self.optimization_results.compliance)/self.optimization_results.compliance[0], label="Objective")
        # # todo: hardcoded constraint 6
        # ax.plot(np.array(self.optimization_results.volume)/self.options["general"]["volume_constraint"], label="Constraint")
        ax2 = ax.twinx()

        # Plot the objective on the left y-axis
        ax.plot(compliance, label="Objective", color='blue')
        ax.set_ylim([0, np.ceil(np.max(compliance))])
        # Plot the constraint on the right y-axis
        ax2.plot(volume, label="Constraint", color='orange')
        ax2.set_ylim([0, np.ceil(np.max(volume))])
        # Set labels for the axes
        ax.set_ylabel('Objective')
        ax2.set_ylabel('Constraint')

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2)

        # Optionally set titles, grid, etc.
        ax.set_xlabel('Iteration')

        # Show the plot
        plt.show()


def load_results(optimization_folder, as_np_array=False):
    with open(optimization_folder/"results.json", "r") as f:
        optimization_results = json.load(f)
        print(f"loaded: {optimization_folder/'results.json'}")
    if as_np_array:
        return np.array([optimization_results["compliance"],
                            optimization_results["volume"],
                            optimization_results["design_vector"]]).T
    else:
        return optimization_results    


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


if __name__ == "__main__":
    optimization = struct_optimization("optimization_runs/test_opti")
    optimization.plot_convergence()