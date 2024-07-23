from optimization.optimization import struct_optimization
import torch
import deep_sdf.workspace as ws
import numpy as np

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimization = struct_optimization("simulations/optimization_double_lattice_flexicubes")
    latent = ws.load_latent_vectors(optimization.experiment_directory, 
                                    optimization.checkpoint).to("cpu").numpy()
    optimization.set_x0(np.zeros_like(latent[0]))
    optimization.run_optimization()