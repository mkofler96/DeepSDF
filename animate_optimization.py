from optimization.opti import struct_optimization, configure_logging
import torch
import deep_sdf.workspace as ws
import deep_sdf
import numpy as np
import argparse
import logging

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run a DeepMS optimization")
    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    optimization_dir = "/storage/mkofler/DeepSDF/optimization_runs/opti_double_lattice"
    experiment_location = "/storage/mkofler/DeepSDF/"
    args.logfile = optimization_dir+"/optimization_logs.log"
    configure_logging(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimization = struct_optimization(optimization_dir, experiment_location)

    optimization.create_animation()