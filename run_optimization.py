from optimization.optimization import struct_optimization
import torch
import deep_sdf.workspace as ws
import deep_sdf
import numpy as np
import logging
import argparse

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run a DeepMS optimization")
    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    deep_sdf.configure_logging(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimization = struct_optimization("optimization_runs/opti_refined_cps")

    optimization.set_x0(None)
    optimization.run_optimization()