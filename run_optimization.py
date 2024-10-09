from optimization.opti import struct_optimization, configure_logging
import torch
import deep_sdf.workspace as ws
import deep_sdf
import numpy as np
import argparse
import logging

        # self.logger.setLevel(logging.DEBUG)
        # formatter = logging.Formatter("%(asctime)s %(module)s - %(levelname)s - %(message)s", datefmt='%H:%M:%S')
        # # Create a FileHandler
        # fh = logging.FileHandler(self.log_filename, mode='w')
        # fh.setLevel(logging.DEBUG)
        # fh.setFormatter(formatter)
        # self.logger.addHandler(fh)
        # # Create a StreamHandler
        # sh = logging.StreamHandler()
        # sh.setLevel(logging.INFO)
        # sh.setFormatter(formatter)
        # self.logger.addHandler(sh)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run a DeepMS optimization")
    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    configure_logging(args)
    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimization = struct_optimization("optimization_runs/test_opti")

    optimization.set_x0(None)
    optimization.run_optimization()