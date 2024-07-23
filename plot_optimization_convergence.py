import argparse
from optimization.optimization import struct_optimization

if __name__ == '__main__':
    input_parser = argparse.ArgumentParser(description='Plot optimization convergence')
    input_parser.add_argument('--optimization_folder', '-o', type=str, help='Path to optimization folder', default="")
    args = input_parser.parse_args()
    optimization_folder = args.optimization_folder


    opti = struct_optimization(optimization_folder) 
    opti.plot_convergence()
