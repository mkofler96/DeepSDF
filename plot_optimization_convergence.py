import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import pathlib


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
    
def plot_convergence(optimization_folder):
    optimization_results = load_results(pathlib.Path(optimization_folder), as_np_array=False)
    print(optimization_results)
    plt.plot(np.array(optimization_results["compliance"])/optimization_results["compliance"][0], label="Objective")
    # todo: hardcoded constraint 6
    plt.plot(np.array(optimization_results["volume"])/6, label="Constraint")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    input_parser = argparse.ArgumentParser(description='Plot optimization convergence')
    input_parser.add_argument('--optimization_folder', '-o', type=str, help='Path to optimization folder', default="")
    args = input_parser.parse_args()

    plot_convergence(args.optimization_folder)