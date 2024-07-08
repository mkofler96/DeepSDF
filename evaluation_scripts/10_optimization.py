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




if __name__ == "__main__":
    max_iter = 30

    init_vals = 