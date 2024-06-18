import deep_sdf.mesh
import os
import json
import torch
import deep_sdf.workspace as ws
import pathlib
import time
import datetime
import splinepy as sp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_directory = "./experiments/simple_geom"
checkpoint = "1000"

# decoder = ws.load_trained_model(experiment_directory, checkpoint)
# latent = ws.load_latent_vectors(experiment_directory, checkpoint).to(device)

box = sp.helpme.create.box(10,10,10).bspline
box.insert_knots(0, [0.5])
box.insert_knots(1, [0.5])
box.insert_knots(2, [0.5])
print(box.control_points)

