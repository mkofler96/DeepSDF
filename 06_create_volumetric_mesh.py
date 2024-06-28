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
import numpy as np
import skimage
import pygalmesh


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_directory = "./experiments/corner_spheres_only"
checkpoint = "1000"

decoder = ws.load_trained_model(experiment_directory, checkpoint)
latent = ws.load_latent_vectors(experiment_directory, checkpoint).to(device)

decoder.eval()

mesh = pygalmesh.generate_volume_mesh_from_surface_mesh(
    "experiments/corner_spheres_only/Reconstructions/1000/Meshes/latent_recon/all/15.ply",
    min_facet_angle=25.0,
    max_radius_surface_delaunay_ball=0.15,
    max_facet_distance=0.008,
    max_circumradius_edge_ratio=3.0,
    verbose=False,
)

mesh.write("test_reconstructed.inp")