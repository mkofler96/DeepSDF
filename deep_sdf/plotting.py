import json
import vedo
import numpy as np
import os
import gustaf as gus
import logging
import pathlib

import deep_sdf.workspace as ws
import deep_sdf.mesh


def extract_paths(data, current_path=''):
    paths = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{current_path}/{key}" if current_path else key
            paths.extend(extract_paths(value, new_path))
    
    elif isinstance(data, list):
        for item in data:
            paths.extend(extract_paths(item, current_path))
    
    else:
        paths.append(f"{current_path}/{data}")

    return paths


def show_random_training_files(experiment_directory, deep_sdf_dir, n_files=4, epoch=None):
    specs = ws.load_experiment_specifications(experiment_directory)
    data_dir = pathlib.Path(deep_sdf_dir)
    split_dir = data_dir/specs["TrainSplit"]
    npz_filenames = extract_paths(json.load(open(split_dir)))
    ids = np.random.choice(len(npz_filenames), n_files, replace=False)
    plots = []
    for plt_id, id in enumerate(ids):
        current_plot = []
        plt = vedo.Plotter(axes=1)
        npz_filename = npz_filenames[id]
        full_filename = os.path.join(data_dir/specs["DataSource"], "SdfSamples", npz_filename +".npz")
        points = np.load(full_filename)
        all_points = np.vstack([points["neg"], points["pos"]])
        points = gus.vertices.Vertices(all_points[:, :3])
        points.vertex_data["sdf"] = all_points[:, -1]
        points.show_options["data"] = "sdf"
        points.show_options["cmap"] = "coolwarm"
        points.show_options["vmin"] = -0.1
        points.show_options["vmax"] = 0.1
        points.show_options["r"] = 10
        points.show_options["axes"] = True
        current_plot.append(points)

        # reconstruct the mesh
        if epoch is not None:
            ply_file = deep_sdf.mesh.create_mesh_from_latent(experiment_directory, epoch, id)
            try:
                mesh = gus.io.meshio.load(ply_file)
            except ValueError:
                logging.warning(f"Reconstruction for {npz_filename} not found")
                continue
            current_plot.append(mesh)
        plots.append(current_plot)
    gus.show(*plots)