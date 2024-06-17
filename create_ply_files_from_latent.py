import deep_sdf.mesh
import os
import json
import torch
import deep_sdf.workspace as ws
import pathlib
import time
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_directory = "./experiments/simple_geom"
checkpoint = "1000"

specs_filename = os.path.join(experiment_directory, "specs.json")

specs = json.load(open(specs_filename))

arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

latent_size = specs["CodeLength"]

decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

decoder = torch.nn.DataParallel(decoder)
decoder.eval()


saved_model_state = torch.load(
    os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth"
    ), map_location=device
)
saved_model_epoch = saved_model_state["epoch"]

decoder.load_state_dict(saved_model_state["model_state_dict"])

decoder = decoder.module.to(device)

latent = ws.load_latent_vectors(experiment_directory, checkpoint).to(device)

for i, latent_in in enumerate(latent):
    epoch = checkpoint
    dataset = "latent_recon"
    class_name = "all"
    instance_name = f"{i}"
    fname = ws.get_reconstructed_mesh_filename(experiment_directory, 
                                                   epoch, dataset, class_name, 
                                                   instance_name)
    fname = pathlib.Path(fname)
    if os.path.isdir(fname.parent) == False:
        os.makedirs(fname.parent)
    if os.path.isfile(fname):
        print(f"Skipping {fname}")
        continue
    print(f"Reconstructing {fname} ({i}/{len(latent)})")
    deep_sdf.mesh.create_mesh(
        decoder, latent_in, str(fname.with_suffix("")), N=256, max_batch=int(8**3)
    )
