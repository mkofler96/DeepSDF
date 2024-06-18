import deep_sdf.mesh
import os
import json
import torch
import deep_sdf.workspace as ws
import pathlib
import time
import datetime



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_directory = "./experiments/corner_spheres_only"
checkpoint = "1000"

decoder = ws.load_trained_model(experiment_directory, checkpoint)
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


indices = [10, 30]
steps = 11
start = time.time()
num_samples = (len(indices)-1)*steps
i_sample = 1
for i_latent, lat in enumerate(indices[:-1]):
    index1 = indices[i_latent]
    index2 = indices[i_latent+1]
    latent1 = latent[index1]
    latent2 = latent[index2]

    for i in range(steps):
        latent_in = latent1 + (latent2 - latent1) * i / (steps - 1)
        epoch = checkpoint
        dataset = "latent_recon"
        class_name = "interpolation"
        instance_name = f"interpolate_{index1}_{index2}_{i}"
        fname = ws.get_reconstructed_mesh_filename(experiment_directory, 
                                                       epoch, dataset, class_name, 
                                                       instance_name)
        fname = pathlib.Path(fname)
        if os.path.isdir(fname.parent) == False:
            os.makedirs(fname.parent)
        if fname.exists():
            print(f"Skipping {fname}")
            continue
        deep_sdf.mesh.create_mesh(
            decoder, latent_in, str(fname.with_suffix("")), N=256, max_batch=int(32**3)
        )

        end = time.time()
        # logging.info("epoch {}...".format(epoch))
        tot_time = time.time() - start
        avg_time_per_sample = tot_time/(i_sample)
        estimated_remaining_time = avg_time_per_sample*(num_samples-(i_sample))
        time_string = str(datetime.timedelta(seconds=round(estimated_remaining_time)))
        print(f"Finished {i_sample} ({i_sample}/{num_samples}) [{i_sample/num_samples*100:.2f}%] in {time_string} ({avg_time_per_sample:.2f}s/epoch)")
        i_sample += 1