
import numpy as np
from sdf_sampler import sdf_sampler
from sdf_sampler.microstructures import CrossMsSDF, CornerSpheresSDF

outdir = "data/SdfSamples"
splitdir = "data/splits"
sdf_sampler = sdf_sampler.SDFSampler(outdir, splitdir)

ms = [CrossMsSDF(r) for r in np.linspace(0.1, 0.75, 20)]
dataset_info = {
    "dataset_name": "microstructure",
    "class_name": "round_cross"}
training_split_info = sdf_sampler.sample_sdfs([MS.SDF for MS in ms], dataset_info, n_samples=1e5, sampling_strategy="uniform", show=False)
sdf_sampler.write_json("round_cross_only.json", dataset_info, training_split_info)

ms = [CornerSpheresSDF(r) for r in np.linspace(0.4, 1, 20)] + [CrossMsSDF(r) for r in np.linspace(0.1, 0.75, 20)]
dataset_info = {
    "dataset_name": "microstructure",
    "class_name": "corner_spheres_and_round_cross"}
training_split_info = sdf_sampler.sample_sdfs([MS.SDF for MS in ms], dataset_info, n_samples=1e5, sampling_strategy="uniform", show=False)
sdf_sampler.write_json("microstructure_round_cross_train.json", dataset_info, training_split_info)


ms = [CornerSpheresSDF(r, limit=0.9) for r in np.linspace(0.4, 1, 20)]
dataset_info = {
    "dataset_name": "microstructure",
    "class_name": "corner_spheres"}
training_split_info = sdf_sampler.sample_sdfs([MS.SDF for MS in ms], dataset_info, n_samples=1e5, sampling_strategy="uniform", show=False)
sdf_sampler.write_json("corner_spheres_only.json", dataset_info, training_split_info)