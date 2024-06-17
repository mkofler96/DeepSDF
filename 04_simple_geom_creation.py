
import numpy as np
import gustaf as gus
import vedo
from sdf_sampler.plotting import scatter_contour_at_z_level
from sdf_sampler import sdf_sampler
from sdf_sampler.microstructures import CrossMsSDF, CornerSpheresSDF

outdir = "data/SdfSamples"
splitdir = "data/splits"
sdf_sampler = sdf_sampler.SDFSampler(outdir, splitdir)



ms = [CornerSpheresSDF(r) for r in np.linspace(0.4, 1, 20)] + [CrossMsSDF(r) for r in np.linspace(0.2, 1.5, 20)]
dataset_info = {
    "dataset_name": "microstructure",
    "class_name": "round_cross"}
training_split_info = sdf_sampler.sample_sdfs([MS.SDF for MS in ms], dataset_info, n_samples=1e5, sampling_strategy="uniform", show=False)
sdf_sampler.write_json("microstructure_round_cross_train.json", dataset_info, training_split_info)


