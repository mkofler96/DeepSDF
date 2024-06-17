
import numpy as np
import gustaf as gus
import vedo
from sdf_sampler.plotting import scatter_contour_at_z_level
from sdf_sampler import sdf_sampler
from sdf_sampler.cross_ms_sdf import CrossMsSDF

outdir = "data/SdfSamples"
splitdir = "data/splits"
sdf_sampler = sdf_sampler.SDFSampler(outdir, splitdir)


sdf = CrossMsSDF(0.8).SDF
sdf(np.array([[1,1,1]]))


scatter_contour_at_z_level(sdf, z_level=1)


import vedo
vedo.settings.default_backend = 'k3d'

dataset_info = {
    "dataset_name": "test_set",
    "class_name": "microstructure"}
training_split_info = sdf_sampler.sample_sdfs([CrossMsSDF(r).SDF for r in np.linspace(0.1, 0.4, 20)], dataset_info, n_samples=1e5, sampling_strategy="uniform", show=True)
sdf_sampler.write_json("microstructure_train.json", dataset_info, training_split_info)


