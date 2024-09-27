from setuptools import setup

setup(
   name='sdf_sampler',
   version='1.0',
   description='creates SDF samples',
   author='Michael Kofler',
   author_email='michael.kofler@tuwien.ac.at',
   install_requires=[
      "plyfile",
      "scikit-image",
      "trimesh",
      "matplotlib",
      "gustaf @ git+https://github.com/mkofler96/gustaf.git@ft-mfem-3D-export",
      "splinepy",
      "vedo",
      "libigl",
      "embreex",
      "tetgenpy",
      "meshio",
   ],
   packages=['sdf_sampler', 'deep_sdf', 'deep_sdf.metrics', 'deep_sdf.networks', 'optimization'],  #same as name
)