from setuptools import setup
from sys import platform

install_requires = [
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
      "numba",
      "napf",
      "mmapy"
   ]

if platform.startswith('linux'):
   install_requires.append("mfem")

setup(
   name='sdf_sampler',
   version='1.1',
   description='creates SDF samples',
   author='Michael Kofler',
   author_email='michael.kofler@tuwien.ac.at',
   install_requires=install_requires,
   packages=['sdf_sampler', 
             'deep_sdf', 
             'deep_sdf.metrics', 
             'deep_sdf.networks',
             'analysis',
             'flexicubes',
             'analysis.problems',
             'optimization'],  #same as name
)