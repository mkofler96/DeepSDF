#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch

import deep_sdf.utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_mesh(
    decoder, latent_vec, filename, N=256, max_batch=32 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].to(device)

        samples[head : min(head + max_batch, num_samples), 3] = (
            deep_sdf.utils.decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    if not isinstance(voxel_size, list):
        voxel_size = [voxel_size]*3
    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=voxel_size
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )


def create_mesh_microstructure(tiling, decoder, latent_vec_interpolation, filename, N=256, max_batch=32 ** 3, offset=None, scale=None, cap_borders=False, save_ply_file = False
):
    if isinstance(tiling, list):
        if len(tiling) != 3:
            raise ValueError("Tiling must be a list of 3 integers")
        tiling = np.array(tiling)
    elif isinstance(tiling, int):
        tiling = np.array([tiling, tiling, tiling])
    else:
        raise ValueError("Tiling must be a list or an integer")
    
    if isinstance(N, list):
        if len(N) != 3:
            raise ValueError("Tiling must be a list of 3 integers")
        N = np.array(N)
    elif isinstance(N, int):
        N = np.array([N, N, N])
    else:
        raise ValueError("Tiling must be a list or an integer")

    start = time.time()
    ply_filename = filename

    decoder.eval()
    N_tot = N[0]*N[1]*N[2]
    overall_index = torch.arange(0, N_tot, 1, out=torch.LongTensor())
    samples_orig = torch.zeros(N_tot, 4)
    samples = torch.zeros(N_tot, 4)
    # transform first 3 columns
    # to be the x, y, z index
    samples_orig[:, 2] = overall_index % N[2]
    samples_orig[:, 1] = (overall_index // N[2]) % N[1]
    samples_orig[:, 0] = ((overall_index // N[2]) // N[1]) % N[0]

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size_x = 2.0 / (N[0]-1)
    voxel_size_y = 2.0 / (N[1]-1)
    voxel_size_z = 2.0 / (N[2]-1)
    # transform first 3 columns
    # to be the x, y, z coordinate
    samples_orig[:, 0] = (samples_orig[:, 0] * voxel_size_x) + voxel_origin[0]
    samples_orig[:, 1] = (samples_orig[:, 1] * voxel_size_y) + voxel_origin[1]
    samples_orig[:, 2] = (samples_orig[:, 2] * voxel_size_z) + voxel_origin[2]

    # samples = [-1, 1]
    px, py, pz = tiling
    px = 1/px
    py = 1/py
    pz = 1/pz
    samples[:, 0] = -(2/px)*torch.abs((samples_orig[:, 0]) % (px*2) - px) + 1
    samples[:, 1] = -(2/py)*torch.abs((samples_orig[:, 1]) % (py*2) - py) + 1
    samples[:, 2] = -(2/pz)*torch.abs((samples_orig[:, 2]) % (pz*2) - pz) + 1
    num_samples = N_tot

    samples.requires_grad = False

    head = 0
    lat_vec_red = latent_vec_interpolation.evaluate(samples_orig[:, 0:3].numpy())
    queries = torch.hstack([torch.tensor(lat_vec_red).to(torch.float32), samples[:, 0:3]])

    while head < num_samples:
        sample_subset = queries[head : min(head + max_batch, num_samples), :].to(device)

        queries[head : min(head + max_batch, num_samples), -1] = (
            deep_sdf.utils.decode_sdf(decoder, None, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch
    sample_time = time.time()
    print("sampling takes: %f" % (sample_time - start))
    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N[0], N[1], N[2])
    sdf_values = queries[:, -1].data.cpu().numpy()
    samples_orig = samples_orig.cpu().numpy()
    if cap_borders:
        for (dim, measure) in zip([0, 0, 1, 1, 2, 2], [-1, 1, -1, 1, -1, 1]):
            border_sdf = (samples_orig[:, dim] - measure*0.9)*-measure
            sdf_values = np.maximum(sdf_values, -border_sdf)
        end = time.time()
        print("Capping takes: %f" % (end - sample_time))
    
    sdf_values = sdf_values.reshape(N[0], N[1], N[2])
    sdf_values = torch.tensor(sdf_values)

    voxel_size = [voxel_size_x, voxel_size_y, voxel_size_z]

    

    if save_ply_file:
        convert_sdf_samples_to_ply(
            sdf_values,
            voxel_origin,
            voxel_size,
            ply_filename + ".ply",
            offset,
            scale,
        )
    else:
        if not isinstance(voxel_size, list):
            voxel_size = [voxel_size]*3
        verts, faces, normals, values = skimage.measure.marching_cubes(
            sdf_values.numpy(), level=0.0, spacing=voxel_size
        )
        # sci-kit measure assumes origin at (0,0,0)
        # input for SDF is -1 to 1
        # scale factor 2 to get to 0 to 1
        verts = verts/2
        return verts, faces