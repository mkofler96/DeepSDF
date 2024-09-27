#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import torch
import numpy as np

def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s DeepSdf - %(levelname)s - %(message)s", datefmt='%H:%M:%S')
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def decode_sdf(decoder, latent_vector, queries):    
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    sdf = decoder(inputs)

    return sdf

def get_MS(decoder, latent_vector):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(latent_vector, np.ndarray):
        latent_vector = torch.from_numpy(latent_vector).float().to(device)
    
    x = torch.linspace(-1,1,100).to(device)
    y = torch.linspace(-1,1,100).to(device)
    xv, yv = torch.meshgrid(x,y)
    
    num_samples = xv.shape[0]*xv.shape[1]
    latent_inputs = latent_vector.expand(num_samples, -1)
    xf, yf = xv.reshape((-1,1)).float(), yv.reshape((-1,1)).float()
    inputs = torch.cat([latent_inputs.float(), xf, yf], 1)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    pred_sdf = decoder(inputs)
    z = pred_sdf.cpu().detach().numpy()
    x = xf.cpu().detach().numpy()
    y = yf.cpu().detach().numpy()
    return x, y, z


import torch
import logging

def format_memory_size(size_in_bytes):
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    size = size_in_bytes
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:6.2f}{units[unit_index]}"

def log_memory_usage():
    memory_allocated = torch.cuda.memory_allocated(0)
    memory_reserved = torch.cuda.memory_reserved(0)
    
    # Format memory values
    memory_allocated_str = format_memory_size(memory_allocated)
    memory_reserved_str = format_memory_size(memory_reserved)
    
    # Single line output with aligned values
    output = (f"torch.cuda.memory_allocated: {memory_allocated_str} | "
              f"torch.cuda.memory_reserved: {memory_reserved_str}")
    
    logging.debug(output)
