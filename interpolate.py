#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import time
import torch
import tools
import tools.workspace as ws
import plotly.graph_objects as go
import plotly.subplots as sp

def create_mesh(
    decoder,scene_mlp1, scene_mlp2, step, filename, N=256, max_batch=32 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()
    scene_mlp1.eval()
    scene_mlp2.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        latent1 = scene_mlp1(sample_subset)
        latent2 = scene_mlp2(sample_subset)
        alpha = 1 - step / num_steps  
        code = alpha * latent1 + (1 - alpha) * latent2
        samples[head : min(head + max_batch, num_samples), 3] = (
             decoder(code)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    tools.mesh.convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )
    ply_list.append(ply_filename + ".ply")
        
if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained locally-conditioned decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for interpolation",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )


    tools.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    tools.configure_logging(args)

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder,Scene_MLP"])
    latent_size = specs["CodeLength"]
    mlp_dims = specs["MlpHiddenDims"]
    device_ids = [7]  # Assign GPU id
    device = torch.device("cuda:{}".format(device_ids[0]))
    torch.cuda.set_device(device)
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).to(device)
    scene_mlp1 = arch.Scene_MLP(latent_size,mlp_dims,False).to(device)
    scene_mlp2 = arch.Scene_MLP(latent_size,mlp_dims,False).to(device)
    if len(device_ids) > 1:
        decoder = torch.nn.DataParallel(decoder,device_ids=device_ids)
        scene_mlp1 = torch.nn.DataParallel(scene_mlp1,device_ids=device_ids)
        scene_mlp2 = torch.nn.DataParallel(scene_mlp2,device_ids=device_ids)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.decoder_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    logging.debug(decoder)

    interpolation_dir = os.path.join(
        args.experiment_directory, ws.interpolations_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(interpolation_dir):
        os.makedirs(interpolation_dir)

    interpolation_meshes_dir = os.path.join(
        interpolation_dir, ws.interpolation_meshes_subdir
    )
    if not os.path.isdir(interpolation_meshes_dir):
        os.makedirs(interpolation_meshes_dir)
    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )
    reconstruction_mlp_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_mlp_subdir
    )

    end_shape = "fdcb11fa39661f0fb08f81d66e854bfa"
    start_shape = "df411aa240fe48d5855eb7906a7a7a04"
    scene_mlp1.load_state_dict(torch.load(os.path.join(reconstruction_mlp_dir,start_shape+".pth")))
    print("Loading",os.path.join(reconstruction_mlp_dir,start_shape+".pth"))
    scene_mlp2.load_state_dict(torch.load(os.path.join(reconstruction_mlp_dir,end_shape+".pth")))
    print("Loading",os.path.join(reconstruction_mlp_dir,end_shape+".pth"))
    num_steps = 4

    ply_list = []
    for step in range(num_steps+1):
        print(f"step {step} start to interpolate.")
        mesh_filename = os.path.join(interpolation_meshes_dir,str(step))
        with torch.no_grad():
            create_mesh(decoder,scene_mlp1,scene_mlp2,step,mesh_filename,N=256, max_batch=int(2 ** 18))


