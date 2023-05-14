#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch
import torch.nn.init as init
import tools
import tools.workspace as ws

def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

def save_mlp_params(experiment_directory,scene,model):
    model_params_dir = ws.get_model_params_dir(experiment_directory, ws.mlp_params_subdir,True)

    torch.save(model.state_dict(),os.path.join(model_params_dir, scene+'.pth'))


def reconstruct(
    decoder,
    scene_mlp,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=True,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    # if type(stat) == type(0.1):
    #     latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    # else:
    #     latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    # latent.requires_grad = True

    optimizer = torch.optim.Adam(scene_mlp.parameters(), lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = tools.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = scene_mlp(xyz)
        pred_sdf = decoder(latent_inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(latent_inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)
       
#         l2_reg_mlp = None
#         for name, param in scene_mlp.named_parameters():
#             if 'weight' in name:
#                 if l2_reg_mlp is None:
#                     l2_reg_mlp = torch.norm(param)
#                 else:
#                     l2_reg_mlp = l2_reg_mlp + torch.norm(param)
        
        loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += 1e-4 * torch.mean(latent_inputs.pow(2))
        
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
        loss_num = loss.cpu().data.numpy()

    return loss_num


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
        + "files to use for reconstruction",
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
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=1000,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    tools.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    tools.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

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
    scene_mlp = arch.Scene_MLP(latent_size,mlp_dims).to(device)
    if len(device_ids) > 1:
        decoder = torch.nn.DataParallel(decoder,device_ids=device_ids)
        scene_mlp = torch.nn.DataParallel(scene_mlp,device_ids=device_ids)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.decoder_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])
    
#     decoder = decoder.module.cuda()
#     scene_mlp = scene_mlp.module.cuda()


    with open(args.split_filename, "r") as f:
        split = json.load(f)

    npz_filenames,scene_names = tools.data.get_instance_filenames(args.data_source, split)

    random.shuffle(npz_filenames)

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_mlp_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_mlp_subdir
    )
    if not os.path.isdir(reconstruction_mlp_dir):
        os.makedirs(reconstruction_mlp_dir)

    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)

        logging.debug("loading {}".format(npz))

        data_sdf = tools.data.read_sdf_samples_into_ram(full_filename)

        scene_name = os.path.basename(npz[:-4])

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz[:-4] + "-" + str(k + rerun)
                )
                mlp_weight_name = ws.load_mlp_parameters(args.experiment_directory, scene_name+ "-" + str(k + rerun), scene_mlp, device)
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
                # initialize the scene mlp
                scene_mlp.apply(weight_init)

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
            ):
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            start = time.time()
            err = reconstruct(
                decoder,
                scene_mlp,
                int(args.iterations),
                latent_size,
                data_sdf,
                0.01,  # [emp_mean,emp_var],
                0.1,
                num_samples=8000,
                lr=1e-3,
                l2reg=False,
            )
            logging.debug("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            decoder.eval()

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    tools.mesh.create_mesh(
                        decoder, scene_mlp, mesh_filename, N=256, max_batch=int(2 ** 18)
                    )
                logging.debug("total time: {}".format(time.time() - start))

            torch.save(scene_mlp.state_dict(),os.path.join(reconstruction_mlp_dir, scene_name+'.pth'))
