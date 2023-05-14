#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
import torch.nn.init as init
import signal
import sys
import os
import logging
import math
import copy
import json
import time
import wandb
import tools
import tools.workspace as ws


# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="Scene_SDF",
#     resume = True,
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate1": 0.0005,
#     "learning_rate2": 0.0005,
#     "architecture": "MLP",
#     "dataset": "ShapeNet",
#     "epochs": 2001,
#     "regularization": "false",
#     "optimizer":"many Adams",
#     "decoder_layers": 4
#     }
# )


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, model, model_params_subdir, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, model_params_subdir,True)

    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        os.path.join(model_params_dir, filename),
    )

def save_mlp_params(experiment_directory,scene,params):
    model_params_dir = ws.get_model_params_dir(experiment_directory, ws.mlp_params_subdir,True)

    torch.save(params,os.path.join(model_params_dir, scene+'.pth'))

def save_mlp(experiment_directory,scene,model):
    model_params_dir = ws.get_model_params_dir(experiment_directory, ws.mlp_params_subdir,True)

    torch.save(model,os.path.join(model_params_dir, scene+'.pth'))


def save_optimizer(experiment_directory, scene, optimizer):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(optimizer,os.path.join(optimizer_params_dir, scene+'.pth'))


def load_optimizer(experiment_directory, scene_name):

    full_filename = os.path.join(ws.get_optimizer_params_dir(experiment_directory), scene_name + ".pth")

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    return torch.load(full_filename)

def load_mlp_parameters(experiment_directory,scene_name,model,device):
    filename = os.path.join(experiment_directory, ws.mlp_params_subdir, scene_name + ".pth")
    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename,map_location=device)

    model.load_state_dict(data)

def load_mlp(experiment_directory,scene_name):
    filename = os.path.join(experiment_directory, ws.mlp_params_subdir, scene_name + ".pth")
    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    return torch.load(filename)

def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    mlp_param_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "mlp_param_magnitude": mlp_param_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["mlp_param_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())

def preload_params(experiment_directory, scene_names,device):
    logging.info("Preloading parameters")
    weight_dict = {}
    optimizer_dict = {}
    for scene in scene_names:
        mlp_filename = os.path.join(experiment_directory, ws.mlp_params_subdir, scene + ".pth")
        optim_filename = os.path.join(ws.get_optimizer_params_dir(experiment_directory), scene + ".pth")
        weight_dict[scene] = torch.load(mlp_filename,map_location=device)
        optimizer_dict[scene] = torch.load(optim_filename,map_location='cpu')
    return weight_dict,optimizer_dict

def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
        
# initialize the mlp optimizer
def optimizer_init(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            state['momentum_buffer'] = torch.zeros_like(p.data)
        
        
def main_function(experiment_directory, continue_from, batch_split):

    logging.debug("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + specs["Description"])

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder,Scene_MLP"])

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch,weight_dict,optimizer_dict):

        save_model(experiment_directory, "latest.pth", decoder, ws.decoder_params_subdir, epoch)
        for scene,param in weight_dict.items():
            save_mlp_params(experiment_directory,scene,param)
        for scene,param in optimizer_dict.items():
            save_optimizer(experiment_directory,scene,param)
        #save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
     #   save_model(experiment_directory, "latest.pth", scene_mlp, ws.mlp_params_subdir, epoch)

    def save_checkpoints(epoch,weight_dict,optimizer_dict):

        save_model(experiment_directory, str(epoch) + ".pth", decoder, ws.decoder_params_subdir,epoch)
        for scene,param in weight_dict.items():
            save_mlp_params(experiment_directory,scene,param)
        for scene,param in optimizer_dict.items():
            save_optimizer(experiment_directory,scene,param)
        #save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
       # save_model(experiment_directory, str(epoch) + ".pth", scene_mlp, ws.mlp_params_subdir, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)


    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_decoder_regularization = get_spec_with_default(specs, "DecoderRegularization", True)
    mlp_reg_lambda = get_spec_with_default(specs, "MLPRegularizationLambda", 1e-4)
    decoder_reg_lambda = get_spec_with_default(specs,"DecoderRegularizationLambda",0.0005)
    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    do_scene_mlp_regularization = get_spec_with_default(specs, "SceneRegularization", False)
    do_pos_enc = get_spec_with_default(specs, "PosEncode", True)
    mlp_dims = get_spec_with_default(specs, "MlpHiddenDims",[64,256,256,512])
    device_ids = [0]  # Assign GPU id
    device = torch.device("cuda:{}".format(device_ids[0]))
    torch.cuda.set_device(device)
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).to(device)
    scene_mlp = arch.Scene_MLP(latent_size,mlp_dims, do_pos_enc).to(device)

    logging.info("training with {} GPU(s)".format(len(device_ids)))
    
    if len(device_ids) > 1:
        decoder = torch.nn.DataParallel(decoder,device_ids=device_ids)
        scene_mlp = torch.nn.DataParallel(scene_mlp,device_ids=device_ids)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    # sdf dataset
    sdf_dataset = tools.data.SDFSamples(
        data_source, train_split,num_samp_per_scene,load_ram=False
    )
    scene_names = sdf_dataset.scene_names
    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    loss_l1 = torch.nn.L1Loss(reduction="sum")
    
    optimizer_decoder = torch.optim.Adam(decoder.parameters(),lr_schedules[0].get_learning_rate(0))
    optimizer_mlp = torch.optim.Adam(scene_mlp.parameters(),lr_schedules[0].get_learning_rate(0))

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}
    mlp_param_mag_log = {}

    start_epoch = 1
    weight_dict_exist = False
    weight_dict = {}
    optimizer_dict = {}
    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        weight_dict, optimizer_dict = preload_params(experiment_directory,scene_names,device)

#         mlp_epoch = ws.load_model_parameters(
#             experiment_directory, ws.mlp_params_subdir,continue_from, scene_mlp
#         )

        model_epoch = ws.load_model_parameters(
            experiment_directory, ws.decoder_params_subdir,continue_from, decoder
        )

        loss_log, lr_log, timing_log, mlp_param_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, mlp_param_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, mlp_param_mag_log, param_mag_log, model_epoch
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))
    decoder_param_num = sum(p.data.nelement() for p in decoder.parameters())
    mlp_param_num = sum(p.data.nelement() for p in scene_mlp.parameters())
    logging.info(
        "Number of decoder parameters: {}".format(decoder_param_num)
        )
    logging.info(
        "Number of scene mlp parameters: {}".format(mlp_param_num)
        )
                              
    #wandb.watch(scene_mlp)
    #wandb.watch(decoder)
    
    for epoch in range(start_epoch, num_epochs + 1):
#         if not weight_dict_exist and epoch!=1:
#             weight_dict = preload_mlp_weights(experiment_directory,scene_names,device)
#             weight_dict_exist = True
        start = time.time()

        logging.info("epoch {}...".format(epoch))

        decoder.train()
        scene_mlp.train()
        
        adjust_learning_rate(lr_schedules, optimizer_decoder, epoch)
        #adjust_learning_rate(lr_schedules, optimizer_mlp, epoch)
        epoch_loss = 0.0
        reg_loss_mlp = 0.0
        reg_loss_decoder = 0.0
        gt_loss = 0.0
        reg_loss_latent = 0.0

        for sdf_data, scene in sdf_loader:

            # Process the input data
            sdf_data = sdf_data.reshape(-1, 4)
            num_sdf_samples = sdf_data.shape[0]
            
            sdf_data.requires_grad = False

            xyz = sdf_data[:, 0:3].cuda()
            sdf_gt = sdf_data[:, 3].unsqueeze(1).cuda()

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            xyz = torch.chunk(xyz, batch_split)

            sdf_gt = torch.chunk(sdf_gt, batch_split)

            # Load MLP parameters:
            if epoch != 1:
                scene_mlp.load_state_dict(weight_dict[scene[0]])
                #load_mlp_parameters(experiment_directory,scene[0],scene_mlp,device)
                #scene_mlp = load_mlp(experiment_directory,scene[0]).to(device)
                #load_optimizer(experiment_directory,scene[0],optimizer_all)
                #optimizer_mlp = load_optimizer(experiment_directory,scene[0])
                optimizer_mlp.load_state_dict(optimizer_dict[scene[0]])
                adjust_learning_rate(lr_schedules, optimizer_mlp, epoch)
                #optimizer_mlp = optimizer_dict[scene[0]]
            else:
                scene_mlp.apply(weight_init)
                #scene_mlp = arch.Scene_MLP(latent_size,mlp_dims).to(device)
                optimizer_mlp = torch.optim.Adam(scene_mlp.parameters(),lr_schedules[1].get_learning_rate(epoch))
                #optimizer_dict[scene[0]] = optimizer_mlp


            batch_loss = 0.0
            reg_loss = 0.0

            optimizer_decoder.zero_grad()
            optimizer_mlp.zero_grad()
            
            for i in range(batch_split):
                batch_vecs = scene_mlp(xyz[i])
                #print(batch_vecs.shape,batch_vecs)  
                #input = torch.cat([batch_vecs, xyz[i]], dim=1)
                # NN optimization
                pred_sdf = decoder(batch_vecs)

                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                chunk_loss = loss_l1(pred_sdf, sdf_gt[i]) / num_sdf_samples
                gt_loss += chunk_loss.item()
                if do_scene_mlp_regularization:
                    l2_reg_mlp = None
                    for name, param in scene_mlp.named_parameters():
                        if 'weight' in name:
                            if l2_reg_mlp is None:
                                l2_reg_mlp = torch.norm(param)
                            else:
                                l2_reg_mlp = l2_reg_mlp + torch.norm(param)
                    reg_loss = (mlp_reg_lambda * l2_reg_mlp)/mlp_param_num
                    reg_loss_mlp += (mlp_reg_lambda * l2_reg_mlp)/mlp_param_num

                if do_decoder_regularization:
                    l2_reg_decoder = None
                    for name, param in decoder.named_parameters():
                        if 'weight' in name:
                            if l2_reg_decoder is None:
                                l2_reg_decoder = torch.norm(param)
                            else:
                                l2_reg_decoder = l2_reg_decoder + torch.norm(param)

                    reg_loss += decoder_reg_lambda * l2_reg_decoder/ decoder_param_num
                    reg_loss_decoder += decoder_reg_lambda * l2_reg_decoder/decoder_param_num
                # calculate l2 reg loss of latent code
                l2_latent_loss = torch.sum(torch.norm(batch_vecs.detach(), dim=1))
                reg_latent_loss_batch = (1e-4 * min(1, epoch / 100) * l2_latent_loss) / num_sdf_samples
                reg_loss_latent += reg_latent_loss_batch.item()

                if do_code_regularization:
                    reg_loss += reg_latent_loss_batch

                chunk_loss = chunk_loss + reg_loss

                chunk_loss.backward()

                batch_loss += chunk_loss.item()

            logging.debug("loss = {}".format(batch_loss))

            loss_log.append(batch_loss)

            if grad_clip is not None:

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(scene_mlp.parameters(), grad_clip)

            optimizer_decoder.step()
            optimizer_mlp.step()

             # Store the weights of mlp
            #save_mlp_params(experiment_directory,scene[0],scene_mlp)
            #save_mlp(experiment_directory,scene[0],scene_mlp)
            weight_dict[scene[0]] = copy.deepcopy(scene_mlp.state_dict())
            optimizer_dict[scene[0]] = copy.deepcopy(optimizer_mlp.state_dict())
            #save_optimizer(experiment_directory,scene[0],optimizer_mlp)
            epoch_loss += batch_loss
        
#         for k,v in weight_dict.items():
#             print(k,v)
        end = time.time()
        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)
        logging.info("Loss:{}...reconstruction_loss:{}...".format(epoch_loss/ (num_scenes),gt_loss/(num_scenes)))
        logging.info("Latent loss:{}...".format(reg_loss_latent/ (num_scenes)))
        #wandb.log({"total_loss": epoch_loss/ num_scenes, "reg_loss_mlp": reg_loss_mlp/ num_scenes,"reg_loss_decoder":reg_loss_decoder/ num_scenes,"gt_loss":gt_loss/ num_scenes,"reg_loss_latent":reg_loss_latent/num_scenes}, step=epoch)
        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

        # lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        append_parameter_magnitudes(param_mag_log, decoder)
        append_parameter_magnitudes(mlp_param_mag_log, scene_mlp)

        if epoch in checkpoints:
            save_checkpoints(epoch,weight_dict,optimizer_dict)

        if epoch % log_frequency == 0:

            save_latest(epoch,weight_dict,optimizer_dict)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                mlp_param_mag_log,
                param_mag_log,
                epoch,
            )


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a locally-conditioned autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    tools.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    tools.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))
