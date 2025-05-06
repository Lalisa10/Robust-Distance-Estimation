# Reference: https://github.com/RoyChao19477/SEMamba/train.py
# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py

import warnings
import wandb
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
# Thêm thư mục 'audio-distance-estimation' vào sys.path
import time
import argparse
import json
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from Mamba_SEUNet.dataloaders.dataloader_vctk import VCTKDemandDataset
from Mamba_SEUNet.models.stfts import mag_phase_stft, mag_phase_istft
from Mamba_SEUNet.models.generator import MambaSEUNet
from joint_model import JointModel
from Mamba_SEUNet.models.loss import pesq_score, phase_losses
from Mamba_SEUNet.models.discriminator import MetricDiscriminator, batch_pesq
from audio_distance_estimation.model import SeldNet
from Mamba_SEUNet.utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed,
    print_gpu_info, log_model_info, initialize_process_group,
)
torch.backends.cudnn.benchmark = True

def setup_optimizers(models, cfg):
    """Set up optimizers for the models."""
    generator, discriminator = models
    learning_rate = cfg['training_cfg']['learning_rate']
    betas = (cfg['training_cfg']['adam_b1'], cfg['training_cfg']['adam_b2'])

    optim_g = optim.AdamW(generator.parameters(), lr=learning_rate, betas=betas)
    optim_d = optim.AdamW(discriminator.parameters(), lr=learning_rate, betas=betas)

    return optim_g, optim_d

def setup_schedulers(optimizers, cfg, last_epoch):
    """Set up learning rate schedulers."""
    optim_g, optim_d = optimizers
    lr_decay = cfg['training_cfg']['lr_decay']

    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay, last_epoch=last_epoch)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay, last_epoch=last_epoch)

    return scheduler_g, scheduler_d



def create_dataset(cfg, train=True, split=True, device='cuda:0'):
    """Create dataset based on cfguration."""
    clean_json = cfg['data_cfg']['train_clean_json'] if train else cfg['data_cfg']['valid_clean_json']
    noises_path = cfg['data_cfg']['train_noises_path'] if train else cfg['data_cfg']['valid_noises_path']
    shuffle = (cfg['env_setting']['num_gpus'] <= 1) if train else False
    pcs = cfg['training_cfg']['use_PCS400'] if train else False
    snr = cfg['data_cfg']['snr']

    return VCTKDemandDataset(
        clean_json=clean_json,
        noises_path=noises_path,
        snr=snr,
        sampling_rate=cfg['stft_cfg']['sampling_rate'],
        segment_size=cfg['training_cfg']['segment_size'],
        n_fft=cfg['stft_cfg']['n_fft'],
        hop_size=cfg['stft_cfg']['hop_size'],
        win_size=cfg['stft_cfg']['win_size'],
        compress_factor=cfg['model_cfg']['compress_factor'],
        split=split,
        n_cache_reuse=0,
        shuffle=shuffle,
        device=device,
        pcs=pcs
    )

def create_dataloader(dataset, cfg, train=True):
    """Create dataloader based on dataset and configuration."""
    if cfg['env_setting']['num_gpus'] > 1:
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(cfg['training_cfg']['training_epochs'])
        batch_size = (cfg['training_cfg']['batch_size'] // max(1, cfg['env_setting']['num_gpus'])) if train else 1
    else:
        sampler = None
        batch_size = cfg['training_cfg']['batch_size'] if train else 1
    num_workers = cfg['env_setting']['num_workers'] if train else 0

    return DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=(sampler is None) and train,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True
    )


def train(rank, args, cfg):
    num_gpus = cfg['env_setting']['num_gpus']
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    batch_size = cfg['training_cfg']['batch_size'] // max(1,cfg['env_setting']['num_gpus'])
    beta = cfg['training_cfg']['loss']['beta']
    if num_gpus >= 1:
        initialize_process_group(cfg, rank)
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        raise RuntimeError("Mamba needs GPU acceleration")

    generator = MambaSEUNet(cfg).to(device)
    discriminator = MetricDiscriminator().to(device)
    seld_net = SeldNet("freq", 2, "all", "onSpec").to(device)

    joint_model = JointModel(generator, discriminator, seld_net, device, n_fft, hop_size, win_size, compress_factor, cfg['stft_cfg']['sampling_rate'], cfg['training_cfg']['segment_size'])
    joint_model.to(device)

    if args.load_pretrained_se != 'None':
        g_path = args.load_pretrained_se
        do_path = args.load_pretrained_se.replace('g_', 'do_')
        state_dict_g = torch.load(g_path, map_location=device)
        state_dict_do = torch.load(do_path, map_location=device)

        if state_dict_g is not None and state_dict_do is not None:
            print("Loading pretrained SE successfully!")
            joint_model.generator.load_state_dict(state_dict_g['generator'], strict=False)
            joint_model.discriminator.load_state_dict(state_dict_do['discriminator'], strict=False)
            steps = 0
            last_epoch = 0
        else:
            print("No pretrained SE found!")
            steps = 0
            last_epoch = 0

        # Create optimizer and schedulers
        optimizers = setup_optimizers((generator, discriminator), cfg)
        load_optimizer_states(optimizers, state_dict_do)
        optim_g, optim_d = optimizers

        scheduler_g, scheduler_d = setup_schedulers(optimizers, cfg, last_epoch)
    else:
        #Continue training
        print("Continue training")
        state_dict_jm, steps, last_epoch = load_ckpts(args, device)
        joint_model.load_state_dict(state_dict_jm['joint_model'], strict=False)
        print("Load pretrained Joint Model successfully!")

        if num_gpus > 1 and torch.cuda.is_available():
            joint_model.generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
            joint_model.discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)
        # Create optimizer and schedulers
        optimizers = setup_optimizers((joint_model.generator, joint_model.discriminator), cfg)
        load_optimizer_states(optimizers, state_dict_jm)
        optim_g, optim_d = optimizers
        scheduler_g, scheduler_d = setup_schedulers(optimizers, cfg, last_epoch)

    optim_s = torch.optim.Adam(seld_net.parameters(), lr = 0.001)
    scheduler_s = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_s, verbose=True, patience = 5, factor = 0.2)
    evaluate = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()

    # # Create trainset and train_loader
    trainset = create_dataset(cfg, train=True, split=True, device=device)
    train_loader = create_dataloader(trainset, cfg, train=True)
    #print(len(train_loader))
    if rank == 0:
        validset = create_dataset(cfg, train=False, split=True, device=device)
        validation_loader = create_dataloader(validset, cfg, train=False)

    joint_model.train()
    run_name = "-10 dB QMULTIMIT"
    wandb.init(project='Distance-Estimation-Mamba-SEUnet-QMULTIMIT-10s', name=run_name)

    best_pesq, best_pesq_step = 0.0, 0
    best_mae, best_mae_step = 1000000.0, 0
    for epoch in range(max(0, last_epoch), cfg['training_cfg']['training_epochs']):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            clean_audio, clean_audio_segments, clean_mag_segments, clean_pha_segments,\
            clean_com_segments, noisy_mag_segments, noisy_pha_segments, norm_factor, true_dist = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes

            #print("train noisy_mag.shape: ", noisy_mag.shape)
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            clean_audio_segments = [torch.autograd.Variable(clean_audio_segments[i].to(device, non_blocking=True)) for i in range(len(clean_audio_segments))]
            clean_mag_segments = [torch.autograd.Variable(clean_mag_segments[i].to(device, non_blocking=True)) for i in range(len(clean_mag_segments))]
            clean_pha_segments = [torch.autograd.Variable(clean_pha_segments[i].to(device, non_blocking=True)) for i in range(len(clean_pha_segments))]
            clean_com_segments = [torch.autograd.Variable(clean_com_segments[i].to(device, non_blocking=True)) for i in range(len(clean_com_segments))]
            noisy_mag_segments = [torch.autograd.Variable(noisy_mag_segments[i].to(device, non_blocking=True)) for i in range(len(noisy_mag_segments))]
            noisy_pha_segments = [torch.autograd.Variable(noisy_pha_segments[i].to(device, non_blocking=True)) for i in range(len(noisy_pha_segments))]

            true_dist = torch.autograd.Variable(true_dist.to(device, non_blocking=True))
            norm_factor = torch.autograd.Variable(true_dist.to(device, non_blocking=True))
            one_labels = torch.ones(batch_size).to(device, non_blocking=True)

            audio_g, denoised_audio_segments, denoised_mag_segments, denoised_pha_segments, denoised_com_segments, time_dist, pred_dist \
            = joint_model(noisy_mag_segments, noisy_pha_segments, norm_factor)
            audio_list_r, audio_list_g = list(clean_audio.cpu().numpy()), list(audio_g.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g, cfg)


    #         # Discriminator
    #         # -------------------------------------------------------
            optim_d.zero_grad()
            loss_disc_all = 0.0

            for clean_mag, mag_g in zip(clean_mag_segments, denoised_mag_segments):
                metric_r = joint_model.discriminator(clean_mag, clean_mag)
                metric_g = joint_model.discriminator(clean_mag, mag_g.detach())

                loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())


                if batch_pesq_score is not None:
                    loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
                else:
                    print("No batch pesq score!")
                    loss_disc_g = 0

                loss_disc_all += loss_disc_r + loss_disc_g

            loss_disc_all.backward()
            optim_d.step()
            # ------------------------------------------------------- #

            # Generator
            # ------------------------------------------------------- #
            optim_g.zero_grad()
            loss_gen_all = 0.0
            # Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py
            for clean_mag, mag_g, clean_pha, pha_g, clean_com, com_g, clean_audio, audio_g in \
             zip(clean_mag_segments, denoised_mag_segments, clean_pha_segments, denoised_pha_segments, \
                 clean_com_segments, denoised_com_segments, clean_audio_segments, denoised_audio_segments):
                # L2 Magnitude Loss

                loss_mag = F.mse_loss(clean_mag, mag_g)

                # Anti-wrapping Phase Loss
                loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, cfg)
                loss_pha = loss_ip + loss_gd + loss_iaf
                # L2 Complex Loss

                loss_com = F.mse_loss(clean_com, com_g) * 2
                # Time Loss

                loss_time = F.l1_loss(clean_audio, audio_g)
                # Metric Loss
                metric_g = joint_model.discriminator(clean_mag, mag_g)
                #print("metric_g.shape", metric_g.shape)

                loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
                # Consistancy Loss
                _, _, rec_com = mag_phase_stft(audio_g, n_fft, hop_size, win_size, compress_factor, addeps=True)

                loss_con = F.mse_loss(com_g, rec_com) * 2

                loss_gen_all += (
                    loss_metric * cfg['training_cfg']['loss']['metric'] +
                    loss_mag * cfg['training_cfg']['loss']['magnitude'] +
                    loss_pha * cfg['training_cfg']['loss']['phase'] +
                    loss_com * cfg['training_cfg']['loss']['complex'] +
                    loss_time * cfg['training_cfg']['loss']['time'] +
                    loss_con * cfg['training_cfg']['loss']['consistancy']
                )

            # ------------------------------------------------------- #

            # seld_net
            optim_s.zero_grad()

            dist_loss = criterion(pred_dist, true_dist)
            timewise_loss = criterion(torch.mean(time_dist, dim = -1), true_dist)
            sde_loss = (dist_loss + timewise_loss)/2

            #Backward Propargation
            joint_loss = sde_loss * (1 - beta) + loss_gen_all * beta
            joint_loss.backward()

            optim_s.step()
            optim_g.step()

            scheduler_s.step(sde_loss)

            if rank == 0:
                # STDOUT logging
                if steps % cfg['env_setting']['stdout_interval'] == 0:
                    with torch.no_grad():
                        print(f'Training at step {steps}:\tSDE Loss: {sde_loss:.4f}\tSE Loss: {loss_gen_all:.4f}\tJoint Loss: {joint_loss:.4f}\t')

                        wandb.log({"train/gen_loss:": loss_gen_all,
                                   "train/disc_loss": loss_disc_all,
                                   "train/sde_loss" : sde_loss,
                                   "train/joint_loss": joint_loss})

                # Checkpointing
                if steps % cfg['env_setting']['checkpoint_interval'] == 0 and steps != 0:
                    exp_name = f"{args.exp_path}/jm_{steps:08d}.pth"
                    save_checkpoint(
                        exp_name,
                        {
                            'joint_model': joint_model.state_dict(),
                            'optim_g': optim_g.state_dict(),
                            'optim_d': optim_d.state_dict(),
                            'optim_s': optim_s.state_dict(),
                            'steps': steps,
                            'epoch': epoch
                        }
                    )


                # If NaN happend in training period, RaiseError
                if torch.isnan(loss_gen_all).any():
                    raise ValueError("NaN values found in loss_gen_all")

                #Validation
                if steps % cfg['env_setting']['validation_interval'] == 0:
                    # Validation
                    joint_model.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_mag_err_tot = 0
                    val_pha_err_tot = 0
                    val_com_err_tot = 0
                    with torch.no_grad():
                        sum_loss = 0.0
                        sum_loss_timewise = 0.0
                        sum_sde_loss = 0.0
                        mae = 0.0
                        for j, batch in enumerate(validation_loader):
                            clean_audio, clean_audio_segments, clean_mag_segments, clean_pha_segments,\
                            clean_com_segments, noisy_mag_segments, noisy_pha_segments, norm_factor, labels = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
                            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
                            clean_audio_segments = [torch.autograd.Variable(clean_audio_segments[i].to(device, non_blocking=True)) for i in range(len(clean_audio_segments))]
                            clean_mag_segments = [torch.autograd.Variable(clean_mag_segments[i].to(device, non_blocking=True)) for i in range(len(clean_mag_segments))]
                            clean_pha_segments = [torch.autograd.Variable(clean_pha_segments[i].to(device, non_blocking=True)) for i in range(len(clean_pha_segments))]
                            clean_com_segments = [torch.autograd.Variable(clean_com_segments[i].to(device, non_blocking=True)) for i in range(len(clean_com_segments))]
                            noisy_mag_segments = [torch.autograd.Variable(noisy_mag_segments[i].to(device, non_blocking=True)) for i in range(len(noisy_mag_segments))]
                            noisy_pha_segments = [torch.autograd.Variable(noisy_pha_segments[i].to(device, non_blocking=True)) for i in range(len(noisy_pha_segments))]
                            norm_factor = torch.autograd.Variable(norm_factor.to(device, non_blocking=True))
                            labels = torch.autograd.Variable(labels.to(device, non_blocking = True))

                            audio_g, denoised_audio_segments, denoised_mag_segments, denoised_pha_segments, denoised_com_segments, time_wise_distance, distance_est \
                    = joint_model(noisy_mag_segments, noisy_pha_segments, norm_factor)


                            dist_loss = criterion(distance_est, labels)
                            time_loss = criterion(torch.mean(time_wise_distance, dim = -1), labels)
                            sde_loss = (dist_loss + time_loss) / 2
                            # print("mae += evaluate(distance_est, labels)", distance_est.shape, labels.shape)
                            mae += evaluate(distance_est, labels)

                            sum_loss += dist_loss
                            sum_loss_timewise += time_loss
                            sum_sde_loss += sde_loss

                        sum_loss /= len(validation_loader)
                        sum_loss_timewise /= len(validation_loader)
                        sum_sde_loss /= len(validation_loader)
                        mae /= len(validation_loader)

                        scheduler_s.step(sum_sde_loss)

                        wandb.log({"val/loss": sum_loss})
                        wandb.log({"val/loss_timewise": sum_loss_timewise})
                        wandb.log({"val/sde_loss": sum_sde_loss})
                        wandb.log({"val/mae": mae})
                    if mae < best_mae:
                        best_mae = mae
                        best_mae_step = steps
                    print(f"Validation step {steps}: validation sde_loss: {sum_sde_loss}, validation mae: {mae}, best mae: {best_mae} at {best_mae_step}")
                    joint_model.train()
            steps += 1
        scheduler_g.step()
        scheduler_d.step()
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='exp')
    parser.add_argument('--exp_name', default='Mamba-SEUnet+SeldNet-10s')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--load_pretrained_se', default='None')

    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg['env_setting']['seed']
    num_gpus = cfg['env_setting']['num_gpus']
    available_gpus = torch.cuda.device_count()

    if num_gpus > available_gpus:
        warnings.warn(
            f"Warning: The actual number of available GPUs ({available_gpus}) is less than the .yaml config ({num_gpus}). Auto reset to num_gpu = {available_gpus}",
            UserWarning
        )
        cfg['env_setting']['num_gpus'] = available_gpus
        num_gpus = available_gpus


    initialize_seed(seed)
    args.exp_path = os.path.join(args.exp_folder, args.exp_name)
    build_env(args.config, 'config.yaml', args.exp_path)

    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_available_gpus}")
        print_gpu_info(num_available_gpus, cfg)
    else:
        print("CUDA is not available.")

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(args, cfg))
    else:
        train(0, args, cfg)

if __name__ == '__main__':
    main()

