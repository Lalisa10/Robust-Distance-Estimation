
import warnings
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
from Mamba_SEUNet.utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed,
    print_gpu_info, log_model_info, initialize_process_group,
)
torch.backends.cudnn.benchmark = True

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
    num_workers = cfg['env_setting']['num_workers']

    return DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=(sampler is None) and train,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True
    )

def main():
    cfg = load_config("config.yaml")
    seed = cfg['env_setting']['seed']
    num_gpus = cfg['env_setting']['num_gpus']
    available_gpus = torch.cuda.device_count()
    train_dataset = create_dataset(cfg, train=True, )
    train_dataloader = create_dataloader(train_dataset, cfg, train=True)
    clean_audio, clean_audio_segments, clean_mag_segments, clean_pha_segments,\
            clean_com_segments, noisy_mag_segments, noisy_pha_segments, norm_factor, true_dist = next(iter(train_dataloader))
    
    print(clean_audio)

if __name__ == "__main__":
    main()