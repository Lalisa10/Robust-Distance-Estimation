import yaml
import torch
import os
import shutil
import glob
from torch.distributed import init_process_group
from scipy.io import wavfile
import numpy as np

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def initialize_seed(seed):
    """Initialize the random seed for both CPU and GPU."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def print_gpu_info(num_gpus, cfg):
    """Print information about available GPUs and batch size per GPU."""
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
        print('Batch size per GPU:', int(cfg['training_cfg']['batch_size'] / num_gpus))

def initialize_process_group(cfg, rank):
    """Initialize the process group for distributed training."""
    init_process_group(
        backend=cfg['env_setting']['dist_cfg']['dist_backend'],
        init_method=cfg['env_setting']['dist_cfg']['dist_url'],
        world_size=cfg['env_setting']['dist_cfg']['world_size'] * cfg['env_setting']['num_gpus'],
        rank=rank
    )

def log_model_info(rank, model, exp_path):
    """Log model information and create necessary directories."""
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("Generator Parameters :", num_params)
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'logs'), exist_ok=True)
    print("checkpoints directory :", exp_path)

def load_ckpts(args, device, retrain = False):
    """Load checkpoints if available."""
    if os.path.isdir(args.exp_path):
        cp_jm = scan_checkpoint(args.exp_path, 'jm_')
        if cp_jm is None:
            return None, None, 0, -1
        state_dict = load_checkpoint(cp_jm, device)
        if not retrain:
            return state_dict, state_dict['steps'] + 1, state_dict['epoch']
        else: return state_dict, 0, 0
    return None, None, 0, -1

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????' + '.pth')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def build_env(config, config_name, exp_path):
    os.makedirs(exp_path, exist_ok=True)
    t_path = os.path.join(exp_path, config_name)
    if config != t_path:
        shutil.copyfile(config, t_path)

def load_optimizer_states(optimizers, state_dict):
    """Load optimizer states from checkpoint."""
    if state_dict is not None:
        optim_g, optim_d = optimizers
        optim_g.load_state_dict(state_dict['optim_g'])
        optim_d.load_state_dict(state_dict['optim_d'])

def export_to_wav(audio_data, sample_rate, output_file, library="scipy", normalize=True):
    """
    Xuất numpy array ra file .wav.
    
    Tham số:
        audio_data (np.ndarray): Dữ liệu âm thanh (shape [n_samples] cho mono hoặc [n_samples, n_channels] cho stereo).
        sample_rate (int): Tần số lấy mẫu (Hz).
        output_file (str): Đường dẫn file đầu ra (ví dụ: 'output.wav').
        library (str): Thư viện sử dụng ('scipy' hoặc 'soundfile').
        normalize (bool): Tự động chuẩn hóa dữ liệu về [-1, 1] nếu dùng 'soundfile'.
    """
    import numpy as np
    # Kiểm tra đầu vào
    if not isinstance(audio_data, np.ndarray):
        raise ValueError("audio_data phải là numpy array")
    if sample_rate <= 0:
        raise ValueError("sample_rate phải > 0")
    if not output_file.endswith(".wav"):
        output_file += ".wav"

    # Xử lý dữ liệu theo thư viện
    if library == "scipy":
        
        # Chuyển đổi sang int16 nếu là float
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
        
        wavfile.write(output_file, sample_rate, audio_data)

    elif library == "soundfile":
        import soundfile as sf
        
        # Tự động chuẩn hóa
        if normalize and (np.max(np.abs(audio_data)) > 1.0):
            audio_data = audio_data / np.max(np.abs(audio_data))
            
        sf.write(output_file, audio_data, sample_rate)

    else:
        raise ValueError("Thư viện không hỗ trợ. Chọn 'scipy' hoặc 'soundfile'")

def calculate_snr(clean_audio: np.ndarray, noisy_audio: np.ndarray) -> float:
    """
    Tính SNR (Signal-to-Noise Ratio) trong đơn vị dB.
    
    Args:
        clean_audio (np.ndarray): Tín hiệu gốc không nhiễu, shape (samples,) hoặc (samples, channels).
        noisy_audio (np.ndarray): Tín hiệu bị nhiễu, cùng shape với clean_audio.
        
    Returns:
        float: Giá trị SNR tính bằng decibel (dB).
        
    Raises:
        ValueError: Nếu clean_audio và noisy_audio không cùng shape.
    """
    # Kiểm tra shape
    if clean_audio.shape != noisy_audio.shape:
        raise ValueError("clean_audio và noisy_audio phải có cùng shape")
    
    # Tính noise (nhiễu) = noisy_audio - clean_audio
    noise = noisy_audio - clean_audio
    
    # Tính công suất tín hiệu và nhiễu (power = mean(signal^2))
    signal_power = np.mean(clean_audio.astype(np.float32) ** 2)
    noise_power = np.mean(noise.astype(np.float32) ** 2)
    
    # Tránh chia cho 0
    if noise_power <= 1e-10:  # Ngưỡng để tránh giá trị cực nhỏ
        raise ValueError("Noise power quá nhỏ, SNR tiến tới vô cùng")
    
    # Công thức SNR (dB)
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db
