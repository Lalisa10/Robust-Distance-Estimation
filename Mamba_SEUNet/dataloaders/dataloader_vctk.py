import os
import json
import random
import torch
import torch.utils.data
import librosa as lb
from ..models.stfts import mag_phase_stft, mag_phase_istft
from ..models.pcs400 import cal_pcs
from ..utils.util import calculate_snr

import numpy as np

def list_files_in_directory(directory_path):
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.wav'):   # only add .wav files
                files.append(os.path.join(root, filename))
    return files

def load_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def extract_identifier(file_path):
    return os.path.basename(file_path)

def get_clean_path_for_noisy(noisy_file_path, clean_path_dict):
    identifier = extract_identifier(noisy_file_path)
    return clean_path_dict.get(identifier, None)

def adjust_array_length(arr, target_length, pad_value=None):
    current_length = len(arr)

    if current_length > target_length:
        # Chọn ngẫu nhiên một đoạn con liên tiếp
        start_idx = np.random.randint(0, current_length - target_length + 1)
        return arr[start_idx:start_idx + target_length]

    elif current_length < target_length:
        # Tính số phần tử cần padding
        num_pad = target_length - current_length
        left_pad = np.random.randint(0, num_pad + 1)  # Chia số phần tử padding cho bên trái
        right_pad = num_pad - left_pad  # Phần còn lại cho bên phải

        # Tạo padding từ chính mảng hoặc giá trị mặc định
        if pad_value is None:
            left_values = np.random.choice(arr, left_pad) if left_pad > 0 else np.array([])
            right_values = np.random.choice(arr, right_pad) if right_pad > 0 else np.array([])
        else:
            left_values = np.full(left_pad, pad_value)
            right_values = np.full(right_pad, pad_value)

        return np.concatenate([left_values, arr, right_values])

    return arr  # Giữ nguyên nếu đã đúng độ dài

def get_noisy_audio(clean_audio, noises_path, list_all_noises, snr, sr = 16000):
    random_index_noise = np.random.randint(low = 0, high = len(list_all_noises))
    selected_noise_file = list_all_noises[random_index_noise]
    audio_noise, _ = lb.load(os.path.join(noises_path, selected_noise_file), sr = sr, mono = True, res_type = "kaiser_fast")

    audio_noise = adjust_array_length(audio_noise, len(clean_audio), 0)

    RMS_s = np.sqrt(np.mean(np.power(clean_audio,2)))
    if snr == "Random": # extractly randomly an SNR
        random_SNR = np.random.rand() * 50
        RMS_n = np.sqrt(np.power(RMS_s,2) / np.power(10, random_SNR/10))
    else:
        RMS_n = np.sqrt(np.power(RMS_s,2) / np.power(10, snr/10))

    RMS_n_current = np.sqrt(np.mean(np.power(audio_noise,2)))

    
    audio_noise = audio_noise * (RMS_n / RMS_n_current)

    noised_audio = clean_audio.squeeze() + audio_noise
    return noised_audio

class VCTKDemandDataset(torch.utils.data.Dataset):
    """
    Dataset for loading clean and noisy audio files.

    Args:
        clean_wavs_json (str): Directory containing clean audio files.
        noisy_wavs_json (str): Directory containing noisy audio files.
        audio_index_file (str): File containing audio indexes.
        sampling_rate (int, optional): Sampling rate of the audio files. Defaults to 16000.
        segment_size (int, optional): Size of the audio segments. Defaults to 32000.
        n_fft (int, optional): FFT size. Defaults to 400.
        hop_size (int, optional): Hop size. Defaults to 100.
        win_size (int, optional): Window size. Defaults to 400.
        compress_factor (float, optional): Magnitude compression factor. Defaults to 1.0.
        split (bool, optional): Whether to split the audio into segments. Defaults to True.
        n_cache_reuse (int, optional): Number of times to reuse cached audio. Defaults to 1.
        device (torch.device, optional): Target device. Defaults to None
        pcs (bool, optional): Use PCS in training period. Defaults to False
    """
    def __init__(
        self,
        clean_json,
        noises_path,
        snr = None,
        sampling_rate=16000,
        segment_size=32000,
        n_fft=400,
        hop_size=100,
        win_size=400,
        compress_factor=1.0,
        split=True,
        n_cache_reuse=1,
        shuffle=True,
        device=None,
        pcs=False
    ):

        self.clean_wavs_path = load_json_file( clean_json )
        self.noises_path = noises_path
        self.snr = snr
        random.seed(1234)

        # if shuffle:
        #     random.shuffle(self.noisy_wavs_path)
        self.clean_path_dict = {extract_identifier(clean_path): clean_path for clean_path in self.clean_wavs_path}

        self.list_all_noises = [f for f in os.listdir(self.noises_path) if os.path.isfile(os.path.join(self.noises_path, f))]
        
        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.split = split
        self.n_cache_reuse = n_cache_reuse

        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self._cache_ref_count = 0
        self.device = device
        self.pcs = pcs

    def get_segments(self, tensor, segment_size, split_dim):
        num_split = tensor.size(split_dim) // segment_size
        last_segment_size = tensor.size(split_dim) % segment_size
        assert(last_segment_size != 0)
        
        cutted_tensor = tensor.narrow(split_dim, 0, num_split * segment_size)
        
        segments = list(torch.split(cutted_tensor, segment_size, dim=split_dim))
        
        last_segment = tensor.narrow(split_dim, tensor.size(split_dim) - segment_size, segment_size)
        
        segments.append(last_segment)
        for i in range(len(segments)):
            segments[i] = segments[i].squeeze()
        return segments
    def __getitem__(self, index):
        """
        Get an audio sample by index.

        Args:
            index (int): Index of the audio sample.

        Returns:
            tuple: clean audio, clean magnitude, clean phase, clean complex, noisy magnitude, noisy phase
        """
        distance = None
        if self._cache_ref_count == 0:
            clean_path = self.clean_wavs_path[index]
            file_name = clean_path.split('/')[-1]
            distance = file_name.split('_')[2]
            distance = float(distance[:-1])
            assert(distance != None)
            distance = torch.tensor(distance).float()
            #clean_path = get_clean_path_for_noisy(noisy_path, self.clean_path_dict)
            clean_audio, _ = lb.load( clean_path, sr=self.sampling_rate, mono = True, res_type = "kaiser_fast")
            noisy_audio = get_noisy_audio(clean_audio, self.noises_path, self.list_all_noises, self.snr)
            if self.pcs == True:
                clean_audio = cal_pcs(clean_audio)
            self.cached_noisy_wav = noisy_audio
            self.cached_clean_wav = clean_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1

        clean_audio, noisy_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(noisy_audio)
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        assert clean_audio.size(1) == noisy_audio.size(1)

        clean_mag, clean_pha, clean_com = mag_phase_stft(clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)
        noisy_mag, noisy_pha, noisy_com = mag_phase_stft(noisy_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)

        clean_audio_segments = self.get_segments(clean_audio, self.segment_size, 1)
        clean_mag_segments = self.get_segments(clean_mag, 256, 2)
        clean_pha_segments = self.get_segments(clean_pha, 256, 2)
        clean_com_segments = self.get_segments(clean_com, 256, 2)
        noisy_mag_segments = self.get_segments(noisy_mag, 256, 2)
        noisy_pha_segments = self.get_segments(noisy_pha, 256, 2)

        return (clean_audio.squeeze(), clean_audio_segments, clean_mag_segments, clean_pha_segments, clean_com_segments, noisy_mag_segments, noisy_pha_segments, norm_factor, distance)

    def __len__(self):
        return len(self.clean_wavs_path)