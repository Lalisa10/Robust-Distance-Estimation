import torch
from Mamba_SEUNet.models.stfts import mag_phase_istft

class JointModel(torch.nn.Module):
    def __init__(self,
        generator,
        discriminator,
        seld_net,
        device,
        n_fft,
        hop_size,
        win_size,
        compress_factor,
        sampling_rate,
        segment_size,
        ):
        super(JointModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.seld_net = seld_net
        self.device = device
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.audio_length = 160000
        self.stft_length = 1334
        self.segment_stft = 256

    def forward(self, noisy_mag_segments, noisy_pha_segments, norm_factor):
        denoised_segments = []
        audio_segments = []
        denoised_mag_segments, denoised_pha_segments, denoised_com_segments = [], [], []
        for i in range(len(noisy_mag_segments)):
            noisy_mag = noisy_mag_segments[i]
            noisy_pha = noisy_pha_segments[i]
            denoised_mag, denoised_pha, denoised_com = self.generator(noisy_mag, noisy_pha)
            audio_g = mag_phase_istft(denoised_mag, denoised_pha, self.n_fft, self.hop_size, self.win_size, self.compress_factor)

            if i == len(noisy_mag_segments) - 1: # last_segment
                last_segment_length = self.audio_length % self.segment_size
                last_stft_length = self.stft_length % self.segment_stft

                if last_segment_length != 0:
                    last_audio_g = audio_g[:, -last_segment_length:]

                    audio_segments.append(last_audio_g)
                else:
                    raise ValueError
            else:
                audio_segments.append(audio_g)
            denoised_segments.append(audio_g)
            denoised_mag_segments.append(denoised_mag)
            denoised_pha_segments.append(denoised_pha)
            denoised_com_segments.append(denoised_com)

        audio_g = torch.cat(audio_segments, dim=1)
        audio_g /= norm_factor
        distance, timewise_distance, _, _ = self.seld_net(audio_g)

        return audio_g, denoised_segments, denoised_mag_segments, denoised_pha_segments, denoised_com_segments, timewise_distance, distance
