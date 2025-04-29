import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa import STFT
import pandas as pd

class SeldNet(nn.Module):
    def __init__(self, kernels, n_grus, features_set, att_conf):
        super(SeldNet, self).__init__()
        self.n_fft = 512
        self.hop_length = 256
        self.nb_cnn2d_filt = 128
        self.pool_size = [8, 8, 2]
        self.rnn_size = [128, 128]
        self.fnn_size = 128
        self.segment = 160000
        self.kernels = kernels
        self.n_grus = n_grus
        self.features_set = features_set
        self.att_conf = att_conf

        # kernels "freq" [1, 3] - "time" [3, 1] - "square" [3, 3]
        if self.kernels == "freq":
            self.kernels = (1,3)
        elif self.kernels == "time":
            self.kernels = (3,1)
        elif self.kernels == "square":
            self.kernels = (3,3)
        else:
            raise ValueError

        self.STFT = STFT(n_fft=self.n_fft, hop_length=self.hop_length)

        # feature set "stft", "sincos", "all"
        num_frames = self.segment // self.hop_length + 1

        if self.features_set == "stft":
            self.data_in = [1, num_frames,  int(self.n_fft/2)]
        elif self.features_set == "sincos":
            self.data_in = [2, num_frames,  int(self.n_fft/2)]
        elif self.features_set == "all":
            self.data_in = [3, num_frames,  int(self.n_fft/2)]
        else:
            raise ValueError

        # ATTENTION MAP False, "onSpec", "onAll"
        if self.att_conf == "Nothing":
            pass
        elif self.att_conf == "onSpec":
            self.heatmap = nn.Sequential(
                nn.Conv2d(in_channels = self.data_in[0], out_channels = 16,
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(in_channels = 16, out_channels = 64,
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 1, padding = "same"),
                nn.Sigmoid()
            )
        elif self.att_conf == "onAll":
            self.heatmap = nn.Sequential(
                nn.Conv2d(in_channels = self.data_in[0], out_channels = 16,
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(in_channels = 16, out_channels = 64,
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels = self.data_in[0], kernel_size = 1, padding = "same"),
                nn.Sigmoid()
            )
        else:
            raise ValueError

        # First Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=self.data_in[0], out_channels=8, kernel_size=self.kernels, padding="same", bias = False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=8)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, self.pool_size[0]))
        self.pool1avg = nn.AvgPool2d(kernel_size=(1, self.pool_size[0]))

        # Second Convolutional layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=self.kernels, padding="same", bias = False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, self.pool_size[1]))
        self.pool2avg = nn.AvgPool2d(kernel_size=(1, self.pool_size[1]))

        # Third Convolutional layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=self.nb_cnn2d_filt, kernel_size=self.kernels, padding="same", bias = False)
        self.batch_norm3 = nn.BatchNorm2d(num_features=self.nb_cnn2d_filt)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, self.pool_size[2]))
        self.pool3avg = nn.AvgPool2d(kernel_size=(1, self.pool_size[2]))

        # GRUS 2, 1, 0
        if self.n_grus == 2:
            self.gru1 = nn.GRU(input_size=int(self.data_in[-1]* self.nb_cnn2d_filt / (self.pool_size[-3]*self.pool_size[-2]*self.pool_size[-1])), hidden_size=self.rnn_size[0], bidirectional=True, batch_first = True)
            self.gru2 = nn.GRU(input_size=self.rnn_size[0]*2, hidden_size=self.rnn_size[1], bidirectional=True, batch_first = True)
        elif self.n_grus == 1:
            self.gru1 = nn.GRU(input_size=int(self.data_in[-1]* self.nb_cnn2d_filt / (self.pool_size[-3]*self.pool_size[-2]*self.pool_size[-1])), hidden_size=self.rnn_size[1], bidirectional=True, batch_first = True)
        elif self.n_grus == 0:
            self.gru_linear1 = nn.Linear(in_features = int(self.data_in[-1]* self.nb_cnn2d_filt / (self.pool_size[-3]*self.pool_size[-2]*self.pool_size[-1])), out_features = self.rnn_size[0])
            self.gru_linear2 = nn.Linear(in_features=self.rnn_size[0], out_features=self.rnn_size[1]*2)
        else:
            raise ValueError

        self.fc1 = nn.Linear(in_features=self.rnn_size[1]*2, out_features=self.fnn_size)
        self.fc2 = nn.Linear(in_features=self.fnn_size, out_features = 1)

        self.final = nn.Linear(in_features = self.data_in[-2], out_features = 1)

    def normalize_tensor(self, x):
        mean = x.mean(dim = (2,3), keepdim = True)
        std = x.std(dim = (2,3), unbiased = False, keepdim = True)
        return torch.div((x - mean), std)

    def forward(self, x):
        # features extraction
        #print(x.shape)
        x_real, x_imm = self.STFT(x)
        b, c, t, f = x_real.size()
        magn = torch.sqrt(torch.pow(x_real, 2) + torch.pow(x_imm, 2))
        magn = torch.log(magn**2 + 1e-7)
        previous_magn = magn
        #print("magn: ", magn.shape)
        angles_cos = torch.cos(torch.angle(x_real + 1j*x_imm))
        angles_sin = torch.sin(torch.angle(x_real + 1j*x_imm))
        magn = magn[:,:,:,:-1]
        angles_cos = angles_cos[:,:,:,:-1]
        angles_sin = angles_sin[:,:,:,:-1]
        #print("sin cos: ", angles_cos.shape)
        # set up feature set
        if self.features_set == "stft":
            x = magn
        elif self.features_set == "sincos":
            x = torch.cat((angles_cos, angles_sin), dim = 1)
        elif self.features_set == "all":
            x = torch.cat((magn, angles_cos, angles_sin), dim = 1)
        else:
            raise ValueError
        #print("x: ", x.shape)
        # normalize features
        x = self.normalize_tensor(x)

        # computation of the heatmap
        if self.att_conf == "Nothing":
            pass
        else:
            hm = self.heatmap(x)
            if self.att_conf == "onSpec":
                magn = magn * hm
                x = torch.cat((magn, angles_cos, angles_sin), dim = 1)
                x = self.normalize_tensor(x)
            elif self.att_conf == "onAll":
                x = x * hm


        # convolutional layers
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.elu(x)
        x = self.pool1(x) + self.pool1avg(x)
        #print("conv layer1: ", x.shape)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.elu(x)
        x = self.pool2(x) + self.pool2avg(x)
        #print("conv layer2: ", x.shape)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.elu(x)
        x = self.pool3(x) + self.pool3avg(x)
        #print("conv layer3: ", x.shape)
        # recurrent layers (if any)
        x = x.permute(0, 2, 1, 3)
        #print("permute: ", x.shape)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        #print("reshape: ", x.shape)
        if self.n_grus == 2:
            x, _ = self.gru1(x)
            #print("gru1: ", x.shape)
            x, _ = self.gru2(x)
        elif self.n_grus == 1:
            x, _ = self.gru1(x)
            #print("gru1: ", x.shape)
        else:
            x = self.gru_linear1(x)
            x = self.gru_linear2(x)

        x = F.elu(self.fc1(x))
        #print("fc1: ", x.shape)
        x = F.elu(self.fc2(x))
        #print("fc2: ", x.shape)
        x = x.squeeze(2) # here [batch_size, time_bins]
        rnn = x
        x = self.final(x).squeeze(1)
        #print("final: ", x.shape)
        if self.att_conf == "Nothing":
            return x, rnn, previous_magn.detach(), None
        else:
            return x, rnn, previous_magn.detach(), hm.detach()
