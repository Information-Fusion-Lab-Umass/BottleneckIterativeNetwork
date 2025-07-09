import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

from src.fusion import FUSION_FUNCTION_FACTORY
from src.get_data import stft_size

import numpy as np


class SeparationModel(nn.Module):
    """
    Partially adapted from https://github.com/aispeech-lab/advr-avss/blob/master/model.py
    """
    def __init__(self,
                 audio_in_size: int,
                 audio_hidden_size: int,
                 audio_out_size: int,
                 audio_num_layers: int,
                 visual_in_size: int,
                 visual_hidden_size: int,
                 visual_out_size: int,
                 visual_num_layers: int,
                 fusion_function: str,
                 fusion_hidden_size: int = None,
                 dropout_rate: float = 0.1,
                 device = torch.device("cpu")
                 ):
        super(SeparationModel, self).__init__()
        # self.audio_encoder = nn.Conv1d(audio_in_size, audio_out_size, kernel_size=WIN_LEN,
        #                                stride = WIN_LEN // 2,
        #                                bias=False)
        # self.LN = nn.GroupNorm(1, audio_out_size, eps=1e-8)
        # self.BN = nn.Conv1d(audio_out_size, audio_hidden_size, 1)
        self.audio_encoder = AudioFeatNet(audio_size=audio_in_size)
        audio_out_size = audio_in_size * self.audio_encoder.last_filter

        if visual_num_layers > 0:
            self.visual_encoder = nn.LSTM(
                input_size=visual_in_size,
                hidden_size=visual_hidden_size,
                num_layers=visual_num_layers,
                bidirectional=True,
                batch_first=True
            )
        else:
            self.visual_encoder = nn.Sequential()

        self.drop = nn.Dropout(p=dropout_rate)

        self.fusion = FUSION_FUNCTION_FACTORY[fusion_function](fusion_hidden_size=fusion_hidden_size,
                                                               primary_dim=audio_out_size,
                                                               aux_dim=visual_out_size,
                                                               device=device,
                                                               mlp_type = "b",)
        if fusion_hidden_size is None:
            fusion_hidden_size = audio_out_size + visual_out_size
        if fusion_function == "hyperfuse":
            fusion_hidden_size = audio_out_size
        # self.separator = TCN_audio(audio_hidden_size, audio_hidden_size*4,
        #                            visual_dim=visual_hidden_size * 2,
        #                            layer=layer, stack=stack,
        #                            skip=SKIP,
        #                                   causal=False, dilated=True)

        self.separator = nn.LSTM(fusion_hidden_size, audio_in_size, num_layers=1, batch_first=True)
        # post-processing layers
        self.output = nn.Linear(audio_in_size, audio_in_size)
        torch.nn.init.xavier_uniform_(self.output.weight)
        # # speech decoder
        # self.decoder = nn.ConvTranspose1d(audio_out_size, 1, WIN_LEN, bias=False, stride=4)

    def central_params(self):
        return [
            {"params": self.fusion.parameters()},
            {"params": self.separator.parameters()},
            {"params": self.output.parameters()},
            {"params": self.decoder.parameters()},
        ]

    def forward(self, mixture, visual_input):
        visual_output = self.visual_encoder(visual_input).unsqueeze(1)
        B, C, T, D_a = mixture.shape
        audio_output = self.audio_encoder(mixture)

        if visual_output.shape[2] != T:
            upsampled_visual_output = F.interpolate(visual_output, (T, VISUAL_DIM), mode='nearest').reshape(-1, T, VISUAL_DIM)
        else:
            upsampled_visual_output = visual_output.squeeze(1)

        # audio-visual separator
        fused = self.fusion(audio_output, upsampled_visual_output)
        output = self.separator(fused)[0]
        # predict the speech features of target speaker
        # output = output.transpose(1, 2)
        # audio_output = audio_output.transpose(1,2)
        masks = torch.sigmoid(self.output(output))
        masked_output = torch.mul(mixture, masks.unsqueeze(1))  # B x C x T x D

        return masked_output


class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear((2 if bidirectional else 1) * hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        lengths = torch.tensor(np.array([lengths]), dtype=torch.int64)
        bs = x.size(0)

        #packed_sequence = pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
        packed_sequence = x.double()
        _, final_states = self.rnn(packed_sequence)

        if self.bidirectional:
            h = self.dropout(torch.cat((final_states[0][0], final_states[0][1]), dim=-1))
        else:
            h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class FusionNet(nn.Module):
    def __init__(self, a_only=False):
        super(FusionNet, self).__init__()
        if a_only:
            visfeat_size = 0
        else:
            visfeat_size = VISUAL_DIM
        self.lstm_conv = nn.LSTM(visfeat_size + 1028, stft_size // 2 + 1, num_layers=1, batch_first=True)
        self.time_distributed_1 = nn.Linear(in_features=stft_size // 2 + 1, out_features=stft_size // 2 + 1)
        torch.nn.init.xavier_uniform_(self.time_distributed_1.weight)
        self.activation = F.sigmoid

    def forward(self, input):
        x = self.lstm_conv(input)[0]
        pred_mask = self.activation(self.time_distributed_1(x))
        return pred_mask

class AudioFeatNet(nn.Module):

    def __init__(self, audio_size: int,
                 num_conv=5, kernel_size=5, filters=64, last_filter=4, dilation=True, batch_norm=True,
                 fc_layers=0, lstm_layers=0, lr=0.0003):
        super(AudioFeatNet, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.last_filter = last_filter
        self.batch_norm = batch_norm
        self.dilation = dilation
        self.num_conv = num_conv
        self.lstm_layers = lstm_layers
        self.fc_layers = fc_layers
        self.lr = lr
        self.embed_size = audio_size

        if batch_norm:
            setattr(self, "bn0", nn.BatchNorm2d(1))
        for i in range(num_conv):
            if i == 0:
                inp_filter = 1
                out_filter = self.filters
            else:
                inp_filter, out_filter = self.filters, self.filters
            if self.dilation:
                dilation_size = 2 ** i
                padding = (self.kernel_size - 1) * dilation_size
            else:
                padding = self.kernel_size - 1
                dilation_size = 1
            setattr(self, "conv{}".format(i + 1),
                    nn.Conv2d(inp_filter, out_filter, (self.kernel_size, self.kernel_size), padding=padding // 2,
                              dilation=dilation_size))
            if batch_norm:
                setattr(self, "bn{}".format(i + 1), nn.BatchNorm2d(out_filter))
        if num_conv == 0:
            inp_filter = 2
        else:
            inp_filter = self.filters
        self.convf = nn.Conv2d(inp_filter, self.last_filter, (1, 1), padding=0, dilation=(1, 1))
        if batch_norm:
            self.bn_last = nn.BatchNorm2d(self.last_filter)
        last_conv = True
        for i in range(lstm_layers):
            if i == 0 and not last_conv and num_conv == 0:
                input_size = 2 * self.embed_size
            elif last_conv and num_conv == 0:
                input_size = self.last_filter * self.embed_size
            elif not last_conv and num_conv != 0:
                input_size = self.filters * self.embed_size
            elif i == 0 and last_conv:
                input_size = self.last_filter * self.embed_size
            else:
                input_size = self.embed_size
            setattr(self, "lstm{}".format(i + 1), nn.LSTM(input_size, self.embed_size))

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, x):
        _, _, num_aud_feat, _ = x.shape
        if self.batch_norm:
            x = getattr(self, "bn0")(x)
        for i in range(self.num_conv):
            x = getattr(self, "conv{}".format(i + 1))(x)
            if self.batch_norm:
                x = getattr(self, "bn{}".format(i + 1))(x)
            x = F.relu(x)
        x = self.convf(x)
        if self.batch_norm:
            x = self.bn_last(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1, 3).reshape(-1, num_aud_feat, self.embed_size * self.last_filter)
        for i in range(self.lstm_layers):
            if i == 0:
                x = x.transpose(1, 0)
            getattr(self, "lstm{}".format(i + 1)).flatten_parameters()
            x = getattr(self, "lstm{}".format(i + 1))(x)
            x = x[0]
        if self.lstm_layers > 0:
            x = x.transpose(1, 0)
        return x

# Parameters for SS model
WIN_LEN = 16
layer = 8
stack = 3
visual_layer = 4
visual_stack = 1
MODAL_FUSION = 'CF' # CF or DCF
FUSION_POSITION = '8' # '0','8','16'
VISUAL_DIM = 512
SKIP = True

class TCN_audio(nn.Module):
    """
    Source: https://github.com/aispeech-lab/advr-avss/blob/master/module.py
    """
    def __init__(self,
                 BN_dim, hidden_dim,
                 visual_dim,
                 layer, stack, num_spk=2, kernel=3, skip=True,
                 causal=True, dilated=True):
        super(TCN_audio, self).__init__()
        # the input is a sequence of features of shape (B, N, L)
        self.receptive_field = 0
        self.dilated = dilated
        self.layer = layer
        self.stack = stack
        self.skip = skip
        self.causal = causal
        # if not self.causal:
        #     self.visual_dim = VISUAL_DIM * 2
        # else:
        self.visual_dim = visual_dim
        # the method of deep concatenate fusion (DCF) use target and other speaker's visual information
        if MODAL_FUSION == 'DCF':
            self.fc = nn.Linear(128 + num_spk * self.visual_dim, 128, bias=True)
        # the method of concatenate fusion (CF) use target speaker's visual information only
        if MODAL_FUSION == 'CF':
            self.fc = nn.Linear(BN_dim + self.visual_dim, BN_dim, bias=True)
        self.TCN = nn.ModuleList([])
        # TCN module that be used to process audio
        base = 2
        for s in range(self.stack):
            for i in range(self.layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=base**i, padding=base**i, skip=self.skip, causal=self.causal))
                else:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=self.skip, causal=self.causal))
                if i == 0 and s == 0:
                    self.receptive_field = self.receptive_field + kernel
                else:
                    if self.dilated:
                        self.receptive_field = self.receptive_field + (kernel - 1) * base**i
                    else:
                        self.receptive_field = self.receptive_field + (kernel - 1)
        print("Receptive field in audio path: {:3d} frames.".format(self.receptive_field))

    def forward(self, input, query, num_spk, CACHE_audio=None):
        # input shape: (B, N, L)
        #  select the multi-modal fusion position
        fusion_position = [FUSION_POSITION]
        output = input

        skip_connection = 0.
        for i in range(len(self.TCN)):
            if str(i) in fusion_position:
                if MODAL_FUSION == 'CF':
                    output = torch.cat((output, query), dim=1)
                    output = output.transpose(1, 2)
                    output = self.fc(output)
                    output = output.transpose(1, 2)
                if MODAL_FUSION == 'DCF':
                    shape = output.shape
                    output = output.view(-1, num_spk, shape[1], shape[2])
                    qshape = query.shape
                    query = query.view(-1, num_spk, qshape[1], qshape[2]) # shape:[B,num_spk,N,L]
                    if num_spk == 1:
                        output = torch.cat((output, query), dim=2)
                        output = output.view(-1, shape[1] + qshape[1], shape[2])
                    if num_spk == 2:
                        output0 = torch.cat((output[:,0,:,:],query[:,0,:,:],query[:,1,:,:]), dim=1)
                        output1 = torch.cat((output[:,1,:,:],query[:,1,:,:],query[:,0,:,:]), dim=1)
                        output = torch.cat((output0.unsqueeze(1), output1.unsqueeze(1)), dim=1)
                        output = output.view(-1, shape[1] + 2*qshape[1], shape[2])
                    if num_spk == 3:
                        output0 = torch.cat((output[:,0,:,:],query[:,0,:,:],query[:,1,:,:]+query[:,2,:,:]), dim=1)
                        output1 = torch.cat((output[:,1,:,:],query[:,1,:,:],query[:,0,:,:]+query[:,2,:,:]), dim=1)
                        output2 = torch.cat((output[:,2,:,:],query[:,2,:,:],query[:,0,:,:]+query[:,1,:,:]), dim=1)
                        output = torch.cat((output0.unsqueeze(1), output1.unsqueeze(1), output2.unsqueeze(1)), dim=1)
                        output = output.view(-1, shape[1] + 2*qshape[1], shape[2])
                    output = output.transpose(1, 2)
                    output = self.fc(output)
                    output = output.transpose(1, 2)

            residual, skip = self.TCN[i](output) # causal=False, causal=True and inference=False
            output = output + residual
            skip_connection = skip_connection + skip

        return output, CACHE_audio


class cLN(nn.Module):
    def __init__(self, dimension, eps = 1e-8, trainable=True):
        super(cLN, self).__init__()
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        batch_size, channel, time_step = input.size(0), input.size(1), input.size(2)
        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class DepthConv1d(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        super(DepthConv1d, self).__init__()
        self.causal = causal
        self.skip = skip
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding

        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
                                     groups=hidden_channel, padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        # now, cLN, BN and LN all can meet streaming inference strategy, but BN and LN are simple

        self.reg1 = cLN(hidden_channel, eps=1e-08)
        self.reg2 = cLN(hidden_channel, eps=1e-08)

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input, cache_last=None):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual
