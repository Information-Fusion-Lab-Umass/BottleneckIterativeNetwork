'''
Adapted from: https://raw.githubusercontent.com/JusperLee/AV-ConvTasNet/main/model/av_model.py
'''
import json
import sys
from typing import Dict

import torch.nn as nn
import torch
import math
import numpy as np

# ----------Basic Part-------------
import torchaudio.transforms

from src.fusion import hyperfuse
from src.util import istft

from speechbrain.pretrained import HIFIGAN
from speechbrain.lobes.models.HifiGAN import mel_spectogram

class Conv1D(nn.Conv1d):
    '''
       Applies a 1D convolution over an input signal composed of several input planes.
    '''

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
            input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True,
           this module has learnable per-element affine parameters
           initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim):
    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    else:
        return nn.BatchNorm1d(dim)


# ----------Audio Part-------------


class Encoder(nn.Module):
    '''
       Audio Encoder
       in_channels: Audio in_Channels is 1
       out_channels: Encoder part output's channels
       kernel_size: Conv1D's kernel size
       stride: Conv1D's stride size
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, non_negative: bool = True, bias=True):
        super(Encoder, self).__init__()
        self.conv = Conv1D(in_channels, out_channels,
                           kernel_size, stride=stride, bias=bias)
        self.non_negative = non_negative
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
           x: [B, T]
           out: [B, N, T]
        '''
        x = self.conv(x)
        if self.non_negative:
            x = self.relu(x)
        return x


class Decoder(nn.ConvTranspose1d):
    '''
        Decoder
        This module can be seen as the gradient of Conv1d with respect to its input.
        It is also known as a fractionally-strided convolution
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)

        return x


class Audio_1DConv(nn.Module):
    '''
       Audio part 1-D Conv Block
       in_channels: Encoder's output channels
       out_channels: 1DConv output channels
       b_conv: the B_conv channels
       sc_conv: the skip-connection channels
       kernel_size: the depthwise conv kernel size
       dilation: the depthwise conv dilation
       norm: 1D Conv normalization's type
       causal: Two choice(causal or noncausal)
       skip_con: Whether to use skip connection
    '''

    def __init__(self,
                 in_channels=256,
                 out_channels=512,
                 b_conv=256,
                 sc_conv=256,
                 kernel_size=3,
                 dilation=1,
                 norm='gln',
                 causal=False,
                 skip_con=False,
                 bias=True):
        super(Audio_1DConv, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1, 1, bias=bias)
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm, out_channels)
        self.pad = (dilation * (kernel_size - 1)
                    ) // 2 if not causal else (dilation * (kernel_size - 1))
        self.dconv = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               padding=self.pad, dilation=dilation, groups=out_channels)
        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm, out_channels)
        self.B_conv = nn.Conv1d(out_channels, b_conv, 1)
        self.Sc_conv = nn.Conv1d(out_channels, sc_conv, 1)
        self.causal = causal
        self.skip_con = skip_con

    def forward(self, x):
        '''
           x: [B, N, T]
           out: [B, N, T]
        '''
        # x: [B, N, T]
        out = self.conv1x1(x)
        out = self.prelu1(out)
        out = self.norm1(out)
        out = self.dconv(out)
        if self.causal:
            out = out[:, :, :-self.pad]
        out = self.prelu2(self.norm2(out))
        if self.skip_con:
            skip = self.Sc_conv(out)
            B = self.B_conv(out)
            # [B, N, T]
            return skip, B + x
        else:
            B = self.B_conv(out)
            # [B, N, T]
            return B + x


class Audio_Sequential(nn.Module):
    def __init__(self, repeats, blocks,
                 in_channels=256,
                 out_channels=512,
                 b_conv=256,
                 sc_conv=256,
                 kernel_size=3,
                 norm='gln',
                 causal=False,
                 skip_con=False):
        super(Audio_Sequential, self).__init__()
        self.lists = nn.ModuleList([])
        self.skip_con = skip_con
        for r in range(repeats):
            for b in range(blocks):
                self.lists.append(Audio_1DConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    b_conv=b_conv,
                    sc_conv=sc_conv,
                    kernel_size=kernel_size,
                    dilation=(2 ** b),
                    norm=norm,
                    causal=causal,
                    skip_con=skip_con))

    def forward(self, x):
        '''
           x: [B, N, T]
           out: [B, N, T]
        '''
        if self.skip_con:
            skip_connection = 0
            for i in range(len(self.lists)):
                skip, out = self.lists[i](x)
                x = out
                skip_connection += skip
            return skip_connection
        else:
            for i in range(len(self.lists)):
                out = self.lists[i](x)
                x = out
            return x


class Video_1Dconv(nn.Module):
    """
    video part 1-D Conv Block
    in_channels: video Encoder output channels
    conv_channels: dconv channels
    kernel_size: the depthwise conv kernel size
    dilation: the depthwise conv dilation
    residual: Whether to use residual connection
    skip_con: Whether to use skip connection
    first_block: first block, not residual
    """

    def __init__(self,
                 in_channels,
                 conv_channels,
                 kernel_size,
                 dilation=1,
                 residual=True,
                 skip_con=True,
                 first_block=True
                 ):
        super(Video_1Dconv, self).__init__()
        self.first_block = first_block
        # first block, not residual
        self.residual = residual and not first_block
        self.bn = nn.BatchNorm1d(in_channels) if not first_block else None
        self.relu = nn.ReLU() if not first_block else None
        self.dconv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            dilation=dilation,
            padding=(dilation * (kernel_size - 1)) // 2,
            bias=True)
        self.bconv = nn.Conv1d(in_channels, conv_channels, 1)
        self.sconv = nn.Conv1d(in_channels, conv_channels, 1)
        self.skip_con = skip_con

    def forward(self, x):
        '''
           x: [B, N, T]
           out: [B, N, T]
        '''
        if not self.first_block:
            y = self.bn(self.relu(x))
            y = self.dconv(y)
        else:
            y = self.dconv(x)
        # skip connection
        if self.skip_con:
            skip = self.sconv(y)
            if self.residual:
                y = y + x
                return skip, y
            else:
                return skip, y
        else:
            y = self.bconv(y)
            if self.residual:
                y = y + x
                return y
            else:
                return y


class Video_2Dconv(nn.Module):
    """
    video part 1-D Conv Block
    in_channels: video Encoder output channels
    conv_channels: dconv channels
    kernel_size: the depthwise conv kernel size
    dilation: the depthwise conv dilation
    residual: Whether to use residual connection
    skip_con: Whether to use skip connection
    first_block: first block, not residual
    """

    def __init__(self,
                 in_channels,
                 conv_channels,
                 kernel_size,
                 dilation=1,
                 residual=True,
                 skip_con=True,
                 first_block=True
                 ):
        super(Video_2Dconv, self).__init__()
        self.first_block = first_block
        # first block, not residual
        self.residual = residual and not first_block
        self.bn = nn.BatchNorm1d(in_channels) if not first_block else None
        self.relu = nn.ReLU() if not first_block else None
        self.dconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            dilation=dilation,
            padding=(dilation * (kernel_size - 1)) // 2,
            bias=True)
        self.bconv = nn.Conv2d(in_channels, conv_channels, 1)
        self.sconv = nn.Conv2d(in_channels, conv_channels, 1)
        self.skip_con = skip_con

    def forward(self, x):
        '''
           x: [B, N, T]
           out: [B, N, T]
        '''
        if not self.first_block:
            y = self.bn(self.relu(x))
            y = self.dconv(y)
        else:
            y = self.dconv(x)
        # skip connection
        if self.skip_con:
            skip = self.sconv(y)
            if self.residual:
                y = y + x
                return skip, y
            else:
                return skip, y
        else:
            y = self.bconv(y)
            if self.residual:
                y = y + x
                return y
            else:
                return y


class Video_Sequential_1D(nn.Module):
    """
    All the Video Part
    in_channels: front3D part in_channels
    out_channels: Video Conv1D part out_channels
    kernel_size: the kernel size of Video Conv1D
    skip_con: skip connection
    repeat: Conv1D repeats
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 skip_con=True,
                 repeat=5):
        super(Video_Sequential_1D, self).__init__()
        self.conv1d_list = nn.ModuleList([])
        self.skip_con = skip_con
        for i in range(repeat):
            in_channels = out_channels if i else in_channels
            self.conv1d_list.append(
                Video_1Dconv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    skip_con=skip_con,
                    residual=True,
                    first_block=(i == 0)))

    def forward(self, x):
        '''
           x: [B, N, T]
           out: [B, N, T]
        '''
        if self.skip_con:
            skip_connection = 0
            for i in range(len(self.conv1d_list)):
                skip, out = self.conv1d_list[i](x)
                x = out
                skip_connection += skip
            return skip_connection
        else:
            for i in range(len(self.conv1d_list)):
                out = self.conv1d_list[i](x)
                x = out
            return x


class VideoSequential2D(nn.Module):
    """
    All the Video Part
    in_channels: front3D part in_channels
    out_channels: Video Conv1D part out_channels
    kernel_size: the kernel size of Video Conv1D
    skip_con: skip connection
    repeat: Conv1D repeats
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 skip_con=True,
                 repeat=5):
        super(VideoSequential2D, self).__init__()
        self.conv1d_list = nn.ModuleList([])
        self.skip_con = skip_con
        for i in range(repeat):
            in_channels = out_channels if i else in_channels
            self.conv1d_list.append(
                Video_2Dconv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    skip_con=skip_con,
                    residual=True,
                    first_block=(i == 0)))

    def forward(self, x):
        '''
           x: [B, N, T]
           out: [B, N, T]
        '''
        if self.skip_con:
            skip_connection = 0
            for i in range(len(self.conv1d_list)):
                skip, out = self.conv1d_list[i](x)
                x = out
                skip_connection += skip
            return skip_connection
        else:
            for i in range(len(self.conv1d_list)):
                out = self.conv1d_list[i](x)
                x = out
            return x


class Concat(nn.Module):
    """
    Audio and Visual Concatenated Part
    audio_channels: Audio Part Channels
    video_channels: Video Part Channels
    out_channels: Concat Net channels
    """

    def __init__(self, audio_channels, video_channels, out_channels):
        super(Concat, self).__init__()
        self.audio_channels = audio_channels
        self.video_channels = video_channels
        # project
        self.conv1d = nn.Conv1d(audio_channels + video_channels, out_channels, 1)

    def forward(self, a, v):
        """
        a: audio features, N x A x Ta
        v: video features, N x V x Tv
        """
        if a.size(1) != self.audio_channels or v.size(1) != self.video_channels:
            raise RuntimeError("Dimention mismatch for audio/video features, "
                               "{:d}/{:d} vs {:d}/{:d}".format(
                a.size(1), v.size(1), self.audio_channels,
                self.video_channels))
        # up-sample video features
        v = torch.nn.functional.interpolate(v, size=a.size(-1))
        # concat: n x (A+V) x Ta
        y = torch.cat([a, v], dim=1)
        # conv1d
        return self.conv1d(y)


class Encoder_STFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, non_negative: bool = True, type="stft",
                 sample_rate=16000):
        super(Encoder_STFT, self).__init__()
        self.type = type
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.non_negative = non_negative

    def forward(self, x):
        if self.type == "stft":
            out = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, center=True,
                             window=torch.hann_window(self.win_length).to(x.device),
                             return_complex=True)
            out_real = torch.view_as_real(out)[:, :, :, 0]
            noisy_phase = torch.view_as_real(out)[:, :, :, 1]
            if self.non_negative:
                out_real = torch.abs(out_real)
            return out_real, noisy_phase
        elif self.type == "mel":
            #x = torch.nn.functional.pad(x, (int((1024-256)/2), int((1024-256)/2)), mode='reflect')
            x = mel_spectogram(
                sample_rate=16000,
                hop_length=256,
                win_length=1024,
                n_fft=1024,
                n_mels=80,
                f_min=0.0,
                f_max=8000,
                power=1,
                normalized=False,
                norm="slaney",
                mel_scale="slaney",
                compression=True,
                audio = x,
            )
            return x, None
        else:
            raise NotImplementedError(f"{self.type} is not supported")


class Decoder_STFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super(Decoder_STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x, noisy_phase=None, length=None):
        if noisy_phase != None:
            x = x + 1.j * noisy_phase
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                        center=True, length=length,
                        window=torch.hann_window(self.win_length).to(x.device))
        return x


class Decoder_HIFI_GAN(nn.Module):
    def __init__(self, device="cpu"):
        super(Decoder_HIFI_GAN, self).__init__()
        self.generator = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                                              savedir="pretrained_models",
                                              run_opts={"device": device})

    def forward(self, x):
        out = self.generator.decode_batch(x)
        out = out.squeeze(1)
        return out


class AV_ConvTasnet(nn.Module):
    """
    Audio and Visual Speech Separation
    Audio Part
        N	Number of ﬁlters in autoencoder
        L	Length of the ﬁlters (in samples)
        B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
        SC  Number of channels in skip-connection paths’ 1 × 1-conv blocks
        H	Number of channels in convolutional blocks
        P	Kernel size in convolutional blocks
        X	Number of convolutional blocks in each repeat
    Video Part
        E   Number of ﬁlters in video autoencoder
        V   Number of channels in convolutional blocks
        K   Kernel size in convolutional blocks
        D   Number of repeats
    Concat Part
        F   Number of channels in convolutional blocks
    Other Setting
        R	Number of all repeats
        skip_con	Skip Connection
        audio_index     Number repeats of audio part
        norm    Normaliztion type
        causal  Two choice(causal or noncausal)
    """

    def __init__(
            self,
            # audio conf
            N=256,
            L=40,
            B=256,
            Sc=256,
            H=512,
            P=3,
            X=8,  # number of blocks/stacks within 1 repeat Conv
            audio_encoder_non_negative: bool = True,
            audio_encoder_type: str = "conv1d",
            audio_encoder_bias: bool = True,
            audio_decoder_type: str = "conv1d",  # conv1d, stft, hifi_gan
            audio_decoder_bias: bool = True,
            hifi_gan_config: str = None,
            hifi_gan_pretrained_checkpoint: str = None,
            hop_length=160,
            n_fft=511,
            win_length=400,
            # video conf
            E=256,
            V=256,
            K=3,
            D=5,
            # fusion index
            F=256,
            # other
            R=4,  # total number of repeats for audio
            skip_con=False,
            audio_index=2,
            norm="gln",
            causal=False,
            fusion_function="concat",
            fusion_hidden_size=None,
            mask_activation_function: str = "relu",
            device_type="cpu",
    ):
        super(AV_ConvTasnet, self).__init__()
        self.audio_encoder_type = audio_encoder_type

        self.video = VideoSequential2D(E, V, K, skip_con=skip_con, repeat=D)
        if audio_encoder_type == "conv1d":
            # n x S > n x N x T
            self.encoder = Encoder(1, N, L, stride=L // 2, non_negative=audio_encoder_non_negative, bias=audio_encoder_bias)
        else:
            self.encoder = Encoder_STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                        non_negative=audio_encoder_non_negative,
                                        type=audio_encoder_type)

        # before repeat blocks, always cLN
        self.cln = CumulativeLayerNorm(N)
        # n x N x T > n x B x T
        self.conv1x1 = Conv1D(N, B, 1)
        # repeat blocks
        # n x B x T => n x B x T
        self.skip_con = skip_con
        self.audio_conv = Audio_Sequential(
            audio_index,
            X,
            in_channels=B,
            out_channels=H,
            b_conv=B,
            sc_conv=Sc,
            kernel_size=P,
            norm=norm,
            causal=causal,
            skip_con=skip_con)
        self.fusion_function = fusion_function
        if fusion_function == "concat":
            self.fusion = Concat(B, V, F)
        elif fusion_function == "hyperfuse":
            self.fusion = hyperfuse.HyperFuse(primary_size=B,
                                              aux_size=V,
                                              hidden_size=fusion_hidden_size,
                                              mlp_type="b")
        self.feats_conv = Audio_Sequential(
            R - audio_index,
            X,
            in_channels=B,
            out_channels=H,
            b_conv=B,
            sc_conv=Sc,
            kernel_size=P,
            norm=norm,
            causal=causal,
            skip_con=skip_con)
        # mask 1x1 conv
        # n x B x T => n x N x T
        self.mask = Conv1D(F, N, 1)
        self.mask_activation_function = mask_activation_function
        # n x N x T => n x 1 x To
        self.audio_decoder_type = audio_decoder_type
        if audio_decoder_type == "conv1d":
            self.decoder = Decoder(
                N, 1, kernel_size=L, stride=L // 2, bias=audio_decoder_bias)
        elif audio_decoder_type == "stft":
            self.decoder = Decoder_STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        elif audio_decoder_type == "hifi_gan":
            self.decoder = nn.Sequential()
        else:
            raise NotImplementedError(f"{audio_decoder_type} is not supported")
        # for stft encoding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # define parameter group
        self.params = nn.ModuleDict({
            "encoder": nn.ModuleList([self.encoder]),
            "separation": nn.ModuleList(
                [self.cln, self.conv1x1, self.audio_conv, self.fusion, self.feats_conv, self.mask]),
            "decoder": nn.ModuleList([self.decoder])
        })

    def check_forward_args(self, x, v):
        if x.dim() != 2:
            raise RuntimeError(
                "{} accept 1/2D tensor as audio input, but got {:d}".format(
                    self.__class__.__name__, x.dim()))
        if v.dim() != 3:
            raise RuntimeError(
                "{} accept 2/3D tensor as video input, but got {:d}".format(
                    self.__class__.__name__, v.dim()))
        if x.size(0) != v.size(0):
            raise RuntimeError(
                "auxiliary input do not have same batch size with input chunk, {:d} vs {:d}"
                .format(x.size(0), v.size(0)))

    def forward_encoder_stft(self, x):
        # when inference, only one utt
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        w, _ = self.encoder(x)
        return w

    def forward(self, x, v, return_mask: bool = False, rescale: bool = False):
        """
        x: raw waveform chunks, N x T
        v: time variant lip embeddings, N x T x D
        """
        # transpose video input
        v = torch.transpose(v, 1, 2)
        # when inference, only one utt
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
            v = torch.unsqueeze(v, 0)
        # check args
        self.check_forward_args(x, v)

        # n x 1 x S => n x N x T
        if self.audio_encoder_type == "conv1d":
            w = self.encoder(x)
        else:
            w, noisy_phase = self.encoder(x)

        # # debug
        # d = self.decoder(w, noisy_phase, x.shape[-1])

        # n x B x T
        a = self.conv1x1(self.cln(w))
        # audio feats: n x B x T
        a = self.audio_conv(a)
        # lip embeddings
        # N x T x D => N x V x T
        v = self.video(v)

        # audio/video fusion
        if self.fusion_function == "hyperfuse":
            # tranpose because hyperfuse receipves input as n x T x D
            a = torch.transpose(a, 1, 2)
            v = torch.transpose(v, 1, 2)
        y = self.fusion(a, v)
        if self.fusion_function == "hyperfuse":
            y = torch.transpose(y, 1, 2)  # n x D x T

        # n x (B+V) x T
        y = self.feats_conv(y)
        # n x N x T
        m = self.mask(y)
        if self.mask_activation_function == "relu":
            m = torch.nn.functional.relu(m)
        elif self.mask_activation_function == "signmoid":
            m = torch.nn.functional.sigmoid(m)
        elif self.mask_activation_function == "softmax":
            m = torch.nn.functional.softmax(m)
        elif self.mask_activation_function == "softplus":
            if self.training:
                m = torch.nn.functional.softplus(m)
            else:
                m = torch.nn.functional.relu(m)
        elif self.mask_activation_function == "none":
            if not self.training:
                m = torch.nn.functional.relu(m)

        # n x To
        d_pre = w * m
        if self.audio_decoder_type != "stft":
            d = self.decoder(d_pre)
        else:
            d = self.decoder(d_pre, noisy_phase=noisy_phase, length=x.shape[-1])

        if self.audio_decoder_type == "hifi_gan":
            if return_mask:
                # n x D x T
                return d, w, m, d_pre
            else:
                return d
        # TODO: figure out ways to do better than clipping
        input_length = x.shape[-1]
        d = d[:,:input_length]

        # rescale
        if rescale:
            d = rescale_amplitude(d)

        if return_mask:
            return d, w, m, d_pre  # decoder output, encoder output, mask and output at tf domain
        else:
            return d

class ConvTasNet(nn.Module):
    '''
       ConvTasNet module
       N	Number of ﬁlters in autoencoder
       L	Length of the ﬁlters (in samples)
       B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
       Sc	Number of channels in skip-connection paths’ 1 × 1-conv blocks
       H	Number of channels in convolutional blocks
       P	Kernel size in convolutional blocks
       X	Number of convolutional blocks in each repeat
       R	Number of repeats
    '''

    def __init__(self,
                 N=512,
                 L=16,
                 B=128,
                 H=512,
                 P=3,
                 X=8,
                 R=3,
                 Sc=256,
                 norm="gln",
                 num_spks=2,
                 mask_activation_function="relu",
                 skip_con=False,
                 causal=False,
                 audio_encoder_non_negative: bool = True,
                 audio_encoder_bias: bool = True,
                 audio_encoder_type: str = "conv1d",
                 audio_decoder_type: str = "conv1d",  # conv1d, stft, hifi_gan
                 audio_decoder_bias: bool = True,
                 device_type="cpu"):
        super(ConvTasNet, self).__init__()
        # n x 1 x T => n x N x T
        self.encoder = Encoder(1, N, L, stride=L // 2, non_negative=audio_encoder_non_negative, bias=audio_encoder_bias)
        # n x N x T  Layer Normalization of Separation
        self.LayerN_S = CumulativeLayerNorm(N)
        # n x B x T  Conv 1 x 1 of  Separation
        self.BottleN_S = Conv1D(N, B, 1)
        # Separation block
        # n x B x T => n x B x T
        self.separation = Audio_Sequential(
            R,
            X,
            in_channels=B,
            out_channels=H,
            b_conv=B,
            sc_conv=Sc,
            kernel_size=P,
            norm=norm,
            causal=causal,
            skip_con=skip_con)
        # n x B x T => n x 2*N x T
        self.mask = Conv1D(B, num_spks*N, 1)
        # n x N x T => n x 1 x L
        self.decoder = Decoder(N, 1, L, stride=L//2)
        # activation function
        active_f = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=0)
        }
        self.activation_type = mask_activation_function
        self.activation = active_f[mask_activation_function]
        self.num_spks = num_spks

    def _Sequential_block(self, num_blocks, **block_kwargs):
        '''
           Sequential 1-D Conv Block
           input:
                 num_block: how many blocks in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        Conv1D_Block_lists = [Audio_1DConv(
            **block_kwargs, dilation=(2**i)) for i in range(num_blocks)]

        return nn.Sequential(*Conv1D_Block_lists)

    def _Sequential_repeat(self, num_repeats, num_blocks, **block_kwargs):
        '''
           Sequential repeats
           input:
                 num_repeats: Number of repeats
                 num_blocks: Number of block in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        repeats_lists = [self._Sequential_block(
            num_blocks, **block_kwargs) for i in range(num_repeats)]
        return nn.Sequential(*repeats_lists)

    def forward(self, x, rescale: bool = False):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        # x: n x 1 x L => n x N x T
        w = self.encoder(x)
        # n x N x L => n x B x L
        e = self.LayerN_S(w)
        e = self.BottleN_S(e)
        # n x B x L => n x B x L
        e = self.separation(e)
        # n x B x L => n x num_spk*N x L
        m = self.mask(e)
        # n x N x L x num_spks
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        # num_spks x n x N x L
        m = self.activation(torch.stack(m, dim=0))
        d = [w*m[i] for i in range(self.num_spks)]
        # decoder part num_spks x n x L
        s = [self.decoder(d[i]) for i in range(self.num_spks)]
        if rescale:
            s = [rescale_amplitude(s[i] for i in range(self.num_spks))]
        s = torch.stack(s, dim=1)

        return s


class AA_ConvTasnet(nn.Module):
    """
    Speech separation using audio reference cues

    Audio Part
        N	Number of ﬁlters in autoencoder
        L	Length of the ﬁlters (in samples)
        B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
        SC  Number of channels in skip-connection paths’ 1 × 1-conv blocks
        H	Number of channels in convolutional blocks
        P	Kernel size in convolutional blocks
        X	Number of convolutional blocks in each repeat
    Audio Reference Part
        E   Number of ﬁlters in video autoencoder
        V   Number of channels in convolutional blocks
        K   Kernel size in convolutional blocks
        D   Number of repeats
    Fusion Part
        F   Number of channels in convolutional blocks
    Other Setting
        R	Number of all repeats
        skip_con	Skip Connection
        audio_index     Number repeats of audio part
        norm    Normaliztion type
        causal  Two choice(causal or noncausal)
    """

    def __init__(
            self,
            # audio conf
            N=256,
            L=40,
            B=256,
            Sc=256,
            H=512,
            P=3,
            X=8,  # number of blocks/stacks within 1 repeat Conv
            audio_encoder_non_negative: bool = True,
            audio_encoder_type: str = "conv1d",
            audio_encoder_bias: bool = True,
            audio_decoder_type: str = "conv1d",  # conv1d, stft, hifi_gan
            audio_decoder_bias: bool = True,
            hifi_gan_config: str = None,
            hifi_gan_pretrained_checkpoint: str = None,
            hop_length=160,
            n_fft=511,
            win_length=400,
            # audio ref conf
            use_pretrained_reference_embedding: bool = False,
            audio_reference_embedding_size=None,
            reference_encoder_n_layers=1,
            reference_encoder_hidden_channels=256,
            reference_encoder_out_channels=400,
            # fusion index
            F=256,
            # other
            R=4,  # total number of repeats for audio
            skip_con=False,
            audio_index=2,
            norm="gln",
            causal=False,
            fusion_function="concat",
            fusion_hidden_size=None,
            mask_activation_function: str = "relu",
            device_type="cpu",

    ):
        super(AA_ConvTasnet, self).__init__()
        self.audio_encoder_type = audio_encoder_type


        if audio_encoder_type == "conv1d":
            # n x S > n x N x T
            self.encoder = Encoder(1, N, L, stride=L // 2, non_negative=audio_encoder_non_negative, bias=audio_encoder_bias)
        else:
            self.encoder = Encoder_STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                        non_negative=audio_encoder_non_negative,
                                        type=audio_encoder_type)

        if use_pretrained_reference_embedding:
            self.reference_audio = nn.Sequential()
        else:
            self.reference_audio = nn.Sequential(
                LSTM(
                    input_size=1, hidden_size=reference_encoder_hidden_channels,
                    num_layers=reference_encoder_n_layers, batch_first=True,
                    bidirectional=True
                ),
                nn.ReLU(),
                nn.Linear(reference_encoder_hidden_channels * 2, reference_encoder_out_channels)
            )
            audio_reference_embedding_size = reference_encoder_out_channels

        # before repeat blocks, always cLN
        self.cln = CumulativeLayerNorm(N)
        # n x N x T > n x B x T
        self.conv1x1 = Conv1D(N, B, 1)
        # repeat blocks
        # n x B x T => n x B x T
        self.skip_con = skip_con
        self.audio_conv = Audio_Sequential(
            audio_index,
            X,
            in_channels=B,
            out_channels=H,
            b_conv=B,
            sc_conv=Sc,
            kernel_size=P,
            norm=norm,
            causal=causal,
            skip_con=skip_con)
        self.fusion_function = fusion_function
        if fusion_function == "concat":
            self.fusion = Concat(B, audio_reference_embedding_size, F)
        elif fusion_function == "hyperfuse":
            self.fusion = hyperfuse.HyperFuse(primary_size=B,
                                              aux_size=V,
                                              hidden_size=fusion_hidden_size,
                                              mlp_type="b")
        self.feats_conv = Audio_Sequential(
            R - audio_index,
            X,
            in_channels=B,
            out_channels=H,
            b_conv=B,
            sc_conv=Sc,
            kernel_size=P,
            norm=norm,
            causal=causal,
            skip_con=skip_con)
        # mask 1x1 conv
        # n x B x T => n x N x T
        self.mask = Conv1D(F, N, 1)
        self.mask_activation_function = mask_activation_function
        # n x N x T => n x 1 x To
        self.audio_decoder_type = audio_decoder_type
        if audio_decoder_type == "conv1d":
            self.decoder = Decoder(
                N, 1, kernel_size=L, stride=L // 2, bias=audio_decoder_bias)
        elif audio_decoder_type == "stft":
            self.decoder = Decoder_STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        elif audio_decoder_type == "hifi_gan":
            self.decoder = nn.Sequential()
        else:
            raise NotImplementedError(f"{audio_decoder_type} is not supported")
        # for stft encoding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # define parameter group
        self.params = nn.ModuleDict({
            "encoder": nn.ModuleList([self.encoder]),
            "separation": nn.ModuleList(
                [self.cln, self.conv1x1, self.audio_conv, self.fusion, self.feats_conv, self.mask]),
            "decoder": nn.ModuleList([self.decoder])
        })

    def check_forward_args(self, x, a):
        if x.dim() != 2 or a.dim() != 2:
            raise RuntimeError(
                "{} accept 1/2D tensor as audio input, but got {:d}".format(
                    self.__class__.__name__, x.dim()))
        if x.size(0) != a.size(0):
            raise RuntimeError(
                "auxiliary input do not have same batch size with input chunk, {:d} vs {:d}"
                .format(x.size(0), v.size(0)))

    def forward_encoder_stft(self, x):
        # when inference, only one utt
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        w, _ = self.encoder(x)
        return w

    def forward(self, x, ref_a, return_mask: bool = False, rescale: bool = False):
        """
        x: raw waveform chunks, N x T
        v: time variant lip embeddings, N x T x D
        """
        # when inference, only one utt
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
            ref_a = torch.unsqueeze(ref_a, 0)
        # check args
        self.check_forward_args(x, ref_a)

        # n x 1 x S => n x N x T
        if self.audio_encoder_type == "conv1d":
            w = self.encoder(x)
        else:
            w, noisy_phase = self.encoder(x)

        # n x B x T
        a = self.conv1x1(self.cln(w))
        # audio feats: n x B x T
        a = self.audio_conv(a)
        # reference audio
        # N x T => N x D x T
        ref_a = self.reference_audio(ref_a.unsqueeze(1)).permute(0,2,1)
        ref_embedding = torch.mean(ref_a,  dim=-1)  # [400]
        ref_embedding_cache = ref_embedding
        ref_embedding = ref_embedding.unsqueeze(1).repeat((1, a.shape[-1], 1)).permute(0, 2, 1)

        # audio/video fusion
        if self.fusion_function == "hyperfuse":
            # tranpose because hyperfuse receipves input as n x T x D
            a = torch.transpose(a, 1, 2)
            ref_embedding = torch.transpose(ref_embedding, 1, 2)
        y = self.fusion(a, ref_embedding)
        if self.fusion_function == "hyperfuse":
            y = torch.transpose(y, 1, 2)  # n x D x T

        # n x (B+V) x T
        y = self.feats_conv(y)
        # n x N x T
        m = self.mask(y)
        if self.mask_activation_function == "relu":
            m = torch.nn.functional.relu(m)
        elif self.mask_activation_function == "signmoid":
            m = torch.nn.functional.sigmoid(m)
        elif self.mask_activation_function == "softmax":
            m = torch.nn.functional.softmax(m)
        elif self.mask_activation_function == "softplus":
            if self.training:
                m = torch.nn.functional.softplus(m)
            else:
                m = torch.nn.functional.relu(m)
        elif self.mask_activation_function == "none":
            if not self.training:
                m = torch.nn.functional.relu(m)

        # n x To
        d_pre = w * m
        if self.audio_decoder_type != "stft":
            d = self.decoder(d_pre)
        else:
            d = self.decoder(d_pre, noisy_phase=noisy_phase, length=x.shape[-1])

        if self.audio_decoder_type == "hifi_gan":
            if return_mask:
                # n x D x T
                return d, w, m, d_pre
            else:
                return d

        # TODO: figure out ways to do better than clipping
        input_length = x.shape[-1]
        d = d[:,:input_length]

        # rescale
        if rescale:
            d = rescale_amplitude(d)

        if return_mask:
            return d, w, m, d_pre  # decoder output, encoder output, mask and output at tf domain
        else:
            return d

class LSTM(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__()
        self.model = nn.LSTM(*args, **kwargs)

    def forward(self, x):
        self.model.flatten_parameters()
        assert x.dim() == 3
        x = x.permute(0, 2, 1)
        o, _ = self.model(x)
        return o

def rescale_amplitude(signal: torch.Tensor):
    signal = signal - torch.mean(signal)
    signal = signal / (torch.max(torch.abs(signal)) + torch.finfo(torch.float64).eps)
    return signal


def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10 ** 6


class AV_ConvTasnet2D(nn.Module):
    """
    Audio and Visual Speech Separation
    Audio Part
        N	Number of ﬁlters in autoencoder
        L	Length of the ﬁlters (in samples)
        B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
        SC  Number of channels in skip-connection paths’ 1 × 1-conv blocks
        H	Number of channels in convolutional blocks
        P	Kernel size in convolutional blocks
        X	Number of convolutional blocks in each repeat
    Video Part
        E   Number of ﬁlters in video autoencoder
        V   Number of channels in convolutional blocks
        K   Kernel size in convolutional blocks
        D   Number of repeats
    Concat Part
        F   Number of channels in convolutional blocks
    Other Setting
        R	Number of all repeats
        skip_con	Skip Connection
        audio_index     Number repeats of audio part
        norm    Normaliztion type
        causal  Two choice(causal or noncausal)
    """

    def __init__(
            self,
            # audio conf
            N=256,
            L=40,
            B=256,
            Sc=256,
            H=512,
            P=3,
            X=8,  # number of blocks/stacks within 1 repeat Conv
            audio_encoder_non_negative: bool = True,
            audio_encoder_type: str = "conv1d",
            audio_encoder_bias: bool = True,
            audio_decoder_type: str = "conv1d",  # conv1d, stft, hifi_gan
            audio_decoder_bias: bool = True,
            hifi_gan_config: str = None,
            hifi_gan_pretrained_checkpoint: str = None,
            hop_length=160,
            n_fft=511,
            win_length=400,
            # video conf
            E=256,
            V=256,
            K=3,
            D=5,
            # fusion index
            F=256,
            # other
            R=4,  # total number of repeats for audio
            skip_con=False,
            audio_index=2,
            norm="gln",
            causal=False,
            fusion_function="concat",
            fusion_hidden_size=None,
            mask_activation_function: str = "relu",
            device_type="cpu",
    ):
        super(AV_ConvTasnet2D, self).__init__()
        self.audio_encoder_type = audio_encoder_type

        self.video = VideoSequential2D(E, V, K, skip_con=skip_con, repeat=D)
        if audio_encoder_type == "conv1d":
            # n x S > n x N x T
            self.encoder = Encoder(1, N, L, stride=L // 2, non_negative=audio_encoder_non_negative, bias=audio_encoder_bias)
        else:
            self.encoder = Encoder_STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                        non_negative=audio_encoder_non_negative,
                                        type=audio_encoder_type)

        # before repeat blocks, always cLN
        self.cln = CumulativeLayerNorm(N)
        # n x N x T > n x B x T
        self.conv1x1 = Conv1D(N, B, 1)
        # repeat blocks
        # n x B x T => n x B x T
        self.skip_con = skip_con
        self.audio_conv = Audio_Sequential(
            audio_index,
            X,
            in_channels=B,
            out_channels=H,
            b_conv=B,
            sc_conv=Sc,
            kernel_size=P,
            norm=norm,
            causal=causal,
            skip_con=skip_con)
        self.fusion_function = fusion_function
        if fusion_function == "concat":
            self.fusion = Concat(B, V, F)
        elif fusion_function == "hyperfuse":
            self.fusion = hyperfuse.HyperFuse(primary_size=B,
                                              aux_size=V,
                                              hidden_size=fusion_hidden_size,
                                              mlp_type="b")
        self.feats_conv = Audio_Sequential(
            R - audio_index,
            X,
            in_channels=B,
            out_channels=H,
            b_conv=B,
            sc_conv=Sc,
            kernel_size=P,
            norm=norm,
            causal=causal,
            skip_con=skip_con)
        # mask 1x1 conv
        # n x B x T => n x N x T
        self.mask = Conv1D(F, N, 1)
        self.mask_activation_function = mask_activation_function
        # n x N x T => n x 1 x To
        self.audio_decoder_type = audio_decoder_type
        if audio_decoder_type == "conv1d":
            self.decoder = Decoder(
                N, 1, kernel_size=L, stride=L // 2, bias=audio_decoder_bias)
        elif audio_decoder_type == "stft":
            self.decoder = Decoder_STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        elif audio_decoder_type == "hifi_gan":
            self.decoder = nn.Sequential()
        else:
            raise NotImplementedError(f"{audio_decoder_type} is not supported")
        # for stft encoding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # define parameter group
        self.params = nn.ModuleDict({
            "encoder": nn.ModuleList([self.encoder]),
            "separation": nn.ModuleList(
                [self.cln, self.conv1x1, self.audio_conv, self.fusion, self.feats_conv, self.mask]),
            "decoder": nn.ModuleList([self.decoder])
        })

    def check_forward_args(self, x, v):
        if x.dim() != 2:
            raise RuntimeError(
                "{} accept 1/2D tensor as audio input, but got {:d}".format(
                    self.__class__.__name__, x.dim()))
        if v.dim() != 4:
            raise RuntimeError(
                "{} accept (B, F, H, W) tensor as video input, but got {:d}".format(
                    self.__class__.__name__, v.dim()))
        if x.size(0) != v.size(0):
            raise RuntimeError(
                "auxiliary input do not have same batch size with input chunk, {:d} vs {:d}"
                .format(x.size(0), v.size(0)))

    def forward_encoder_stft(self, x):
        # when inference, only one utt
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        w, _ = self.encoder(x)
        return w

    def forward(self, x, v, return_mask: bool = False, rescale: bool = False):
        """
        x: raw waveform chunks, N x T
        v: time variant lip embeddings, N x T x D
        """
        # transpose video input
        v = torch.transpose(v, 1, 2)
        # when inference, only one utt
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
            v = torch.unsqueeze(v, 0)
        # check args
        self.check_forward_args(x, v)

        # n x 1 x S => n x N x T
        if self.audio_encoder_type == "conv1d":
            w = self.encoder(x)
        else:
            w, noisy_phase = self.encoder(x)

        # # debug
        # d = self.decoder(w, noisy_phase, x.shape[-1])

        # n x B x T
        a = self.conv1x1(self.cln(w))
        # audio feats: n x B x T
        a = self.audio_conv(a)
        # lip embeddings
        # N x T x D => N x V x T
        v = self.video(v)

        # audio/video fusion
        if self.fusion_function == "hyperfuse":
            # tranpose because hyperfuse receipves input as n x T x D
            a = torch.transpose(a, 1, 2)
            v = torch.transpose(v, 1, 2)
        y = self.fusion(a, v)
        if self.fusion_function == "hyperfuse":
            y = torch.transpose(y, 1, 2)  # n x D x T

        # n x (B+V) x T
        y = self.feats_conv(y)
        # n x N x T
        m = self.mask(y)
        if self.mask_activation_function == "relu":
            m = torch.nn.functional.relu(m)
        elif self.mask_activation_function == "signmoid":
            m = torch.nn.functional.sigmoid(m)
        elif self.mask_activation_function == "softmax":
            m = torch.nn.functional.softmax(m)
        elif self.mask_activation_function == "softplus":
            if self.training:
                m = torch.nn.functional.softplus(m)
            else:
                m = torch.nn.functional.relu(m)
        elif self.mask_activation_function == "none":
            if not self.training:
                m = torch.nn.functional.relu(m)

        # n x To
        d_pre = w * m
        if self.audio_decoder_type != "stft":
            d = self.decoder(d_pre)
        else:
            d = self.decoder(d_pre, noisy_phase=noisy_phase, length=x.shape[-1])

        if self.audio_decoder_type == "hifi_gan":
            if return_mask:
                # n x D x T
                return d, w, m, d_pre
            else:
                return d
        # TODO: figure out ways to do better than clipping
        input_length = x.shape[-1]
        d = d[:,:input_length]

        # rescale
        if rescale:
            d = rescale_amplitude(d)

        if return_mask:
            return d, w, m, d_pre  # decoder output, encoder output, mask and output at tf domain
        else:
            return d


if __name__ == "__main__":
    opt = {
        # audio conf
        "N": 256,
        "L": 40,
        "B": 256,
        "H": 512,
        "P": 3,
        "X": 8,
        # audio reference
        "reference_encoder_n_layers" : 1,
        "reference_encoder_hidden_channels": 256,
        "reference_encoder_out_channels": 400,
        # fusion index
        "F": 256,
        # other
        "R": 4,
        "audio_index": 1,
        "norm": "gln",
        "causal": False
    }
    ref_a = torch.randn(1, 32000)
    audio = torch.randn(1, 32000)
    aa_model = AA_ConvTasnet(**opt)
    out = aa_model(audio, ref_a)
    print(out.shape)
    print(check_parameters(aa_model))
