import torch.nn as nn
import torch

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


class VideoConv1D(nn.Module):
    """
    video part 1-D Conv Block
    in_channels: video Encoder output channels
    conv_channels: dconv channels
    kernel_size: the depthwise conv kernel size
    dilation: the depthwise conv dilation
    residual: Whether to use residual connection
    skip_con: Whether to use skip connection
    first_block: first block, not residual

    Requires input video to be of size (B, H*W, T) and convolve on time
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
        super(VideoConv1D, self).__init__()
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

           Requires input video to be of size (B, H*W, T) and convolve on time
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


class VideoSequential1D(nn.Module):
    """
    All the Video Part
    in_channels: front3D part in_channels
    out_channels: Video Conv1D part out_channels
    kernel_size: the kernel size of Video Conv1D
    skip_con: skip connection
    repeat: Conv1D repeats

    Requires input video to be of size (B, H*W, T) and convolve on time
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 skip_con=True,
                 repeat=5):
        super(VideoSequential1D, self).__init__()
        self.conv1d_list = nn.ModuleList([])
        self.skip_con = skip_con
        for i in range(repeat):
            in_channels = out_channels if i else in_channels
            self.conv1d_list.append(
                VideoConv1D(
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

            Requires input video to be of size (B, H*W, T) and convolve on time
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
