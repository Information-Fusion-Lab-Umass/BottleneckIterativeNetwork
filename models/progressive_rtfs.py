import torch


from models.rtfs.layers import ConvNormAct
from models.rtfs.TDAVNet import (
    BaseAVModel,
    encoder,
    mask_generator,
    decoder,
)

import inspect
import torch
import torch.nn as nn

from models.rtfs.TDAVNet.fusion import MultiModalFusion
from models.rtfs.utils import get_MACS_params
from models.rtfs import separators


class ProgressiveRefinementModule(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        audio_fusion_dim: int,
        video_fusion_dim: int,
        video_len: int,
        audio_F: int,
        audio_T: int,
        audio_params: dict,
        video_params: dict,
        audio_bn_chan: int,
        video_bn_chan: int,
        fusion_params: dict,
    ):
        super().__init__()
        self.audio_params = audio_params
        self.video_params = video_params
        self.audio_bn_chan = audio_bn_chan
        self.video_bn_chan = video_bn_chan
        self.fusion_params = fusion_params

        self.audio_token = nn.Parameter(torch.rand(audio_fusion_dim, audio_F, audio_T))
        self.video_token = nn.Parameter(torch.rand(video_fusion_dim, video_len))

        # self.fusion_repeats = self.video_params.get("repeats", 0)
        # self.audio_repeats = self.audio_params["repeats"] - self.fusion_repeats
        self.num_blocks = num_blocks
        self.audio_fusion_dim = audio_fusion_dim
        self.video_fusion_dim = video_fusion_dim

        self.audio_net = separators.get(self.audio_params.get("audio_net", None))(
            **self.audio_params,
            in_chan=self.audio_bn_chan + self.audio_fusion_dim,
        )
        self.video_net = separators.get(self.video_params.get("video_net", None))(
            **self.video_params,
            in_chan=self.video_bn_chan + self.video_fusion_dim,
        )

        self.crossmodal_fusion = MultiModalFusion(
            **self.fusion_params,
            # audio_bn_chan=self.audio_bn_chan,
            # video_bn_chan=self.video_bn_chan,
            # fusion_repeats=self.fusion_repeats,
            audio_bn_chan=self.audio_fusion_dim,
            video_bn_chan=self.video_fusion_dim,
            fusion_repeats=1
        )

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        b = audio.shape[0]

        print(audio.shape, video.shape)

        audio_token = self.audio_token.unsqueeze(0).repeat(b, 1, 1, 1)
        video_token = self.video_token.unsqueeze(0).repeat(b, 1, 1)

        audio_expand = torch.cat((audio, audio_token), dim=1)
        video_expand = torch.cat((video, video_token), dim=1)

        for i in range(self.num_blocks):
            audio_latent = self.audio_net.get_block(i)(audio_expand + audio_residual if i > 0 else audio_expand)
            video_latent = self.video_net.get_block(i)(video_expand + video_residual if i > 0 else video_expand)

            audio_fusion = audio_latent[:, -self.audio_fusion_dim:, :]
            video_fusion = video_latent[:, -self.video_fusion_dim:, :]

            # cross modal fusion
            audio_fused, video_fused = self.crossmodal_fusion.get_fusion_block(i)(audio_fusion, video_fusion)

            audio_residual = torch.cat((audio_latent[:, :-self.audio_fusion_dim, :], audio_fused), dim=1)
            video_residual = torch.cat((video_latent[:, :-self.video_fusion_dim, :], video_fused), dim=1)

        return audio

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args

    def get_MACs(self, bn_audio, bn_video):
        macs = []

        macs += get_MACS_params(self.audio_net, (bn_audio,))

        macs += get_MACS_params(self.video_net, (bn_video,))

        macs += get_MACS_params(self.crossmodal_fusion, (bn_audio, bn_video))

        return macs



class AVNet(BaseAVModel):
    def __init__(
        self,
        n_src: int,
        num_blocks: int,
        audio_fusion_dim: int,
        video_fusion_dim: int,
        video_len: int,
        audio_T: int,
        audio_F: int,
        enc_dec_params: dict,
        audio_bn_params: dict,
        audio_params: dict,
        mask_generation_params: dict,
        pretrained_vout_chan: int = -1,
        video_bn_params: dict = dict(),
        video_params: dict = dict(),
        fusion_params: dict = dict(),
        print_macs: bool = False,
        *args,
        **kwargs,
    ):
        super(AVNet, self).__init__()

        self.n_src = n_src
        self.pretrained_vout_chan = pretrained_vout_chan
        self.audio_bn_params = audio_bn_params
        self.video_bn_params = video_bn_params
        self.enc_dec_params = enc_dec_params
        self.audio_params = audio_params
        self.video_params = video_params
        self.fusion_params = fusion_params
        self.mask_generation_params = mask_generation_params
        self.print_macs = print_macs
        self.num_blocks = num_blocks

        self.audio_fusion_dim = audio_fusion_dim
        self.video_fusion_dim = video_fusion_dim
        self.video_len = video_len
        self.audio_F = audio_F
        self.audio_T = audio_T

        self.encoder: encoder.BaseEncoder = encoder.get(self.enc_dec_params["encoder_type"])(
            **self.enc_dec_params,
            in_chan=1,
            upsampling_depth=self.audio_params.get("upsampling_depth", 1),
        )

        self.init_modules()

    def init_modules(self):
        self.enc_out_chan = self.encoder.get_out_chan()

        self.mask_generation_params["mask_generator_type"] = self.mask_generation_params.get("mask_generator_type", "MaskGenerator")
        self.audio_bn_chan = self.audio_bn_params.get("out_chan", self.enc_out_chan)
        self.audio_bn_params["out_chan"] = self.audio_bn_chan
        self.video_bn_chan = self.video_bn_params.get("out_chan", self.pretrained_vout_chan) * self.n_src

        self.audio_bottleneck = ConvNormAct(**self.audio_bn_params, in_chan=self.enc_out_chan)
        self.video_bottleneck = ConvNormAct(**self.video_bn_params, in_chan=self.pretrained_vout_chan * self.n_src)

        self.refinement_module = ProgressiveRefinementModule(
            num_blocks=self.num_blocks,
            audio_fusion_dim=self.audio_fusion_dim,
            video_fusion_dim=self.video_fusion_dim,
            video_len=self.video_len,
            audio_F=self.audio_F,
            audio_T=self.audio_T,
            fusion_params=self.fusion_params,
            audio_params=self.audio_params,
            video_params=self.video_params,
            audio_bn_chan=self.audio_bn_chan,
            video_bn_chan=self.video_bn_chan,
        )

        self.mask_generator: mask_generator.BaseMaskGenerator = mask_generator.get(self.mask_generation_params["mask_generator_type"])(
            **self.mask_generation_params,
            n_src=self.n_src,
            audio_emb_dim=self.enc_out_chan,
            bottleneck_chan=self.audio_bn_chan,
        )

        self.decoder: decoder.BaseDecoder = decoder.get(self.enc_dec_params["decoder_type"])(
            **self.enc_dec_params,
            # in_chan=self.enc_out_chan * self.n_src,
            in_chan=self.enc_out_chan,
            n_src=self.n_src,
        )

        if self.print_macs:
            self.get_MACs()

    def forward(self, audio_mixture: torch.Tensor, mouth_embedding: torch.Tensor = None):
        # reshape mouth_embedding
        # from (B, n_src, length, dim) to  (B, dim, n_src * length)
        if len(mouth_embedding.shape) > 3:
            b = audio_mixture.shape[0]
            mouth_embedding = mouth_embedding.permute(0, 1, 3, 2).reshape(b, self.video_bn_chan, -1)
            # print(mouth_embedding.shape)

        audio_mixture_embedding = self.encoder(audio_mixture)  # B, 1, L -> B, N, T, (F)

        # print(audio_mixture_embedding.shape)

        audio = self.audio_bottleneck(audio_mixture_embedding)  # B, C, T, (F)

        # print(audio.shape)

        video = self.video_bottleneck(mouth_embedding)  # B, N2, T2, (F2) -> B, C2, T2, (F2)

        # print(video.shape)

        refined_features = self.refinement_module(audio, video)  # B, C, T, (F)

        # print(refined_features.shape)

        separated_audio_embeddings = self.mask_generator(refined_features, audio_mixture_embedding)  # B, n_src, N, T, (F)

        # print(separated_audio_embeddings.shape)

        separated_audios = self.decoder(separated_audio_embeddings, audio_mixture.shape)  #  B, n_src, L

        return separated_audios

    def get_config(self):
        model_args = {}
        model_args["encoder"] = self.encoder.get_config()
        model_args["audio_bottleneck"] = self.audio_bottleneck.get_config()
        model_args["video_bottleneck"] = self.video_bottleneck.get_config()
        model_args["refinement_module"] = self.refinement_module.get_config()
        model_args["mask_generator"] = self.mask_generator.get_config()
        model_args["decoder"] = self.decoder.get_config()

        return model_args
