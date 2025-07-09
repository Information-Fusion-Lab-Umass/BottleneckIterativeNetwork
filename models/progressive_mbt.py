import torch
import torch.nn.functional as torch_f
from torch import nn
import models.afrcnn as afrcnn
import models.autoencoder as autoencoder
import models.sequential_conv as seq_conv
import math
import models.avlit as avlit


def rescale_amplitude(signal: torch.Tensor):
    signal = signal - torch.mean(signal)
    signal = signal / (torch.max(torch.abs(signal)) + torch.finfo(torch.float64).eps)
    return signal


def compute_conv_len(in_len, kernel_size, stride=1, padding=0, dilation=1):
    return math.floor((in_len + 2 * padding - dilation * (kernel_size-1) - 1) / stride + 1)


class IterativeBranch(nn.Module):
    def __init__(
        self,
        num_sources: int = 2,
        hidden_channels: int = 512,
        bottleneck_channels: int = 128,
        fusion_channels: int = 64,
        num_blocks: int = 8,
        states: int = 5
    ) -> None:
        super().__init__()

        # Branch attributes
        self.num_sources = num_sources
        self.hidden_channels = hidden_channels
        self.bottleneck_channels = bottleneck_channels
        self.fusion_channels = fusion_channels

        self.num_blocks = num_blocks
        self.states = states

        # Modules
        self.afrcnn_block = afrcnn.AFRCNN(
            in_channels=hidden_channels,
            out_channels=bottleneck_channels + fusion_channels,
            states=states,
        )
        self.adapt_audio = nn.Sequential(
            nn.Conv1d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=1,
                stride=1,
                groups=bottleneck_channels,
            ),
            nn.PReLU(),
        )

        self.adapt_fusion = nn.Sequential(
            nn.Conv1d(
                bottleneck_channels + fusion_channels,
                bottleneck_channels + fusion_channels,
                kernel_size=1,
                stride=1,
                groups=bottleneck_channels + fusion_channels,
            ),
            nn.PReLU(),
        )

    def forward(
        self,
        fa: torch.Tensor,
        fv_p=None,
        fused=False
    ) -> torch.Tensor:
        Ri = self.adapt_audio(fa)  # (B, bottleneck_channels, _ ) -> (B, bottleneck_channels, _)

        if fv_p is not None:
            f = self._modality_fusion(Ri,
                                      fv_p)  # (B, bottleneck_channels, _ ) -> (B, bottleneck_channels + fusion_channels, _)
            Ri = self.adapt_fusion(
                f)  # (B, bottleneck_channels + fusion_channels, _) -> (B, bottleneck_channels + fusion_channels, _)

        # 3) Apply the A-FRCNN block
        Ri = self.afrcnn_block(Ri)  # (B, bottleneck_channels + fusion_channels, _) -> (B, bottleneck_channels + fusion_channels, _)
        return Ri

    def _modality_fusion(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape[-1] > b.shape[-1]:
            b = torch_f.interpolate(b, size=a.shape[2:])
        return torch.cat([a, b], dim=1)


class ProgressiveBranch(nn.Module):
    def __init__(self,
                 num_blocks,
                 num_sources: int = 2,
                 video_dim: int = 1024,
                 audio_dim: int = 32000,
                 kernel_size: int = 40,
                 fusion_channels: int = 512,
                 audio_hidden_channels: int = 512,
                 audio_bottleneck_channels: int = 128,
                 audio_states: int = 5,
                 video_hidden_channels: int = 128,
                 video_bottleneck_channels: int = 128,
                 video_states: int = 5,
                 ):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_sources = num_sources
        self.kernel_size = kernel_size

        self.fusion_channels = fusion_channels

        self.audio_states = audio_states
        self.audio_hidden_channels = audio_hidden_channels
        # self.video_embedding_dim = 2048
        self.video_embedding_dim = video_dim

        audio_conv_len = compute_conv_len(audio_dim, kernel_size=kernel_size,
                                          stride=kernel_size // 2,
                                          padding=kernel_size // 2)

        print("model params: ", locals(), "  audio conv len:, ", audio_conv_len)

        self.audio_encoder = nn.Conv1d(
            in_channels=1,
            out_channels=audio_hidden_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            padding=kernel_size // 2,
            bias=False,
        )

        self.audio_decoder = nn.ConvTranspose1d(
            in_channels=audio_hidden_channels,
            out_channels=1,
            output_padding=kernel_size // 2 - 1,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            padding=kernel_size // 2,
            groups=1,
            bias=False,
        )

        # Audio adaptation
        self.audio_norm = afrcnn.GlobLN(audio_hidden_channels)
        self.audio_bottleneck = nn.Conv1d(
            in_channels=audio_hidden_channels,
            out_channels=audio_bottleneck_channels,
            kernel_size=1,
        )

        # Video adaptation
        self.video_bottleneck = nn.Conv1d(
            in_channels=self.video_embedding_dim * self.num_sources,
            out_channels=video_bottleneck_channels,
            kernel_size=1,
        )

        self.audio_branch = IterativeBranch(
            num_sources=num_sources,
            hidden_channels=audio_hidden_channels,
            bottleneck_channels=audio_bottleneck_channels,
            states=audio_states,
            fusion_channels=fusion_channels
        )

        # Video branch
        self.video_branch = IterativeBranch(
            num_sources=num_sources,
            hidden_channels=video_hidden_channels,
            bottleneck_channels=video_bottleneck_channels,
            states=video_states,
            fusion_channels=fusion_channels
        )

        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(
                in_channels=audio_bottleneck_channels,
                out_channels=num_sources * audio_hidden_channels,
                kernel_size=1,
            ),
        )
        self.mask_activation = nn.ReLU()

        self.audio_token = nn.Parameter(torch.rand(fusion_channels, audio_conv_len))
        self.video_token = nn.Parameter(torch.rand(fusion_channels, audio_conv_len))
        # self.audio_token = nn.Parameter(torch.rand(fusion_channels, 3201))
        # self.video_token = nn.Parameter(torch.rand(fusion_channels, 3201))

    def _masking(self, f, m):
        m = self.mask_net(m)
        m = m.view(
            m.shape[0],
            self.num_sources,
            self.audio_hidden_channels,
            -1,
        )
        m = self.mask_activation(m)
        masked = m * f.unsqueeze(1)
        return masked

    def lcm(self):
        half_kernel = self.kernel_size // 2
        pow_states = 2 ** self.audio_states
        return abs(half_kernel * pow_states) // math.gcd(half_kernel, pow_states)

    def _pad_input(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm()
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(
                list(appropriate_shape[:-1])
                + [appropriate_shape[-1] + self.lcm() - values_to_pad],
                dtype=torch.float32,
            ).to(x.device)
            padded_x[..., : x.shape[-1]] = x
            return padded_x
        return x

    def _trim_output(self, x, T):
        if x.shape[-1] >= T:
            return x[..., 0:T]
        return x

    def forward(self, audio, video, clean_audio=None):
        """
        :param audio: (B, 1, time * sample rate)
        :param video: (B, 1, time * fps, latent)
        :return:
        """
        if clean_audio is None:
            b, T = audio.shape[0], audio.shape[-1]
            M, F = video.shape[1], video.shape[2]

            # Get audio features, fa
            x = self._pad_input(audio)
            fa_in = self.audio_encoder(x)
            fa = self.audio_norm(fa_in)
            audio_embed = self.audio_bottleneck(fa)

            # Get video features, fv
            # fv = video
            fv = video.permute(0, 1, 3, 2).reshape(b, M * self.video_embedding_dim, -1)
            video_embed = self.video_bottleneck(fv)
            video_embed = torch_f.interpolate(video_embed, size=audio_embed.shape[2:])
            #
            # print(audio_embed.shape)  # (B, 512, 1601)
            # print(video_embed.shape)  # (B, 512, 1601)

            audio_token = self.audio_token.unsqueeze(0).repeat(b, 1, 1)
            video_token = self.video_token.unsqueeze(0).repeat(b, 1, 1)

            audio_rec = 0
            video_rec = 0

            # progressive loop
            for i in range(self.num_blocks):
                # audio_latent = self.audio_block(torch.cat((audio_embed, fusion_latent), dim=1))
                # video_latent = self.video_block(torch.cat((video_embed, fusion_latent), dim=1))
                fused_token = (audio_token + video_token) / 2
                audio_latent = self.audio_branch(audio_embed + audio_rec, fused_token)
                video_latent = self.video_branch(video_embed + video_rec, fused_token)

                audio_token = audio_latent[:, -self.fusion_channels:]
                video_token = video_latent[:, -self.fusion_channels:]

                audio_rec = audio_latent[:, :-self.fusion_channels]
                video_rec = video_latent[:, :-self.fusion_channels]

            mask = audio_rec + video_rec

            fa_m = self._masking(fa_in, mask)

            fa_m = fa_m.view(b * self.num_sources, self.audio_hidden_channels, -1)
            s = self.audio_decoder(fa_m)
            s = s.view(b, self.num_sources, -1)
            s = self._trim_output(s, T)
            return s
        else:
            return self.visualize(audio, video, clean_audio)

    def visualize(self, audio, video, clean_audio):
        def mask_visual(f, m):
            m = self.mask_net(m)
            m = m.view(
                m.shape[0],
                self.num_sources,
                self.audio_hidden_channels,
                -1,
            )
            m = self.mask_activation(m)
            # print(f.shape, m.shape)
            masked = m * f.unsqueeze(1)
            return masked, m

        b, T = audio.shape[0], audio.shape[-1]
        M, F = video.shape[1], video.shape[2]

        # Get audio features, fa
        x = self._pad_input(audio)
        fa_in = self.audio_encoder(x)  # latent audio, (B, 512, 1601)

        fa = self.audio_norm(fa_in)
        audio_embed = self.audio_bottleneck(fa)

        # Get clean audio embedding
        clean_encoded = []
        for i in range(M):
            clean_encoded.append(self.audio_encoder(clean_audio[:, i:i+1]))

        clean_encoded = torch.stack(clean_encoded, dim=1).detach().cpu().numpy()
        # print("clean encode", clean_encoded.shape)

        # Get video features, fv
        # fv = video
        fv = video.permute(0, 1, 3, 2).reshape(b, M * self.video_embedding_dim, -1)
        video_embed = self.video_bottleneck(fv)
        video_embed = torch_f.interpolate(video_embed, size=audio_embed.shape[2:])
        #
        # print(audio_embed.shape)  # (B, 512, 1601)
        # print(video_embed.shape)  # (B, 512, 1601)

        audio_token = self.audio_token.unsqueeze(0).repeat(b, 1, 1)
        video_token = self.video_token.unsqueeze(0).repeat(b, 1, 1)

        audio_rec = 0
        video_rec = 0

        mask_list = []
        sep_encoded_list = []
        audio_list = []

        # progressive loop
        for i in range(self.num_blocks):

            # audio_latent = self.audio_block(torch.cat((audio_embed, fusion_latent), dim=1))
            # video_latent = self.video_block(torch.cat((video_embed, fusion_latent), dim=1))
            fused_token = (audio_token + video_token) / 2
            audio_latent = self.audio_branch(audio_embed + audio_rec, fused_token)
            video_latent = self.video_branch(video_embed + video_rec, fused_token)

            audio_token = audio_latent[:, -self.fusion_channels:]
            video_token = video_latent[:, -self.fusion_channels:]

            audio_rec = audio_latent[:, :-self.fusion_channels]
            video_rec = video_latent[:, :-self.fusion_channels]

            fa_m, sep_mask = mask_visual(fa_in, audio_rec + video_rec)

            # print("sep encode: ", fa_m.shape, sep_mask.shape)
            sep_encoded_list.append(fa_m.detach().cpu().numpy())

            fa_m = fa_m.view(b * self.num_sources, self.audio_hidden_channels, -1)
            s = self.audio_decoder(fa_m)
            s = s.view(b, self.num_sources, -1)
            s = self._trim_output(s, T)

            mask_list.append(sep_mask.detach().cpu().numpy())

            audio_list.append(s.detach())

        return audio_list, mask_list, sep_encoded_list, clean_encoded


class ProgressiveBranchCVOnly(ProgressiveBranch):
    def forward(self, audio, video, clean_audio=None):
        """
        :param audio: (B, 1, time * sample rate)
        :param video: (B, 1, time * fps, latent)
        :return:
        """
        if clean_audio is None:
            b, T = audio.shape[0], audio.shape[-1]
            M, F = video.shape[1], video.shape[2]

            # Get audio features, fa
            x = self._pad_input(audio)
            fa_in = self.audio_encoder(x)
            fa = self.audio_norm(fa_in)
            audio_embed = self.audio_bottleneck(fa)

            # Get video features, fv
            # fv = video
            fv = video.permute(0, 1, 3, 2).reshape(b, M * self.video_embedding_dim, -1)
            video_embed = self.video_bottleneck(fv)
            video_embed = torch_f.interpolate(video_embed, size=audio_embed.shape[2:])
            #
            # print(audio_embed.shape)  # (B, 512, 1601)
            # print(video_embed.shape)  # (B, 512, 1601)
            video_token = self.video_token.unsqueeze(0).repeat(b, 1, 1)

            audio_rec = 0
            video_rec = 0

            # progressive loop
            for i in range(self.num_blocks):
                # audio_latent = self.audio_block(torch.cat((audio_embed, fusion_latent), dim=1))
                # video_latent = self.video_block(torch.cat((video_embed, fusion_latent), dim=1))
                fused_token = video_token
                audio_latent = self.audio_branch(audio_embed + audio_rec, fused_token)
                video_latent = self.video_branch(video_embed + video_rec, fused_token)

                video_token = video_latent[:, -self.fusion_channels:]

                audio_rec = audio_latent[:, :-self.fusion_channels]
                video_rec = video_latent[:, :-self.fusion_channels]

            mask = audio_rec + video_rec

            fa_m = self._masking(fa_in, mask)

            fa_m = fa_m.view(b * self.num_sources, self.audio_hidden_channels, -1)
            s = self.audio_decoder(fa_m)
            s = s.view(b, self.num_sources, -1)
            s = self._trim_output(s, T)
            return s
        else:
            return self.visualize(audio, video, clean_audio)


class ProgressiveBranchCAOnly(ProgressiveBranch):

    def forward(self, audio, video, clean_audio=None):
        """
        :param audio: (B, 1, time * sample rate)
        :param video: (B, 1, time * fps, latent)
        :return:
        """
        if clean_audio is None:
            b, T = audio.shape[0], audio.shape[-1]
            M, F = video.shape[1], video.shape[2]

            # Get audio features, fa
            x = self._pad_input(audio)
            fa_in = self.audio_encoder(x)
            fa = self.audio_norm(fa_in)
            audio_embed = self.audio_bottleneck(fa)

            # Get video features, fv
            # fv = video
            fv = video.permute(0, 1, 3, 2).reshape(b, M * self.video_embedding_dim, -1)
            video_embed = self.video_bottleneck(fv)
            video_embed = torch_f.interpolate(video_embed, size=audio_embed.shape[2:])
            #
            # print(audio_embed.shape)  # (B, 512, 1601)
            # print(video_embed.shape)  # (B, 512, 1601)
            audio_token = self.audio_token.unsqueeze(0).repeat(b, 1, 1)

            audio_rec = 0
            video_rec = 0

            # progressive loop
            for i in range(self.num_blocks):
                # audio_latent = self.audio_block(torch.cat((audio_embed, fusion_latent), dim=1))
                # video_latent = self.video_block(torch.cat((video_embed, fusion_latent), dim=1))
                fused_token = audio_token
                audio_latent = self.audio_branch(audio_embed + audio_rec, fused_token)
                video_latent = self.video_branch(video_embed + video_rec, fused_token)

                audio_token = audio_latent[:, -self.fusion_channels:]

                audio_rec = audio_latent[:, :-self.fusion_channels]
                video_rec = video_latent[:, :-self.fusion_channels]

            mask = audio_rec + video_rec

            fa_m = self._masking(fa_in, mask)

            fa_m = fa_m.view(b * self.num_sources, self.audio_hidden_channels, -1)
            s = self.audio_decoder(fa_m)
            s = s.view(b, self.num_sources, -1)
            s = self._trim_output(s, T)
            return s
        else:
            return self.visualize(audio, video, clean_audio)


class ProgressiveBranchNoCR(ProgressiveBranch):
    def forward(self, audio, video, clean_audio=None):
        """
        :param audio: (B, 1, time * sample rate)
        :param video: (B, 1, time * fps, latent)
        :return:
        """
        if clean_audio is None:
            b, T = audio.shape[0], audio.shape[-1]
            M, F = video.shape[1], video.shape[2]

            # Get audio features, fa
            x = self._pad_input(audio)
            fa_in = self.audio_encoder(x)
            fa = self.audio_norm(fa_in)
            audio_embed = self.audio_bottleneck(fa)

            # Get video features, fv
            # fv = video
            fv = video.permute(0, 1, 3, 2).reshape(b, M * self.video_embedding_dim, -1)
            video_embed = self.video_bottleneck(fv)
            video_embed = torch_f.interpolate(video_embed, size=audio_embed.shape[2:])
            #
            # print(audio_embed.shape)  # (B, 512, 1601)
            # print(video_embed.shape)  # (B, 512, 1601)

            audio_token = self.audio_token.unsqueeze(0).repeat(b, 1, 1)
            video_token = self.video_token.unsqueeze(0).repeat(b, 1, 1)

            audio_rec = 0
            video_rec = 0

            # progressive loop
            for i in range(self.num_blocks):
                # audio_latent = self.audio_block(torch.cat((audio_embed, fusion_latent), dim=1))
                # video_latent = self.video_block(torch.cat((video_embed, fusion_latent), dim=1))
                audio_latent = self.audio_branch(audio_embed + audio_rec, audio_token)
                video_latent = self.video_branch(video_embed + video_rec, video_token)

                audio_token = audio_latent[:, -self.fusion_channels:]
                video_token = video_latent[:, -self.fusion_channels:]

                audio_rec = audio_latent[:, :-self.fusion_channels]
                video_rec = video_latent[:, :-self.fusion_channels]

            mask = audio_rec + video_rec

            fa_m = self._masking(fa_in, mask)

            fa_m = fa_m.view(b * self.num_sources, self.audio_hidden_channels, -1)
            s = self.audio_decoder(fa_m)
            s = s.view(b, self.num_sources, -1)
            s = self._trim_output(s, T)
            return s
        else:
            return self.visualize(audio, video, clean_audio)


class ProgressiveBranchNoC(ProgressiveBranch):
    def __init__(self,
                 num_blocks,
                 num_sources: int = 2,
                 video_dim: int = 1024,
                 kernel_size: int = 40,
                 fusion_channels: int = 512,
                 audio_hidden_channels: int = 512,
                 audio_bottleneck_channels: int = 128,
                 audio_states: int = 5,
                 video_hidden_channels: int = 128,
                 video_bottleneck_channels: int = 128,
                 video_states: int = 5,
                 ):
        super().__init__(num_blocks=num_blocks, num_sources=num_sources, video_dim=video_dim, kernel_size=kernel_size,
                         fusion_channels=fusion_channels,
                         audio_hidden_channels=audio_hidden_channels,
                         audio_bottleneck_channels=audio_bottleneck_channels,
                         audio_states=audio_states,
                         video_hidden_channels=video_hidden_channels,
                         video_bottleneck_channels=video_bottleneck_channels,
                         video_states=video_states)

        self.audio_branch = IterativeBranch(
            num_sources=num_sources,
            hidden_channels=audio_hidden_channels,
            bottleneck_channels=audio_bottleneck_channels,
            states=audio_states,
            fusion_channels=0
        )

        self.video_branch = IterativeBranch(
            num_sources=num_sources,
            hidden_channels=video_hidden_channels,
            bottleneck_channels=video_bottleneck_channels,
            states=audio_states,
            fusion_channels=0
        )

        self.audio_token = None
        self.video_token = None

    def forward(self, audio, video, clean_audio=None):
        """
        :param audio: (B, 1, time * sample rate)
        :param video: (B, 1, time * fps, latent)
        :return:
        """
        if clean_audio is None:
            b, T = audio.shape[0], audio.shape[-1]
            M, F = video.shape[1], video.shape[2]

            # Get audio features, fa
            x = self._pad_input(audio)
            fa_in = self.audio_encoder(x)
            fa = self.audio_norm(fa_in)
            audio_embed = self.audio_bottleneck(fa)

            # Get video features, fv
            # fv = video
            fv = video.permute(0, 1, 3, 2).reshape(b, M * self.video_embedding_dim, -1)
            video_embed = self.video_bottleneck(fv)
            video_embed = torch_f.interpolate(video_embed, size=audio_embed.shape[2:])
            #
            # print(audio_embed.shape)  # (B, 512, 1601)
            # print(video_embed.shape)  # (B, 512, 1601)

            audio_rec = 0
            video_rec = 0

            # progressive loop
            for i in range(self.num_blocks):
                # audio_latent = self.audio_block(torch.cat((audio_embed, fusion_latent), dim=1))
                # video_latent = self.video_block(torch.cat((video_embed, fusion_latent), dim=1))
                audio_latent = self.audio_branch(audio_embed + audio_rec)
                video_latent = self.video_branch(video_embed + video_rec)

                audio_rec = audio_latent
                video_rec = video_latent

            mask = audio_rec + video_rec

            fa_m = self._masking(fa_in, mask)

            fa_m = fa_m.view(b * self.num_sources, self.audio_hidden_channels, -1)
            s = self.audio_decoder(fa_m)
            s = s.view(b, self.num_sources, -1)
            s = self._trim_output(s, T)
            return s
        else:
            return self.visualize(audio, video, clean_audio)


class ProgressiveBranchEarly(ProgressiveBranch):

    def __init__(self,
                 num_blocks,
                 num_sources: int = 2,
                 video_dim: int = 1024,
                 kernel_size: int = 40,
                 fusion_channels: int = 512,
                 audio_hidden_channels: int = 512,
                 audio_bottleneck_channels: int = 128,
                 audio_states: int = 5,
                 video_hidden_channels: int = 128,
                 video_bottleneck_channels: int = 128,
                 video_states: int = 5,
                 ):
        super().__init__(num_blocks=num_blocks, num_sources=num_sources, video_dim=video_dim, kernel_size=kernel_size,
                         fusion_channels=fusion_channels,
                         audio_hidden_channels=audio_hidden_channels,
                         audio_bottleneck_channels=audio_bottleneck_channels,
                         audio_states=audio_states,
                         video_hidden_channels=video_hidden_channels,
                         video_bottleneck_channels=video_bottleneck_channels,
                         video_states=video_states)

        self.audio_branch = None

        self.video_branch = None

        self.audio_token = None
        self.video_token = None

        self.fusion_branch = IterativeBranch(
            num_sources=num_sources,
            hidden_channels=audio_hidden_channels + video_hidden_channels,
            bottleneck_channels=audio_bottleneck_channels + video_bottleneck_channels,
            states=audio_states,
            fusion_channels=0
        )

        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(
                in_channels=audio_bottleneck_channels + video_bottleneck_channels,
                out_channels=num_sources * audio_hidden_channels,
                kernel_size=1,
            ),
        )

    def forward(self, audio, video, clean_audio=None):
        """
        :param audio: (B, 1, time * sample rate)
        :param video: (B, 1, time * fps, latent)
        :return:
        """
        if clean_audio is None:
            b, T = audio.shape[0], audio.shape[-1]
            M, F = video.shape[1], video.shape[2]

            # Get audio features, fa
            x = self._pad_input(audio)
            fa_in = self.audio_encoder(x)
            fa = self.audio_norm(fa_in)
            audio_embed = self.audio_bottleneck(fa)

            # Get video features, fv
            # fv = video
            fv = video.permute(0, 1, 3, 2).reshape(b, M * self.video_embedding_dim, -1)
            video_embed = self.video_bottleneck(fv)
            video_embed = torch_f.interpolate(video_embed, size=audio_embed.shape[2:])
            #
            # print(audio_embed.shape)  # (B, 512, 1601)
            # print(video_embed.shape)  # (B, 512, 1601)

            fusion_embed = torch.cat((audio_embed, video_embed), dim=1)

            fusion_rec = 0

            # progressive loop
            for i in range(self.num_blocks):
                # audio_latent = self.audio_block(torch.cat((audio_embed, fusion_latent), dim=1))
                # video_latent = self.video_block(torch.cat((video_embed, fusion_latent), dim=1))
                fusion_latent = self.fusion_branch(fusion_embed + fusion_rec)
                fusion_rec = fusion_latent

            mask = fusion_rec

            fa_m = self._masking(fa_in, mask)

            fa_m = fa_m.view(b * self.num_sources, self.audio_hidden_channels, -1)
            s = self.audio_decoder(fa_m)
            s = s.view(b, self.num_sources, -1)
            s = self._trim_output(s, T)
            return s
        else:
            return self.visualize(audio, video, clean_audio)


class ProgressiveBranchLowDim(ProgressiveBranch):
    def __init__(self,
                 num_blocks,
                 num_sources: int = 2,
                 video_dim: int = 1024,
                 kernel_size: int = 40,
                 fusion_channels: int = 8,
                 audio_hidden_channels: int = 512,
                 audio_bottleneck_channels: int = 8,
                 audio_states: int = 5,
                 video_hidden_channels: int = 128,
                 video_bottleneck_channels: int = 8,
                 video_states: int = 5,
                 ):
        super().__init__(num_blocks=num_blocks, num_sources=num_sources, video_dim=video_dim, kernel_size=kernel_size,
                         fusion_channels=fusion_channels,
                         audio_hidden_channels=audio_hidden_channels,
                         audio_bottleneck_channels=audio_bottleneck_channels,
                         audio_states=audio_states,
                         video_hidden_channels=video_hidden_channels,
                         video_bottleneck_channels=video_bottleneck_channels,
                         video_states=video_states)


class ProgressiveBranchLowDimLate(ProgressiveBranch):
    def __init__(self,
                 num_blocks,
                 num_sources: int = 2,
                 video_dim: int = 1024,
                 kernel_size: int = 40,
                 fusion_channels: int = 8,
                 audio_hidden_channels: int = 512,
                 audio_bottleneck_channels: int = 8,
                 audio_states: int = 5,
                 video_hidden_channels: int = 128,
                 video_bottleneck_channels: int = 8,
                 video_states: int = 5,
                 ):
        super().__init__(num_blocks=num_blocks, num_sources=num_sources, video_dim=video_dim, kernel_size=kernel_size,
                         fusion_channels=fusion_channels,
                         audio_hidden_channels=audio_hidden_channels,
                         audio_bottleneck_channels=audio_bottleneck_channels,
                         audio_states=audio_states,
                         video_hidden_channels=video_hidden_channels,
                         video_bottleneck_channels=video_bottleneck_channels,
                         video_states=video_states)

        self.audio_branch = IterativeBranch(
            num_sources=num_sources,
            hidden_channels=audio_hidden_channels,
            bottleneck_channels=audio_bottleneck_channels,
            states=audio_states,
            fusion_channels=0
        )

        self.video_branch = IterativeBranch(
            num_sources=num_sources,
            hidden_channels=video_hidden_channels,
            bottleneck_channels=video_bottleneck_channels,
            states=audio_states,
            fusion_channels=0
        )

        self.audio_token = None
        self.video_token = None

    def forward(self, audio, video, clean_audio=None):
        """
        :param audio: (B, 1, time * sample rate)
        :param video: (B, 1, time * fps, latent)
        :return:
        """
        if clean_audio is None:
            b, T = audio.shape[0], audio.shape[-1]
            M, F = video.shape[1], video.shape[2]

            # Get audio features, fa
            x = self._pad_input(audio)
            fa_in = self.audio_encoder(x)
            fa = self.audio_norm(fa_in)
            audio_embed = self.audio_bottleneck(fa)

            # Get video features, fv
            # fv = video
            fv = video.permute(0, 1, 3, 2).reshape(b, M * self.video_embedding_dim, -1)
            video_embed = self.video_bottleneck(fv)
            video_embed = torch_f.interpolate(video_embed, size=audio_embed.shape[2:])
            #
            # print(audio_embed.shape)  # (B, 512, 1601)
            # print(video_embed.shape)  # (B, 512, 1601)

            audio_rec = 0
            video_rec = 0

            # progressive loop
            for i in range(self.num_blocks):
                # audio_latent = self.audio_block(torch.cat((audio_embed, fusion_latent), dim=1))
                # video_latent = self.video_block(torch.cat((video_embed, fusion_latent), dim=1))
                audio_latent = self.audio_branch(audio_embed + audio_rec)
                video_latent = self.video_branch(video_embed + video_rec)

                audio_rec = audio_latent
                video_rec = video_latent

            mask = audio_rec + video_rec

            fa_m = self._masking(fa_in, mask)

            fa_m = fa_m.view(b * self.num_sources, self.audio_hidden_channels, -1)
            s = self.audio_decoder(fa_m)
            s = s.view(b, self.num_sources, -1)
            s = self._trim_output(s, T)
            return s
        else:
            return self.visualize(audio, video, clean_audio)