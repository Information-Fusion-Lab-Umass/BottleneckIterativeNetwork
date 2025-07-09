import os
import torch
import torch.nn as nn
import itertools

from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models.Xvector import Xvector
from torch_pesq import PesqLoss
from torch_stoi import NegSTOILoss

from models.device_check import *
import utils.audio as audio

pretrain_root_dir = "/project/pi_mfiterau_umass_edu/dolby/speech-enhancement-fusion/pretrained_models"


class PermInvariantSISDR(nn.Module):
    """!
    Class for SISDR computation between reconstructed signals and
    target wavs by also regulating it with learned target masks.
    ABORTED
    """

    def __init__(self,
                 batch_size=None,
                 zero_mean=False,
                 n_sources=None,
                 backward_loss=True,
                 improvement=False,
                 return_individual_results=False):
        """
        Initialization for the results and torch tensors that might
        be used afterwards

        :param batch_size: The number of the samples in each batch
        :param zero_mean: If you want to perform zero-mean across
        last dimension (time dim) of the signals before SDR computation
        """
        super().__init__()
        self.bs = batch_size
        self.perform_zero_mean = zero_mean
        self.backward_loss = backward_loss
        self.permutations = list(itertools.permutations(
            torch.arange(n_sources)))
        self.permutations_tensor = torch.LongTensor(self.permutations)
        self.improvement = improvement
        self.n_sources = n_sources
        self.return_individual_results = return_individual_results

    def normalize_input(self, pr_batch, t_batch, initial_mixtures=None):
        min_len = min(pr_batch.shape[-1],
                      t_batch.shape[-1])
        if initial_mixtures is not None:
            min_len = min(min_len, initial_mixtures.shape[-1])
            initial_mixtures = initial_mixtures[:, :, :min_len]
        pr_batch = pr_batch[:, :, :min_len]
        t_batch = t_batch[:, :, :min_len]

        if self.perform_zero_mean:
            pr_batch = pr_batch - torch.mean(
                pr_batch, dim=-1, keepdim=True)
            t_batch = t_batch - torch.mean(
                t_batch, dim=-1, keepdim=True)
            if initial_mixtures is not None:
                initial_mixtures = initial_mixtures - torch.mean(
                    initial_mixtures, dim=-1, keepdim=True)
        return pr_batch, t_batch, initial_mixtures

    @staticmethod
    def dot(x, y):
        return torch.sum(x * y, dim=-1, keepdim=True)

    def compute_permuted_sisnrs(self,
                                permuted_pr_batch,
                                t_batch,
                                t_t_diag, eps=10e-8):
        s_t = (self.dot(permuted_pr_batch, t_batch) /
               (t_t_diag + eps) * t_batch)
        e_t = permuted_pr_batch - s_t
        sisnrs = 10 * torch.log10(self.dot(s_t, s_t) /
                                  (self.dot(e_t, e_t) + eps))
        return sisnrs

    def compute_sisnr(self,
                      pr_batch,
                      t_batch,
                      initial_mixtures=None,
                      eps=10e-8):

        t_t_diag = self.dot(t_batch, t_batch)

        sisnr_l = []
        for perm in self.permutations:
            permuted_pr_batch = pr_batch[:, perm, :]
            sisnr = self.compute_permuted_sisnrs(permuted_pr_batch,
                                                 t_batch,
                                                 t_t_diag, eps=eps)
            sisnr_l.append(sisnr)
        all_sisnrs = torch.cat(sisnr_l, -1)
        # best_sisdr, best_perm_ind = torch.max(all_sisnrs.mean(-2), -1)
        # change to use the max
        best_sisdr, best_perm_ind = torch.max(all_sisnrs[:, 0, :], -1)

        if self.improvement:
            initial_mix = initial_mixtures.repeat(1, self.n_sources, 1)
            base_sisdr = self.compute_permuted_sisnrs(initial_mix,
                                                      t_batch,
                                                      t_t_diag, eps=eps)
            best_sisdr -= base_sisdr.mean()

        print(best_sisdr.shape)

        if not self.return_individual_results:
            best_sisdr = best_sisdr.mean()

        if self.backward_loss:
            return -best_sisdr, best_perm_ind
        return best_sisdr, best_perm_ind

    def forward(self,
                pr_batch,
                t_batch,
                eps=1e-9,
                initial_mixtures=None,
                return_best_permutation=False):
        """!
        :param pr_batch: Reconstructed wavs: Torch Tensors of size:
                         batch_size x self.n_sources x length_of_wavs
        :param t_batch: Target wavs: Torch Tensors of size:
                        batch_size x self.n_sources x length_of_wavs
        :param eps: Numerical stability constant.
        :param initial_mixtures: Initial Mixtures for SISDRi: Torch Tensor
                                 of size: batch_size x 1 x length_of_wavs

        :returns results_buffer Buffer for loading the results directly
                 to gpu and not having to reconstruct the results matrix: Torch
                 Tensor of size: batch_size x 1
        """
        pr_batch, t_batch, initial_mixtures = self.normalize_input(
            pr_batch, t_batch, initial_mixtures=initial_mixtures)

        sisnr_l, best_perm_ind = self.compute_sisnr(
            pr_batch, t_batch, eps=eps,
            initial_mixtures=initial_mixtures)

        if return_best_permutation:
            best_permutations = self.permutations_tensor[best_perm_ind]
            return sisnr_l, best_perm_ind
        else:
            return sisnr_l


class SISDR_Recover(nn.Module):
    def __init__(self, num_chunks, recover_weight=0.05):
        super().__init__()
        self.sisdr_criterion = ScaleInvariantSignalDistortionRatio(zero_mean=True)

        self.num_chunks = num_chunks
        self.recover_weight = recover_weight

        if recover_weight != 0:
            self.audio_fbank = Fbank(n_mels=24).to(device)
            self.audio_embedding_model = Xvector(device=device,
                                                 activation=torch.nn.LeakyReLU,
                                                 tdnn_blocks=5,
                                                 tdnn_channels=[512, 512, 512, 512, 1500],
                                                 tdnn_kernel_sizes=[5, 3, 3, 1, 1],
                                                 tdnn_dilations=[1, 2, 3, 1, 1],
                                                 lin_neurons=512,
                                                 in_channels=24).to(device)
            self.audio_embedding_model.load_state_dict(
                torch.load(os.path.join(pretrain_root_dir, "spkrec-xvect-voxceleb/embedding_model.ckpt"),
                           map_location=device))
            self.audio_embedding_model.eval()
        else:
            self.audio_embedding_model = None
            self.audio_fbank = None

    def forward(self, predict_audio, clean_audio):
        """
        :param predict_audio: (batch_size, 1, dim)
        :param clean_audio: (batch_size, 1, dim)
        :return:
        """
        loss_t = -self.sisdr_criterion(predict_audio, clean_audio).mean()  # negative sisdr

        num_source = predict_audio.shape[1]

        full_loss_rec = 0

        if self.recover_weight == 0:
            return {"main": loss_t,
                    "si_sdr": loss_t,
                    "recover": torch.tensor(0, dtype=torch.float32)}

        for i in range(num_source):
            predict_audio_emb = audio.get_audio_embedding(predict_audio[:, i, :],
                                                          audio_fbank=self.audio_fbank,
                                                          pretrained_audio_embedding_model=self.audio_embedding_model,
                                                          num_chunks=self.num_chunks)

            clean_audio_emb = audio.get_audio_embedding(clean_audio[:, i, :],
                                                        audio_fbank=self.audio_fbank,
                                                        pretrained_audio_embedding_model=self.audio_embedding_model,
                                                        num_chunks=self.num_chunks)

            if self.num_chunks == 1:
                loss_rec = torch.linalg.norm(predict_audio_emb - clean_audio_emb)
            else:
                loss_rec = torch.max(torch.linalg.norm(predict_audio_emb - clean_audio_emb, dim=-1))

            full_loss_rec += loss_rec / num_source

        return {"main": loss_t + self.recover_weight * full_loss_rec,
                "si_sdr": loss_t,
                "recover": self.recover_weight * full_loss_rec}


class SISDRPesq(nn.Module):
    def __init__(self, num_chunks, recover_weight=0.5):
        super().__init__()
        self.sisdr_criterion = ScaleInvariantSignalDistortionRatio(zero_mean=True)

        # pesq_loss.forward returns a negative value for maximization
        self.pesq_loss = PesqLoss(recover_weight, sample_rate=16000)

        self.num_chunks = num_chunks
        self.recover_weight = recover_weight

    def forward(self, predict_audio, clean_audio):
        """
        :param predict_audio: (batch_size, 1, dim)
        :param clean_audio: (batch_size, 1, dim)
        :return:
        """
        loss_t = -self.sisdr_criterion(predict_audio, clean_audio).mean()  # negative sisdr

        num_source = predict_audio.shape[1]

        # full_loss_rec = 0
        #
        if self.recover_weight == 0:
            return {"main": loss_t,
                    "si_sdr": loss_t,
                    "pesq": torch.tensor(0, dtype=torch.float32)}
        #
        # for i in range(num_source):
        #     predict_chunks = torch.split(predict_audio[:, i, :], self.num_chunks, dim=1)
        #     clean_chunks = torch.split(clean_audio[:, i, :], self.num_chunks, dim=1)
        #
        #     loss_rec = torch.mean(torch.linalg.norm(torch.stack(predict_chunks, dim=1) -
        #                                            torch.stack(clean_chunks, dim=1), dim=-1))
        #
        #     full_loss_rec += loss_rec / num_source

        pesq_loss = 0
        for i in range(num_source):
            pred = predict_audio[:, i, :]
            clean = clean_audio[:, i, :]
            pesq_loss += self.pesq_loss(clean, pred).mean() / num_source

        # return {"main": loss_t + self.recover_weight * full_loss_rec,
        #         "si_sdr": loss_t,
        #         "recover": self.recover_weight * full_loss_rec}

        # return {"main": loss_t + self.recover_weight * pesq_loss,
        #         "si_sdr": loss_t,
        #         "pesq": self.recover_weight * pesq_loss}
        return {"main": loss_t + pesq_loss,
                "si_sdr": loss_t,
                "pesq": pesq_loss / self.recover_weight}


class SISDRStoi(nn.Module):
    def __init__(self, num_chunks, recover_weight=1):
        super().__init__()
        self.sisdr_criterion = ScaleInvariantSignalDistortionRatio(zero_mean=True)

        # pesq_loss.forward returns a negative value for maximization
        self.stoi_loss = NegSTOILoss(sample_rate=16000)

        self.num_chunks = num_chunks
        self.recover_weight = recover_weight

    def forward(self, predict_audio, clean_audio):
        """
        :param predict_audio: (batch_size, 1, dim)
        :param clean_audio: (batch_size, 1, dim)
        :return:
        """
        loss_t = -self.sisdr_criterion(predict_audio, clean_audio).mean()  # negative sisdr

        num_source = predict_audio.shape[1]

        # full_loss_rec = 0
        #
        if self.recover_weight == 0:
            return {"main": loss_t,
                    "si_sdr": loss_t,
                    "stoi": torch.tensor(0, dtype=torch.float32)}
        #
        # for i in range(num_source):
        #     predict_chunks = torch.split(predict_audio[:, i, :], self.num_chunks, dim=1)
        #     clean_chunks = torch.split(clean_audio[:, i, :], self.num_chunks, dim=1)
        #
        #     loss_rec = torch.mean(torch.linalg.norm(torch.stack(predict_chunks, dim=1) -
        #                                            torch.stack(clean_chunks, dim=1), dim=-1))
        #
        #     full_loss_rec += loss_rec / num_source

        stoi_loss = 0
        for i in range(num_source):
            pred = predict_audio[:, i, :]
            clean = clean_audio[:, i, :]
            stoi_loss += self.stoi_loss(pred, clean).mean() / num_source

        # return {"main": loss_t + self.recover_weight * full_loss_rec,
        #         "si_sdr": loss_t,
        #         "recover": self.recover_weight * full_loss_rec}

        # return {"main": loss_t + self.recover_weight * pesq_loss,
        #         "si_sdr": loss_t,
        #         "pesq": self.recover_weight * pesq_loss}
        return {"main": loss_t + self.recover_weight * stoi_loss,
                "si_sdr": loss_t,
                "stoi": stoi_loss}