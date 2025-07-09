import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.optim import lr_scheduler
import numpy as np
import random
import csv
import pandas as pd
from scipy.io.wavfile import write
from PIL import Image
import yaml

import models.device_check as dc
import models.avlit as avlit
import models.progressive_fusion as pro_fusion
import models.progressive_mbt as pro_mbt
import models.progressive_separation as pro_sep
import engines.abs_separation as abs_sep
import engines.double_separation as double_sep

import dataset.lrs3.dataset as lrs3_set
import dataset.lrs3_wham.dataset as lrs3_wham_set
import models.model_config
import plot.curve as curve
import utils.dynamic as dynamic
import models.loss as loss
import utils.audio as audio_util
import utils.distributed as distributed_util
import dataset.tcd_timit.dataset as ntcd_set
import models.rtfs.tdavnet as rtfs_net
import models.progressive_rtfs as pro_rtfs

import matplotlib.pylab as plt

import matspy
from scipy.io.wavfile import write

from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

pretrain_root_dir = "/project/pi_mfiterau_umass_edu/dolby/speech-enhancement-fusion/pretrained_models"
original_root_dir = "/project/pi_mfiterau_umass_edu/dolby/speech-enhancement-fusion"


matspy.params.title = False
matspy.params.indices = False


class LRS3RTFSSingle(double_sep.AbsDoubleSeparation):
    def init_data(self):
        self.train_set = lrs3_wham_set.LRS3ProcessedSingleSet(split="train")

        self.val_set = lrs3_wham_set.LRS3ProcessedSingleSet(split="val")

        self.test_set = lrs3_wham_set.LRS3ProcessedSingleSet(split="test")

        self.train_sampler = torch.utils.data.DistributedSampler(self.train_set,
                                                                 num_replicas=max(self.args.n_gpu_per_node, 1),
                                                                 rank=self.args.local_rank)

        self.test_sampler = torch.utils.data.DistributedSampler(self.test_set,
                                                                num_replicas=max(self.args.n_gpu_per_node, 1),
                                                                rank=self.args.local_rank)
        self.val_sampler = torch.utils.data.DistributedSampler(self.val_set,
                                                               num_replicas=max(self.args.n_gpu_per_node, 1),
                                                               rank=self.args.local_rank)

        self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size=self.cfg["train"]["batch_size"],
                                                        sampler=self.train_sampler,
                                                        pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(self.val_set,
                                                      batch_size=self.cfg["train"]["batch_size"],
                                                      sampler=self.val_sampler,
                                                      pin_memory=True)

        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=self.cfg["train"]["batch_size"],
                                                       sampler=self.test_sampler,
                                                       pin_memory=True)

    def init_model(self):
        cfg_train = self.cfg["train"]

        with open(os.path.join('./config/lrs3/rtfs12_single_model.yaml'), 'r') as f:
            rtfs_cfg = yaml.load(f, Loader=yaml.FullLoader)

        fusion_model = rtfs_net.AVNet(**rtfs_cfg["audionet"])

        self.fusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(fusion_model.to(self.device))

        # self.fusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        #     pro_fusion.ProgressiveAFRCNN(num_blocks=4).to(self.device))

        self.fusion_model = torch.nn.parallel.DistributedDataParallel(
            self.fusion_model,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True,
        )

        self.optim = torch.optim.AdamW(self.fusion_model.parameters(),
                                       lr=cfg_train["lr"], betas=(0.9, 0.98),
                                       weight_decay=cfg_train.get("weight_decay", 0.01))

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=10)
        # self.scheduler = lr_scheduler.StepLR(self.optim, step_size=25, gamma=1 / 3)

        loss_config = cfg_train.get("loss_config", "PermInvariantSISDR_Train")
        loss_config_dict = dynamic.import_string("models.model_config.{}".format(loss_config))
        loss_config_dict["batch_size"] = cfg_train["batch_size"]

        # self.loss_func = loss.SISDR_Recover(loss_config_dict,
        #                                     num_chunks=cfg_train.get("num_chunks", 5),
        #                                     recover_weight=cfg_train.get("recover_weight", 0.05)).to(self.device)

        self.loss_func = loss.SISDRPesq(num_chunks=cfg_train.get("num_chunks", 5),
                                        recover_weight=cfg_train.get("recover_weight", 0)).to(self.device)

    def scheduler_step(self, cur_loss):
        self.scheduler.step(cur_loss)

    def forward_pass(self, input_tuple):
        # mixture = input_tuple["mix_audio"].float().to(self.device)
        # visual_feature = input_tuple["video_npy"].float().to(self.device)
        # clean_audio = input_tuple["clean_audio"].to(self.device)
        mixture = input_tuple["mix_audio"].float().to(self.device)
        visual_feature = input_tuple["video_npy"].float().to(self.device)
        spk_idx = input_tuple["target_spk"].cpu().detach().numpy().astype(int)

        clean_audio = input_tuple["clean_audio"].float()
        clean_audio = clean_audio[torch.arange(len(clean_audio)), spk_idx, :].unsqueeze(1)

        # mixture: (batch_size, 32000)
        # visual: (batch_size, 1, 50, 1024)
        # clean: (batch_size, 1, 32000)

        predict_audio = self.fusion_model(mixture.unsqueeze(1), visual_feature)
        # predict_audio: (batch_size, 1, 32000)
        mixture = mixture.detach().cpu()
        visual_feature = visual_feature.detach().cpu()

        # del visual_feature
        # del mixture
        # torch.cuda.empty_cache()

        loss_dict = self.loss_func(predict_audio, clean_audio.to(self.device))

        return predict_audio, clean_audio, loss_dict

    def evaluate(self, data_loader):
        cfg_train = self.cfg["train"]
        reference_drop_type = cfg_train.get("reference_drop_type", None)  # random, end, start
        reference_drop_rate = float(cfg_train.get("reference_drop_rate", 0))
        sampling_rate = int(cfg_train.get("sampling_rate", 16000))

        # eval_sisdr = loss.PermInvariantSISDR(**models.model_config.PermInvariantSISDR_Eval).to(self.device)

        dropped_frames = []
        num_dropped_samples = int(self.video_len * reference_drop_rate)
        if reference_drop_type == "random":
            dropped_frames = random.choices(range(self.video_len), k=num_dropped_samples)
        elif reference_drop_type == "start":
            dropped_frames = range(num_dropped_samples, self.video_len)
        elif reference_drop_type == "end":
            dropped_frames = range(self.video_len - num_dropped_samples)

        test_details_output_file = os.path.join(self.result_path,
                                                f"test_details_{reference_drop_type}_{reference_drop_rate}_{self.args.local_rank}.csv")
        if os.path.exists(test_details_output_file):
            os.remove(test_details_output_file)

        pesq_metric = 0
        stoi_metric = 0
        sisdri_metric = 0
        sdr_metric = 0
        sdri_metric = 0

        test_details = {
            "overlap_ms": [],
            "predict_pesq_loss": [],
            "predict_loss": [],
            "predict_sisdr": [],
            "predict_sisdri": [],
            "predict_pesq": [],
            "predict_estoi": [],
            "predict_sdr": [],
            "mix_sdr": [],
            "sdri": [],
            "mixture_path": []
            # "clean_audio_path": [],
        }

        with torch.no_grad():
            for batch_i, batch_data in enumerate(data_loader):
                self.logger.info(f"Local Rank {self.args.local_rank}: Evaluation: Batch {batch_i} / {len(data_loader)}")
                # ve = batch_data["visual_feature"].float()
                ve = batch_data["video_npy"].float()
                spk_idx = batch_data["target_spk"].cpu().detach().numpy().astype(int)

                clean_audio = batch_data["clean_audio"].float()
                clean_audio = clean_audio[torch.arange(len(clean_audio)), spk_idx, :].unsqueeze(1)

                for i in range(len(ve)):
                    test_details["mixture_path"].append(
                        f'{batch_data["spk1_id"][i]}_{batch_data["spk1_vid"][i]}_{batch_data["spk2_id"][i]}_{batch_data["spk2_vid"][i]}')

                    overlap_ms = batch_data["overlap_ms"][i]
                    if overlap_ms >= 0:
                        main_speaker_start = 0
                    else:
                        main_speaker_start = int(overlap_ms / 1000 * 25) + 75
                    dropped_frames_i = [d + main_speaker_start for d in dropped_frames]
                    for f in dropped_frames_i:
                        ve[i, f, :] = 0.0

                # predict_audio = self.fusion_model(batch_data["mixture"].float().unsqueeze(1).to(self.device),
                #                                   ve.unsqueeze(1).to(self.device))

                predict_audio = self.fusion_model(batch_data["mix_audio"].float().unsqueeze(1).to(self.device),
                                                  ve.to(self.device))  # (B, num sources, len)
                # del ve
                # torch.cuda.empty_cache()

                clean_audio = clean_audio.to(self.device)

                cur_loss = self.loss_func(predict_audio, clean_audio)

                predict_pesq = cur_loss["pesq"]
                predict_pesq = predict_pesq.detach().cpu().numpy()
                test_details["predict_pesq_loss"].append(predict_pesq)

                predict_loss = cur_loss["main"]
                predict_loss = predict_loss.detach().cpu().numpy()
                test_details["predict_loss"].append(predict_loss)

                # predict_sisdr = eval_sisdr(predict_audio, clean_audio).detach().cpu().numpy()

                # Compute PESQ and ESTOI and SISDR

                # true_wav = batch_data["clean_audio"].to(self.device)  # (B, num sources, len)

                if "overlap_ms" in batch_data.keys():
                    test_details["overlap_ms"].append(batch_data["overlap_ms"].detach().cpu().numpy())

                sdr = audio_util.cal_multisource_sdr(clean_audio, predict_audio)
                test_details["predict_sdr"].append(sdr.detach().cpu().numpy())

                sdr_mix = audio_util.cal_SDR(clean_audio.mean(dim=1),
                                             batch_data["mix_audio"].to(self.device))
                test_details["mix_sdr"].append(sdr_mix.detach().cpu().numpy())
                sdri = sdr - sdr_mix

                test_details["sdri"].append(sdri.detach().cpu().numpy())

                sdr_metric += torch.mean(sdr).detach().cpu().numpy()
                sdri_metric += torch.mean(sdri).detach().cpu().numpy()

                num_source = predict_audio.shape[1]

                pesq = 0  # (batch_size, )
                estoi = 0  # (batch_size, )
                sisdr = 0  # (batch_size, )

                mix_sisdr = 0

                for s in range(num_source):
                    pesq += perceptual_evaluation_speech_quality(predict_audio[:, s, :],
                                                                 clean_audio[:, s, :].to(self.device),
                                                                 fs=sampling_rate,
                                                                 mode="wb") / num_source
                    estoi += short_time_objective_intelligibility(predict_audio[:, s, :],
                                                                  clean_audio[:, s, :].to(self.device),
                                                                  extended=True,
                                                                  fs=sampling_rate) / num_source
                    sisdr += scale_invariant_signal_distortion_ratio(predict_audio[:, s, :],
                                                                     clean_audio[:, s, :],
                                                                     zero_mean=True) / num_source
                    mix_sisdr += scale_invariant_signal_distortion_ratio(
                        batch_data["mix_audio"].float().to(self.device),
                        clean_audio[:, s, :].to(self.device),
                        zero_mean=True) / num_source

                pesq = pesq.detach().cpu().numpy()
                test_details["predict_pesq"].append(pesq)

                estoi = estoi.detach().cpu().numpy()
                test_details["predict_estoi"].append(estoi)

                sisdr = sisdr.detach().cpu().numpy()
                test_details["predict_sisdr"].append(sisdr)

                mix_sisdr = mix_sisdr.detach().cpu().numpy()
                test_details["predict_sisdri"].append(sisdr - mix_sisdr)

                pesq_metric += np.mean(pesq)
                stoi_metric += np.mean(estoi)
                sisdri_metric += np.mean(sisdr - mix_sisdr)

            test_sisdri = sisdri_metric / len(data_loader)
            test_pesq = pesq_metric / len(data_loader)
            test_stoi = stoi_metric / len(data_loader)
            test_sdr = sdr_metric / len(data_loader)
            test_sdri = sdri_metric / len(data_loader)

            for c in test_details.keys():
                if "path" not in c:
                    if "loss" not in c:
                        test_details[c] = np.concatenate(test_details[c], axis=0)
                    else:
                        test_details[c] = np.stack(test_details[c], axis=0)
                    # print(c, test_details[c].shape)

            with open(test_details_output_file, "w") as outfile:
                writer = csv.writer(outfile)
                writer.writerow(test_details.keys())
                writer.writerows(zip(*test_details.values()))

            # LOGGER.info(f"Test loss: {test_loss}")
            # LOGGER.info(f"Test PESQ: {test_pesq}")
            # LOGGER.info(f"Test STOI: {test_stoi}")
            # LOGGER.info(f"Test SDR: {test_sdr}")
            # LOGGER.info(f"Test SDRi: {test_sdri}")

            return {"main": test_details["predict_loss"].mean(),
                    "sisdr": test_details["predict_sisdr"].mean(),
                    "pesq": test_details["predict_pesq_loss"].mean()}, \
                {"PESQ": test_details["predict_pesq"].mean(),
                 "ESTOI": test_details["predict_estoi"].mean(),
                 "SDR": test_details["predict_sdr"].mean(),
                 "SI-SDRi": test_details["predict_sisdri"].mean()}


class LRS3RFS(double_sep.AbsDoubleSeparation):
    def init_data(self):
        self.train_set = lrs3_wham_set.LRS3ProcessedSingleSet(split="train")

        self.val_set = lrs3_wham_set.LRS3ProcessedSingleSet(split="val")

        self.test_set = lrs3_wham_set.LRS3ProcessedSingleSet(split="test")

        self.train_sampler = torch.utils.data.DistributedSampler(self.train_set,
                                                                 num_replicas=max(self.args.n_gpu_per_node, 1),
                                                                 rank=self.args.local_rank)

        self.test_sampler = torch.utils.data.DistributedSampler(self.test_set,
                                                                num_replicas=max(self.args.n_gpu_per_node, 1),
                                                                rank=self.args.local_rank)
        self.val_sampler = torch.utils.data.DistributedSampler(self.val_set,
                                                               num_replicas=max(self.args.n_gpu_per_node, 1),
                                                               rank=self.args.local_rank)

        self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size=self.cfg["train"]["batch_size"],
                                                        sampler=self.train_sampler,
                                                        pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(self.val_set,
                                                      batch_size=self.cfg["train"]["batch_size"],
                                                      sampler=self.val_sampler,
                                                      pin_memory=True)

        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=self.cfg["train"]["batch_size"],
                                                       sampler=self.test_sampler,
                                                       pin_memory=True)

    def init_model(self):
        cfg_train = self.cfg["train"]

        with open(os.path.join('./config/lrs3/rtfs12_model.yaml'), 'r') as f:
            rtfs_cfg = yaml.load(f, Loader=yaml.FullLoader)

        fusion_model = rtfs_net.AVNet(**rtfs_cfg["audionet"])

        self.fusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(fusion_model.to(self.device))

        # self.fusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        #     pro_fusion.ProgressiveAFRCNN(num_blocks=4).to(self.device))

        self.fusion_model = torch.nn.parallel.DistributedDataParallel(
            self.fusion_model,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True,
        )

        self.optim = torch.optim.AdamW(self.fusion_model.parameters(),
                                       lr=cfg_train["lr"], betas=(0.9, 0.98),
                                       weight_decay=cfg_train.get("weight_decay", 0.01))

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=10)
        # self.scheduler = lr_scheduler.StepLR(self.optim, step_size=25, gamma=1 / 3)

        loss_config = cfg_train.get("loss_config", "PermInvariantSISDR_Train")
        loss_config_dict = dynamic.import_string("models.model_config.{}".format(loss_config))
        loss_config_dict["batch_size"] = cfg_train["batch_size"]

        # self.loss_func = loss.SISDR_Recover(loss_config_dict,
        #                                     num_chunks=cfg_train.get("num_chunks", 5),
        #                                     recover_weight=cfg_train.get("recover_weight", 0.05)).to(self.device)

        self.loss_func = loss.SISDRPesq(num_chunks=cfg_train.get("num_chunks", 5),
                                        recover_weight=cfg_train.get("recover_weight", 0)).to(self.device)

    def scheduler_step(self, cur_loss):
        self.scheduler.step(cur_loss)


class LRS3RTFSEval(LRS3RFS):
    def evaluate(self, data_loader):
        sample_rate = 16000
        cfg_train = self.cfg["train"]

        eval_path = os.path.join(self.result_path, f"eval_{cfg_train.get('num_blocks', 12)}")
        os.makedirs(eval_path, exist_ok=True)

        reference_drop_type = cfg_train.get("reference_drop_type", None)  # random, end, start
        reference_drop_rate = float(cfg_train.get("reference_drop_rate", 0))
        sampling_rate = int(cfg_train.get("sampling_rate", 16000))

        # eval_sisdr = loss.PermInvariantSISDR(**models.model_config.PermInvariantSISDR_Eval).to(self.device)

        dropped_frames = []
        num_dropped_samples = int(self.video_len * reference_drop_rate)
        if reference_drop_type == "random":
            dropped_frames = random.choices(range(self.video_len), k=num_dropped_samples)
        elif reference_drop_type == "start":
            dropped_frames = range(num_dropped_samples, self.video_len)
        elif reference_drop_type == "end":
            dropped_frames = range(self.video_len - num_dropped_samples)

        test_details_output_file = os.path.join(eval_path,
                                                f"final_eval_details_{reference_drop_type}_{reference_drop_rate}_{self.args.local_rank}.csv")

        if os.path.exists(test_details_output_file):
            os.remove(test_details_output_file)

        test_details = {
            "overlap_ms": [],
            # evaluation
            "predict_sisdr": [],
            "predict_sisdri": [],
            "predict_pesq": [],
            "predict_estoi": [],
            # separated audio path
            "spk1_sep_path": [],
            "spk2_sep_path": [],
            # clean audio path
            "spk1_path": [],
            "spk2_path": [],
            # id information
            "spk1_id": [],
            "spk1_vid": [],
            "spk2_id": [],
            "spk2_vid": []
        }

        loss_batch_avg = 0
        pesq_batch_avg = 0

        with torch.no_grad():
            for batch_i, batch_data in enumerate(data_loader):
                batch_eval_path = os.path.join(eval_path, f"batch_{batch_i}")
                os.makedirs(batch_eval_path, exist_ok=True)

                self.logger.info(f"Local Rank {self.args.local_rank}: Evaluation: Batch {batch_i} / {len(data_loader)}")
                # ve = batch_data["visual_feature"].float()
                ve = batch_data["video_npy"].float()
                for i in range(len(ve)):
                    test_details["spk1_id"].append(batch_data["spk1_id"][i])
                    test_details["spk1_vid"].append(batch_data["spk1_vid"][i])
                    test_details["spk2_id"].append(batch_data["spk2_id"][i])
                    test_details["spk2_vid"].append(batch_data["spk2_vid"][i])

                    # test_details["mixture_path"].append(
                    #     f'{batch_data["spk1_id"][i]}_{batch_data["spk1_vid"][i]}_{batch_data["spk2_id"][i]}_{batch_data["spk2_vid"][i]}')

                    overlap_ms = batch_data["overlap_ms"][i]
                    if overlap_ms >= 0:
                        main_speaker_start = 0
                    else:
                        main_speaker_start = int(overlap_ms / 1000 * 25) + 75
                    dropped_frames_i = [d + main_speaker_start for d in dropped_frames]
                    for f in dropped_frames_i:
                        ve[i, f, :] = 0.0

                predict_audio = self.fusion_model(batch_data["mix_audio"].float().unsqueeze(1).to(self.device),
                                                  ve.to(self.device))  # (B, num sources, len)

                true_wav = batch_data["clean_audio"].to(self.device)  # (B, num sources, len)

                # compute the main avg loss
                cur_loss = self.loss_func(predict_audio, true_wav)
                predict_pesq = cur_loss["pesq"]
                predict_pesq = predict_pesq.detach().cpu().item()
                pesq_batch_avg += predict_pesq

                predict_loss = cur_loss["main"]
                predict_loss = predict_loss.detach().cpu().item()
                loss_batch_avg += predict_loss

                if "overlap_ms" in batch_data.keys():
                    test_details["overlap_ms"].append(batch_data["overlap_ms"].detach().cpu().numpy())

                num_source = predict_audio.shape[1]

                pesq = 0  # (batch_size, )
                estoi = 0  # (batch_size, )
                sisdr = 0  # (batch_size, )

                mix_sisdr = 0

                for s in range(num_source):
                    pesq += perceptual_evaluation_speech_quality(predict_audio[:, s, :],
                                                                 true_wav[:, s, :].to(self.device),
                                                                 fs=sampling_rate,
                                                                 mode="wb") / num_source
                    estoi += short_time_objective_intelligibility(predict_audio[:, s, :],
                                                                  true_wav[:, s, :].to(self.device),
                                                                  extended=True,
                                                                  fs=sampling_rate) / num_source
                    sisdr += scale_invariant_signal_distortion_ratio(predict_audio[:, s, :],
                                                                     true_wav[:, s, :],
                                                                     zero_mean=True) / num_source
                    mix_sisdr += scale_invariant_signal_distortion_ratio(
                        batch_data["mix_audio"].float().to(self.device),
                        true_wav[:, s, :].to(self.device),
                        zero_mean=True) / num_source

                pesq = pesq.detach().cpu().numpy()
                test_details["predict_pesq"].append(pesq)

                estoi = estoi.detach().cpu().numpy()
                test_details["predict_estoi"].append(estoi)

                sisdr = sisdr.detach().cpu().numpy()
                test_details["predict_sisdr"].append(sisdr)

                mix_sisdr = mix_sisdr.detach().cpu().numpy()
                test_details["predict_sisdri"].append(sisdr - mix_sisdr)

                # dump the audio file from predict_audio and true wav (both in shape (B, num_sources, 32000))
                for b in range(predict_audio.shape[0]):
                    sample_eval_path = os.path.join(batch_eval_path, f"sample{b}_gpu{self.args.local_rank}")
                    os.makedirs(sample_eval_path, exist_ok=True)

                    spk1_id = batch_data["spk1_id"][b]
                    spk2_id = batch_data["spk2_id"][b]
                    spk1_vid = batch_data["spk1_vid"][b]
                    spk2_vid = batch_data["spk2_vid"][b]

                    spk1_path = os.path.join(sample_eval_path, f"{spk1_id}_{spk1_vid}_clean.wav")
                    spk2_path = os.path.join(sample_eval_path, f"{spk2_id}_{spk2_vid}_clean.wav")

                    spk1_sep_path = os.path.join(sample_eval_path, f"{spk1_id}_{spk1_vid}_sep.wav")
                    spk2_sep_path = os.path.join(sample_eval_path, f"{spk2_id}_{spk2_vid}_sep.wav")

                    write(spk1_path, sample_rate, true_wav[b, 0, :].detach().cpu().numpy())
                    write(spk2_path, sample_rate, true_wav[b, 1, :].detach().cpu().numpy())

                    write(spk1_sep_path, sample_rate, predict_audio[b, 0, :].detach().cpu().numpy())
                    write(spk2_sep_path, sample_rate, predict_audio[b, 1, :].detach().cpu().numpy())

                    write(os.path.join(sample_eval_path, f"mixture.wav"), sample_rate, batch_data["mix_audio"][b].float().detach().cpu().numpy())

                    test_details["spk1_path"].append(spk1_path)
                    test_details["spk1_sep_path"].append(spk1_sep_path)
                    test_details["spk2_path"].append(spk2_path)
                    test_details["spk2_sep_path"].append(spk2_sep_path)

            loss_batch_avg /= len(data_loader)
            pesq_batch_avg /= len(data_loader)

            for c in test_details.keys():
                if "predict" in c or "overlap_ms" in c:
                    test_details[c] = np.concatenate(test_details[c], axis=0)
                    print(c, test_details[c].shape)

            # save the csv file
            df = pd.DataFrame.from_dict(test_details)
            df.to_csv(test_details_output_file)

            self.logger.debug(f"Local Rank {self.args.local_rank}: wait for synchronization. Final evaluation")
            torch.distributed.barrier()

            # merge all test csv
            if self.args.local_rank == 0:
                df_list = []
                for rank in range(self.args.n_gpu_per_node):
                    df = pd.read_csv(os.path.join(eval_path,
                                                f"final_eval_details_{reference_drop_type}_{reference_drop_rate}_{rank}.csv"))
                    df_list.append(df)

                concat_df = pd.concat(df_list)
                concat_df.to_csv(os.path.join(eval_path,
                                                f"final_eval_details_{reference_drop_type}_{reference_drop_rate}.csv"))

            self.logger.debug(f"Local Rank {self.args.local_rank}: wait for synchronization. GPU 0 merge all csv records")
            torch.distributed.barrier()

            return {"main": loss_batch_avg,
                    "sisdr": test_details["predict_sisdr"].mean(),
                    "pesq": pesq_batch_avg}, \
                {"PESQ": test_details["predict_pesq"].mean(),
                 "ESTOI": test_details["predict_estoi"].mean(),
                 "SI-SDRi": test_details["predict_sisdri"].mean()}


class LRS3ProRTFS(LRS3RFS):
    def init_model(self):
        cfg_train = self.cfg["train"]

        with open(os.path.join('./config/lrs3/profusion_rtfs12_model.yaml'), 'r') as f:
            rtfs_cfg = yaml.load(f, Loader=yaml.FullLoader)

        if "num_blocks" in cfg_train:
            rtfs_cfg["audionet"]["num_blocks"] = cfg_train["num_blocks"]

        if "audio_fusion_dim" in cfg_train:
            rtfs_cfg["audionet"]["audio_fusion_dim"] = cfg_train["audio_fusion_dim"]

        if "video_fusion_dim" in cfg_train:
            rtfs_cfg["audionet"]["video_fusion_dim"] = cfg_train["video_fusion_dim"]

        print("model param: ", rtfs_cfg)

        fusion_model = pro_rtfs.AVNet(**rtfs_cfg["audionet"])

        self.fusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(fusion_model.to(self.device))

        # self.fusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        #     pro_fusion.ProgressiveAFRCNN(num_blocks=4).to(self.device))

        self.fusion_model = torch.nn.parallel.DistributedDataParallel(
            self.fusion_model,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True,
        )

        self.optim = torch.optim.AdamW(self.fusion_model.parameters(),
                                       lr=cfg_train["lr"], betas=(0.9, 0.98),
                                       weight_decay=cfg_train.get("weight_decay", 0.01))

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=10)
        # self.scheduler = lr_scheduler.StepLR(self.optim, step_size=25, gamma=1 / 3)

        loss_config = cfg_train.get("loss_config", "PermInvariantSISDR_Train")
        loss_config_dict = dynamic.import_string("models.model_config.{}".format(loss_config))
        loss_config_dict["batch_size"] = cfg_train["batch_size"]

        # self.loss_func = loss.SISDR_Recover(loss_config_dict,
        #                                     num_chunks=cfg_train.get("num_chunks", 5),
        #                                     recover_weight=cfg_train.get("recover_weight", 0.05)).to(self.device)

        self.loss_func = loss.SISDRPesq(num_chunks=cfg_train.get("num_chunks", 5),
                                        recover_weight=cfg_train.get("recover_weight", 0)).to(self.device)



class NTCDRTFS(double_sep.AbsDoubleSeparation):
    def init_data(self):
        self.train_set = ntcd_set.NTCDProcessedPairSet(split="train")

        self.val_set = ntcd_set.NTCDProcessedPairSet(split="val")

        self.test_set = ntcd_set.NTCDProcessedPairSet(split="test")

        self.train_sampler = torch.utils.data.DistributedSampler(self.train_set,
                                                                 num_replicas=max(self.args.n_gpu_per_node, 1),
                                                                 rank=self.args.local_rank)

        self.test_sampler = torch.utils.data.DistributedSampler(self.test_set,
                                                                num_replicas=max(self.args.n_gpu_per_node, 1),
                                                                rank=self.args.local_rank)
        self.val_sampler = torch.utils.data.DistributedSampler(self.val_set,
                                                               num_replicas=max(self.args.n_gpu_per_node, 1),
                                                               rank=self.args.local_rank)

        self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size=self.cfg["train"]["batch_size"],
                                                        sampler=self.train_sampler,
                                                        pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(self.val_set,
                                                      batch_size=self.cfg["train"]["batch_size"],
                                                      sampler=self.val_sampler,
                                                      pin_memory=True)

        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=self.cfg["train"]["batch_size"],
                                                       sampler=self.test_sampler,
                                                       pin_memory=True)

    def init_model(self):
        cfg_train = self.cfg["train"]

        with open(os.path.join('./config/ntcd/rtfs8_model.yaml'), 'r') as f:
            rtfs_cfg = yaml.load(f, Loader=yaml.FullLoader)

        fusion_model = rtfs_net.AVNet(**rtfs_cfg["audionet"])

        self.fusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(fusion_model.to(self.device))

        # self.fusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        #     pro_fusion.ProgressiveAFRCNN(num_blocks=4).to(self.device))

        self.fusion_model = torch.nn.parallel.DistributedDataParallel(
            self.fusion_model,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True,
        )

        self.optim = torch.optim.AdamW(self.fusion_model.parameters(),
                                       lr=cfg_train["lr"], betas=(0.9, 0.98),
                                       weight_decay=cfg_train.get("weight_decay", 0.01))

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=10)
        # self.scheduler = lr_scheduler.StepLR(self.optim, step_size=25, gamma=1 / 3)

        loss_config = cfg_train.get("loss_config", "PermInvariantSISDR_Train")
        loss_config_dict = dynamic.import_string("models.model_config.{}".format(loss_config))
        loss_config_dict["batch_size"] = cfg_train["batch_size"]

        # self.loss_func = loss.SISDR_Recover(loss_config_dict,
        #                                     num_chunks=cfg_train.get("num_chunks", 5),
        #                                     recover_weight=cfg_train.get("recover_weight", 0.05)).to(self.device)

        self.loss_func = loss.SISDRPesq(num_chunks=cfg_train.get("num_chunks", 5),
                                        recover_weight=cfg_train.get("recover_weight", 0)).to(self.device)

    def scheduler_step(self, cur_loss):
        self.scheduler.step(cur_loss)


class NTCDRTFSSingle(LRS3RTFSSingle):
    def init_data(self):
        self.train_set = ntcd_set.NTCDProcessedSingleSet(split="train")

        self.val_set = ntcd_set.NTCDProcessedSingleSet(split="val")

        self.test_set = ntcd_set.NTCDProcessedSingleSet(split="test")

        self.train_sampler = torch.utils.data.DistributedSampler(self.train_set,
                                                                 num_replicas=max(self.args.n_gpu_per_node, 1),
                                                                 rank=self.args.local_rank)

        self.test_sampler = torch.utils.data.DistributedSampler(self.test_set,
                                                                num_replicas=max(self.args.n_gpu_per_node, 1),
                                                                rank=self.args.local_rank)
        self.val_sampler = torch.utils.data.DistributedSampler(self.val_set,
                                                               num_replicas=max(self.args.n_gpu_per_node, 1),
                                                               rank=self.args.local_rank)

        self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size=self.cfg["train"]["batch_size"],
                                                        sampler=self.train_sampler,
                                                        pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(self.val_set,
                                                      batch_size=self.cfg["train"]["batch_size"],
                                                      sampler=self.val_sampler,
                                                      pin_memory=True)

        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=self.cfg["train"]["batch_size"],
                                                       sampler=self.test_sampler,
                                                       pin_memory=True)

    def init_model(self):
        cfg_train = self.cfg["train"]

        with open(os.path.join('./config/ntcd/rtfs12_single_model.yaml'), 'r') as f:
            rtfs_cfg = yaml.load(f, Loader=yaml.FullLoader)

        fusion_model = rtfs_net.AVNet(**rtfs_cfg["audionet"])

        self.fusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(fusion_model.to(self.device))

        # self.fusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        #     pro_fusion.ProgressiveAFRCNN(num_blocks=4).to(self.device))

        self.fusion_model = torch.nn.parallel.DistributedDataParallel(
            self.fusion_model,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True,
        )

        self.optim = torch.optim.AdamW(self.fusion_model.parameters(),
                                       lr=cfg_train["lr"], betas=(0.9, 0.98),
                                       weight_decay=cfg_train.get("weight_decay", 0.01))

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=10)
        # self.scheduler = lr_scheduler.StepLR(self.optim, step_size=25, gamma=1 / 3)

        loss_config = cfg_train.get("loss_config", "PermInvariantSISDR_Train")
        loss_config_dict = dynamic.import_string("models.model_config.{}".format(loss_config))
        loss_config_dict["batch_size"] = cfg_train["batch_size"]

        # self.loss_func = loss.SISDR_Recover(loss_config_dict,
        #                                     num_chunks=cfg_train.get("num_chunks", 5),
        #                                     recover_weight=cfg_train.get("recover_weight", 0.05)).to(self.device)

        self.loss_func = loss.SISDRPesq(num_chunks=cfg_train.get("num_chunks", 5),
                                        recover_weight=cfg_train.get("recover_weight", 0)).to(self.device)

    def scheduler_step(self, cur_loss):
        self.scheduler.step(cur_loss)