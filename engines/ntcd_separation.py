import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.optim import lr_scheduler
import numpy as np
import random
import csv
from scipy.io.wavfile import write

import models.avlit as avlit
import models.progressive_fusion as pro_fusion
import engines.abs_separation as abs_sep
import dataset.tcd_timit.dataset as ntcd_set
import models.model_config
import plot.curve as curve
import utils.dynamic as dynamic
import models.loss as loss
import utils.audio as audio_util
import utils.distributed as distributed_util

from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality


class DistributedAVLITSeparation(abs_sep.DistributedSeparation):
    def init_data(self):
        self.train_set = ntcd_set.NTCDDataset(split="train")

        self.val_set = ntcd_set.NTCDDataset(split="val")

        self.test_set = ntcd_set.NTCDDataset(split="test")

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

        # train_subset = torch.utils.data.Subset(self.train_set, range(48))
        # test_subset = torch.utils.data.Subset(self.test_set, range(48))
        # val_subset = torch.utils.data.Subset(self.val_set, range(48))
        #
        # self.train_sampler = torch.utils.data.DistributedSampler(train_subset,
        #                                                          num_replicas=max(self.args.n_gpu_per_node, 1),
        #                                                          rank=self.args.local_rank)
        #
        # self.test_sampler = torch.utils.data.DistributedSampler(test_subset,
        #                                                         num_replicas=max(self.args.n_gpu_per_node, 1),
        #                                                         rank=self.args.local_rank)
        # self.val_sampler = torch.utils.data.DistributedSampler(val_subset,
        #                                                        num_replicas=max(self.args.n_gpu_per_node, 1),
        #                                                        rank=self.args.local_rank)
        #
        # self.train_loader = torch.utils.data.DataLoader(train_subset,
        #                                                 batch_size=self.cfg["train"]["batch_size"],
        #                                                 sampler=self.train_sampler,
        #                                                 pin_memory=True)
        #
        # self.val_loader = torch.utils.data.DataLoader(val_subset,
        #                                               batch_size=self.cfg["train"]["batch_size"],
        #                                               sampler=self.val_sampler,
        #                                               pin_memory=True)
        #
        # self.test_loader = torch.utils.data.DataLoader(test_subset,
        #                                                batch_size=self.cfg["train"]["batch_size"],
        #                                                sampler=self.test_sampler,
        #                                                pin_memory=True)

    def init_model(self):
        cfg_train = self.cfg["train"]

        model_config = cfg_train.get("model_config", "avlit_default")
        model_config_dict = dynamic.import_string("models.model_config.{}".format(model_config))

        fusion_model = avlit.AVLIT(**model_config_dict)
        fusion_model.video_encoder.requires_grad_(False)
        fusion_model.video_encoder.load_state_dict(torch.load(
            "/project/pi_mfiterau_umass_edu/sidong/speech/results/video_autoencoder_ntcd_e500/model.pth.tar")[
                                                       "autoencoder"])

        self.fusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            fusion_model.to(self.device))

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

        # self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.8, patience=5)
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=25, gamma=1/3)

        loss_config = cfg_train.get("loss_config", "PermInvariantSISDR_Train")
        loss_config_dict = dynamic.import_string("models.model_config.{}".format(loss_config))
        loss_config_dict["batch_size"] = cfg_train["batch_size"]

        self.loss_func = loss.SISDR_Recover(loss_config_dict,
                                            num_chunks=cfg_train.get("num_chunks", 5),
                                            recover_weight=cfg_train.get("recover_weight", 0.05)).to(self.device)

    def forward_pass(self, input_tuple):
        # mixture = input_tuple["mixture"].float().to(self.device)
        # visual_feature = input_tuple["visual_feature"].float().to(self.device)
        # clean_audio = input_tuple["clean_audio"].to(self.device)

        mixture = input_tuple["mix_audio"].float().to(self.device)
        visual_feature = input_tuple["video_npy"].float().to(self.device)
        clean_audio = input_tuple["clean_audio"].to(self.device)

        # mixture: (batch_size, 48000)
        # visual: (batch_size, 75, 512)

        predict_audio = self.fusion_model(mixture.unsqueeze(1), visual_feature.unsqueeze(1))
        # predict_audio: (batch_size, 1, 48000)

        loss_dict = self.loss_func(predict_audio, clean_audio.unsqueeze(1))

        return predict_audio, clean_audio.unsqueeze(1), loss_dict

    def evaluate(self, data_loader):
        cfg_train = self.cfg["train"]
        reference_drop_type = cfg_train.get("reference_drop_type", None)  # random, end, start
        reference_drop_rate = float(cfg_train.get("reference_drop_rate", 0))
        sampling_rate = int(cfg_train.get("sampling_rate", 16000))

        eval_sisdr = loss.PermInvariantSISDR(**models.model_config.PermInvariantSISDR_Eval).to(self.device)

        dropped_frames = []
        num_dropped_samples = int(75 * reference_drop_rate)
        if reference_drop_type == "random":
            dropped_frames = random.choices(range(75), k=num_dropped_samples)
        elif reference_drop_type == "start":
            dropped_frames = range(num_dropped_samples, 75)
        elif reference_drop_type == "end":
            dropped_frames = range(75 - num_dropped_samples)

        pesq_metric = 0
        stoi_metric = 0
        sdr_metric = 0
        sdri_metric = 0

        test_details = {
            # "background_speaker_db": [],
            "overlap_ms": [],
            "predict_recover_loss": [],
            "predict_loss": [],
            "predict_sisdr": [],
            "predict_pesq": [],
            "predict_stoi": [],
            "predict_sdr": [],
            "mix_sdr": [],
            "sdri": []
            # "mixture_path": [],
            # "clean_audio_path": [],
        }

        test_details_output_file = os.path.join(self.result_path,
                                                    f"test_details_{reference_drop_type}_{reference_drop_rate}_{self.args.local_rank}.csv")
        if os.path.exists(test_details_output_file):
            os.remove(test_details_output_file)

        with torch.no_grad():
            for batch_i, batch_data in enumerate(data_loader):
                self.logger.info(f"Local Rank {self.args.local_rank}: Evaluation: Batch {batch_i} / {len(data_loader)}")
                # ve = batch_data["visual_feature"].float()
                ve = batch_data["video_npy"].float()
                clean_audio = batch_data["clean_audio"].to(self.device)
                for i in range(len(ve)):
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
                                                  ve.unsqueeze(1).to(self.device))

                predict_wav = predict_audio.squeeze(1)
                cur_loss = self.loss_func(predict_audio, clean_audio.unsqueeze(1))

                predict_recover = cur_loss["recover"]
                predict_recover = predict_recover.detach().cpu().numpy()
                test_details["predict_recover_loss"].append(predict_recover)

                predict_loss = cur_loss["main"]
                predict_loss = predict_loss.detach().cpu().numpy()
                test_details["predict_loss"].append(predict_loss)

                predict_sisdr = eval_sisdr(predict_audio, clean_audio.unsqueeze(1)).detach().cpu().numpy()
                test_details["predict_sisdr"].append(predict_sisdr)

                # Compute PESQ and STOI

                true_wav = batch_data["clean_audio"].to(self.device)
                # test_details["background_speaker_db"].append(batch_data["background_speaker_db"].detach().cpu().numpy())

                if "overlap_ms" in batch_data.keys():
                    test_details["overlap_ms"].append(batch_data["overlap_ms"].detach().cpu().numpy())

                sdr = audio_util.cal_SDR(true_wav.to(self.device), predict_wav)
                test_details["predict_sdr"].append(sdr.detach().cpu().numpy())
                # sdr_mix = audio_util.cal_SDR(true_wav.to(self.device), batch_data["mixture"].to(self.device))
                sdr_mix = audio_util.cal_SDR(true_wav.to(self.device), batch_data["mix_audio"].to(self.device))
                test_details["mix_sdr"].append(sdr_mix.detach().cpu().numpy())
                sdri = sdr - sdr_mix
                test_details["sdri"].append(sdri.detach().cpu().numpy())
                sdr_metric += torch.mean(sdr).detach().cpu().numpy()
                sdri_metric += torch.mean(sdri).detach().cpu().numpy()
                # test_details["clean_audio_path"].extend(batch_data["clean_audio_path"])
                # test_details["mixture_path"].extend(batch_data["mixture_path"])

                pesq = perceptual_evaluation_speech_quality(predict_wav, true_wav.to(self.device), fs=sampling_rate,
                                                            mode="wb")
                pesq = pesq.detach().cpu().numpy()
                test_details["predict_pesq"].append(pesq)

                stoi = short_time_objective_intelligibility(predict_wav, true_wav.to(self.device), fs=sampling_rate)
                stoi = stoi.detach().cpu().numpy()
                test_details["predict_stoi"].append(stoi)

                pesq_metric += np.mean(pesq)
                stoi_metric += np.mean(stoi)

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
                    "recover": test_details["predict_recover_loss"].mean()}, \
                {"PESQ": test_pesq, "STOI": test_stoi, "SDR": test_sdr, "SDRi": test_sdri}

    def separation(self):
        self.set_eval()

        cp_state, epoch = self.load_model(
            self.result_path + '/model.pth.tar'
        )

        # speaker_1_dir,speaker_2_dir,mixture_dir

        for idx, batch_data in tqdm(enumerate(self.test_loader)):
            mixture = batch_data["mix_audio"].float().to(self.device)
            visual_feature = batch_data["video_npy"].float().to(self.device)
            clean_audio = batch_data["clean_audio"].to(self.device)
            # mixture: (batch_size, 48000)
            # visual: (batch_size, 75, 512)

            predict_audio = self.fusion_model(mixture.unsqueeze(1), visual_feature.unsqueeze(1)).squeeze(1)
            spk1_id = batch_data["spk1_id"]
            spk2_id = batch_data["spk2_id"]
            spk1_vid = batch_data["spk1_vid"]
            spk2_vid = batch_data["spk2_vid"]

            for i, _ in enumerate(spk1_vid):
                video_id = f"{spk1_id[i]}_{spk1_vid[i]}_{spk2_id[i]}_{spk2_vid[i]}"

                os.makedirs(os.path.join(self.result_path, "test_separation", video_id), exist_ok=True)

                predict_filename = os.path.join(self.result_path, "test_separation", video_id, "separation.wav")
                clean_filename = os.path.join(self.result_path, "test_separation", video_id, f"{spk1_id[i]}_{spk1_vid[i]}.wav")
                mix_filename = os.path.join(self.result_path, "test_separation", video_id, f"{video_id}.wav")
                write(predict_filename, 16000, predict_audio[i].detach().cpu().numpy())
                write(clean_filename, 16000, clean_audio[i].detach().cpu().numpy())
                write(mix_filename, 16000, mixture[i].detach().cpu().numpy())


class DistributedAVLITSeparationWhamPretrainAE(DistributedAVLITSeparation):
    def init_data(self):

        self.train_set = ntcd_set.NTCDDataset(split="train")

        self.val_set = ntcd_set.NTCDDataset(split="val")

        self.test_set = ntcd_set.NTCDDataset(split="test")

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
        model_config = cfg_train.get("model_config", "avlit_default")
        model_config_dict = dynamic.import_string("models.model_config.{}".format(model_config))

        fusion_model = avlit.AVLITFixAE(**model_config_dict)
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

        # self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.8, patience=5)
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=25, gamma=1 / 3)

        loss_config = cfg_train.get("loss_config", "PermInvariantSISDR_Train")
        loss_config_dict = dynamic.import_string("models.model_config.{}".format(loss_config))
        loss_config_dict["batch_size"] = cfg_train["batch_size"]

        self.loss_func = loss.SISDR_Recover(loss_config_dict,
                                            num_chunks=cfg_train.get("num_chunks", 5),
                                            recover_weight=cfg_train.get("recover_weight", 0.05)).to(self.device)

    def forward_pass(self, input_tuple):
        # mixture = input_tuple["mix_audio"].float().to(self.device)
        # visual_feature = input_tuple["video_npy"].float().to(self.device)
        # clean_audio = input_tuple["clean_audio"].to(self.device)
        mixture = input_tuple["mix_audio"].float().to(self.device)
        visual_feature = input_tuple["video_npy"].float().to(self.device)
        clean_audio = input_tuple["clean_audio"].float()

        # mixture: (batch_size, 48000)
        # visual: (batch_size, 75, 512)

        predict_audio = self.fusion_model(mixture.unsqueeze(1), visual_feature)
        # predict_audio: (batch_size, 1, 48000)
        # del visual_feature
        # del mixture
        # torch.cuda.empty_cache()

        loss_dict = self.loss_func(predict_audio, clean_audio.unsqueeze(1).to(self.device))

        return predict_audio, clean_audio.unsqueeze(1), loss_dict

    def evaluate(self, data_loader):
        cfg_train = self.cfg["train"]
        reference_drop_type = cfg_train.get("reference_drop_type", None)  # random, end, start
        reference_drop_rate = float(cfg_train.get("reference_drop_rate", 0))
        sampling_rate = int(cfg_train.get("sampling_rate", 16000))

        eval_sisdr = loss.PermInvariantSISDR(**models.model_config.PermInvariantSISDR_Eval).to(self.device)

        dropped_frames = []
        num_dropped_samples = int(75 * reference_drop_rate)
        if reference_drop_type == "random":
            dropped_frames = random.choices(range(75), k=num_dropped_samples)
        elif reference_drop_type == "start":
            dropped_frames = range(num_dropped_samples, 75)
        elif reference_drop_type == "end":
            dropped_frames = range(75 - num_dropped_samples)

        pesq_metric = 0
        stoi_metric = 0
        sdr_metric = 0
        sdri_metric = 0

        test_details = {
            # "background_speaker_db": [],
            "overlap_ms": [],
            "predict_recover_loss": [],
            "predict_loss": [],
            "predict_sisdr": [],
            "predict_pesq": [],
            "predict_stoi": [],
            "predict_sdr": [],
            "mix_sdr": [],
            "sdri": []
            # "mixture_path": [],
            # "clean_audio_path": [],
        }

        test_details_output_file = os.path.join(self.result_path,
                                                    f"test_details_{reference_drop_type}_{reference_drop_rate}_{self.args.local_rank}.csv")
        if os.path.exists(test_details_output_file):
            os.remove(test_details_output_file)

        with torch.no_grad():
            for batch_i, batch_data in enumerate(data_loader):
                self.logger.info(f"Local Rank {self.args.local_rank}: Evaluation: Batch {batch_i} / {len(data_loader)}")
                # ve = batch_data["visual_feature"].float()
                ve = batch_data["video_npy"].float()
                clean_audio = batch_data["clean_audio"]
                for i in range(len(ve)):
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
                                                  ve.to(self.device))
                # del ve
                # torch.cuda.empty_cache()

                clean_audio = clean_audio.to(self.device)

                predict_wav = predict_audio.squeeze(1)
                cur_loss = self.loss_func(predict_audio, clean_audio.unsqueeze(1))

                predict_recover = cur_loss["recover"]
                predict_recover = predict_recover.detach().cpu().numpy()
                test_details["predict_recover_loss"].append(predict_recover)

                predict_loss = cur_loss["main"]
                predict_loss = predict_loss.detach().cpu().numpy()
                test_details["predict_loss"].append(predict_loss)

                predict_sisdr = eval_sisdr(predict_audio, clean_audio.unsqueeze(1)).detach().cpu().numpy()
                test_details["predict_sisdr"].append(predict_sisdr)

                # Compute PESQ and STOI

                true_wav = batch_data["clean_audio"].to(self.device)
                # test_details["background_speaker_db"].append(batch_data["background_speaker_db"].detach().cpu().numpy())

                if "overlap_ms" in batch_data.keys():
                    test_details["overlap_ms"].append(batch_data["overlap_ms"].detach().cpu().numpy())

                sdr = audio_util.cal_SDR(true_wav.to(self.device), predict_wav)
                test_details["predict_sdr"].append(sdr.detach().cpu().numpy())
                # sdr_mix = audio_util.cal_SDR(true_wav.to(self.device), batch_data["mixture"].to(self.device))
                sdr_mix = audio_util.cal_SDR(true_wav.to(self.device), batch_data["mix_audio"].to(self.device))
                test_details["mix_sdr"].append(sdr_mix.detach().cpu().numpy())
                sdri = sdr - sdr_mix
                test_details["sdri"].append(sdri.detach().cpu().numpy())
                sdr_metric += torch.mean(sdr).detach().cpu().numpy()
                sdri_metric += torch.mean(sdri).detach().cpu().numpy()
                # test_details["clean_audio_path"].extend(batch_data["clean_audio_path"])
                # test_details["mixture_path"].extend(batch_data["mixture_path"])

                pesq = perceptual_evaluation_speech_quality(predict_wav, true_wav.to(self.device), fs=sampling_rate,
                                                            mode="wb")
                pesq = pesq.detach().cpu().numpy()
                test_details["predict_pesq"].append(pesq)

                stoi = short_time_objective_intelligibility(predict_wav, true_wav.to(self.device), fs=sampling_rate)
                stoi = stoi.detach().cpu().numpy()
                test_details["predict_stoi"].append(stoi)

                pesq_metric += np.mean(pesq)
                stoi_metric += np.mean(stoi)

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
                    "recover": test_details["predict_recover_loss"].mean()}, \
                {"PESQ": test_pesq, "STOI": test_stoi, "SDR": test_sdr, "SDRi": test_sdri}


class DistributedProfusionSeparationWhamPretrainAEOffload(DistributedAVLITSeparationWhamPretrainAE):
    def init_model(self):
        cfg_train = self.cfg["train"]

        # model_config = cfg_train.get("model_config", "avlit_default")
        # model_config_dict = dynamic.import_string("models.model_config.{}".format(model_config))
        #
        # self.fusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        #     avlit.AVLIT(**model_config_dict).to(self.device))

        fusion_model = pro_fusion.ProgressiveAFRCNNFixAE(num_blocks=8)

        self.fusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(fusion_model.to(self.device))

        self.fusion_model = torch.nn.parallel.DistributedDataParallel(
            self.fusion_model,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True,
        )

        self.optim = torch.optim.AdamW(self.fusion_model.parameters(),
                                       lr=cfg_train["lr"], betas=(0.9, 0.98),
                                       weight_decay=cfg_train.get("weight_decay", 0.01))

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=5)
        # self.scheduler = lr_scheduler.StepLR(self.optim, step_size=25, gamma=1 / 3)

        loss_config = cfg_train.get("loss_config", "PermInvariantSISDR_Train")
        loss_config_dict = dynamic.import_string("models.model_config.{}".format(loss_config))
        loss_config_dict["batch_size"] = cfg_train["batch_size"]

        self.loss_func = loss.SISDR_Recover(loss_config_dict,
                                            num_chunks=cfg_train.get("num_chunks", 5),
                                            recover_weight=cfg_train.get("recover_weight", 0.05)).to(self.device)

    def forward_pass(self, input_tuple):
        # mixture = input_tuple["mix_audio"].float().to(self.device)
        # visual_feature = input_tuple["video_npy"].float().to(self.device)
        # clean_audio = input_tuple["clean_audio"].to(self.device)
        mixture = input_tuple["mix_audio"].float().to(self.device)
        visual_feature = input_tuple["video_npy"].float().to(self.device)
        clean_audio = input_tuple["clean_audio"].float()

        # mixture: (batch_size, 48000)
        # visual: (batch_size, 75, 512)

        predict_audio = self.fusion_model(mixture.unsqueeze(1), visual_feature)
        # predict_audio: (batch_size, 1, 48000)
        visual_feature = visual_feature.detach().cpu()
        mixture = mixture.detach().cpu()
        # del visual_feature
        # del mixture
        # torch.cuda.empty_cache()

        loss_dict = self.loss_func(predict_audio, clean_audio.unsqueeze(1).to(self.device))

        return predict_audio.detach(), clean_audio.unsqueeze(1), loss_dict
