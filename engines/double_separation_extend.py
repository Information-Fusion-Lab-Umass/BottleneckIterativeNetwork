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
import models.progressive_stft as pro_stft
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


class LRS3ProfusionSTOI(double_sep.AbsDoubleSeparation):
    def init_data(self):
        # self.train_set = lrs3_wham_set.LRS3LazyPairSet(split="train", args=self.args, logger=self.logger)
        # self.val_set = lrs3_wham_set.LRS3LazyPairSet(split="val", args=self.args, logger=self.logger)
        # self.test_set = lrs3_wham_set.LRS3LazyPairSet(split="test", args=self.args, logger=self.logger)
        #
        # self.train_loader = torch.utils.data.DataLoader(self.train_set,
        #                                                 batch_size=self.cfg["train"]["batch_size"],
        #                                                 pin_memory=True)
        #
        # self.val_loader = torch.utils.data.DataLoader(self.val_set,
        #                                               batch_size=self.cfg["train"]["batch_size"],
        #                                               pin_memory=True)
        #
        # self.test_loader = torch.utils.data.DataLoader(self.test_set,
        #                                                batch_size=self.cfg["train"]["batch_size"],
        #                                                pin_memory=True)

        self.train_set = lrs3_wham_set.LRS3ProcessedPairSet(split="train")

        self.val_set = lrs3_wham_set.LRS3ProcessedPairSet(split="val")

        self.test_set = lrs3_wham_set.LRS3ProcessedPairSet(split="test")

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

        # train_subset = torch.utils.data.Subset(self.train_set, range(96))
        # test_subset = torch.utils.data.Subset(self.test_set, range(96))
        # val_subset = torch.utils.data.Subset(self.val_set, range(96))
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
        self.aux_loss_name = "stoi"
        cfg_train = self.cfg["train"]

        num_blocks = cfg_train.get("num_blocks", 8)
        fusion_dim = cfg_train.get("fusion_dim", 256)

        fusion_model = pro_mbt.ProgressiveBranch(num_blocks=num_blocks, fusion_channels=fusion_dim, video_dim=1024)

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

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=5)
        # self.scheduler = lr_scheduler.StepLR(self.optim, step_size=25, gamma=1 / 3)

        loss_config = cfg_train.get("loss_config", "PermInvariantSISDR_Train")
        loss_config_dict = dynamic.import_string("models.model_config.{}".format(loss_config))
        loss_config_dict["batch_size"] = cfg_train["batch_size"]

        # self.loss_func = loss.SISDR_Recover(loss_config_dict,
        #                                     num_chunks=cfg_train.get("num_chunks", 5),
        #                                     recover_weight=cfg_train.get("recover_weight", 0.05)).to(self.device)

        self.loss_func = loss.SISDRStoi(num_chunks=cfg_train.get("num_chunks", 5),
                                        recover_weight=cfg_train.get("recover_weight", 1)).to(self.device)

    def scheduler_step(self, cur_loss):
        self.scheduler.step(cur_loss)


class LRS3ProfusionSTFT(double_sep.AbsDoubleSeparation):
    def init_data(self):
        # self.train_set = lrs3_wham_set.LRS3LazyPairSet(split="train", args=self.args, logger=self.logger)
        # self.val_set = lrs3_wham_set.LRS3LazyPairSet(split="val", args=self.args, logger=self.logger)
        # self.test_set = lrs3_wham_set.LRS3LazyPairSet(split="test", args=self.args, logger=self.logger)
        #
        # self.train_loader = torch.utils.data.DataLoader(self.train_set,
        #                                                 batch_size=self.cfg["train"]["batch_size"],
        #                                                 pin_memory=True)
        #
        # self.val_loader = torch.utils.data.DataLoader(self.val_set,
        #                                               batch_size=self.cfg["train"]["batch_size"],
        #                                               pin_memory=True)
        #
        # self.test_loader = torch.utils.data.DataLoader(self.test_set,
        #                                                batch_size=self.cfg["train"]["batch_size"],
        #                                                pin_memory=True)

        self.train_set = lrs3_wham_set.LRS3ProcessedPairSet(split="train")

        self.val_set = lrs3_wham_set.LRS3ProcessedPairSet(split="val")

        self.test_set = lrs3_wham_set.LRS3ProcessedPairSet(split="test")

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

        # train_subset = torch.utils.data.Subset(self.train_set, range(96))
        # test_subset = torch.utils.data.Subset(self.test_set, range(96))
        # val_subset = torch.utils.data.Subset(self.val_set, range(96))
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
        self.aux_loss_name = "stoi"
        cfg_train = self.cfg["train"]

        num_blocks = cfg_train.get("num_blocks", 8)
        fusion_dim = cfg_train.get("fusion_dim", 256)

        with open(os.path.join('./config/lrs3/profusion_stft_model.yaml'), 'r') as f:
            stft_cfg = yaml.load(f, Loader=yaml.FullLoader)

        stft_cfg["audionet"]["num_blocks"] = num_blocks
        stft_cfg["audionet"]["audio_fusion_dim"] = fusion_dim
        stft_cfg["audionet"]["video_fusion_dim"] = fusion_dim

        self.logger.info(str(stft_cfg))

        fusion_model = pro_stft.ProgressiveSTFT(**stft_cfg["audionet"])

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

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=5)
        # self.scheduler = lr_scheduler.StepLR(self.optim, step_size=25, gamma=1 / 3)

        loss_config = cfg_train.get("loss_config", "PermInvariantSISDR_Train")
        loss_config_dict = dynamic.import_string("models.model_config.{}".format(loss_config))
        loss_config_dict["batch_size"] = cfg_train["batch_size"]

        # self.loss_func = loss.SISDR_Recover(loss_config_dict,
        #                                     num_chunks=cfg_train.get("num_chunks", 5),
        #                                     recover_weight=cfg_train.get("recover_weight", 0.05)).to(self.device)

        self.loss_func = loss.SISDRStoi(num_chunks=cfg_train.get("num_chunks", 5),
                                        recover_weight=cfg_train.get("recover_weight", 0)).to(self.device)

    def scheduler_step(self, cur_loss):
        self.scheduler.step(cur_loss)


class NTCDProfusionSTOI(double_sep.AbsDoubleSeparation):
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
        self.aux_loss_name = "stoi"
        cfg_train = self.cfg["train"]

        num_blocks = cfg_train.get("num_blocks", 8)
        fusion_dim = cfg_train.get("fusion_dim", 256)

        fusion_model = pro_mbt.ProgressiveBranch(num_blocks=num_blocks,
                                                 fusion_channels=fusion_dim,
                                                 video_dim=1024,
                                                 audio_dim=64000
                                                 )

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

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=5)
        # self.scheduler = lr_scheduler.StepLR(self.optim, step_size=25, gamma=1 / 3)

        loss_config = cfg_train.get("loss_config", "PermInvariantSISDR_Train")
        loss_config_dict = dynamic.import_string("models.model_config.{}".format(loss_config))
        loss_config_dict["batch_size"] = cfg_train["batch_size"]

        # self.loss_func = loss.SISDR_Recover(loss_config_dict,
        #                                     num_chunks=cfg_train.get("num_chunks", 5),
        #                                     recover_weight=cfg_train.get("recover_weight", 0.05)).to(self.device)

        self.loss_func = loss.SISDRStoi(num_chunks=cfg_train.get("num_chunks", 5),
                                        recover_weight=cfg_train.get("recover_weight", 1)).to(self.device)

    def scheduler_step(self, cur_loss):
        self.scheduler.step(cur_loss)