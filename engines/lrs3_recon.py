import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
from torch.optim import lr_scheduler
import engines.abs_reconstruction as abs_recon
import dataset.lrs3.dataset as lrs3_set
import dataset.lrs3_wham.dataset as lrs3_wham_set
import models.model_config
import models.autoencoder as ae_model
import pickle


class DistributedVideoRecon(abs_recon.DistributedReconstruction):
    def init_data(self):
        self.train_set = lrs3_set.LRS3VideoSet(split="train")

        self.val_set = lrs3_set.LRS3VideoSet(split="val")

        self.test_set = lrs3_set.LRS3VideoSet(split="test")

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

        # train_subset = torch.utils.data.Subset(self.train_set, range(16))
        # test_subset = torch.utils.data.Subset(self.test_set, range(16))
        # val_subset = torch.utils.data.Subset(self.val_set, range(16))
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

        self.autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ae_model.FrameAutoEncoder().to(self.device))

        self.autoencoder = torch.nn.parallel.DistributedDataParallel(
            self.autoencoder,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=False,
        )

        self.optim = torch.optim.AdamW(self.autoencoder.parameters(),
                                       lr=cfg_train["lr"], betas=(0.9, 0.98),
                                       weight_decay=cfg_train.get("weight_decay", 0.01))

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.8, patience=5)
        # self.scheduler = lr_scheduler.StepLR(self.optim, step_size=25, gamma=1/3)
        self.loss_func = torch.nn.MSELoss()

    def forward_pass(self, input_tuple):
        video = input_tuple["video_npy"].float().to(self.device)
        assert video.shape[1] == 50
        recon = self.autoencoder(video.unsqueeze(1))
        loss_dict = {"main": self.loss_func(video.unsqueeze(1), recon)}

        return recon, video, loss_dict

    def evaluate(self, data_loader):
        with torch.no_grad():
            recon_mse = 0
            for batch_i, batch_data in enumerate(data_loader):
                self.logger.info(f"Local Rank {self.args.local_rank}: Evaluation: Batch {batch_i} / {len(data_loader)}")
                # ve = batch_data["visual_feature"].float()
                _, _, loss_dict = self.forward_pass(batch_data)
                recon_mse += loss_dict["main"].item()

            test_recon = recon_mse / len(data_loader)

            return {"main": test_recon}


def generate():
    import models.device_check as dc
    autoencoder = ae_model.FrameAutoEncoder()
    autoencoder.load_state_dict(torch.load(
            "/work/pi_mfiterau_umass_edu/sidong/speech/results/video_autoencoder_lrs3_e1000_single_gpu/model.pth.tar")["autoencoder"])
    autoencoder = autoencoder.to(dc.device)

    print("load from: video_autoencoder_lrs3_e1000_single/model.pth.tar")
    def save(data_loader, split):
        latent_dict = {}
        with torch.no_grad():
            for batch_i, batch_data in tqdm(enumerate(data_loader)):
                video = batch_data["pair_video"].float().to(dc.device)
                latent = autoencoder.encode(video).cpu().detach().numpy()
                for i in range(latent.shape[0]):
                    latent_dict[batch_data["video_id"][i]] = latent[i]

        pickle.dump(latent_dict,
                    open(f"/work/pi_mfiterau_umass_edu/LRS3/train_full_wham/videopair_latent_{split}.pkl", "wb"))

    train_set = lrs3_set.LRS3VideoSet(split="train")

    val_set = lrs3_set.LRS3VideoSet(split="val")

    test_set = lrs3_set.LRS3VideoSet(split="test")

    train_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=32,
                                                    pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=32,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_set,
                                               batch_size=32,
                                               pin_memory=True)

    save(train_loader, "train")
    save(test_loader, "test")
    save(val_loader, "val")


def generate_single():
    import models.device_check as dc
    autoencoder = ae_model.FrameAutoEncoder()
    autoencoder.load_state_dict(torch.load(
            "/work/pi_mfiterau_umass_edu/sidong/speech/results/video_autoencoder_lrs3_e1000_single_gpu/model.pth.tar")["autoencoder"])
    autoencoder = autoencoder.to(dc.device)

    print("load from: lrs3_e1000_single_gpu/model.pth.tar")
    # def save(data_loader, split):
    #     latent_dict = {}
    #     with torch.no_grad():
    #         for batch_i, batch_data in tqdm(enumerate(data_loader)):
    #             video = batch_data["video_npy"].unsqueeze(1).float().to(dc.device)
    #             latent = autoencoder.encode(video).cpu().detach().numpy()
    #             assert latent.shape[-1] == 1024
    #             for i in range(latent.shape[0]):
    #                 video_id = f'{batch_data["spk1_id"][i]}_{batch_data["spk1_vid"][i]}'
    #                 latent_dict[video_id] = latent[i]
    #
    #     pickle.dump(latent_dict,
    #                 open(f"/work/pi_mfiterau_umass_edu/LRS3/train_full_wham/video_single_latent_{split}.pkl", "wb"))

    def save(data_loader, split):
        with torch.no_grad():
            for batch_i, batch_data in tqdm(enumerate(data_loader)):
                video = batch_data["video_npy"].unsqueeze(1).float().to(dc.device)
                latent = autoencoder.encode(video).cpu().detach().numpy()
                assert latent.shape[-1] == 1024
                for i in range(latent.shape[0]):
                    video_id = f'{batch_data["spk1_id"][i]}_{batch_data["spk1_vid"][i]}'
                    np.save(os.path.join("/work/pi_mfiterau_umass_edu/LRS3/avlit_ae", f"{video_id}.npy"), latent[i])

    train_set = lrs3_set.LRS3VideoSet(split="train")

    val_set = lrs3_set.LRS3VideoSet(split="val")

    test_set = lrs3_set.LRS3VideoSet(split="test")

    train_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=32,
                                                    pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=32,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_set,
                                               batch_size=32,
                                               pin_memory=True)

    save(train_loader, "train")
    save(test_loader, "test")
    save(val_loader, "val")

