import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm
import pickle
from torch.optim import lr_scheduler
import engines.abs_reconstruction as abs_recon
import dataset.tcd_timit.dataset as ntcd_set
import models.model_config
import models.autoencoder as ae_model

class DistributedVideoRecon(abs_recon.DistributedReconstruction):
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
            "/work/pi_mfiterau_umass_edu/sidong/speech/results/video_autoencoder_ntcd_e3000_lr1e-2/model.pth.tar")["autoencoder"])
    autoencoder = autoencoder.to(dc.device)
    print("load from video_autoencoder_ntcd_e3000_lr1e-2")
    def save(data_loader, split):
        latent_dict = {}
        with torch.no_grad():
            for batch_i, batch_data in tqdm(enumerate(data_loader)):
                video = batch_data["video_npy"].float().to(dc.device)
                latent = autoencoder.encode(video.unsqueeze(1)).cpu().detach().numpy()
                for i in range(latent.shape[0]):
                    if batch_data["spk1_id"][i] not in latent_dict:
                        latent_dict[batch_data["spk1_id"][i]] = {batch_data["spk1_vid"][i]: latent[i]}
                    else:
                        latent_dict[batch_data["spk1_id"][i]][batch_data["spk1_vid"][i]] = latent[i]

        pickle.dump(latent_dict,
                    open(f"/work/pi_mfiterau_umass_edu/TCD_TIMIT/ntcd/video_latent_{split}_1024.pkl", "wb"))

    train_set = ntcd_set.NTCDDataset(split="train")

    val_set = ntcd_set.NTCDDataset(split="val")

    test_set = ntcd_set.NTCDDataset(split="test")

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
