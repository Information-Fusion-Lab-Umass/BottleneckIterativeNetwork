import os
import torch
from tqdm import tqdm
import pickle
import models.lipreading.lip_embedding as lip_embed
import dataset.lrs3.dataset as lrs3_set
from definition import *


def generate():
    import models.device_check as dc
    encoder = lip_embed.get_model_from_json(os.path.join(ROOT_PATH, "config", "pretrained", "resnet_dctcn_video.json"))
    encoder.load_state_dict(torch.load(
            "/work/pi_mfiterau_umass_edu/sidong/speech/pretrained/lrw_resnet18_dctcn_video.pth.tar")["model_state_dict"])
    encoder = encoder.to(dc.device)

    def save(data_loader, split):
        latent_dict = {}
        with torch.no_grad():
            for batch_i, batch_data in tqdm(enumerate(data_loader)):
                video = batch_data["video_npy"].float().to(dc.device)
                latent = encoder(video.unsqueeze(1), video.shape[1]).cpu().detach().numpy()
                assert latent.shape[1] == 50
                assert latent.shape[2] == 512

                for i in range(latent.shape[0]):
                    if batch_data["spk1_id"][i] not in latent_dict:
                        latent_dict[batch_data["spk1_id"][i]] = {batch_data["spk1_vid"][i]: latent[i]}
                    else:
                        latent_dict[batch_data["spk1_id"][i]][batch_data["spk1_vid"][i]] = latent[i]

        pickle.dump(latent_dict,
                    open(f"/work/pi_mfiterau_umass_edu/TCD_TIMIT_EXP/ntcd/video_latent_{split}_dctcn.pkl", "wb"))

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