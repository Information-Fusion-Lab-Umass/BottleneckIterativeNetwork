import argparse
import datetime
import math
import pickle

import librosa
import numpy as np
import torch
from scipy import signal
import os
import random
import logging

import cv2
from skimage.transform import resize

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.audio import process_audio


stft_size = 511
window_size = 400
window_shift = 160
window_length = None
sampling_rate = 16000
fps = 25
fading = False
windows = signal.windows.hann


class NTCDDataset(Dataset):
    def __init__(self, split):
        # ntcd_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT_EXP"
        ntcd_path = "/work/pi_mfiterau_umass_edu/TCD_TIMIT"
        data_dir = os.path.join(ntcd_path, "ntcd", f"mix_2_spk_{split}_fully_overlapped_only_5k_generation.pkl")
        # data_dir = os.path.join(ntcd_path, "ntcd", f"mix_2_spk_{split}_fully_overlapped_only_5k_videolatent.pkl")
        self.full_data = pickle.load(open(data_dir, "rb"))

    def __len__(self):
        return len(self.full_data)

    def __getitem__(self, idx):
        # return_dict = {}
        # for key in self.full_data[idx]:
        #     if key == "mix_audio":
        #         return_dict[key] = self.full_data[idx][key][0]
        #     elif key == "clean_audio":
        #         return_dict[key] = self.full_data[idx][key][0][:32000]
        #     else:
        #         return_dict[key] = self.full_data[idx][key]
        # return return_dict
        # video = self.full_data[idx]["video_npy"]
        # pad_video = np.zeros((250, 64, 64))
        # if video.shape[0] < 250:
        #     pad_video[:video.shape[0]] = video
        # else:
        #     pad_video = video[:250]
        # self.full_data[idx]["video_npy"] = pad_video
        return self.full_data[idx]


class NTCDProcessedPairSet(Dataset):
    def __init__(self, split):
        ntcd_path = "/work/pi_mfiterau_umass_edu/TCD_TIMIT/ntcd"

        audio_data_dir = os.path.join(ntcd_path, f"mix_2_spk_{split}_fully_overlapped_only_full_double.pkl")

        self.audio_data = pickle.load(open(audio_data_dir, "rb"))

        video_data_dir = os.path.join(ntcd_path, f"video_latent_{split}_1024.pkl")

        self.video_data = pickle.load(open(video_data_dir, "rb"))

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):

        record = self.audio_data[idx]

        video_record = np.concatenate([self.video_data[record['spk1_id']][record['spk1_vid']],
                                       self.video_data[record['spk2_id']][record['spk2_vid']]], axis=0)   # 2 source videos
        record["video_npy"] = video_record

        return record


class NTCDProcessedSingleSet(Dataset):
    def __init__(self, split):
        ntcd_path = "/work/pi_mfiterau_umass_edu/TCD_TIMIT/ntcd"

        audio_data_dir = os.path.join(ntcd_path, f"mix_2_spk_{split}_fully_overlapped_only_full_double.pkl")

        self.audio_data = pickle.load(open(audio_data_dir, "rb"))

        video_data_dir = os.path.join(ntcd_path, f"video_latent_{split}_1024.pkl")

        self.video_data = pickle.load(open(video_data_dir, "rb"))

    def __len__(self):
        return len(self.audio_data) * 2

    def __getitem__(self, idx):
        audio_idx = idx // 2
        spk_idx = f"spk{idx % 2 + 1}_id"
        spk_vidx = f"spk{idx % 2 + 1}_vid"

        record = self.audio_data[audio_idx]

        # video_id = f"{record[spk_idx]}_{record[spk_vidx]}"

        video_record = self.video_data[record[spk_idx]][record[spk_vidx]]  # 1 source videos
        record["video_npy"] = video_record
        record["target_spk"] = idx % 2
        return record
