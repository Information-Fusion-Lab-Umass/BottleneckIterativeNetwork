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
import pandas as pd
import utils.audio as audio
import logging

import cv2

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.audio import process_audio

logging.getLogger('numba').setLevel(logging.WARNING)

stft_size = 511
window_size = 400
window_shift = 160
window_length = None
sampling_rate = 16000
fps = 25
fading = False
windows = signal.windows.hann

original_root_dir = "/project/pi_mfiterau_umass_edu/dolby/speech-enhancement-fusion"

# Instruction file dictionary
CACHE_FILE_DICTIONARY = {
    # 100% overlap setting
    # small size
    "mix_2_spk_train_fully_overlapped_only_6000.txt": {
        "audio": "train_fully_overlapped_only_raw_6000_include_ref_audio.pkl",
        "visual": "train_fully_overlapped_only_raw_6000_include_visual.pkl"
    },
    "mix_2_spk_valid_fully_overlapped_only_2000.txt": {
        "audio": "valid_fully_overlapped_only_raw_2000_include_ref_audio.pkl",
        "visual": "valid_fully_overlapped_only_raw_2000_include_visual.pkl"
    },
    "mix_2_spk_test_fully_overlapped_only_3000.txt": {
        "audio": "test_fully_overlapped_only_raw_3000_include_ref_audio.pkl",
        "visual": "test_fully_overlapped_only_raw_3000_include_visual.pkl",
        "visual_raw": "test_fully_overlapped_only_raw_3000_include_visual_raw.pkl"
    },
    # mixed overlap setting
    # medium size
    "mix_2_spk_train_overlap_10000.txt": {
        "audio": "train_overlap_raw_10000_include_ref_audio.pkl",
        "visual": "train_overlap_raw_10000_include_visual.pkl"
    },
    "mix_2_spk_valid_overlap_3000.txt": {
        "audio": "valid_overlap_raw_3000_include_ref_audio.pkl",
        "visual": "valid_overlap_raw_3000_include_visual.pkl"
    },
    # small size
    "mix_2_spk_train_overlap_6000.txt": {
        "audio": "train_overlap_raw_6000_include_ref_audio.pkl",
        "visual": "train_overlap_raw_6000_include_visual.pkl"
    },
    "mix_2_spk_valid_overlap_2000.txt": {
        "audio": "valid_overlap_raw_2000_include_ref_audio.pkl",
        "visual": "valid_overlap_raw_2000_include_visual.pkl",
    },
    "mix_2_spk_test_overlap_3000.txt": {
        "audio": "test_overlap_raw_3000_include_ref_audio.pkl",
        "visual": "test_overlap_raw_3000_include_visual.pkl",
        "visual_raw": "test_overlap_raw_3000_include_visual_raw.pkl"
    },
}


class LRS3RawDataset(Dataset):
    def __init__(self, split):
        lrs3_path = "/work/pi_mfiterau_umass_edu/LRS3"
        # data_dir = os.path.join(lrs3_path, "train10k", f"mix_2_spk_{split}_fully_overlapped_only_full_generation.pkl")
        data_dir = os.path.join(lrs3_path, "train_full_wham", f"mix_2_spk_{split}_fully_overlapped_only_full_generation.pkl")
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
        return self.full_data[idx]


class LRS3ProcessedSet(Dataset):
    def __init__(self, split):
        lrs3_path = "/work/pi_mfiterau_umass_edu/LRS3"
        data_dir = os.path.join(lrs3_path, "train_full_wham", f"mix_2_spk_{split}_fully_overlapped_only_full_videolatent.pkl")
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
        return self.full_data[idx]


class LRS3LazyPairSet(Dataset):
    def preload_audio(self):
        mouth_path = "/project/pi_mfiterau_umass_edu/LRS3/video_mouth_64x64"
        local_list = []
        for idx, row in self.meta.iterrows():
            spk1_dir = row["speaker_1_dir"]
            spk2_dir = row["speaker_2_dir"]
            speaker_1, sr = audio.process_audio(spk1_dir, target_sr=16000)
            speaker_2, sr = audio.process_audio(spk2_dir, target_sr=16000)

            s1_start = row["s1_start"]
            s2_start = row["s2_start"]

            mix_start = 0
            s1_start = int(s1_start / 1000 * 16000)
            s2_start = int(s2_start / 1000 * 16000)

            max_audio_samples = 32000
            s1_dur = min(len(speaker_1), max_audio_samples)
            s2_dur = min(len(speaker_2), max_audio_samples)

            s1_end = s1_start + s1_dur
            s2_end = s2_start + s2_dur

            mix_end = max(s1_end, s2_end, max_audio_samples)

            # mix the audio and add noise
            clean_audio = np.stack([
                np.pad(speaker_1[:(s1_end - s1_start)],
                       (s1_start - mix_start, np.abs(s1_end - mix_end)), constant_values=0.0),
                np.pad(speaker_2[:(s2_end - s2_start)],
                       (s2_start - mix_start, np.abs(s2_end - mix_end)), constant_values=0.0)], axis=0)

            assert clean_audio.shape[1] == 32000
            assert clean_audio.shape[0] == 2

            spk1_split = spk1_dir.split("/")
            spk2_split = spk2_dir.split("/")

            spk1_id = spk1_split[-2]
            spk1_vid = spk1_split[-1].split(".")[0]

            spk2_id = spk2_split[-2]
            spk2_vid = spk2_split[-1].split(".")[0]

            record = {"spk1_id": spk1_id,
                      "spk2_id": spk2_id,
                      "spk1_vid": spk1_vid,
                      "spk2_vid": spk2_vid,
                      "overlap_ms": row["overlap_ms"],
                      "clean_audio": clean_audio,
                      "mix_audio":
                          audio.process_audio(row["mixture_dir"].replace("audio_mixture", "audio_mixture_wham"),
                                              target_sr=16000)[0]}
            local_list.append(record)

        return local_list

    def preload_video(self):
        video_ids = set([])
        for idx, row in self.meta.iterrows():
            spk1_path = row["speaker_1_dir"].split("/")
            spk2_path = row["speaker_2_dir"].split("/")

            spk1_id = spk1_path[-2]
            spk1_vid = spk1_path[-1].replace(".wav", "")

            spk2_id = spk2_path[-2]
            spk2_vid = spk2_path[-1].replace(".wav", "")

            video_ids.add(f"{spk1_id}_{spk1_vid}")
            video_ids.add(f"{spk2_id}_{spk2_vid}")

        video_dict = {}
        mouth_path = "/work/pi_mfiterau_umass_edu/LRS3/avlit_ae"
        for video_id in video_ids:
            video_dict[video_id] = np.load(os.path.join(os.path.join(mouth_path, f"{video_id}.npy")))

        return video_dict

    def __init__(self, split, args, logger):
        rank = args.local_rank
        num_gpus = int(os.environ["LOCAL_WORLD_SIZE"])
        meta = pd.read_csv(f"/project/pi_mfiterau_umass_edu/LRS3/meta/mix_2_spk_{split}_fully_overlapped_only_full_wham.csv")
        total_size = meta.shape[0]
        block_size = total_size // num_gpus

        self.logger = logger

        self.meta = meta[rank * block_size: (rank + 1) * block_size] if rank < num_gpus-1 else meta[rank * block_size:]

        self.logger.info(f"Dataset: local rank {rank} preloading audio")
        self.audio = self.preload_audio()

        self.logger.info(f"Dataset: local rank {rank} preloading video")
        self.video = self.preload_video()

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        record = self.audio[idx]

        video1_id = f"{record['spk1_id']}_{record['spk1_vid']}"
        video2_id = f"{record['spk2_id']}_{record['spk2_vid']}"

        video_record = np.concatenate([self.video[video1_id], self.video[video2_id]],
                                      axis=0)  # 2 source videos
        record["video_npy"] = video_record

        return record


class LRS3ProcessedPairSet(Dataset):
    def __init__(self, split):
        lrs3_path = "/work/pi_mfiterau_umass_edu/LRS3"

        audio_data_dir = os.path.join(lrs3_path, "train_full_wham",
                                      f"mix_2_spk_{split}_fully_overlapped_only_full_double_dist.pkl")

        self.audio_data = pickle.load(open(audio_data_dir, "rb"))

        video_data_dir = os.path.join(lrs3_path, "train_full_wham",
                                      f"video_single_latent_{split}.pkl")

        self.video_data = pickle.load(open(video_data_dir, "rb"))

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):

        record = self.audio_data[idx]

        video1_id = f"{record['spk1_id']}_{record['spk1_vid']}"
        video2_id = f"{record['spk2_id']}_{record['spk2_vid']}"

        video_record = np.concatenate([self.video_data[video1_id], self.video_data[video2_id]], axis=0)   # 2 source videos
        record["video_npy"] = video_record

        return record


class LRS3ProcessedSingleSet(Dataset):
    def __init__(self, split):
        lrs3_path = "/work/pi_mfiterau_umass_edu/LRS3"

        audio_data_dir = os.path.join(lrs3_path, "train_full_wham",
                                      f"mix_2_spk_{split}_fully_overlapped_only_full_double_dist.pkl")

        self.audio_data = pickle.load(open(audio_data_dir, "rb"))

        video_data_dir = os.path.join(lrs3_path, "train_full_wham",
                                      f"video_single_latent_{split}.pkl")

        self.video_data = pickle.load(open(video_data_dir, "rb"))

    def __len__(self):
        return len(self.audio_data) * 2

    def __getitem__(self, idx):
        audio_idx = idx // 2
        spk_idx = f"spk{idx % 2 + 1}_id"
        spk_vidx = f"spk{idx % 2 + 1}_vid"

        record = self.audio_data[audio_idx]

        video_id = f"{record[spk_idx]}_{record[spk_vidx]}"

        video_record = self.video_data[video_id]  # 2 source videos
        record["video_npy"] = video_record
        record["target_spk"] = idx % 2 
        return record


class LRS3ProcessedPairSubSet(Dataset):
    def __init__(self, split):
        lrs3_path = "/work/pi_mfiterau_umass_edu/LRS3"

        audio_data_dir = os.path.join(lrs3_path, "train_full_wham", "subset",
                                      f"mix_2_spk_{split}_fully_overlapped_only_full_double_dist_10p.pkl")

        self.audio_data = pickle.load(open(audio_data_dir, "rb"))

        video_data_dir = os.path.join(lrs3_path, "train_full_wham", "subset",
                                      f"video_single_latent_{split}_10p.pkl")

        self.video_data = pickle.load(open(video_data_dir, "rb"))

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):

        record = self.audio_data[idx]

        video1_id = f"{record['spk1_id']}_{record['spk1_vid']}"
        video2_id = f"{record['spk2_id']}_{record['spk2_vid']}"

        video_record = np.concatenate([self.video_data[video1_id], self.video_data[video2_id]], axis=0)   # 2 source videos
        record["video_npy"] = video_record

        return record


class LRS3TCNSet(Dataset):
    def __init__(self, split):
        lrs3_path = "/work/pi_mfiterau_umass_edu/LRS3"
        data_dir = os.path.join(lrs3_path, "train_full_wham", f"mix_2_spk_{split}_fully_overlapped_only_full_videotcn.pkl")
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
        self.full_data[idx]["video_npy"] = np.expand_dims(self.full_data[idx]["video_npy"], axis=0)
        return self.full_data[idx]

