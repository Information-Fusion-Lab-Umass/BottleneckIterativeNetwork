import sys
import os.path
sys.path.append("../../..")

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.io.wavfile import write
import utils.audio as audio

raw_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT"
gray_processed_path = "/project/pi_mfiterau_umass_edu/sidong/speech/dataset/tcd_timit/script/dda_gray_nmcroi_dataout"
opticalflow_processed_path = "/project/pi_mfiterau_umass_edu/sidong/speech/dataset/tcd_timit/script/dda_optical_flow_nmcroi_dataout"

meta_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT_meta/"


def mixture_white(split, audio_ms=4000, sample_rate=16000):
    audio_len = int(audio_ms / 1000 * sample_rate)
    meta = pd.read_csv(os.path.join(meta_path, f"mix_{split}_noisy.csv"))
    overlap_ms_list = []
    s1_silence_start_list = []
    s1_silence_end_list = []

    for idx, row in tqdm(meta.iterrows()):
        spk1_path = row["spk1_audio_path"]
        spk2_path = row["spk2_audio_path"]
        noise_path = row["noise_path"]

        noise_ratio = 10 ** (row["noise_db"] / 20.0)

        spk1_audio, _ = audio.process_audio(spk1_path, target_sr=sample_rate)
        spk2_audio, _ = audio.process_audio(spk2_path, target_sr=sample_rate)
        noise_audio, _ = audio.process_audio(noise_path, target_sr=sample_rate)

        s1_dur = int(min(len(spk1_audio), audio_len))
        s2_dur = int(min(len(spk2_audio), audio_len))

        s1_start = int(0)
        s2_start = int(0)

        s1_end = s1_start + s1_dur
        s2_end = s2_start + s2_dur

        if s1_start == s2_start:
            overlap = s1_dur
        elif s1_start > s2_start:
            overlap = max(s2_end - s1_start, 0)
        else:
            overlap = - max(s1_end - s2_start, 0)
        overlap_ms = int(overlap / sample_rate * 1000)

        mix_start = 0
        mix_end = max(s1_end, s2_end)

        if s1_start > 0:
            s1_silence_start_ms = 0
            s1_silence_end_ms = int(s1_start / sample_rate * 1000)
        else:
            s1_silence_start_ms = s1_end / sample_rate * 1000
            s1_silence_end_ms = mix_end / sample_rate * 1000

        # mix the audio and add noise
        mixture = np.pad(spk1_audio[:(s1_end - s1_start)],
                         (s1_start - mix_start, np.abs(s1_end - mix_end)), constant_values=0.0)
        mixture += np.pad(spk2_audio[:(s2_end - s2_start)],
                          (s2_start - mix_start, np.abs(s2_end - mix_end)), constant_values=0.0)
        mixture += noise_ratio * np.pad(noise_audio[:(s1_end - s1_start)],
                         (s1_start - mix_start, np.abs(s1_end - mix_end)), constant_values=0.0)

        write(row["mix_path"], sample_rate, mixture)

        overlap_ms_list.append(overlap_ms)
        s1_silence_start_list.append(s1_silence_start_ms)
        s1_silence_end_list.append(s1_silence_end_ms)

    meta["overlap_ms"] = overlap_ms_list
    meta["s1_silence_start_ms"] = s1_silence_start_list
    meta["s1_silence_end_ms"] = s1_silence_end_list

    meta.to_csv(os.path.join(meta_path, f"mix_{split}_noisy_generation.csv"))


def mixture_5k(split, audio_ms=4000, sample_rate=16000):
    audio_len = int(audio_ms / 1000 * sample_rate)
    meta = pd.read_csv(os.path.join(meta_path, f"{split}_mixture.csv"))
    overlap_ms_list = []
    s1_silence_start_list = []
    s1_silence_end_list = []

    for idx, row in tqdm(meta.iterrows()):
        spk1_path = row["spk1_path"]
        spk2_path = row["spk2_path"]
        noise_path = row["noise_path"]

        spk1_audio, _ = audio.process_audio(spk1_path, target_sr=sample_rate)
        spk2_audio, _ = audio.process_audio(spk2_path, target_sr=sample_rate)
        noise_audio, _ = audio.process_audio(noise_path, target_sr=sample_rate)

        s1_dur = int(min(len(spk1_audio), audio_len))
        s2_dur = int(min(len(spk2_audio), audio_len))

        s1_start = int(0)
        s2_start = int(0)

        s1_end = s1_start + s1_dur
        s2_end = s2_start + s2_dur

        if s1_start == s2_start:
            overlap = s1_dur
        elif s1_start > s2_start:
            overlap = max(s2_end - s1_start, 0)
        else:
            overlap = - max(s1_end - s2_start, 0)
        overlap_ms = int(overlap / sample_rate * 1000)

        mix_start = 0
        mix_end = max(s1_end, s2_end, audio_len)

        if s1_start > 0:
            s1_silence_start_ms = 0
            s1_silence_end_ms = int(s1_start / sample_rate * 1000)
        else:
            s1_silence_start_ms = s1_end / sample_rate * 1000
            s1_silence_end_ms = mix_end / sample_rate * 1000

        # mix the audio and add noise
        mixture = np.pad(spk1_audio[:(s1_end - s1_start)],
                         (s1_start - mix_start, np.abs(s1_end - mix_end)), constant_values=0.0)
        mixture += np.pad(spk2_audio[:(s2_end - s2_start)],
                          (s2_start - mix_start, np.abs(s2_end - mix_end)), constant_values=0.0)
        mixture += np.pad(noise_audio[:(s1_end - s1_start)],
                                        (s1_start - mix_start, np.abs(s1_end - mix_end)), constant_values=0.0)

        assert mixture.shape[0] == audio_len

        write(row["mixture_path"], sample_rate, mixture)

        overlap_ms_list.append(overlap_ms)
        s1_silence_start_list.append(s1_silence_start_ms)
        s1_silence_end_list.append(s1_silence_end_ms)

    meta["overlap_ms"] = overlap_ms_list
    meta["s1_silence_start_ms"] = s1_silence_start_list
    meta["s1_silence_end_ms"] = s1_silence_end_list

    meta.to_csv(os.path.join(meta_path, f"{split}_mixture_generation.csv"))


if __name__ == "__main__":
    print("generate valid")
    mixture_5k(split="val")

    print("generate train")
    mixture_5k(split="train")

    print("generate test")
    mixture_5k(split="test")
