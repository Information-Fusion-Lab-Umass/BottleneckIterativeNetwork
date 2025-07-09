import os.path
import sys

sys.path.append("../../..")

import pandas as pd
import numpy as np
import os
import librosa
from tqdm import tqdm
from scipy.io.wavfile import write

import utils.audio as audio


"""
Wham train split has 20,000 records, lrs3 mixture audio train split has 50,000  records
the script is to match wham to lrs3 mixture audio with the following algo:
match one wham to one mixture for the first 20,000 lrs3 mixture audio
if there are still unmatched lrs3 mixture audio
loop back to the first wham, select the noise starting from what previous sample ends
i.e. lrs3 audio mixture sample 1 uses wham sample 1's first 2 seconds
then lrs3 audio mixture sample 20,001 uses wham sample 1's 2 to 4 second
if current wham sample is not long enough, look for the next wham record
"""

WHAM_ROOT_PATH = "/project/pi_mfiterau_umass_edu/wham/wham_noise"
LRS3_ROOT_PATH = "/project/pi_mfiterau_umass_edu/LRS3"

split_dict = {"tr": "train", "cv": "val", "tt": "test"}
reverse_split_dict = {"train": "tr", "val": "cv", "test": "tt"}


def generate_wham_len_meta(split="tr"):
    """
    :param split: tr for train, tt for test, cv for validation
    generate a new meta file of wham based on mix_param_meta_{split},csv
    by adding one column recording the full time length of the noise audio
    generated meta files use train/val/test as split name
    :return:
    """
    wham_path = os.path.join(WHAM_ROOT_PATH, split)
    wham_meta_path = os.path.join(WHAM_ROOT_PATH, "metadata")
    meta_file = pd.read_csv(os.path.join(wham_meta_path, f"mix_param_meta_{split}.csv"))

    time_list = []

    for idx, row in meta_file.iterrows():
        filename = row["utterance_id"]
        audio_path = os.path.join(wham_path, filename)
        assert os.path.exists(audio_path)
        time_list.append(librosa.get_duration(filename=audio_path))

    meta_file["time_duration"] = time_list
    meta_file.to_csv(os.path.join(WHAM_ROOT_PATH, "metadata", f"noise_meta_{split_dict[split]}.csv"))


def generate_lrs3_wham_mixture(split="train", lrs3_duration=2000, sample_rate=16000):
    """
    :param split:
    :param lrs3_duration: lrs audio time duration in miliseconds
    :param sample_rate:
    :return:

    TODO: currently the code does not defend the case when total length of lrs3 exceeds the total length of wham
    """
    lrs3_meta_path = os.path.join(LRS3_ROOT_PATH, "meta", f"mix_2_spk_{split}_fully_overlapped_only_full.csv")
    lrs3_meta = pd.read_csv(lrs3_meta_path)

    wham_meta_path = os.path.join(WHAM_ROOT_PATH, "metadata", f"noise_meta_{split}.csv")
    wham_meta = pd.read_csv(wham_meta_path)

    # record the usage of wham noise
    wham_len = {}

    for idx, row in wham_meta.iterrows():
        wham_len[row["utterance_id"]] = {"total_duration": row["time_duration"],
                                         "last_stop": 0.0}

    # record wham ids in list
    wham_ids = list(wham_len.keys())
    wham_id_pointer = 0

    # three new columns to be added to lrs3 meta recording the wham information
    selected_wham_ids = []
    wham_starttime = []
    wham_endtime = []
    noise_db = []

    # additional lrs3 info columns
    overlap_ms_list = []
    s1_silence_start_ms_list = []
    s1_silence_end_ms_list = []

    for idx, row in tqdm(lrs3_meta.iterrows()):
        selected_wham_id = wham_ids[wham_id_pointer]

        # if the current wham noise does not have sufficiently long unused audio, visit the next wham noise
        while (wham_len[selected_wham_id]["total_duration"] - wham_len[selected_wham_id]["last_stop"]) < (lrs3_duration / 1000):
            wham_id_pointer += 1

            # if we've visited all the wham noise records, go back to the first one
            if wham_id_pointer >= len(wham_ids):
                wham_id_pointer -= len(wham_ids)

            selected_wham_id = wham_ids[wham_id_pointer]

        # record the current wham selection
        selected_wham_ids.append(selected_wham_id)
        wham_starttime.append(wham_len[selected_wham_id]["last_stop"])

        starttime = wham_len[selected_wham_id]["last_stop"]
        endtime = wham_len[selected_wham_id]["last_stop"] + lrs3_duration / 1000

        wham_endtime.append(endtime)
        wham_len[selected_wham_id]["last_stop"] = endtime

        # random sample noise db level
        ratio = round(np.random.uniform(-6.0, 3.0), 3)
        noise_db.append(ratio)

        # move to the next wham for the next lrs3 record
        wham_id_pointer += 1

        # if we've visited all the wham noise records, go back to the first one
        if wham_id_pointer >= len(wham_ids):
            wham_id_pointer -= len(wham_ids)

        # load the selected noise
        noise, sr = audio.process_audio(os.path.join(WHAM_ROOT_PATH, reverse_split_dict[split], selected_wham_id),
                                        target_sr=sample_rate)
        noise_ratio = 10 ** (ratio / 20.0)
        noise = noise[int(starttime * sample_rate): int(endtime * sample_rate)]

        # generate the current mixture
        speaker_1_path = row["speaker_1_dir"]
        speaker_2_path = row["speaker_2_dir"]
        speaker_1_db = row["speaker_1_db"]
        s1_start = row["s1_start"]
        s2_start = row["s2_start"]
        speaker_1, sr = audio.process_audio(speaker_1_path, target_sr=sample_rate)
        speaker_2, sr = audio.process_audio(speaker_2_path, target_sr=sample_rate)

        speaker_1_ratio = 10 ** (speaker_1_db / 20.0)
        speaker_2_ratio = 10 ** (-speaker_1_db / 20.0)

        mix_start = 0
        s1_start = int(s1_start / 1000 * sample_rate)
        s2_start = int(s2_start / 1000 * sample_rate)

        max_audio_samples = int(lrs3_duration / 1000 * sample_rate)
        s1_dur = min(len(speaker_1), max_audio_samples)
        s2_dur = min(len(speaker_2), max_audio_samples)

        s1_end = s1_start + s1_dur
        s2_end = s2_start + s2_dur

        if s1_start == s2_start:
            overlap = s1_dur
        elif s1_start > s2_start:
            overlap = max(s2_end - s1_start, 0)
        else:
            overlap = - max(s1_end - s2_start, 0)
        overlap_ms = int(overlap / sample_rate * 1000)

        mix_end = max(s1_end, s2_end)

        if s1_start > 0:
            s1_silence_start_ms = 0
            s1_silence_end_ms = int(s1_start / sample_rate * 1000)
        else:
            s1_silence_start_ms = s1_end / sample_rate * 1000
            s1_silence_end_ms = mix_end / sample_rate * 1000

        # mix the audio and add noise
        mixture = np.pad(speaker_1[:(s1_end - s1_start)] * speaker_1_ratio,
                         (s1_start - mix_start, np.abs(s1_end - mix_end)), constant_values=0.0)
        mixture += np.pad(speaker_2[:(s2_end - s2_start)] * speaker_2_ratio,
                          (s2_start - mix_start, np.abs(s2_end - mix_end)), constant_values=0.0)
        mixture += noise * noise_ratio

        write(row["mixture_dir"], sample_rate, mixture)

        overlap_ms_list.append(overlap_ms)
        s1_silence_start_ms_list.append(s1_silence_start_ms)
        s1_silence_end_ms_list.append(s1_silence_end_ms)

    lrs3_meta["wham_id"] = selected_wham_ids
    lrs3_meta["wham_starttime"] = wham_starttime
    lrs3_meta["wham_endtime"] = wham_endtime
    lrs3_meta["noise_db"] = noise_db

    lrs3_meta["overlap_ms"] = overlap_ms_list
    lrs3_meta["s1_silence_start_ms"] = s1_silence_start_ms_list
    lrs3_meta["s1_silence_end_ms"] = s1_silence_end_ms_list

    lrs3_meta.to_csv(os.path.join(LRS3_ROOT_PATH, "meta", f"mix_2_spk_{split}_fully_overlapped_only_full_wham.csv"))


if __name__ == "__main__":
    print("generate train")
    # generate_wham_len_meta("tr")
    generate_lrs3_wham_mixture("train", lrs3_duration=2000)

    print("generate val")
    # generate_wham_len_meta("cv")
    generate_lrs3_wham_mixture("val", lrs3_duration=2000)

    print("generate test")
    # generate_wham_len_meta("tt")
    generate_lrs3_wham_mixture("test", lrs3_duration=2000)






