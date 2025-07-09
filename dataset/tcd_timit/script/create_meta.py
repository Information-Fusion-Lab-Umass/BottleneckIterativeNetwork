import sys
import os.path
sys.path.append("../../..")

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import utils.audio as audio
import random

raw_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT"
gray_processed_path = "/project/pi_mfiterau_umass_edu/sidong/speech/dataset/tcd_timit/script/dda_gray_nmcroi_dataout"
opticalflow_processed_path = "/project/pi_mfiterau_umass_edu/sidong/speech/dataset/tcd_timit/script/dda_optical_flow_nmcroi_dataout"

meta_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT_meta/"
noise_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT_NOISE/"
mixture_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT_EXP/audio_mixture"


categories = ["Babble", "Cafe", "Car", "LR", "Street", "White"]
db_level = ["-5", "0", "5", "10", "15", "20"]


def init_meta():
    record = {"spk": [], "fid": [],
              "audio_path": [], "gray_video_path": [], "optical_flow_path": [],
              "audio_length": [],
              "video_length": [],
              "video_H": [],
              "video_W": []}

    for spk in tqdm(os.listdir(raw_path)):
        spk_gray_path = os.path.join(gray_processed_path, spk)
        spk_of_path = os.path.join(opticalflow_processed_path, spk)
        spk_raw_path = os.path.join(raw_path, spk)

        for filename in os.listdir(spk_gray_path):
            fid = filename.replace(".npy", "")

            record["spk"].append(spk)
            record["fid"].append(fid)

            wavname = f"{fid}.wav"

            wav_path = os.path.join(spk_raw_path, "straightcam", wavname)
            gray_path = os.path.join(spk_gray_path, filename)
            of_path = os.path.join(spk_of_path, filename)
            assert os.path.exists(of_path)

            record["audio_path"].append(wav_path)
            record["gray_video_path"].append(gray_path)
            record["optical_flow_path"].append(of_path)

            T, H, W = np.load(gray_path).shape
            record["video_length"].append(T)
            record["video_H"].append(H)
            record["video_W"].append(W)

            wavfile, _ = audio.process_audio(wav_path, target_sr=16000)
            record["audio_length"].append(wavfile.shape[0])


            # print(wavfile.shape[0], T, H, W)

    df = pd.DataFrame.from_dict(record)
    df.to_csv(os.path.join(meta_path, "single_spk_meta.csv"))


def init_meta_5k():
    valtest_path = os.path.join(meta_path, "valtest_split.txt")
    train_path = os.path.join(meta_path, "train_split.txt")

    valtest_spks = set([])
    with open(valtest_path, "r") as f:
        for line in tqdm(f.readlines()):
            line = line[:-1]
            fields = line.split("\\")
            valtest_spks.add(fields[3])

    valtest_spks = list(valtest_spks)
    val_spks = valtest_spks[:8]
    test_spks = valtest_spks[8:]
    print("num val speakers: ", len(val_spks))
    print("num test speakers: ", len(test_spks))

    def create_pd(spk_dict, split):
        df_dict = {
            "spk1": [],
            "spk1_fid": [],
            "spk1_path": [],
            "spk2": [],
            "spk2_fid": [],
            "spk2_path": [],
            "noise_type": [],
            "noise_db": [],
            "noise_path": [],
            "mixture_path": []
        }

        spk_list = list(spk_dict.keys())

        for spk1 in tqdm(spk_list):
            other_spks = [s for s in spk_list if s != spk1]
            spk1_fids = spk_dict[spk1]
            for spk1_fid in spk1_fids:
                spk2 = random.choice(other_spks)
                spk2_fid = random.choice(spk_dict[spk2])

                noise_type = random.choice(categories)
                noise_db = random.choice(db_level)

                df_dict["spk1"].append(spk1)
                df_dict["spk1_fid"].append(spk1_fid)
                df_dict["spk1_path"].append(os.path.join(raw_path, spk1, "straightcam", f"{spk1_fid}.wav"))

                df_dict["spk2"].append(spk2)
                df_dict["spk2_fid"].append(spk2_fid)
                df_dict["spk2_path"].append(os.path.join(raw_path, spk2, "straightcam", f"{spk2_fid}.wav"))

                df_dict["noise_type"].append(noise_type)
                df_dict["noise_db"].append(noise_db)
                df_dict["noise_path"].append(os.path.join(noise_path, noise_type, noise_db, "volunteers",
                                                          spk1, "straightcam", f"{spk1_fid}.wav"))

                df_dict["mixture_path"].append(os.path.join(mixture_path, f"{spk1}_{spk1_fid}_{spk2}_{spk2_fid}.wav"))

        df = pd.DataFrame.from_dict(df_dict)
        df.to_csv(os.path.join(meta_path, f"{split}_mixture.csv"))

    val_dict = {}
    test_dict = {}

    with open(valtest_path, "r") as f:
        for line in tqdm(f.readlines()):
            line = line[:-1]
            fields = line.split("\\")
            spk = fields[3]
            fid = fields[-1].replace(".mfc", "")
            if spk in val_spks:
                if spk not in val_dict:
                    val_dict[spk] = [fid]
                else:
                    val_dict[spk].append(fid)
            elif spk in test_spks:
                if spk not in test_dict:
                    test_dict[spk] = [fid]
                else:
                    test_dict[spk].append(fid)
    train_dict = {}

    with open(train_path, "r") as f:
        for line in tqdm(f.readlines()):
            line = line[:-1]
            fields = line.split("\\")
            spk = fields[3]
            fid = fields[-1].replace(".mfc", "")
            if spk not in train_dict:
                train_dict[spk] = [fid]
            else:
                train_dict[spk].append(fid)

    print("val meta")
    create_pd(val_dict, "val")

    print("train meta")
    create_pd(train_dict, "train")

    print("test meta")
    create_pd(test_dict, "test")




def create_mixture_meta(split, time_duration=4000, sample_rate=16000):
    raw_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT"
    meta_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT_meta"
    noise_root_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT_NOISE"
    txt_meta_path = os.path.join(meta_path, f"mix_2_spk_{split}.txt")
    # ratio = round(np.random.uniform(-5.0, 20.0), 3)

    # single_meta = pd.read_csv(os.path.join(meta_path, "single_spk_meta.csv"))
    mix_dict = {"spk1": [], "spk1_fid": [],
                "spk1_audio_path": [], "spk1_db": [],
                "spk2": [], "spk2_fid": [],
                "spk2_audio_path": [], "spk2_db": [],
                "noise_path": [], "noise_db": [],
                "mix_path": []
                }

    with open(txt_meta_path, "r") as f:
        for line in tqdm(f.readlines()):
            line = line.replace("\n", "")
            fields = line.split(" ")
            s1_fid = fields[0].split("/")[-1].replace(".wav", "")
            s1_name = fields[0].split("/")[-2]
            s1_path = os.path.join(raw_path, s1_name, "straightcam", f"{s1_fid}.wav")

            s2_fid = fields[2].split("/")[-1].replace(".wav", "")
            s2_name = fields[2].split("/")[-2]
            s2_path = os.path.join(raw_path, s2_name, "straightcam", f"{s2_fid}.wav")

            assert os.path.exists(s1_path)
            assert os.path.exists(s2_path)

            s1_db = float(fields[1])
            s2_db = float(fields[3])

            noise_db = round(np.random.uniform(-5.0, 20.0), 3)
            noise_path = os.path.join(noise_root_path, "White", "0", "volunteers", s1_name, "straightcam", f"{s1_fid}.wav")
            assert os.path.exists(noise_path)

            mix_dict["spk1"].append(s1_name)
            mix_dict["spk1_fid"].append(s1_fid)
            mix_dict["spk1_audio_path"].append(s1_path)
            mix_dict["spk1_db"].append(s1_db)

            mix_dict["spk2"].append(s2_name)
            mix_dict["spk2_fid"].append(s2_fid)
            mix_dict["spk2_audio_path"].append(s2_path)
            mix_dict["spk2_db"].append(s2_db)

            mix_dict["noise_path"].append(noise_path)
            mix_dict["noise_db"].append(noise_db)

            mix_dir = os.path.join(
                "/project/pi_mfiterau_umass_edu/TCD_TIMIT_meta/audio_mixture",
                f"{s1_name}_{s1_fid}_{s2_name}_{s2_fid}.wav")

            mix_dict["mix_path"].append(mix_dir)


    df = pd.DataFrame.from_dict(mix_dict)
    df.to_csv(os.path.join(meta_path, f"mix_{split}_noisy.csv"))


if __name__ == "__main__":
    # init_meta()
    # print("train meta")
    # create_mixture_meta("train")
    #
    # print("val meta")
    # create_mixture_meta("valid")
    #
    # print("text meta")
    # create_mixture_meta("test")
    init_meta_5k()




