import sys

sys.path.append("../../..")

import pickle
import os
import pandas as pd
import utils.audio as audio
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import functools


def dump_lrs3(split):
    meta_path = f"/project/pi_mfiterau_umass_edu/LRS3/meta/mix_2_spk_{split}_fully_overlapped_only_full_wham.csv"
    meta = pd.read_csv(meta_path)

    # if split == "train":
    #     meta = meta.head()

    lrs3_path = "/project/pi_mfiterau_umass_edu/LRS3"

    mouth_path = "/project/pi_mfiterau_umass_edu/LRS3/video_mouth_64x64"

    dump_data = []
    for idx, row in tqdm(meta.iterrows()):
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

        mix_end = max(s1_end, s2_end)

        # mix the audio and add noise
        clean_audio = np.pad(speaker_1[:(s1_end - s1_start)],
                         (s1_start - mix_start, np.abs(s1_end - mix_end)), constant_values=0.0)

        assert clean_audio.shape[0] == 32000

        spk1_split = spk1_dir.split("/")
        spk2_split = spk2_dir.split("/")
        video_path = os.path.join(mouth_path, f"{spk1_split[-2]}_{spk1_split[-1].split('.')[0]}")
        record = {"spk1_id": spk1_split[-2],
                  "spk2_id": spk2_split[-2],
                  "spk1_vid": spk1_split[-1].split(".")[0],
                  "spk2_vid": spk2_split[-1].split(".")[0],
                  "overlap_ms": row["overlap_ms"],
                  "clean_audio": clean_audio,
                  "mix_audio": audio.process_audio(row["mixture_dir"].replace("audio_mixture", "audio_mixture_wham"),
                                                   target_sr=16000)[0],
                  "video_npy": np.load(os.path.join(video_path, "video.npy"))}

        dump_data.append(record)

    pickle.dump(dump_data,
                open(os.path.join(lrs3_path, "train_full_wham", f"mix_2_spk_{split}_fully_overlapped_only_full_generation.pkl"),
                     "wb")
                )


def batch_dump(full_list, df):
    mouth_path = "/project/pi_mfiterau_umass_edu/LRS3/video_mouth_64x64"

    for idx, row in tqdm(df.iterrows()):

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

        mix_end = max(s1_end, s2_end)

        # mix the audio and add noise
        clean_audio = np.pad(speaker_1[:(s1_end - s1_start)],
                             (s1_start - mix_start, np.abs(s1_end - mix_end)), constant_values=0.0)

        spk1_split = spk1_dir.split("/")
        spk2_split = spk2_dir.split("/")
        video_path = os.path.join(mouth_path, f"{spk1_split[-2]}_{spk1_split[-1].split('.')[0]}")

        assert clean_audio.shape[0] == 32000

        record = {"spk1_id": spk1_split[-2],
                  "spk2_id": spk2_split[-2],
                  "spk1_vid": spk1_split[-1].split(".")[0],
                  "spk2_vid": spk2_split[-1].split(".")[0],
                  "overlap_ms": row["overlap_ms"],
                  "clean_audio": clean_audio,
                  "mix_audio": audio.process_audio(row["mixture_dir"], target_sr=16000)[0],
                  "video_npy": np.load(os.path.join(video_path, "video.npy"))}

        full_list.append(record)


def parallel_dump_lrs3(split):
    lrs3_path = "/project/pi_mfiterau_umass_edu/LRS3"

    meta_path = f"/project/pi_mfiterau_umass_edu/LRS3/meta/mix_2_spk_{split}_fully_overlapped_only_full_wham.csv"
    meta = pd.read_csv(meta_path)

    manager = Manager()
    full_list = manager.list()

    number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    meta_split = np.array_split(meta, number_of_cores)
    args = []
    for i in range(number_of_cores):
        args.append((meta_split[i]))

    # multiprocssing pool to distribute tasks to:
    with Pool(number_of_cores) as pool:
        # distribute computations and collect results:
        # pool.starmap(batch_dump, args)
        pool.starmap(functools.partial(batch_dump, full_list), args)

    # print(len(video_frame_batch_list))
    # set_start_method("spawn", force=True)
    # # torch.set_num_threads(1)
    # pool = Pool(10)
    # pool.map(partial(process_batch, input_dir=lrs3_path, output_dir=mouth_path), video_frame_batch_list)

    # complete the processe
    pool.close()
    pool.join()

    pickle.dump(full_list,
                open(os.path.join(lrs3_path, "train_full_wham", f"mix_2_spk_{split}_fully_overlapped_only_full_generation_dist.pkl"),
                     "wb")
                )


def replace_video(split):
    lrs3_path = "/work/pi_mfiterau_umass_edu/LRS3"
    data_dict = pickle.load(open(os.path.join(lrs3_path, "train_full_wham", f"mix_2_spk_{split}_fully_overlapped_only_full_generation.pkl"), "rb"))
    processed_video = pickle.load(open(os.path.join(lrs3_path, "train_full_wham", f"video_latent_{split}_dctcn.pkl"), "rb"))

    for record in tqdm(data_dict):
        record.pop('video_npy', None)
        record["video_npy"] = processed_video[record["spk1_id"]][record["spk1_vid"]]

    pickle.dump(data_dict,
                open(os.path.join(lrs3_path, "train_full_wham",
                                  f"mix_2_spk_{split}_fully_overlapped_only_full_videotcn.pkl"),
                     "wb")
                )


def dump_double_lrs3(split):
    print("dump two speakers: ")
    meta_path = f"/project/pi_mfiterau_umass_edu/LRS3/meta/mix_2_spk_{split}_fully_overlapped_only_full_wham.csv"
    meta = pd.read_csv(meta_path)

    # if split == "train":
    #     meta = meta.head()

    lrs3_path = "/work/pi_mfiterau_umass_edu/LRS3"

    dump_data = []
    for idx, row in tqdm(meta.iterrows()):
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
                  "mix_audio": audio.process_audio(row["mixture_dir"].replace("audio_mixture", "audio_mixture_wham"),
                                                   target_sr=16000)[0]}

        dump_data.append(record)

    pickle.dump(dump_data,
                open(os.path.join(lrs3_path, "train_full_wham", f"mix_2_spk_{split}_fully_overlapped_only_full_double.pkl"),
                     "wb")
                )


def replace_video_audio(split):
    lrs3_path = "/work/pi_mfiterau_umass_edu/LRS3"
    data_dict = pickle.load(open(os.path.join(lrs3_path, "train_full_wham", f"mix_2_spk_{split}_fully_overlapped_only_full_generation.pkl"), "rb"))
    processed_video = pickle.load(open(os.path.join(lrs3_path, "train_full_wham", f"video_latent_{split}_dctcn.pkl"), "rb"))

    for record in tqdm(data_dict):
        record.pop('video_npy', None)
        record["video_npy"] = processed_video[record["spk1_id"]][record["spk1_vid"]]
        record[""]

    pickle.dump(data_dict,
                open(os.path.join(lrs3_path, "train_full_wham",
                                  f"mix_2_spk_{split}_fully_overlapped_only_full_double.pkl"),
                     "wb")
                )



if __name__ == "__main__":
    print("generate train dump")
    dump_double_lrs3("train")
    # dump_lrs3("train")
    # parallel_dump_lrs3("train")

    print("generate test dump")
    dump_double_lrs3("test")
    # dump_lrs3("test")
    # parallel_dump_lrs3("test")

    print("generate val dump")
    dump_double_lrs3("val")
    # dump_lrs3("val")
    # parallel_dump_lrs3("val")