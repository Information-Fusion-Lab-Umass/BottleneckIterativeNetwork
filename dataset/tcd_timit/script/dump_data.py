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
import cv2


def dump_ntcd(split):
    meta_path = f"/work/pi_mfiterau_umass_edu/TCD_TIMIT/meta/{split}_mixture_generation.csv"
    meta = pd.read_csv(meta_path)

    # if split == "train":
    #     meta = meta.head()

    tcd_exp_path = "/work/pi_mfiterau_umass_edu/TCD_TIMIT/ntcd"

    mouth_path = "/work/pi_mfiterau_umass_edu/TCD_TIMIT/dda_gray_nmcroi_dataout"

    dump_data = []

    for idx, row in tqdm(meta.iterrows()):
        spk1_dir = row["spk1_path"]
        spk2_dir = row["spk2_path"]
        speaker_1, sr = audio.process_audio(spk1_dir, target_sr=16000)
        speaker_2, sr = audio.process_audio(spk2_dir, target_sr=16000)

        s1_start = int(0)
        s2_start = int(0)

        mix_start = 0
        s1_start = int(s1_start / 1000 * 16000)
        s2_start = int(s2_start / 1000 * 16000)

        max_audio_samples = 64000
        s1_dur = min(len(speaker_1), max_audio_samples)
        s2_dur = min(len(speaker_2), max_audio_samples)

        s1_end = s1_start + s1_dur
        s2_end = s2_start + s2_dur

        mix_end = max(s1_end, s2_end, max_audio_samples)

        # mix the audio and add noise
        clean_audio = np.pad(speaker_1[:(s1_end - s1_start)],
                         (s1_start - mix_start, np.abs(s1_end - mix_end)), constant_values=0.0)

        assert clean_audio.shape[0] == 64000

        video_path = os.path.join(mouth_path, row["spk1"], f"{row['spk1_fid']}.npy")
        video = np.load(video_path)
        resize_imgs = []
        for f in range(video.shape[0]):
            resize_imgs.append(cv2.resize(video[f], (64, 64)))
        resize_imgs = np.stack(resize_imgs, axis=0)

        pad_video = np.zeros((250, 64, 64))
        if resize_imgs.shape[0] < 250:
            pad_video[:video.shape[0]] = resize_imgs
        else:
            pad_video = resize_imgs[:250]

        if idx == 0:
            for f in range(resize_imgs.shape[0]):
                cv2.imwrite(os.path.join(tcd_exp_path, "video_sample", f"frame{f}.png"), resize_imgs[f])

        record = {"spk1_id": row["spk1"],
                  "spk2_id": row["spk2"],
                  "spk1_vid": row["spk1_fid"],
                  "spk2_vid": row["spk2_fid"],
                  "overlap_ms": row["overlap_ms"],
                  "clean_audio": clean_audio,
                  "mix_audio": audio.process_audio(row["mixture_path"], target_sr=16000)[0],
                  "video_npy": pad_video}

        dump_data.append(record)

    pickle.dump(dump_data,
                open(os.path.join(tcd_exp_path, f"mix_2_spk_{split}_fully_overlapped_only_5k_generation.pkl"),
                     "wb")
                )


def dump_ntcd_video(split):
    mouth_path = "/project/pi_mfiterau_umass_edu/sidong/speech/dataset/tcd_timit/script/dda_gray_nmcroi_dataout"
    meta_path = f"/project/pi_mfiterau_umass_edu/TCD_TIMIT_meta/{split}_mixture_generation.csv"
    tcd_exp_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT_EXP"

    meta = pd.read_csv(meta_path)
    dump_data = []
    for idx, row in tqdm(meta.iterrows()):
        video_path = os.path.join(mouth_path, row["spk1"], f"{row['spk1_fid']}.npy")
        record = {"spk1_id": row["spk1"],
                  "spk1_vid": row['spk1_fid'],
                  "video_npy": np.load(os.path.join(video_path, "video.npy"))}

        dump_data.append(record)
    print(len(dump_data))
    pickle.dump(dump_data,
                open(os.path.join(tcd_exp_path, "ntcd", f"spk_{split}_video.pkl"),
                     "wb")
                )


def batch_dump(full_list, df):

    for idx, row in tqdm(df.iterrows()):
        spk1_dir = row["spk1_path"]
        spk2_dir = row["spk2_path"]
        speaker_1, sr = audio.process_audio(spk1_dir, target_sr=16000)
        speaker_2, sr = audio.process_audio(spk2_dir, target_sr=16000)

        s1_start = int(0)
        s2_start = int(0)

        mix_start = 0
        s1_start = int(s1_start / 1000 * 16000)
        s2_start = int(s2_start / 1000 * 16000)

        max_audio_samples = 64000
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


        assert clean_audio.shape[1] == 64000

        record = {"spk1_id": row["spk1"],
                  "spk2_id": row["spk2"],
                  "spk1_vid": row["spk1_fid"],
                  "spk2_vid": row["spk2_fid"],
                  "overlap_ms": row["overlap_ms"],
                  "clean_audio": clean_audio,
                  "mix_audio": audio.process_audio(row["mixture_path"], target_sr=16000)[0]}

        full_list.append(record)


def parallel_dump(split):
    meta_path = f"/work/pi_mfiterau_umass_edu/TCD_TIMIT/meta/{split}_mixture_generation.csv"
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

    full_list = list(full_list)

    pickle.dump(full_list,
                open(os.path.join("/work/pi_mfiterau_umass_edu/TCD_TIMIT/ntcd", f"mix_2_spk_{split}_fully_overlapped_only_full_double_dist.pkl"),
                     "wb")
                )


def double_dump(split):
    meta_path = f"/work/pi_mfiterau_umass_edu/TCD_TIMIT/meta/{split}_mixture_generation.csv"
    meta = pd.read_csv(meta_path)
    full_list = []
    for idx, row in tqdm(meta.iterrows()):
        spk1_dir = row["spk1_path"]
        spk2_dir = row["spk2_path"]
        speaker_1, sr = audio.process_audio(spk1_dir, target_sr=16000)
        speaker_2, sr = audio.process_audio(spk2_dir, target_sr=16000)

        s1_start = int(0)
        s2_start = int(0)

        mix_start = 0
        s1_start = int(s1_start / 1000 * 16000)
        s2_start = int(s2_start / 1000 * 16000)

        max_audio_samples = 64000
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

        assert clean_audio.shape[1] == 64000

        record = {"spk1_id": row["spk1"],
                  "spk2_id": row["spk2"],
                  "spk1_vid": row["spk1_fid"],
                  "spk2_vid": row["spk2_fid"],
                  "overlap_ms": row["overlap_ms"],
                  "clean_audio": clean_audio,
                  "mix_audio": audio.process_audio(row["mixture_path"], target_sr=16000)[0]}

        full_list.append(record)

    pickle.dump(full_list,
                open(os.path.join("/work/pi_mfiterau_umass_edu/TCD_TIMIT/ntcd", f"mix_2_spk_{split}_fully_overlapped_only_full_double.pkl"),
                     "wb")
                )



def replace_video(split):
    ntcd_path = "/work/pi_mfiterau_umass_edu/TCD_TIMIT/ntcd"
    data_dict = pickle.load(open(os.path.join(ntcd_path, f"mix_2_spk_{split}_fully_overlapped_only_5k_generation.pkl"), "rb"))
    processed_video = pickle.load(open(os.path.join(ntcd_path, f"video_latent_{split}.pkl"), "rb"))

    for record in tqdm(data_dict):
        record.pop('video_npy', None)
        record["video_npy"] = processed_video[record["spk1_id"]][record["spk1_vid"]]

    pickle.dump(data_dict,
                open(os.path.join(ntcd_path,
                                  f"mix_2_spk_{split}_fully_overlapped_only_5k_videolatent.pkl"),
                     "wb")
                )


def remove_video(split):
    ntcd_path = "/work/pi_mfiterau_umass_edu/TCD_TIMIT/ntcd"
    data_dict = pickle.load(open(os.path.join(ntcd_path, f"mix_2_spk_{split}_fully_overlapped_only_5k_generation.pkl"), "rb"))
    processed_video = pickle.load(open(os.path.join(ntcd_path, f"video_latent_{split}.pkl"), "rb"))

    for record in tqdm(data_dict):
        record.pop('video_npy', None)

    pickle.dump(data_dict,
                open(os.path.join(ntcd_path,
                                  f"mix_2_spk_{split}_fully_overlapped_only_5k.pkl"),
                     "wb")
                )


if __name__ == "__main__":
    print("generate train dump")
    # replace_video()
    # dump_ntcd_video("train")
    # dump_ntcd("train")
    # remove_video("train")
    # # parallel_dump_lrs3("train")
    # parallel_dump("train")
    double_dump("train")

    #
    print("generate test dump")
    # dump_ntcd_video("test")
    # dump_ntcd("test")
    # remove_video("test")
    # # parallel_dump_lrs3("test")
    # parallel_dump("test")
    double_dump("test")

    print("generate val dump")
    # parallel_dump("val")
    # dump_ntcd_video("val")
    # dump_ntcd("val")
    # remove_video("val")
    # parallel_dump_lrs3("val")
    double_dump("val")