import pickle
import os
import math

root_path = "/work/pi_mfiterau_umass_edu/LRS3/train_full_wham"


def subset(split, ratio=0.1):
    audio_path = os.path.join(root_path, f"mix_2_spk_{split}_fully_overlapped_only_full_double_dist.pkl")
    audio = pickle.load(open(audio_path, "rb"))

    video_path = os.path.join(root_path, f"video_single_latent_{split}.pkl")
    video = pickle.load(open(video_path, "rb"))

    sub_audio = audio[:math.ceil(len(audio) * ratio)]
    print(len(sub_audio))
    sub_video = {}
    for record in sub_audio:
        video1_id = f"{record['spk1_id']}_{record['spk1_vid']}"
        video2_id = f"{record['spk2_id']}_{record['spk2_vid']}"

        if video1_id not in sub_video:
            sub_video[video1_id] = video[video1_id]
        if video2_id not in sub_video:
            sub_video[video2_id] = video[video2_id]

    pickle.dump(sub_audio,
                open(os.path.join(root_path, "subset", f"mix_2_spk_{split}_fully_overlapped_only_full_double_dist_{int(ratio * 100)}p.pkl"), "wb"))

    pickle.dump(sub_video,
                open(os.path.join(root_path, "subset",
                                  f"video_single_latent_{split}_{int(ratio * 100)}p.pkl"),
                     "wb"))


if __name__ == "__main__":
    subset("train")
    subset("val")
    subset("test")