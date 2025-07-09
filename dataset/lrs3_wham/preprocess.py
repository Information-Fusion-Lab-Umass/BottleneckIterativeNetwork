import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from scipy import signal
from scipy.io import wavfile
import cv2 as cv
from scipy.special import softmax
from tqdm import tqdm


def audio_mixture():
    self.audio_max_samples = int(max_duration_ms / 1000 * sampling_rate)
    self.visual_max_samples = int(max_duration_ms / 1000 * fps)
    logging.info(f"Reading mixture file")
    mix_speaker_list = []

    with open(mixture_file_path, "r") as f:
        for line in f.readlines():
            fields = line.replace("\n", "").split(" ")
            mix_speaker_list.append(fields)
    mix_speaker_dict = []
    for mix_speaker_test in tqdm(mix_speaker_list):
        speaker_1, speaker_2, mixture_filename, speaker_1_db, s1_start, s2_start, overlap_ms, s1_silence_start, s1_silence_end = mix_speaker_test
        main_speaker_name, main_file_id, background_speaker_name, background_file_id = mixture_filename.split("/")[
            -1].replace(".wav", "").split("_")
        temp = {
            "clean_audio_filepath": speaker_1,
            "mixture_audio_filepath": mixture_filename,
            "clean_audio_speaker_name": main_speaker_name,
            "clean_audio_file_id": main_file_id,
            "background_db": -float(speaker_1_db),
            "overlap_ms": float(overlap_ms),
            "main_speaker_start_ms": float(s1_start),
            "background_speaker_start_ms": float(s2_start),
            "main_speaker_silence_start_ms": float(s1_silence_start),
            "main_speaker_silence_end_ms": float(s1_silence_end)
        }
        if include_visual:
            if use_mouth_raw:
                main_visual_feature_filepath = os.path.join(visual_feature_dir,
                                                            f"{main_speaker_name}_{main_file_id}/raw.npz")
                background_visual_feature_filepath = os.path.join(visual_feature_dir,
                                                                  f"{background_speaker_name}_{background_file_id}/raw.npz")
            else:
                main_visual_feature_filepath = os.path.join(visual_feature_dir,
                                                            f"{main_speaker_name}_{main_file_id}/embedding.npy")
                background_visual_feature_filepath = os.path.join(visual_feature_dir,
                                                                  f"{background_speaker_name}_{background_file_id}/embedding.npy")
            temp["visual_feature_filepath"] = main_visual_feature_filepath
            temp["background_visual_feature_filepath"] = background_visual_feature_filepath

        if include_reference_audio:
            possible_reference_audio = [f for f in os.listdir(os.path.join(clean_audio_dir, f"{main_speaker_name}"))
                                        if f.endswith(".wav") and f != f"{main_file_id}.wav"]
            if len(possible_reference_audio) == 0:
                continue
            main_reference_audio_file_id = random.sample(possible_reference_audio, 1)[0]
            temp["reference_audio_filepath"] = os.path.join(clean_audio_dir,
                                                            f"{main_speaker_name}/{main_reference_audio_file_id}")

            if include_background_feature:
                possible_reference_audio = [f for f in
                                            os.listdir(os.path.join(clean_audio_dir, f"{background_speaker_name}"))
                                            if f.endswith(".wav") and f != f"{background_file_id}.wav"]
                if len(possible_reference_audio) == 0:
                    continue
                background_reference_audio_file_id = random.sample(possible_reference_audio, 1)[0]
                temp["background_reference_audio_filepath"] = os.path.join(clean_audio_dir,
                                                                           f"{background_speaker_name}/{background_reference_audio_file_id}")

        mix_speaker_dict.append(temp)

    print(f"Number of actual samples: {len(mix_speaker_dict)}")
    self.mix_speaker_dict = mix_speaker_dict
    full_data = []

    print(f"Processing data")
    for idx in tqdm(range(len(mix_speaker_dict))):
        data = {}
        data["mixture"] = self._get_audio_feat(self.mix_speaker_dict[idx]["mixture_audio_filepath"],
                                               start_pad=0,
                                               total_length=self.audio_max_samples
                                               )

        main_speaker_start_audio = int(self.mix_speaker_dict[idx]["main_speaker_start_ms"] / 1000 * sampling_rate)

        data["clean_audio"] = self._get_audio_feat(self.mix_speaker_dict[idx]["clean_audio_filepath"],
                                                   start_pad=main_speaker_start_audio,
                                                   total_length=self.audio_max_samples
                                                   )
        if include_visual:
            main_speaker_start_visual = int(self.mix_speaker_dict[idx]["main_speaker_start_ms"] / 1000 * 25)
            background_speaker_start_visual = int(self.mix_speaker_dict[idx]["background_speaker_start_ms"] / 1000 * 25)
            if use_mouth_raw:
                data["visual_feature"] = self._get_visual_feat_raw(
                    self.mix_speaker_dict[idx]["visual_feature_filepath"],
                    start_pad=main_speaker_start_visual,
                    total_length=self.visual_max_samples
                    )
            else:
                data["visual_feature"] = self._get_visual_feat(self.mix_speaker_dict[idx]["visual_feature_filepath"],
                                                               start_pad=main_speaker_start_visual,
                                                               total_length=self.visual_max_samples
                                                               )
            if include_background_feature:
                if use_mouth_raw:
                    data["background_visual_feature"] = self._get_visual_feat_raw(
                        self.mix_speaker_dict[idx]["background_visual_feature_filepath"],
                        start_pad=background_speaker_start_visual,
                        total_length=self.visual_max_samples)
                else:
                    data["background_visual_feature"] = self._get_visual_feat(
                        self.mix_speaker_dict[idx]["background_visual_feature_filepath"],
                        start_pad=background_speaker_start_visual,
                        total_length=self.visual_max_samples)
        if include_reference_audio:
            data["reference_audio"] = self._get_audio_feat(self.mix_speaker_dict[idx]["reference_audio_filepath"],
                                                           total_length=self.audio_max_samples)
            if include_background_feature:
                data["background_reference_audio"] = self._get_audio_feat(
                    self.mix_speaker_dict[idx]["background_reference_audio_filepath"],
                    total_length=self.audio_max_samples)

        if mode == "test":
            data["clean_audio_path"] = self.mix_speaker_dict[idx]["clean_audio_filepath"]
            if include_reference_audio:
                data["reference_audio_path"] = self.mix_speaker_dict[idx]["reference_audio_filepath"]
            if include_visual:
                data["visual_feature_filepath"] = self.mix_speaker_dict[idx]["visual_feature_filepath"]
            data["mixture_path"] = self.mix_speaker_dict[idx]["mixture_audio_filepath"]
            data["clean_audio_speaker_name"] = self.mix_speaker_dict[idx]["clean_audio_speaker_name"]
            data["clean_audio_file_id"] = self.mix_speaker_dict[idx]["clean_audio_file_id"]
            data["background_speaker_db"] = self.mix_speaker_dict[idx]["background_db"]
            data["overlap_ms"] = self.mix_speaker_dict[idx]["overlap_ms"]
            data["main_speaker_start_ms"] = self.mix_speaker_dict[idx]["main_speaker_start_ms"]
            data["main_speaker_silence_start_ms"] = self.mix_speaker_dict[idx]["main_speaker_silence_start_ms"]
            data["main_speaker_silence_end_ms"] = self.mix_speaker_dict[idx]["main_speaker_silence_end_ms"]

        full_data.append(data)

    if cache_file_path is not None:
        pickle.dump(full_data, open(cache_file_path, "wb"))