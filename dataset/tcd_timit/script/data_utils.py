import librosa
import os
import numpy as np
import cv2
import math
from librosa import util


# Define parameters:
# global_frame_rate = 29.970030  # frames per second
wlen_sec = 0.064  # window length in seconds
hop_percent = 0.25  # math.floor((1 / (wlen_sec * global_frame_rate)) * 1e4) / 1e4  # hop size as a percentage of the window length
win = 'hann'  # type of window function (to perform filtering in the time domain)
center = False  # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect'  # This argument is ignored if center = False
pad_at_end = True  # pad audio file at end to match same size after stft + istft

# Noise robust VAD
vad_quantile_fraction_begin = 0.5  # 0.93
vad_quantile_fraction_end = 0.55  # 0.99
vad_quantile_weight = 1.0  # 0.999
vad_threshold = 1.7

# Other parameters:
sampling_rate = 16000
dtype = 'complex64'
eps = 1e-8


def clean_speech_VAD(speech_t,
                     fs=16e3,
                     wlen_sec=50e-3,
                     hop_percent=0.25,
                     center=True,
                     pad_mode='reflect',
                     pad_at_end=True,
                     vad_threshold=1.70):
    """ Computes VAD based on threshold in the time domain

    Args:
        speech_t ([type]): [description]
        fs ([type]): [description]
        wlen_sec ([type]): [description]
        hop_percent ([type]): [description]
        center ([type]): [description]
        pad_mode ([type]): [description]
        pad_at_end ([type]): [description]
        eps ([type], optional): [description]. Defaults to 1e-8.

    Returns:
        ndarray: [description]
    """
    nfft = int(wlen_sec * fs)  # STFT window length in samples
    hopsamp = int(hop_percent * nfft)  # hop size in samples
    # Sometimes stft / istft shortens the output due to window size
    # so you need to pad the end with hopsamp zeros
    if pad_at_end:
        utt_len = len(speech_t) / fs
        if math.ceil(utt_len / wlen_sec / hop_percent) != int(utt_len / wlen_sec / hop_percent):
            y = np.pad(speech_t, (0, hopsamp), mode='constant')
        else:
            y = speech_t.copy()
    else:
        y = speech_t.copy()

    if center:
        y = np.pad(y, int(nfft // 2), mode=pad_mode)

    y_frames = util.frame(y, frame_length=nfft, hop_length=hopsamp)

    power = np.power(y_frames, 2).sum(axis=0)
    vad = power > np.power(10, vad_threshold) * np.min(power)
    vad = np.float32(vad)
    vad = vad[None]
    return vad


def clean_speech_IBM(speech_tf,
                     eps=1e-8,
                     ibm_threshold=50):
    """ Calculate softened mask
    """
    mag = abs(speech_tf)
    power_db = 20 * np.log10(mag + eps)  # Smoother mask with log
    mask = power_db > np.max(power_db) - ibm_threshold
    mask = np.float32(mask)
    return mask


def noise_robust_clean_speech_IBM(speech_t,
                                  speech_tf,
                                  fs=16e3,
                                  wlen_sec=50e-3,
                                  hop_percent=0.25,
                                  center=True,
                                  pad_mode='reflect',
                                  pad_at_end=True,
                                  vad_threshold=1.70,
                                  eps=1e-8,
                                  ibm_threshold=50):
    """
    Create IBM labels robust to noisy speech recordings using noise-robst VAD.
    In particular, the labels are robust to noise occurring before / after speech.
    """
    # Compute vad
    vad = clean_speech_VAD(speech_t,
                           fs=fs,
                           wlen_sec=wlen_sec,
                           hop_percent=hop_percent,
                           center=center,
                           pad_mode=pad_mode,
                           pad_at_end=pad_at_end,
                           vad_threshold=vad_threshold)

    # Binary mask
    ibm = clean_speech_IBM(speech_tf,
                           eps=eps,
                           ibm_threshold=ibm_threshold)

    # Noise-robust binary mask
    ibm = ibm * vad
    return ibm

def create_ground_truth_labels_from_path(audio_path):
    raw_clean_audio, Fs = librosa.load(audio_path, sr=sampling_rate)

    mask_labels = clean_speech_VAD(raw_clean_audio,
                           fs=sampling_rate,
                           wlen_sec=wlen_sec,
                           hop_percent=hop_percent,
                           center=center,
                           pad_mode=pad_mode,
                           pad_at_end=pad_at_end,
                           vad_threshold=vad_threshold)

    return mask_labels.T


def create_ground_truth_labels(raw_clean_audio):

    mask_labels = clean_speech_VAD(raw_clean_audio,
                           fs=sampling_rate,
                           wlen_sec=wlen_sec,
                           hop_percent=hop_percent,
                           center=center,
                           pad_mode=pad_mode,
                           pad_at_end=pad_at_end,
                           vad_threshold=vad_threshold)

    return mask_labels.T


def create_video_paths_list(base_path):
    video_paths_list = []
    speaker_folders = sorted([x for x in os.listdir(base_path)])
    for speaker in speaker_folders:
            speaker_path = os.path.join(base_path, speaker)
            speaker_mat_files = sorted([y for y in os.listdir(speaker_path)])

            for sentence_mat_file in speaker_mat_files:
                sentence_video_path = os.path.join(speaker_path, sentence_mat_file)
                video_paths_list.append(sentence_video_path)

    return video_paths_list


def create_audio_paths_list(base_path):
    audio_paths_list = []
    speaker_folders = sorted([x for x in os.listdir(base_path)])
    for speaker in speaker_folders:
            speaker_path = os.path.join(base_path, speaker, "straightcam")
            speaker_wav_files = sorted([y for y in os.listdir(speaker_path)])

            for sentence_wav_file in speaker_wav_files:
                sentence_audio_path = os.path.join(speaker_path, sentence_wav_file)
                audio_paths_list.append(sentence_audio_path)

    return audio_paths_list


def animate_npy_file(npy_path, scale_by_255=False):
    with open(npy_path, 'rb') as in_f:
        loaded_np_image = np.load(in_f)
        nr_frames = loaded_np_image.shape[0]
        print("Data in matrix form: ", loaded_np_image)
        for frame in range(nr_frames):
            if scale_by_255:
                cv2.imshow("Loaded image: ", 255 * loaded_np_image[frame, :, :])
            else:
                cv2.imshow("Loaded image: ", loaded_np_image[frame, :, :])

            cv2.waitKey(0)