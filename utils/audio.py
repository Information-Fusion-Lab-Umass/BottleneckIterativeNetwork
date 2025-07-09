import time

import cv2
# import dlib
import librosa
import numpy as np
import torch
import torch.nn as nn

from typing import Dict

import logging
import random

# from mir_eval.separation import bss_eval_sources
# from pesq import pesq
# from pystoi import stoi

EPS = 1e-8
MAX_INT16 = np.iinfo(np.int16).max


def process_audio(path, target_sr = None):
    """
    Resample (if target_sr is not None) and normalize audio

    :param path:
    :param target_sr:
    :return:
    """
    # load audio
    signal, sr = librosa.load(path)
    if target_sr is not None:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
    else:
        target_sr = sr
    # if this audio has more than one channel, select the first channel
    if len(signal.shape) > 1:
        signal = signal[:, 0]
    # do normalization for the audio
    signal -= np.mean(signal)
    signal /= (np.max(np.abs(signal)) + np.spacing(1))

    return signal, target_sr


def cal_multisource_sdr(source, estimate_source):
    num_source = source.shape[1]
    avg_sdr = 0
    for i in range(num_source):
        avg_sdr += cal_SDR(source[:, i, :], estimate_source[:, i, :]) / num_source
    return avg_sdr


def cal_SDR(source, estimate_source):
    assert source.size() == estimate_source.size()
    # flatten the input to (B, num sources * len)
    # B = source.shape[0]
    # source = source.flatten(start_dim=1)
    # estimate_source = estimate_source.flatten(start_dim=1)

    # estimate_source += EPS # the estimated source is zero sometimes

    noise = source - estimate_source
    ratio = torch.sum(source ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sdr = 10 * torch.log10(ratio + EPS)

    return sdr


def get_audio_embedding(ref_audio,
                        audio_fbank,
                        pretrained_audio_embedding_model,
                        num_chunks = 1):
    if len(ref_audio.shape) > 2:
        ref_audio = ref_audio.squeeze(1)
    chunk_size = int(ref_audio.shape[-1] / num_chunks)
    if num_chunks > 1:
        signal_chunks = torch.split(ref_audio, chunk_size, dim=1)
        emb = []
        for chunk_i in range(num_chunks):
            chunk_emb = audio_fbank(signal_chunks[chunk_i])
            chunk_emb = pretrained_audio_embedding_model(chunk_emb).squeeze(1)
            emb.append(chunk_emb)
        return torch.stack(emb, dim=1)
    ref_audio = audio_fbank(ref_audio)
    ref_audio = pretrained_audio_embedding_model(ref_audio).squeeze(1)
    return ref_audio