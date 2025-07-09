import sys
sys.path.append(".")

import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.optim import lr_scheduler
import numpy as np
import random
import csv
from scipy.io.wavfile import write
import argparse
import yaml

import models.avlit as avlit
import engines.abs_separation as abs_sep
import dataset.lrs3.dataset as lrs3_set
import models.model_config
import plot.curve as curve
import utils.dynamic as dynamic
import models.loss as loss
import utils.audio as audio_util
import utils.distributed as distributed_util
import engines.lrs3_separation as lrs3_separation
from models.device_check import *

from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality

pretrain_root_dir = "/project/pi_mfiterau_umass_edu/dolby/speech-enhancement-fusion/pretrained_models"
original_root_dir = "/project/pi_mfiterau_umass_edu/dolby/speech-enhancement-fusion"


def separation(model, data_loader, result_path):
    count = 0
    for idx, batch_data in tqdm(enumerate(data_loader)):
        mixture = batch_data["mixture"].float().to(device)
        visual_feature = batch_data["visual_feature"].float().to(device)
        clean_audio = batch_data["clean_audio"].to(device)
        # mixture: (batch_size, 48000)
        # visual: (batch_size, 75, 512)

        predict_audio = model(mixture.unsqueeze(1), visual_feature.unsqueeze(1)).squeeze(1)
        clean_paths = batch_data["clean_audio_path"]
        mix_paths = batch_data["mixture_path"]

        for i, m_path in enumerate(mix_paths):
            if not os.path.exists(os.path.join(result_path, f"sample{count}")):
                os.makedirs(os.path.join(result_path, f"sample{count}"))
            predict_filename = os.path.join(result_path, f"sample{count}", "separation.wav")
            clean_filename = os.path.join(result_path, f"sample{count}", clean_paths[i].split("/")[-1])
            mix_filename = os.path.join(result_path, f"sample{count}", mix_paths[i].split("/")[-1])

            write(predict_filename, 16000, predict_audio[i].detach().cpu().numpy())
            write(clean_filename, 16000, clean_audio[i].detach().cpu().numpy())
            write(mix_filename, 16000, mixture[i].detach().cpu().numpy())

            count += 1


def run_audio_separation(exp_name, cfg_data):
    result_path = os.path.join("/project/pi_mfiterau_umass_edu/sidong/speech/results", exp_name, "separation")
    model = avlit.AVLIT(**models.model_config.avlit_default).to(device)
    d = torch.load(os.path.join("/project/pi_mfiterau_umass_edu/sidong/speech/results", exp_name, "model.pth.tar"))

    for key in d:
        if key not in ["epoch", "optimizer", "eval"]:
            model.load_state_dict(d[key])

    mixture_index_dir = os.path.join(original_root_dir, cfg_data["mixture_index_dir"])
    clean_audio_dir = os.path.join(original_root_dir, cfg_data["clean_audio_dir"])
    visual_feature_dir = os.path.join(original_root_dir, cfg_data["visual_feature_dir"])

    test_file = cfg_data["test_file"]

    test_set = lrs3_set.LRS3Dataset(os.path.join(mixture_index_dir, test_file),
                                         clean_audio_dir=clean_audio_dir,
                                         visual_feature_dir=visual_feature_dir,
                                         mode="test",
                                         cache_file_path=os.path.join(mixture_index_dir, "temp",
                                                                      lrs3_set.CACHE_FILE_DICTIONARY[test_file][
                                                                          "visual"]),
                                         force_reprocess=False,
                                         include_visual=True,
                                         include_reference_audio=False)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=4,
                                              # sampler=self.test_sampler,
                                              pin_memory=True)
    separation(model, test_loader, result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--exp_version', required=False)
    args = parser.parse_args()

    with open(os.path.join('./config/lrs3', args.config), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = cfg["exp_name"]
    if args.exp_version:
        exp_name += "_{}".format(args.exp_version)
        cfg["exp_name"] = exp_name

    run_audio_separation(exp_name, cfg["data"])

