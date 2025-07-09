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

import models.device_check as dc
import models.avlit as avlit
import models.progressive_fusion as pro_fusion
import models.progressive_mbt as pro_mbt
import models.progressive_separation as pro_sep
import engines.abs_separation as abs_sep
import dataset.lrs3.dataset as lrs3_set
import dataset.lrs3_wham.dataset as lrs3_wham_set
import models.model_config
import plot.curve as curve
import utils.dynamic as dynamic
import models.loss as loss
import utils.audio as audio_util
import utils.distributed as distributed_util
import dataset.tcd_timit.dataset as ntcd_set

from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio


def lrs3wham_prombt():
    def evaluate(data_loader, result_path):
        test_details_output_file = os.path.join(result_path, "eval_details.csv")
        if os.path.exists(test_details_output_file):
            os.remove(test_details_output_file)

        test_details = {
            "predict_recover_loss": [],
            "predict_loss": [],
            "predict_sisdr": [],
            "predict_sisdri": [],
            "predict_pesq": [],
            "predict_estoi": [],
            "mixture_path": [],
            "spk1_path": [],
            "spk2_path": [],
            "spk1_id": [],
            "spk1_vid": [],
            "spk2_id": [],
            "spk2_vid": []
        }

        with torch.no_grad():
            for batch_i, batch_data in enumerate(data_loader):
                self.logger.info(f"Local Rank {self.args.local_rank}: Evaluation: Batch {batch_i} / {len(data_loader)}")
                # ve = batch_data["visual_feature"].float()
                ve = batch_data["video_npy"].float()
                clean_audio = batch_data["clean_audio"]
                for i in range(len(ve)):
                    test_details["mixture_path"].append(
                        f'{batch_data["spk1_id"][i]}_{batch_data["spk1_vid"][i]}_{batch_data["spk2_id"][i]}_{batch_data["spk2_vid"][i]}')

                    overlap_ms = batch_data["overlap_ms"][i]
                    if overlap_ms >= 0:
                        main_speaker_start = 0
                    else:
                        main_speaker_start = int(overlap_ms / 1000 * 25) + 75
                    dropped_frames_i = [d + main_speaker_start for d in dropped_frames]
                    for f in dropped_frames_i:
                        ve[i, f, :] = 0.0

                # predict_audio = self.fusion_model(batch_data["mixture"].float().unsqueeze(1).to(self.device),
                #                                   ve.unsqueeze(1).to(self.device))

                predict_audio = self.fusion_model(batch_data["mix_audio"].float().unsqueeze(1).to(self.device),
                                                  ve.to(self.device))  # (B, num sources, len)
                # del ve
                # torch.cuda.empty_cache()

                clean_audio = clean_audio.to(self.device)

                cur_loss = self.loss_func(predict_audio, clean_audio)

                predict_recover = cur_loss["recover"]
                predict_recover = predict_recover.detach().cpu().numpy()
                test_details["predict_recover_loss"].append(predict_recover)

                predict_loss = cur_loss["main"]
                predict_loss = predict_loss.detach().cpu().numpy()
                test_details["predict_loss"].append(predict_loss)

                # predict_sisdr = eval_sisdr(predict_audio, clean_audio).detach().cpu().numpy()

                # Compute PESQ and ESTOI and SISDR

                true_wav = batch_data["clean_audio"].to(self.device)  # (B, num sources, len)

                if "overlap_ms" in batch_data.keys():
                    test_details["overlap_ms"].append(batch_data["overlap_ms"].detach().cpu().numpy())

                sdr = audio_util.cal_multisource_sdr(true_wav.to(self.device), predict_audio)
                test_details["predict_sdr"].append(sdr.detach().cpu().numpy())

                sdr_mix = audio_util.cal_SDR(true_wav.to(self.device).mean(dim=1),
                                             batch_data["mix_audio"].to(self.device))
                test_details["mix_sdr"].append(sdr_mix.detach().cpu().numpy())
                sdri = sdr - sdr_mix

                test_details["sdri"].append(sdri.detach().cpu().numpy())

                sdr_metric += torch.mean(sdr).detach().cpu().numpy()
                sdri_metric += torch.mean(sdri).detach().cpu().numpy()

                num_source = predict_audio.shape[1]

                pesq = 0  # (batch_size, )
                estoi = 0  # (batch_size, )
                sisdr = 0  # (batch_size, )

                mix_sisdr = 0

                for s in range(num_source):
                    pesq += perceptual_evaluation_speech_quality(predict_audio[:, s, :],
                                                                 true_wav[:, s, :].to(self.device),
                                                                 fs=sampling_rate,
                                                                 mode="wb") / num_source
                    estoi += short_time_objective_intelligibility(predict_audio[:, s, :],
                                                                  true_wav[:, s, :].to(self.device),
                                                                  extended=True,
                                                                  fs=sampling_rate) / num_source
                    sisdr += scale_invariant_signal_distortion_ratio(predict_audio[:, s, :],
                                                                     true_wav[:, s, :],
                                                                     zero_mean=True) / num_source
                    mix_sisdr += scale_invariant_signal_distortion_ratio(
                        batch_data["mix_audio"].float().to(self.device),
                        true_wav[:, s, :].to(self.device),
                        zero_mean=True) / num_source

                pesq = pesq.detach().cpu().numpy()
                test_details["predict_pesq"].append(pesq)

                estoi = estoi.detach().cpu().numpy()
                test_details["predict_estoi"].append(estoi)

                sisdr = sisdr.detach().cpu().numpy()
                test_details["predict_sisdr"].append(sisdr)

                mix_sisdr = mix_sisdr.detach().cpu().numpy()
                test_details["predict_sisdri"].append(sisdr - mix_sisdr)

                pesq_metric += np.mean(pesq)
                stoi_metric += np.mean(estoi)
                sisdri_metric += np.mean(sisdr - mix_sisdr)

            test_sisdri = sisdri_metric / len(data_loader)
            test_pesq = pesq_metric / len(data_loader)
            test_stoi = stoi_metric / len(data_loader)
            test_sdr = sdr_metric / len(data_loader)
            test_sdri = sdri_metric / len(data_loader)

            for c in test_details.keys():
                if "path" not in c:
                    if "loss" not in c:
                        test_details[c] = np.concatenate(test_details[c], axis=0)
                    else:
                        test_details[c] = np.stack(test_details[c], axis=0)

            with open(test_details_output_file, "w") as outfile:
                writer = csv.writer(outfile)
                writer.writerow(test_details.keys())
                writer.writerows(zip(*test_details.values()))

            # LOGGER.info(f"Test loss: {test_loss}")
            # LOGGER.info(f"Test PESQ: {test_pesq}")
            # LOGGER.info(f"Test STOI: {test_stoi}")
            # LOGGER.info(f"Test SDR: {test_sdr}")
            # LOGGER.info(f"Test SDRi: {test_sdri}")

            return {"main": test_details["predict_loss"].mean(),
                    "sisdr": test_details["predict_sisdr"].mean(),
                    "recover": test_details["predict_recover_loss"].mean()}, \
                {"PESQ": test_details["predict_pesq"].mean(),
                 "ESTOI": test_details["predict_estoi"].mean(),
                 "SDR": test_details["predict_sdr"].mean(),
                 "SI-SDRi": test_details["predict_sisdri"].mean()}