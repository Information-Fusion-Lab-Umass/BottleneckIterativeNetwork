import time

import cv2
import dlib
import librosa
import numpy as np
import torch
import torch.nn as nn

from typing import Dict

import logging
import random

from mir_eval.separation import bss_eval_sources
from pesq import pesq
from pystoi import stoi

EPS = 1e-8
MAX_INT16 = np.iinfo(np.int16).max

# this is Keunwoo Choi's implementation of istft.
# https://gist.github.com/keunwoochoi/2f349e72cc941f6f10d4adf9b0d3f37e#file-istft-torch-py
def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
          center=True, normalized=False, onesided=True, length=None):
    """stft_matrix = (batch, freq, time, complex)

    All based on librosa
        - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
    What's missing?
        - normalize by sum of squared window --> do we need it here?
        Actually the result is ok by simply dividing y by 2.
    """
    assert normalized == False
    assert onesided == True
    assert window == "hann"
    assert center == True

    device = stft_matrix.device
    n_fft = 2 * (stft_matrix.shape[-2] - 1)

    batch = stft_matrix.shape[0]

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    istft_window = torch.hann_window(n_fft).to(device).view(1, -1)  # (batch, freq)

    n_frames = stft_matrix.shape[-2]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    y = torch.zeros(batch, expected_signal_len, device=device)
    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, :, i]
        iffted = torch.fft.irfft(spec, dim=1)

        ytmp = istft_window *  iffted
        y[:, sample:(sample+n_fft)] += ytmp

    y = y[:, n_fft//2:]

    if length is not None:
        if y.shape[1] > length:
            y = y[:, :length]
        elif y.shape[1] < length:
            y = torch.cat([y[:, :length], torch.zeros(y.shape[0], length - y.shape[1], device=y.device)], dim=1)

    coeff = n_fft/float(hop_length) / 2.0  # -> this might go wrong if curretnly asserted values (especially, `normalized`) changes.
    return y / coeff

def set_seed(seed):
    #torch.set_default_tensor_type('torch.FloatTensor')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_logging(logger, *, log_file: str = None, log_level=logging.INFO) -> logging:
    """
    Make sure output is logged to both file and console
    :param logger:
    :param log_file:
    :param log_level:
    :return:
    """
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(levelname)s <%(thread)d> [%(asctime)s] %(name)s <%(filename)s:%(lineno)d> %(message)s"
    )
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
    log_handler = logging.StreamHandler()
    log_handler.setLevel(log_level)
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    if log_file is not None:
        logger.addHandler(file_handler)
    return logger

def load_checkpoint(model: nn.Module, state_dict: Dict, is_distributed: bool = False):
    for key in list(state_dict.keys()):
        # due to changes in convtasnet architecture
        if key.startswith("concat."):
            state_dict[key.replace('concat.', 'fusion.')] = state_dict.pop(key)
    if is_distributed:
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


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


def convert_to_wav(augmented_stft: np.ndarray, ref_stft: np.ndarray,
                   window_size,
                   window_shift):

    noisy_phase = np.angle(ref_stft)
    estimated = augmented_stft * (np.cos(noisy_phase) + 1.j * np.sin(noisy_phase))
    if len(estimated.shape) > 2:
        estimated = np.transpose(estimated, (0,2,1))
    else:
        estimated = estimated.T
    wav = librosa.istft(estimated,
                                    win_length=window_size, hop_length=window_shift,
                                    window="hann", center=True, length=48000)
    return wav


def SDRi(est, egs, mix):
    '''
        calculate SDR
        est: Network generated audio
        egs: Ground Truth
    '''
    length = est.numpy().shape[0]
    sdr, _, _, _ = bss_eval_sources(egs.numpy()[:length], est.numpy()[:length])
    mix_sdr, _, _, _ = bss_eval_sources(egs.numpy()[:length], mix.numpy()[:length])
    return float(sdr), float(sdr-mix_sdr)

def cal_SISNR(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    assert source.size() == estimate_source.size()

    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)

    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)

    return sisnr


def cal_SDR(source, estimate_source, return_mean: bool = False):
    assert source.size() == estimate_source.size()

    # estimate_source += EPS # the estimated source is zero sometimes

    noise = source - estimate_source
    ratio = torch.sum(source ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sdr = 10 * torch.log10(ratio + EPS)

    if return_mean:
        sdr = s

    return sdr


def compute_metric(source, predict, mix):
    source = source.squeeze(1).data.cpu().numpy()
    predict = predict.squeeze(1).data.cpu().numpy()
    mix = mix.data.cpu().numpy()
    B = source.shape[0]
    STOI = []
    STOIn = []
    PESQ = []
    PESQn = []

    for i in range(int(B)):
        source_idx = source[i,:]
        predict_idx = predict[i,:]
        STOI_ = stoi(source_idx, predict_idx, 16000)
        STOI_n = stoi(source[i,:], mix[int(i/2),:], 16000)
        PESQ_ = pesq(source_idx, predict_idx, 16000)
        PESQ_n = pesq(source[i,:], mix[int(i/2),:], 16000)
        STOI.append(STOI_)
        PESQ.append(PESQ_)
        STOIn.append(STOI_n)
        PESQn.append(PESQ_n)
    STOI = np.array(STOI)
    PESQ = np.array(PESQ)
    STOIn = np.array(STOIn)
    PESQn = np.array(PESQn)
    STOIavg = STOI.mean()
    PESQavg = PESQ.mean()
    STOInavg = STOIn.mean()
    PESQnavg = PESQn.mean()
    print('STOI PESQ STOI_n PESQ_n this batch:{} {}'.format(STOI, PESQ, STOIn, PESQn))
    return STOI, PESQ, STOIn, PESQn, STOIavg, PESQavg, STOInavg, PESQnavg


def get_audio_feat(filepath, return_stft: bool = True, add_channel_dim: bool = True,
                   sampling_rate: int = 16000,
                   window_size: int = 400,
                   window_shift: int = 160,
                   stft_size: int = 511):
    signal, _ = process_audio(filepath, target_sr=sampling_rate)
    if return_stft:
        signal = librosa.stft(signal,
                              win_length=window_size,
                              n_fft=stft_size, hop_length=window_shift,
                              center=True).T # (D, L) -> (L, D)
    else:
        return signal, signal # (,D)

    if add_channel_dim:
        signal_aug = np.abs(signal).astype(np.float32)[np.newaxis, ...]
    else:
        signal_aug = np.abs(signal).astype(np.float32)
    return signal_aug, signal

def get_visual_feat(filepath):
    ret = np.load(filepath).squeeze() # (n_frames, D)
    if ret.shape[0] < 75:
        #print(f"Missing frame ({ret.shape[0]}): {filepath}")
        shape = ret.shape
        ret = np.vstack((ret, np.zeros((75 - shape[0], shape[1]))))
    return ret

def extract_embeddings(f_path, lip_embedding_model, device, process_beginning_only, save_video=False,
                       save_picture=False,
                       audio_file_name=None,
                       predictor=None):
    # Reference for multi face tracking: https://www.guidodiepen.nl/2017/02/tracking-multiple-faces/

    if audio_file_name is None:
        audio_file_name = f_path

    visual_embeddings = []
    audio_embeddings = []

    # which face in the video to process
    KEY_TO_PROCESS = 0

    face_detector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(f_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = './output_files/output_video_faces.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_time = time.time()
    processing_interval = 5  # seconds
    next_print_time = processing_interval
    num_frames_processed = 0
    timestamp_to_be_processed = 0

    face_trackers = {}
    current_face_id = 0
    sequence = []

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # Update all trackers
        fidsToDelete = []
        for fid in face_trackers.keys():
            trackingQuality = face_trackers[fid].update(frame)

            # If the tracking quality is good enough, we must delete this tracker
            if trackingQuality < 7:
                fidsToDelete.append(fid)
            else:  # else, display face box
                pos = face_trackers[fid].get_position()
                startX, startY, endX, endY = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())

                if save_video:
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, f"Face {fid}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)

        for fid in fidsToDelete:
            print("Removing fid " + str(fid) + " from list of trackers")
            face_trackers.pop(fid, None)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector(gray_frame)

        matchedID = -1

        for rect in faces:
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

            x_cent = x + 0.5 * w
            y_cent = y + 0.5 * h

            for id in face_trackers.keys():
                tracked_position = face_trackers[id].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                # calculate the centerpoint
                t_x_bar = t_x + 0.5 * t_w
                t_y_bar = t_y + 0.5 * t_h

                # check if the centerpoint of the face is within the
                # rectangleof a tracker region. Also, the centerpoint
                # of the tracker region must be within the region
                # detected as a face. If both of these conditions hold
                # we have a match
                if ((t_x <= x_cent <= (t_x + t_w)) and
                        (t_y <= y_cent <= (t_y + t_h)) and
                        (x <= t_x_bar <= (x + w)) and
                        (y <= t_y_bar <= (y + h))):
                    matchedID = id

            # Initialize new tracker if no match found
            if matchedID < 0:
                tracker = dlib.correlation_tracker()
                tracker.start_track(frame, dlib.rectangle(x - 10, y - 20, x + w + 10, y + h + 20))
                face_trackers[current_face_id] = tracker

                current_face_id += 1

        # Print progress every 5 seconds cpu time
        elapsed_time = time.time() - start_time
        if elapsed_time >= next_print_time:
            print(
                f"Processed {cap.get(cv2.CAP_PROP_POS_FRAMES)} out of {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames.")
            next_print_time += processing_interval

        if save_video:
            out.write(frame)

        if (KEY_TO_PROCESS in face_trackers.keys()) == True:
            tracker = face_trackers[KEY_TO_PROCESS]
            t_pos = tracker.get_position()
            rect = dlib.rectangle(int(t_pos.left()),
                                  int(t_pos.top()),
                                  int(t_pos.right()),
                                  int(t_pos.bottom()))
            shape = predictor(gray_frame, rect)
            face_rect = face_utils.rect_to_bb(rect)
            crop_image = crop_mouth(frame, face_utils.shape_to_np(shape))
            sequence.append(crop_image)
            if save_picture:
                cv2.imwrite(f"face_{KEY_TO_PROCESS}_frame_{num_frames_processed}.jpg", crop_image)

        num_frames_processed = num_frames_processed + 1

        # store embeddings every 3 seconds of video
        if (num_frames_processed % (3 * fps)) == 0 and num_frames_processed != 0:
            sequence = np.array(sequence)  # n_frames, H, W
            lip_embedding_model.eval()
            try:
                embedding = lip_embedding_model(torch.FloatTensor(sequence)[None, None, :, :, :].to(device),
                                                lengths=[sequence.shape[0]])
            except Exception as e:
                print('Error generating lip embeddings @ num_frames = {}'.format(num_frames_processed))
                return visual_embeddings, audio_embeddings

            visual_embeddings.append(embedding)
            sequence = []

            ## manually pass audio path for videos that don't contain audio
            # embedding_aud, _ = extract_audio_feature('./sample/s6_bbae7n_s17_bbbr1a.wav', start_t=timestamp_to_be_processed)

            embedding_aud, _ = get_audio_feat(audio_file_name, start_t=timestamp_to_be_processed)
            timestamp_to_be_processed = int(1000 * num_frames_processed / fps)
            audio_embeddings.append(embedding_aud)

            # only processing first 3 seconds of video for debugging purpose
            if process_beginning_only:
                break

    cap.release()
    cv2.destroyAllWindows()

    return visual_embeddings, audio_embeddings


def crop_mouth(frame, shape, output_size=(80, 80)):
    # Code from crop_mouth() in extract_visual_features.py

    height, width = output_size
    height = height // 2
    width = width // 2

    xmouthpoints = [shape[i][0] for i in range(48, 67)]
    ymouthpoints = [shape[i][1] for i in range(48, 67)]
    maxx = max(xmouthpoints)
    minx = min(xmouthpoints)
    maxy = max(ymouthpoints)
    miny = min(ymouthpoints)

    center_y = (miny + maxy) // 2
    center_x = (minx + maxx) // 2
    threshold = 5

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception('too much bias in height')
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception('too much bias in width')

    if center_y + height > frame.shape[0]:
        center_y = frame.shape[0] - height
    if center_y + height > frame.shape[0] + threshold:
        raise Exception('too much bias in height')
    if center_x + width > frame.shape[1]:
        center_x = frame.shape[1] - width
    if center_x + width > frame.shape[1] + threshold:
        raise Exception('too much bias in width')

    crop_image = frame[center_y - height:center_y + height, center_x - width: center_x + width]
    crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

    return crop_image


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