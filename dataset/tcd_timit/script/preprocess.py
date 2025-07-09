import sys
sys.path.append(".")

import numpy as np
import dlib
import cv2
import skvideo.io
import ffmpeg

from multiprocessing import Process
import time
from collections import OrderedDict
import os
from data_utils import create_ground_truth_labels_from_path

################################################################
# The abbreviation nmc in this file means nose/mouth/chin region of interest
# dda means digital differential analyzer
base_dir = "/project/pi_mfiterau_umass_edu/TCD_TIMIT"
processed_dir = "/project/pi_mfiterau_umass_edu/sidong/speech/dataset/tcd_timit/script/dda_gray_nmcroi_dataout"
SHOW_OPTICAL_FLOW = False

# Parameters:
epsilon = 1e-8  # for numerical stability
sampling_rate = 16000
visual_frame_rate_in = 29.970030
visual_frame_rate_out = 62.5
crf = 17

################################################################

# Indices for dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def landmark_to_np(landmarks_in):
    # print("num_parts: ", landmarks_in.num_parts)  # this prints 68, i.e. total nr of landmarks
    landmark_arr = np.zeros(shape=(landmarks_in.num_parts, 2), dtype=int)
    for i in range(landmarks_in.num_parts):
        landmark_arr[i] = (landmarks_in.part(i).x, landmarks_in.part(i).y)

    return landmark_arr


def calculate_optical_flow(video_file):

    of_cap = cv2.VideoCapture(video_file)  # you can set nondda_mp4_file here and you will see the mouth moving more clearly
    dda_nmc_file = skvideo.io.vread(video_file)
    optical_flow_array = np.zeros((dda_nmc_file.shape[0], 67, 67))

    ret, of_frame = of_cap.read()
    prev_frame = cv2.cvtColor(of_frame, cv2.COLOR_BGR2GRAY)  # Note that this gray still has 3 channels
    hsv = np.zeros_like(of_frame)
    hsv[..., 1] = 255  # this is the saturation component. 255 means maximum saturation.

    ret = True  # ret means frame return flag
    opt_flow_count = 0
    while ret:  # while frame was successfully returned
        ret, of_frame2 = of_cap.read()
        try:
            next_frame = cv2.cvtColor(of_frame2, cv2.COLOR_BGR2GRAY)
        except:
            continue
        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 5, 0)
        # args above are: prev, next, flow, pyr_scale, levels, winsize, iterations, size of pixel neighborhood, sigma of smoothing Gaussian, flags

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        optical_flow_array[opt_flow_count] = mag
        opt_flow_count += 1

        hsv[..., 0] = ang * 180 / np.pi / 2  # hue is represented by an angle on the color wheel
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # value component
        # args above are: input src array, output array of the same size as src, lower bound, upper bound, normalization type

        if SHOW_OPTICAL_FLOW:
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            grayactivationmap = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # cv2.imshow(str(video_file), mag)  # originally I used bgr here instead of mag.

            k = cv2.waitKey(30) & 0xff
            if k == 27:  # if escape is pressed
                break
            elif k == ord('s'):  # if letter s is pressed
                cv2.imwrite('grayroi.png', of_frame2)
                cv2.imwrite('opticalhsv.png', grayactivationmap)

        prev_frame = next_frame

    of_cap.release()
    # cv2.destroyAllWindows()
    return optical_flow_array  # this stores the magnitude of the optical flow in x and y direction using hypotenuse formula


def my_processing_per_cpu_process(speaker_sublist):
    print(f"Starting process with sublist {speaker_sublist}")

    for speaker in speaker_sublist:
        print(f"Processing speaker {speaker}")
        speaker_path = os.path.join(base_dir, speaker, "straightcam")

        speaker_mp4_files = sorted([file for file in os.listdir(speaker_path) if ".mp4" in file])
        speaker_audio_files = sorted([file for file in os.listdir(speaker_path) if ".wav" in file])

        for video_mp4_file, audio_wav_file in zip(speaker_mp4_files, speaker_audio_files):
            video_mp4_pa = os.path.join(speaker_path, video_mp4_file)
            audio_wav_pa = os.path.join(speaker_path, audio_wav_file)

            original_video_array = cv2.VideoCapture(video_mp4_pa)

            ############################################################################################################
            # First find the mean center of mouth using first couple of frames:
            mouth_mean_i_video_list = []
            mouth_mean_j_video_list = []

            success, frame = original_video_array.read()
            count_frame = 0

            while success:
                if count_frame % 4 == 0:
                    count_frame += 1
                    continue

                if count_frame > 150:
                    break

                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)  # For down-sampling INTER_AREA is the best

                detected_faces = detector(frame, 1)
                # 1 is the number of image pyramid layers to apply when upscaling the image

                # loop over all detected faces, which for TIMIT should be 1
                for (count_face, detected_face) in enumerate(detected_faces):
                    landmarks = predictor(frame, detected_face)
                    landmarks_arr = landmark_to_np(landmarks)

                    mean_i_list = []
                    mean_j_list = []
                    for (i, j) in landmarks_arr[start_roi_index:end_roi_index]:
                        mean_i_list.append(i)
                        mean_j_list.append(j)

                mean_i = np.mean(mean_i_list)
                mean_j = np.mean(mean_j_list)
                mouth_mean_i_video_list.append(mean_i)
                mouth_mean_j_video_list.append(mean_j)
                success, frame = original_video_array.read()
                count_frame += 1

            overall_mean_i = int(np.mean(mouth_mean_i_video_list))
            overall_mean_j = int(np.mean(mouth_mean_j_video_list))

            # for other datasets, the mean calculation from above would have to be replaced with a smoothing Kalman
            # filter for the purpose of optical stabilization.
            ############################################################################################################
            # Read original file again, this time knowing roughly the position of the mouth center:

            original_video_array1 = cv2.VideoCapture(video_mp4_pa)
            original_video_length = int(original_video_array1.get(cv2.CAP_PROP_FRAME_COUNT))

            success1, frame1 = original_video_array1.read()
            count_frame1 = 0

            nmc_rois = np.zeros((original_video_length, 67, 67), dtype=np.uint8)
            while success1:
                frame1 = cv2.resize(frame1, (224, 224), interpolation=cv2.INTER_AREA)
                # INTER_AREA is best for downsampling
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

                # Uncomment line below if you want nose + mouth + chin
                nosemouthchin_crop = frame1[overall_mean_j - 20:overall_mean_j + 40, overall_mean_i - 20:overall_mean_i + 20]

                # Keep line below if you want just the mouth (although the variable is called nosemouthchin_crop)
                #nosemouthchin_crop = frame1[overall_mean_j - 10:overall_mean_j + 10, overall_mean_i - 15:overall_mean_i + 15]

                nmc_rois[count_frame1] = cv2.resize(nosemouthchin_crop, (67, 67), interpolation=cv2.INTER_CUBIC)
                # INTER_CUBIC is best for upsampling

                success1, frame1 = original_video_array1.read()
                count_frame1 += 1

            ############################################################################################################
            # Create the dda video with gray mouth rois:

            os.makedirs(f'nmcroi_preprocessing_steps/{speaker}/', exist_ok=True)  # speaker is set by main, not in this function
            dda_gray_nmc_roi_video = f'nmcroi_preprocessing_steps/{speaker}/{video_mp4_pa.split("/")[-1].replace(".mp4", "_dda_gray_nmc_roi.mp4")}'
            video_writer0 = skvideo.io.FFmpegWriter(dda_gray_nmc_roi_video,
                                                    inputdict={'-r': str(visual_frame_rate_in),
                                                               '-s': '{}x{}'.format(67, 67)},
                                                    outputdict={'-filter:v': 'fps=fps={}'.format(
                                                        visual_frame_rate_out),
                                                        '-c:v': 'libx264',
                                                        '-crf': str(crf),
                                                        '-preset': 'veryslow'}
                                                    )

            for nmc_frame in nmc_rois:
                video_writer0.writeFrame(nmc_frame)

            video_writer0.close()

            ############################################################################################################
            # Calculate the optical flow for the dda mouth video and store it in npy file:
            dda_optical_flow = calculate_optical_flow(dda_gray_nmc_roi_video)
            os.makedirs(f'dda_optical_flow_nmcroi_dataout/{speaker}/', exist_ok=True)
            dda_optical_flow_out_path = f'dda_optical_flow_nmcroi_dataout/{speaker}/{dda_gray_nmc_roi_video.split("/")[-1].replace("_dda_gray_nmc_roi.mp4", ".npy")}'
            np.save(dda_optical_flow_out_path, dda_optical_flow)  # this is saved as float64

            ############################################################################################################
            # Store the gray mouth frames from the mp4 file to npy files:
            dda_nmc_file_reread = skvideo.io.vread(dda_gray_nmc_roi_video)
            os.makedirs(f'dda_gray_nmcroi_dataout/{speaker}/', exist_ok=True)
            dda_gray_nmc_roi_path = f'dda_gray_nmcroi_dataout/{speaker}/{dda_gray_nmc_roi_video.split("/")[-1].replace("_dda_gray_nmc_roi.mp4", ".npy")}'
            np.save(dda_gray_nmc_roi_path, dda_nmc_file_reread[:, :, :, 0])  # this is saved as uint8

            #############################################   VERIFY   ###################################################

            # Next code is for adding white squares on video frames and audio channel to mp4 to verify qualitatively if the person is speaking or not.
            video_writer_white = skvideo.io.FFmpegWriter(dda_gray_nmc_roi_video.replace(".mp4", "_white.mp4"),
                                                                    inputdict={'-r': str(visual_frame_rate_out),
                                                                               '-s': '{}x{}'.format(67, 67, 3)},
                                                                    outputdict={'-filter:v': 'fps=fps={}'.format(
                                                                        visual_frame_rate_out),
                                                                        '-c:v': 'libx264',
                                                                        '-crf': str(crf),
                                                                        '-preset': 'veryslow'}
                                                                    )
            gt_labels = create_ground_truth_labels_from_path(audio_wav_pa)
            white_frame = 255 * np.ones((67, 67, 3))

            white_dda_array = skvideo.io.vread(dda_gray_nmc_roi_video)
            sync_len = min(gt_labels.shape[0], white_dda_array.shape[0])

            for white_frame_count, up_frame2 in enumerate(white_dda_array[:sync_len]):
                if gt_labels[white_frame_count, 0] == 1:
                    video_writer_white.writeFrame(up_frame2)
                elif gt_labels[white_frame_count, 0] == 0:
                    video_writer_white.writeFrame(white_frame)
                else:
                    print("VAD Label is not 0 or 1 !")  # this shouldn't happen

            video_writer_white.close()

            ###########################################################################################################
            # Now save the white frame dda video with the audio track added as well
            video = ffmpeg.input(dda_gray_nmc_roi_video.replace(".mp4", "_white.mp4"))
            audio = ffmpeg.input(audio_wav_pa)
            out = ffmpeg.output(video, audio, dda_gray_nmc_roi_video.replace(".mp4", "_white_withaudio.mp4"), vcodec='copy',
                                acodec='aac', strict='experimental')
            out = out.overwrite_output()
            out.run()


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    start_roi_index = FACIAL_LANDMARKS_68_IDS["mouth"][0]
    end_roi_index = FACIAL_LANDMARKS_68_IDS["mouth"][1]

    def distributed():
        number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
        all_speaker_folders = sorted(os.listdir(base_dir))
        existing_speaker_folders = os.listdir(processed_dir)

        unprocessed_dirs = []
        for s in all_speaker_folders:
            if s not in existing_speaker_folders:
                unprocessed_dirs.append(s)

        print("All speakers: ", unprocessed_dirs)

        pool_list = np.array_split(
            unprocessed_dirs,
            number_of_cores)
        # pool_list = [all_speaker_folders[5 * i:5 * i + 5] for i in
        #              range(12)]  # each of the 12 CPU cores will process a maximum of 5 speaker folders.

        print("Pool list: ", pool_list)

        start_time = time.perf_counter()

        processes = []
        for split_list in pool_list:
            p = Process(target=my_processing_per_cpu_process,
                        args=(split_list,))
            p.start()
            processes.append(p)
            # p.join()  # if this join is uncommented then the multicore preprocessing is transformed to just single core sequential processing.

        for proc in processes:
            proc.join()

        end_time = time.perf_counter()

        print(f"Elapsed time for the ROI creation: {(end_time - start_time) / 3600} hours")

    def single():
        my_processing_per_cpu_process(["15F"])

    single()


# Comment to put cursor in for waitkey(0), not to shift code by mistake while debugging