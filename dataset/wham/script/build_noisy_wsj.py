import os
import numpy as np
import pandas as pd
import soundfile as sf
import argparse


def main(wsj_root_16k, wsj_root_8k, noise_root, output_root):
    DATALENS = ['max', 'min']
    SAMPLERATES = ['wav16k', 'wav8k']
    SPLITS = ['tr', 'cv', 'tt']

    NOISE_CSV = 'dataset_meta.csv'
    SINGLE_DIR = 'mix_single'
    BOTH_DIR = 'mix_both'
    CLEAN_DIR = 'mix_clean'
    S1_DIR = 's1'
    S2_DIR = 's2'
    NOISE_DIR = 'noise'

    for sr_dir, wsj_root in zip(SAMPLERATES, [wsj_root_16k, wsj_root_8k]):
        for datalen_dir in DATALENS:
            for splt in SPLITS:
                wsj_path = os.path.join(wsj_root, datalen_dir, splt)
                noise_path = os.path.join(noise_root, sr_dir, splt, NOISE_DIR)
                output_path = os.path.join(output_root, sr_dir, datalen_dir, splt)

                os.makedirs(os.path.join(output_path, CLEAN_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, SINGLE_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, BOTH_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, S1_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, S2_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, NOISE_DIR), exist_ok=True)

                print('{} {} dataset, {} split'.format(sr_dir, datalen_dir, splt))

                csv_path = os.path.join(noise_root, sr_dir, splt, NOISE_CSV)
                nz_df = pd.read_csv(csv_path)
                for i_utt, (utt_id, start_samp, end_samp, targ_snr_db, speech_gain_db, clipping_gain_db) \
                        in enumerate(nz_df.itertuples(False, None)):

                    # noise is already scaled, but we will apply gain to the speech
                    speech_gain_db = speech_gain_db + clipping_gain_db
                    speech_gain_lin = 10 ** (speech_gain_db / 20)
                    fname = utt_id + '.wav'

                    # read speech
                    s1_path = os.path.join(wsj_path, S1_DIR, fname)
                    s2_path = os.path.join(wsj_path, S2_DIR, fname)
                    s1_samples, sr = sf.read(s1_path)
                    s1_samples *= speech_gain_lin
                    s2_samples, sr = sf.read(s2_path)
                    s2_samples *= speech_gain_lin

                    # read noise
                    nz_path = os.path.join(noise_path, fname)
                    noise_samples, sr = sf.read(nz_path)

                    # truncate to shortest utterance (min) or append silence to speech (max)
                    speech_end_sample = start_samp + len(s1_samples)
                    if datalen_dir == 'max':
                        s1_len = np.zeros_like(noise_samples)
                        s2_len = np.zeros_like(noise_samples)
                        s1_len[start_samp:speech_end_sample] = s1_samples
                        s2_len[start_samp:speech_end_sample] = s2_samples
                    elif datalen_dir == 'min':
                        s1_len = s1_samples
                        s2_len = s2_samples
                        noise_samples = noise_samples[start_samp:speech_end_sample]

                    # mix audio
                    mix_clean = s1_len + s2_len
                    mix_single = noise_samples + s1_len
                    mix_both = noise_samples + s1_len + s2_len

                    # write audio
                    samps = [mix_clean, mix_single, mix_both, s1_len, s2_len, noise_samples]
                    dirs = [CLEAN_DIR, SINGLE_DIR, BOTH_DIR, S1_DIR, S2_DIR, NOISE_DIR]
                    for dir, samp in zip(dirs, samps):
                        sf.write(os.path.join(output_path, dir, fname), samp,
                                 sr, subtype='FLOAT')

                    if (i_utt + 1) % 500 == 0:
                        print('Completed {} of {} utterances'.format(i_utt + 1, len(nz_df)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsjmix-dir-16k', type=str,
                        help='Folder containing original wsj0-2mix 2speakers 16 kHz dataset. Input argument from \
                             create_wav_2speakers.m Matlab script')
    parser.add_argument('--wsjmix-dir-8k', type=str,
                        help='Folder containing original wsj0-2mix 2speakers 8 kHz dataset. Input argument from \
                                                     create_wav_2speakers.m Matlab script')
    parser.add_argument('--noise-dir', type=str, help='Folder containing noise files.  The root directory containing \
                                                       both wav8k and wav16k')
    parser.add_argument('--output-dir', type=str, help='Directory to write the new dataset')
    args = parser.parse_args()
    main(args.wsjmix_dir_16k, args.wsjmix_dir_8k, args.noise_dir, args.output_dir)