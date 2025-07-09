import os
import subprocess
from tqdm import tqdm
import zipfile
import numpy as np
from multiprocessing import Pool, cpu_count
import tarfile


root_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT"
noise_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT_NOISE"


def unzip_all(dir_list):
    for dirname in tqdm(dir_list):
        spk_path = os.path.join(root_path, dirname)
        cam30d_path = os.path.join(spk_path, "30degcam.zip")

        camstraight_path = os.path.join(spk_path, "straightcam.zip")
        
        # with zipfile.ZipFile(cam30d_path, 'r') as zip_ref:
        #     zip_ref.extractall(spk_path)

        with zipfile.ZipFile(camstraight_path, 'r') as zip_ref:
            zip_ref.extractall(spk_path)


def unzip_noise():
    categories = ["Babble", "Cafe", "Car", "LR", "Street", "White"]
    db_level = ["-5", "0", "5", "10", "15", "20"]
    for cat in tqdm(categories):
        for db in db_level:
            filename = os.path.join(noise_path, f"{cat}_{db}.tar.gz")
            tar = tarfile.open(filename, "r:gz")
            tar.extractall()


def distributed():
    number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    # dir_list = os.listdir(root_path)
    # print(dir_list)
    dir_list = ["04M", "05F", "56M", "55F"]
    dir_list_split = np.array_split(
        dir_list,
        number_of_cores)

    print(number_of_cores)

    args = []
    for i in range(number_of_cores):
        args.append((dir_list_split[i],))

    # multiprocssing pool to distribute tasks to:
    with Pool(number_of_cores) as pool:
        # distribute computations and collect results:
        results = pool.starmap(unzip_all, args)

    # print(len(video_frame_batch_list))
    # set_start_method("spawn", force=True)
    # # torch.set_num_threads(1)
    # pool = Pool(10)
    # pool.map(partial(process_batch, input_dir=lrs3_path, output_dir=mouth_path), video_frame_batch_list)

    # complete the processe
    pool.close()
    pool.join()


if __name__ == "__main__":
    # distributed()
    unzip_noise()