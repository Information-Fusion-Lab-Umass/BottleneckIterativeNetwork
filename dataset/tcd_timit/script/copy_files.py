import os
import subprocess
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil


# root_path = "/project/pi_mfiterau_umass_edu/LRS3/TCD_TIMIT"
target_path = "/project/pi_mfiterau_umass_edu/TCD_TIMIT"
root_path = "/work/pi_mfiterau_umass_edu/TCD_TIMIT"


def copy_files(file_list):
    for dirname in tqdm(file_list):
        os.makedirs(os.path.join(target_path, dirname), exist_ok=True)

        spk_path = os.path.join(root_path, dirname)
        # tg_path = os.path.join(target_path, dirname)
        for filename in os.listdir(spk_path):
            print("copy: ", os.path.join(spk_path, filename))
            shutil.copy2(os.path.join(spk_path, filename),
                            os.path.join(target_path, dirname))
        #
        # command = f"cp -Rv {spk_path} -d {target_path}"
        # p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # p.wait()


def distributed():
    number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    dir_list = os.listdir(root_path)
    print(dir_list)

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
        results = pool.starmap(copy_files, args)

    # print(len(video_frame_batch_list))
    # set_start_method("spawn", force=True)
    # # torch.set_num_threads(1)
    # pool = Pool(10)
    # pool.map(partial(process_batch, input_dir=lrs3_path, output_dir=mouth_path), video_frame_batch_list)

    # complete the processe
    pool.close()
    pool.join()


def linear():
    dir_list = os.listdir(root_path)
    copy_files(dir_list)


if __name__ == "__main__":
    linear()