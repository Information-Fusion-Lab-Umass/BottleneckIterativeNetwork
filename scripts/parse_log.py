import os
import re
import json
from tqdm import tqdm

root_path = "/work/pi_mfiterau_umass_edu/sidong/speech/results"
exp_name = "rtfs12_lstm_lrs3wham_b12_a100"
log_path = os.path.join(root_path, exp_name, "run.log")

train_time_log = {}
val_time_log = {}
train_attempts = {}
val_attempts = {}

key_regs = {
    "epoch": r"epoch [0-9.]+",
    "device": r"Local Rank [0-9.]+",
    "time_length": r"[0-9.]+:[0-9.]+:[0-9.]+",
    "train": r"Model epoch [0-9.]+ training time duration",
    "val": r"Model epoch [0-9.]+ validation time duration"
}

with open(log_path) as f:
    f = f.readlines()

for line in tqdm(f):
    print(line)
    if re.match(key_regs["train"], line):
        epoch = re.findall(key_regs["epoch"], line)[0]
        device = re.findall(key_regs["device"], line)[0]

        # decide how many attempts are made for a specific epoch & device
        if epoch not in train_attempts:
            train_attempts[epoch] = {}

        if device not in train_attempts[epoch]:
            train_attempts[epoch][device] = 1
        else:
            train_attempts[epoch][device] += 1

        cur_tag = f"{epoch}_attempt{train_attempts[epoch][device]}"

        if cur_tag not in train_time_log:
            train_time_log[cur_tag] = {}

        train_time_log[cur_tag][device] = re.findall(key_regs["time_length"], line)[-1]

    elif re.match(key_regs["val"], line):
        epoch = re.findall(key_regs["epoch"], line)[0]

        if epoch not in val_attempts:
            val_attempts[epoch] = 1
        else:
            val_attempts[epoch] += 1

        cur_tag = f"{epoch}_attempt{val_attempts[epoch]}"

        val_time_log[cur_tag] = re.findall(key_regs["time_length"], line)[-1]


result_dict = {
    "train_time": train_time_log,
    "train_attempt": train_attempts,
    "val_time": val_time_log,
    "val_attempt": val_attempts
}


with open(os.path.join(root_path, exp_name, "time_log.json"), 'w') as fp:
    json.dump(result_dict, fp)