import sys
sys.path.append(".")

import yaml
import os
import torch
import numpy as np
import random
import argparse
import json
import engines.ntcd_recon as ntcd_recon
import utils.distributed as distributed_util

# lock all random seed to make the experiment replicable
seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

ROOT_PATH = "/project/pi_mfiterau_umass_edu/sidong/speech"

# python -m torch.distributed.launch --nproc_per_node 2 --master_port=44146 scripts/run_lrs3_avlit_distributed.py --config avlit.yaml --exp_version reg5e-2_distributed --recover_weight 0.05 --batch_size 6


if __name__ == "__main__":
    """
    main.py is only the entrance to the pipeline
    """
    ntcd_recon.generate()

