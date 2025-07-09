import sys
sys.path.append(".")

import yaml
import os
import torch
import numpy as np
import random
import argparse
import json
import engines.double_separation as sep_engine
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

ROOT_PATH = "/work/pi_mfiterau_umass_edu/sidong/speech"

# python -m torch.distributed.launch --nproc_per_node 2 --master_port=44146 scripts/run_lrs3_avlit_distributed.py --config avlit.yaml --exp_version reg5e-2_distributed --recover_weight 0.05 --batch_size 6


if __name__ == "__main__":
    """
    main.py is only the entrance to the pipeline
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--exp_version', required=False)
    parser.add_argument("--recover_weight", type=float, required=False)
    parser.add_argument("--num_blocks", type=int, required=False)
    parser.add_argument("--fusion_dim", type=int, required=False)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--lr", type=float, required=False)
    parser.add_argument("--main_port", type=int, default=-1,
                        help="Main port (for multi-node SLURM jobs)")
    # parser.add_argument("--local-rank", type=int, help="Local rank. "
    #                                                    "Necessary for using the torch.distributed.launch utility.")
    # parser.add_argument("--is_distributed", type=bool)
    args = parser.parse_args()

    distributed_util.init_distributed_mode_torchrun(args)
    distributed_util.init_signal_handler()
    torch.distributed.barrier()

    with open(os.path.join('./config/lrs3', args.config), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = cfg["exp_name"]

    if args.exp_version:
        exp_name += "_{}".format(args.exp_version)
        cfg["exp_name"] = exp_name

    if args.recover_weight or args.recover_weight == 0:
        cfg["train"]["recover_weight"] = args.recover_weight

    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size

    if args.num_blocks:
        cfg["train"]["num_blocks"] = args.num_blocks

    if args.fusion_dim:
        cfg["train"]["fusion_dim"] = args.fusion_dim

    if args.lr:
        cfg["train"]["lr"] = args.lr

    # create the directory for the output file
    if args.local_rank == 0:
        if not os.path.exists(os.path.join(ROOT_PATH, "results", exp_name)):
            os.makedirs(os.path.join(ROOT_PATH, "results", exp_name))

    # path to save output files, like losses, scores, figures etc
    report_path = os.path.join(ROOT_PATH, "results", exp_name)

    p = sep_engine.LRS3PrombtEval(cfg, report_path, args)
    test_result_path = p.test()

    if args.local_rank == 0:
        test_result = json.load(open(test_result_path, "r"))
        print("test loss: ", test_result["loss"])
        print("test performance:", test_result["performance"])

    # p = sep_engine.LRS3ProfusionEval(cfg, report_path)
    # print(p.test())
