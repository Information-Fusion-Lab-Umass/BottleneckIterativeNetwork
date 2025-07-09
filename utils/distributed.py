import logging
from logging import getLogger
import os
import sys
import torch
import socket
import signal
import subprocess
import datetime
import logging


class CustomFormatter(logging.Formatter):

    green = "\x1b[1;32m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(is_main=True, is_distributed=False, filename=None):
    # logger = getLogger()
    logger = logging.getLogger("speech")

    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]

    if filename is not None:
        handlers.append(logging.FileHandler(filename = filename))

    # add colors
    for ch in handlers:
        ch.setFormatter(CustomFormatter())
        # logger.addHandler(ch)

    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S %Z",
        level=logging.DEBUG if is_main else logging.WARN,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)",
        handlers=handlers,
    )
    return logger


def sig_handler(signum, frame):
    logging.warning("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ['SLURM_PROCID'])
    logging.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        logging.warning("Requeuing job " + os.environ['SLURM_JOB_ID'])
        os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    else:
        logging.warning("Not the main process, no need to requeue.")
    sys.exit(-1)


def term_handler(signum, frame):
    logging.warning("Signal handler called with signal " + str(signum))
    logging.warning("Bypassing SIGTERM.")


def init_signal_handler():
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)


def init_distributed_mode(params):

    has_local_rank = hasattr(params, 'local_rank')

    if has_local_rank and params.local_rank != -1:

        assert params.main_port == -1

        # read environment variables
        params.global_rank = int(os.environ['RANK'])
        params.world_size = int(os.environ['WORLD_SIZE'])
        params.n_gpu_per_node = int(os.environ['NGPU'])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
        params.is_distributed = True
    else:
        n_gpu = torch.cuda.device_count()
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = n_gpu
        params.n_gpu_per_node = n_gpu
        params.is_distributed = False

    # define whether this is the master process / if we are in distributed mode
    params.is_main = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1

    # set GPU device
    if params.is_distributed:
        torch.cuda.set_device(params.local_rank)
        device = torch.device("cuda", params.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.device = device

    # initialize multi-GPU
    if params.is_distributed:
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
            timeout = datetime.timedelta(seconds=36000)
        )


def init_distributed_mode_torchrun(params):

    params.local_rank = int(os.environ["LOCAL_RANK"])
    assert params.main_port == -1

    # read environment variables
    params.global_rank = int(os.environ['RANK'])
    params.world_size = int(os.environ['WORLD_SIZE'])
    params.n_gpu_per_node = int(os.environ['NGPU'])

    # number of nodes / node ID
    params.n_nodes = params.world_size // params.n_gpu_per_node
    params.node_id = params.global_rank // params.n_gpu_per_node
    params.is_distributed = True

    # define whether this is the master process / if we are in distributed mode
    params.is_main = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1

    # set GPU device
    if params.is_distributed:
        torch.cuda.set_device(params.local_rank)
        device = torch.device("cuda", params.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.device = device

    # initialize multi-GPU
    if params.is_distributed:
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
            timeout = datetime.timedelta(seconds=36000)
        )