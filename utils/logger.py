from logging import getLogger
import os
import sys
import torch
import socket
import signal
import subprocess
import datetime
import logging

logger = getLogger()


logger = logging.getLogger(__name__)


def init_logger(is_main=True, is_distributed=False, filename=None):
    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename = filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S %Z",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logger
