import os
import yaml
import torch
from torch.profiler import ProfilerActivity
from torch.profiler import record_function
import pickle

from thop import clever_format
from thop import profile

import sys
sys.path.append(".")

import utils.dynamic as dynamic
import models.loss as loss
import models.progressive_mbt as pro_mbt
import models.iianet.iianet as iianet
import models.avlit as avlit
import models.rtfs.tdavnet as rtfs_net

from models.device_check import *


MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


def compute_mem(device_snapshot):
    hist = []
    mem = 0
    for record in device_snapshot:
        if record["action"] == "alloc":
            mem += record["size"]
        elif record["action"] == "free_completed":
            mem -= record["size"]
        hist.append(mem)
    return max(hist), hist    


def param_count(model, batch_size=1):
    mixture = torch.randn(batch_size, 1, 32000)
    visual = torch.randn(batch_size, 2, 50, 1024)
    print("#" * 10)
    macs, params = profile(model, inputs=(mixture, visual))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)


def memory_usage(model, cfg, batch_size=1):
    cfg_train = cfg["train"]

    loss_func = loss.SISDRPesq(num_chunks=cfg_train.get("num_chunks", 5), recover_weight=cfg_train.get("recover_weight", 0)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg_train["lr"], betas=(0.9, 0.98), weight_decay=cfg_train.get("weight_decay", 0.01))

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("data_load"):
            mixture = torch.randn(batch_size, 1, 32000).to(device)
            visual = torch.randn(batch_size, 2, 50, 1024).to(device)
            clean = torch.randn(batch_size, 2, 32000).to(device)
        with record_function("model_forward"):
            model = model.to(device)
            predict = model(mixture, visual)
        with record_function("model_backprop"):
            loss_dict = loss_func(predict, clean)
            main_loss = loss_dict["main"]
            main_loss.backward()
            optim.step()
    
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=-1))
    prof.export_chrome_trace(f"{type(model)}_{cfg_train.get('num_blocks', '')}_time.json")
    prof.export_memory_timeline(f"{type(model)}_{cfg_train.get('num_blocks', '')}_memory.json")


def memory_usage_visualize_training(model, cfg, batch_size=1):
    cfg_train = cfg["train"]

    loss_func = loss.SISDRPesq(num_chunks=cfg_train.get("num_chunks", 5), recover_weight=cfg_train.get("recover_weight", 0)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg_train["lr"], betas=(0.9, 0.98), weight_decay=cfg_train.get("weight_decay", 0.01))

    mixture = torch.randn(batch_size, 1, 32000).to(device)
    visual = torch.randn(batch_size, 2, 50, 1024).to(device)
    clean = torch.randn(batch_size, 2, 32000).to(device)

    # warm up
    for i in range(5):
        model = model.to(device)
        predict = model(mixture, visual)
        loss_dict = loss_func(predict, clean)
        main_loss = loss_dict["main"]
        main_loss.backward()
        optim.step()

    torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
       )
    
    model = model.to(device)
    predict = model(mixture, visual)
    loss_dict = loss_func(predict, clean)
    main_loss = loss_dict["main"]
    main_loss.backward()
    optim.step()
   
    try:
       torch.cuda.memory._dump_snapshot(f"{type(model)}_{cfg_train.get('num_blocks', '')}_train_memory.pickle")
    except Exception as e:
      print(f"Failed to capture memory snapshot {e}")
    
    # memory_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    # torch.cuda.reset_peak_memory_stats()
    # print("inference peak memory: ", memory_usage)

    mem_track = pickle.load(open(f"{type(model)}_{cfg_train.get('num_blocks', '')}_train_memory.pickle", "rb"))
    max_mem, _ = compute_mem(mem_track["device_traces"][0])
    print("training peak memory: ", max_mem)
        
   # Stop recording memory snapshot history.
    torch.cuda.memory._record_memory_history(enabled=None)


def memory_usage_visualize_inference(model, cfg, batch_size=1):
    cfg_train = cfg["train"]

    mixture = torch.randn(batch_size, 1, 32000).to(device)
    visual = torch.randn(batch_size, 2, 50, 1024).to(device)
    
    with torch.no_grad():
        # warm up
        for i in range(5):
            model = model.to(device)
            predict = model(mixture, visual)

        torch.cuda.memory._record_memory_history(
            max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
            )
        model = model.to(device)
        predict = model(mixture, visual)
   
        try:
            torch.cuda.memory._dump_snapshot(f"{type(model)}_{cfg_train.get('num_blocks', '')}_inference_memory.pickle")
        except Exception as e:
            print(f"Failed to capture memory snapshot {e}")
        
        # memory_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # torch.cuda.reset_peak_memory_stats()
        # print("inference peak memory: ", memory_usage)

        mem_track = pickle.load(open(f"{type(model)}_{cfg_train.get('num_blocks', '')}_inference_memory.pickle", "rb"))
        max_mem, _ = compute_mem(mem_track["device_traces"][0])
        print("inference peak memory: ", max_mem)

        # Stop recording memory snapshot history.
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    # print("### IIANet")
    # with open(os.path.join('./config/lrs3/iia12.yaml'), 'r') as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # model = iianet.IIANet(num_blocks=5)
    # # memory_usage(model, cfg)
    # param_count(model)
    # memory_usage_visualize_training(model, cfg)
    # memory_usage_visualize_inference(model, cfg)

    # print("### AVLIT")
    # with open(os.path.join('./config/lrs3/avlit.yaml'), 'r') as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # model_config = cfg["train"].get("model_config", "avlit_default")
    # model_config_dict = dynamic.import_string("models.model_config.{}".format(model_config))
    # model = avlit.AVLITFixAE(**model_config_dict)
    # # memory_usage(model, cfg)
    # param_count(model)
    # memory_usage_visualize_training(model, cfg)
    # memory_usage_visualize_inference(model, cfg)

    # print("### RTFS")
    # with open(os.path.join('./config/lrs3/rtfs12_model.yaml'), 'r') as f:
    #     rtfs_cfg = yaml.load(f, Loader=yaml.FullLoader)
    # model = rtfs_net.AVNet(**rtfs_cfg["audionet"])
    # with open(os.path.join('./config/lrs3/rtfs12.yaml'), 'r') as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # # memory_usage(model, cfg)
    # param_count(model)
    # memory_usage_visualize_training(model, cfg)
    # memory_usage_visualize_inference(model, cfg)

    # print("### Profusion 8")
    # with open(os.path.join('./config/lrs3/avlit.yaml'), 'r') as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # cfg_train = cfg["train"]
    # num_blocks = 8
    # fusion_dim = 256
    # cfg_train["num_blocks"] = num_blocks
    # model = pro_mbt.ProgressiveBranch(num_blocks=num_blocks, fusion_channels=fusion_dim, video_dim=1024)
    # # memory_usage(model, cfg)
    # param_count(model)
    # memory_usage_visualize_training(model, cfg)
    # memory_usage_visualize_inference(model, cfg)

    print("### Profusion 12")
    with open(os.path.join('./config/lrs3/avlit.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_train = cfg["train"]
    num_blocks = 12
    fusion_dim = 256
    cfg_train["num_blocks"] = num_blocks
    model = pro_mbt.ProgressiveBranch(num_blocks=num_blocks, fusion_channels=fusion_dim, video_dim=1024)
    # memory_usage(model, cfg)
    param_count(model)
    # memory_usage_visualize_training(model, cfg)
    memory_usage_visualize_inference(model, cfg)

    # print("### Profusion 16")
    # with open(os.path.join('./config/lrs3/avlit.yaml'), 'r') as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # cfg_train = cfg["train"]
    # num_blocks = 16
    # fusion_dim = 256
    # cfg_train["num_blocks"] = num_blocks
    # model = pro_mbt.ProgressiveBranch(num_blocks=num_blocks, fusion_channels=fusion_dim, video_dim=1024)
    # # memory_usage(model, cfg)
    # param_count(model)
    # memory_usage_visualize_training(model, cfg)
    # memory_usage_visualize_inference(model, cfg)

    # print("### Profusion 20")
    # with open(os.path.join('./config/lrs3/avlit.yaml'), 'r') as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # cfg_train = cfg["train"]
    # num_blocks = 20
    # fusion_dim = 256
    # cfg_train["num_blocks"] = num_blocks
    # model = pro_mbt.ProgressiveBranch(num_blocks=num_blocks, fusion_channels=fusion_dim, video_dim=1024)
    # # memory_usage(model, cfg)
    # param_count(model)
    # memory_usage_visualize_training(model, cfg)
    # memory_usage_visualize_inference(model, cfg)
