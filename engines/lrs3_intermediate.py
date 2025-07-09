import torch
import torch.nn as nn

from abc import ABC, abstractmethod
import models.avlit as avlit


class Intermediate:
    def __init__(self, cfg, model_path, save_path):
        self.cfg = cfg
        self.model_path = model_path
        self.save_path = save_path

        self.model = None
        self.init_model()

        self.train_set = None
        self.val_set = None
        self.test_set = None
        src
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.init_data()

    def init_model(self):
        pass

    def init_data(self):
        pass

    @abstractmethod
    def forward_pass(self, input_tuple):
        pass

    def get_intermediate(self, data_loader):
        intermediate = {}
        for idx, input_tuple in enumerate(data_loader):
            cur_intermediate = self.forward_pass(input_tuple)
            B = cur_intermediate[0].shape[0]

            for num_layer, feature in enumerate(cur_intermediate):
                if f"layer{num_layer}" not in intermediate:
                    intermediate[f"layer{num_layer}"] = {}

                for b in range(B):
                    intermediate[f"layer{num_layer}"][input_tuple["id"][b]] = feature[b]

        return intermediate