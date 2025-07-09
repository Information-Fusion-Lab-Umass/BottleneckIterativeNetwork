import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
import numpy as np
import math
from collections import OrderedDict

import models.misc as misc

SUPPORTED_TYPES = ('linear', 'conv2d', 'flatten', 'conv3d', 'maxpool3d', "maxpool2d", "dropout",
                    "batch_norm", "layer_norm", "relu", "leaky_relu", "sigmoid", "tanh")


def get_named_layers(layers, supported_types):
    layer_count = {}

    named_layers = []

    for k in supported_types:
        layer_count[k] = 0

    for i, description in enumerate(layers):
        module_type = description["layer"]
        if "name" not in description:
            module_name = '{}_{}'.format(module_type, layer_count[module_type])
            layer_count[module_type] += 1
            description.update({"name": module_name})
        named_layers.append(description)

    return named_layers


class ConvNet(nn.Module):
    """
    Use config dict to initialize a conv net
    """

    def get_conv_output_size(self, input_shape, conv_params):
        input_conv_shape = np.array(input_shape[1:])
        out_channels = conv_params.get("out_channels")
        kernel_size = np.array(conv_params.get("kernel_size"))
        stride = np.array(conv_params.get("stride", 1))
        padding = np.array(conv_params.get("padding", 0))
        if padding == "same":
            return out_channels, *input_conv_shape

        dilation = np.array(conv_params.get("dilation", 1))
        groups = np.array(conv_params.get("groups", 1))

        return (out_channels,
                *((input_conv_shape + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1).tolist())

    def get_maxpool_outptut_size(self, input_shape, pool_params):
        input_conv_shape = np.array(input_shape[1:])
        channels = input_shape[0]
        kernel_size = np.array(pool_params.get("kernel_size"))
        stride = np.array(pool_params.get("stride", kernel_size))
        padding = np.array(pool_params.get("padding", 0))
        dilation = np.array(pool_params.get("dilation", 1))

        return (channels,
                *((input_conv_shape + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1).tolist())

    def built_layers(self, description, input_shape):
        if description['layer'] == 'conv2d':
            description["args"]["in_channels"] = input_shape[0]
            layer = nn.Conv2d(**description["args"])
            nn.init.xavier_uniform_(layer.weight)
            out_shape = self.get_conv_output_size(input_shape, description["args"])

        elif description['layer'] == 'conv3d':
            description["args"]["in_channels"] = input_shape[0]
            layer = nn.Conv3d(**description["args"])
            nn.init.xavier_uniform_(layer.weight)
            out_shape = self.get_conv_output_size(input_shape, description["args"])

        elif description['layer'] == 'linear':
            if len(input_shape) >= 2:
                layer = nn.Sequential()
                shape = (math.prod(input_shape),)
                layer.add_module('flatten', misc.View(-1, shape[0]))
            else:
                shape = input_shape

            dim_in, = shape
            dim_out = description['args']["dim_out"]
            fc = nn.Linear(dim_in, dim_out)
            nn.init.xavier_uniform_(fc.weight)
            out_shape = (dim_out,)

            if len(input_shape) >= 2:
                layer.add_module('linear', fc)
            else:
                layer = fc

        elif description['layer'] == 'flatten':
            shape = (math.prod(input_shape),)
            layer = misc.View(-1, shape[0])
            out_shape = shape

        elif description['layer'] == 'maxpool2d':
            layer = nn.MaxPool2d(**description["args"])
            out_shape = self.get_maxpool_outptut_size(input_shape, description["args"])
        elif description['layer'] == 'maxpool3d':
            layer = nn.MaxPool3d(**description["args"])
            out_shape = self.get_maxpool_outptut_size(input_shape, description["args"])
        elif description["layer"] == "dropout":
            layer = nn.Dropout(**description["args"])
            out_shape = input_shape
        elif description["layer"] == "batch_norm":
            if len(input_shape) == 3:
                layer = nn.BatchNorm2d(input_shape[0])
            elif len(input_shape) == 4:
                layer = nn.BatchNorm3d(input_shape[0])
            else:
                layer = nn.BatchNorm1d(input_shape[0])
            out_shape = input_shape
        elif description["layer"] == "layer_norm":
            if "args" not in description or "type" not in description["args"]:
                raise NotImplementedError(
                    "please specify the type of layernorm"
                )
            if description["args"]["type"] == "sequential":
                layer = nn.LayerNorm(input_shape[-1])
            elif description["args"]["type"] == "image":
                layer = nn.LayerNorm(input_shape)
            else:
                raise NotImplementedError(
                    "{} in LayerNorm is not implemented".format(description["args"]["type"])
                )
            out_shape = input_shape
        elif description["layer"] == "relu":
            layer = nn.ReLU()
            out_shape = input_shape
        elif description["layer"] == "leaky_relu":
            layer = nn.LeakyReLU()
            out_shape = input_shape
        elif description["layer"] == "sigmoid":
            layer = nn.Sigmoid()
            out_shape = input_shape
        elif description["layer"] == "tanh":
            layer = nn.Tanh()
            out_shape = input_shape
        else:
            raise NotImplementedError(
                'Layer {} not supported. Use {}'.format(description['layer'], SUPPORTED_TYPES))

        return layer, out_shape, description["layer"]

    def __init__(self, input_shape, layers, verbose=False):
        super(ConvNet, self).__init__()

        self.modules_seq = nn.Sequential()

        self.output_info = OrderedDict()

        out_shape = input_shape

        named_layers = get_named_layers(layers, SUPPORTED_TYPES)

        for i, description in enumerate(named_layers):
            layer, out_shape, module_type = self.built_layers(description, out_shape)

            module_name = description["name"]

            self.modules_seq.add_module(module_name, layer)
            self.output_info[module_name] = out_shape

        self.out_shape = out_shape

        if verbose:
            print("configdriven_conv initiated: ", self.output_info)

    def forward(self, inputs):
        return self.modules_seq(inputs)

    def forward_layerwise(self, inputs, selected_layers=None):
        if selected_layers is None:
            layers = {}
            for key in self.output_info:
                layers[key] = key
        else:
            layers = {}
            for key in selected_layers:
                layers[key] = key

        return IntermediateLayerGetter(self.modules_seq, layers)(inputs)

    def load_model(self, state_dict, include_seq=True):
        """
        :param state_dict: checkpoint
        :param include_seq: if True then the state dict param names start with "modules_seq.", i.e. modules_seq.conv1.weight
                            Otherwise the state dict name only has the layer name, i.e. conv1.weight
        :return:
        """
        if include_seq:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in self.state_dict()}
            self.state_dict().update(pretrained_dict)
            self.load_state_dict(pretrained_dict)
        else:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in self.modules_seq.state_dict()}
            self.modules_seq.state_dict().update(pretrained_dict)
            self.modules_seq.load_state_dict(pretrained_dict)


if __name__ == '__main__':
    test_config = {
        "input_shape": (3, 224, 224),
        "layers": [
            dict(layer='conv2d',
                 args=dict(
                     out_channels=8,
                     kernel_size=5
                 ),
                 name="postl1"),
            dict(layer="layer_norm", args=dict(type="image")),
            dict(layer="relu"),
            dict(
                layer="maxpool2d",
                args=dict(
                    kernel_size=2
                )
            ),
            dict(layer='conv2d',
                 args=dict(
                     out_channels=16,
                     kernel_size=3
                 ),
                 name="postl2"),
            dict(layer="layer_norm", args=dict(type="image")),
            dict(layer="relu"),
            dict(
                layer="maxpool2d",
                args=dict(
                    kernel_size=2
                )
            ),
            dict(layer='conv2d',
                 args=dict(
                     out_channels=32,
                     kernel_size=3
                 ),
                 name="postl3")]
    }

    test_encoder = ConvNet(**test_config)
    x = torch.ones(4, 3, 224, 224)
    output = test_encoder.forward_layerwise(x, ["postl1", "relu_1"])
    for key in output:
        print(key, output[key].shape)



