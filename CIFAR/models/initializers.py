# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math

def initializations(init_type, density):
    def binary(w):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(w.weight)
            sigma = w.weight.data.std()
            w.weight.data = torch.sign(w.weight.data) * sigma

    def kaiming_normal(w):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(w.weight)

    def scaled_kaiming_normal(w):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            fan = nn.init._calculate_correct_fan(w.weight, mode='fan_in')
            fan = fan * density
            gain = nn.init.calculate_gain('leaky_relu')
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                w.weight.data.normal_(0, std)

    def kaiming_uniform(w):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(w.weight)

    def orthogonal(w):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            torch.nn.init.orthogonal_(w.weight)

    if init_type == 'binary':
        return binary
    elif init_type == 'kaiming_normal':
        return kaiming_normal
    elif init_type == 'scaled_kaiming_normal':
        return scaled_kaiming_normal
    elif init_type == 'kaiming_uniform':
        return kaiming_uniform
    elif init_type == 'orthogonal':
        return orthogonal


# def binary(w):
#     if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
#         torch.nn.init.kaiming_normal_(w.weight)
#         sigma = w.weight.data.std()
#         w.weight.data = torch.sign(w.weight.data) * sigma

#
# def kaiming_normal(w):
#     if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
#         torch.nn.init.kaiming_normal_(w.weight)
#
# def scaled_kaiming_normal(w):
#     if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
#         fan = nn.init._calculate_correct_fan(w.weight, mode='fan_in')
#         fan = fan * args.density
#         gain = nn.init.calculate_gain('leaky_relu')
#         std = gain / math.sqrt(fan)
#         with torch.no_grad():
#             w.weight.data.normal_(0, std)
#
# def kaiming_uniform(w):
#     if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
#         torch.nn.init.kaiming_uniform_(w.weight)
#
#
# def orthogonal(w):
#     if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
#         torch.nn.init.orthogonal_(w.weight)