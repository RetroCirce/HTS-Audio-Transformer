# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# Some Useful Common Methods

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import logging
import os
import sys
import h5py
import csv
import time
import json
import museval
import librosa
from datetime import datetime
from tqdm import tqdm
from scipy import stats
import torch.nn as nn
import torch.nn.functional as F


# import from https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        # x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x # without sigmoid since it has been computed
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()


def get_mix_lambda(mixup_alpha, batch_size):
    mixup_lambdas = [np.random.beta(mixup_alpha, mixup_alpha, 1)[0] for _ in range(batch_size)]
    return np.array(mixup_lambdas).astype(np.float32)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def dump_config(config, filename, include_time = False):
    save_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    config_json = {}
    for key in dir(config):
        if not key.startswith("_"):
            config_json[key] = eval("config." + key)
    if include_time:
        filename = filename + "_" + save_time
    with open(filename + ".json", "w") as f:      
        json.dump(config_json, f ,indent=4)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min = -1., a_max = 1.)
    return (x * 32767.).astype(np.int16)


# index for each class
def process_idc(index_path, classes_num, filename):
    # load data
    logging.info("Load Data...............")
    idc = [[] for _ in range(classes_num)]
    with h5py.File(index_path, "r") as f:
        for i in tqdm(range(len(f["target"]))):
            t_class = np.where(f["target"][i])[0]
            for t in t_class:
                idc[t].append(i)
    print(idc)
    np.save(filename, idc)
    logging.info("Load Data Succeed...............")

def clip_bce(pred, target):
    """Binary crossentropy loss.
    """
    return F.binary_cross_entropy(pred, target)
    # return F.binary_cross_entropy(pred, target)


def clip_ce(pred, target):
    return F.cross_entropy(pred, target)

def d_prime(auc):
    d_prime = stats.norm().ppf(auc) * np.sqrt(2.0)
    return d_prime


def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce
    if loss_type == 'clip_ce':
        return clip_ce
    if loss_type == 'asl_loss':
        loss_func = AsymmetricLoss(gamma_neg=4, gamma_pos=0,clip=0.05)
        return loss_func

def do_mixup_label(x):
    out = torch.logical_or(x, torch.flip(x, dims = [0])).float()
    return out

def do_mixup(x, mixup_lambda):
    """
    Args:
      x: (batch_size , ...)
      mixup_lambda: (batch_size,)

    Returns:
      out: (batch_size, ...)
    """
    out = (x.transpose(0,-1) * mixup_lambda + torch.flip(x, dims = [0]).transpose(0,-1) * (1 - mixup_lambda)).transpose(0,-1)
    return out
    
def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate

    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value 
    is the same as the value of the last frame.

    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad

    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output

# set the audio into the format that can be fed into the model
# resample -> convert to mono -> output the audio  
# track [n_sample, n_channel]
def prepprocess_audio(track, ofs, rfs, mono_type = "mix"):
    if track.shape[-1] > 1:
        # stereo
        if mono_type == "mix":
            track = np.transpose(track, (1,0))
            track = librosa.to_mono(track)
        elif mono_type == "left":
            track = track[:, 0]
        elif mono_type == "right":
            track = track[:, 1]
    else:
        track = track[:, 0]
    # track [n_sample]
    if ofs != rfs:
        track = librosa.resample(track, ofs, rfs)
    return track

def init_hier_head(class_map, num_class):
    class_map = np.load(class_map, allow_pickle = True)
    
    head_weight = torch.zeros(num_class,num_class).float()
    head_bias = torch.zeros(num_class).float()

    for i in range(len(class_map)):
        for d in class_map[i][1]:
            head_weight[d][i] = 1.0
        for d in class_map[i][2]:
            head_weight[d][i] = 1.0 / len(class_map[i][2])
        head_weight[i][i] = 1.0
    return head_weight, head_bias
