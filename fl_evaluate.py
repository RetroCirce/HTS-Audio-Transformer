# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# Evalute the localization performance in DESED

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import config
from utils import int16_to_float32
import soundfile as sf
import h5py
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import math


def scratch_optim(preds, threshold, sec_threshold, mode):
    new_preds = np.zeros(preds.shape)
    if threshold == 0:
        return preds
    for i in range(len(preds)):
        new_preds[i] = preds[i]
        if threshold - sec_threshold < preds[i] < threshold + sec_threshold:
            if preds[i] < threshold:
                if mode > 1:
                    new_preds[i] = 2 * threshold - preds[i]
            else:
                if mode < 3:
                    new_preds[i] = 2 * preds[i] - threshold 
    return new_preds
     


def label_enhance(preds):
    class_map = np.load(config.class_map_path, allow_pickle=True)
    new_preds = np.zeros(preds.shape)
    for i in tqdm(range(len(preds))):
        for j in range(preds.shape[-1]):
            new_preds[i, j] += preds[i, j]
            for add_key in class_map[j][2]:
                new_preds[i, add_key] += preds[i, j] / len(class_map[j][2])
                # new_preds[i, add_key] = max(1.0, preds[i, j] / len(class_map[j][1]) + new_preds[i, add_key])
    return new_preds
 
                
def process(f_dir, f_name, f_class, f_map):
    # load data
    f_idx = 0
    rf_name = os.path.join(f_dir, f_name + "_cuda:" + str(f_idx) + ".npy")
    load_file = []
    while os.path.exists(rf_name):
        load_file.append(np.load(rf_name, allow_pickle = True))
        print("Load:", rf_name)
        f_idx += 1
        rf_name = os.path.join(f_dir, f_name + "_cuda:" + str(f_idx) + ".npy")
    load_file = np.concatenate(load_file, axis = 0)
    print(len(load_file))
    print(load_file[0]["heatmap"].shape)
    output_maps = {
        "filename": [],
        "onset":[],
        "offset":[],
        "event_label":[]
    }
    meta_maps = {
        "filename": [],
        "duration": []
    }
    for d in tqdm(load_file):
        audio_name = d["audio_name"]
        real_len = math.ceil(d["real_len"] // config.hop_size * 1.024) # add ratio
        pred_map = d["heatmap"][:real_len]
        pred_map = fl_mapping(pred_map, f_map)
        output_map = draw_timeline(pred_map, f_class)
        for ops in output_map:
            output_maps["filename"].append(audio_name)
            output_maps["onset"].append(ops[0])
            output_maps["offset"].append(ops[1])
            output_maps["event_label"].append(ops[2])
        meta_maps["filename"].append(audio_name)
        meta_maps["duration"].append(d["real_len"] / config.sample_rate)
    q_filename = os.path.join(f_dir, f_name + "_outputmap.tsv")
    m_filename = os.path.join(f_dir, f_name + "_meta.tsv")
    q = pd.DataFrame(
        output_maps
    )
    q.to_csv(q_filename, index = False, sep="\t")
    m = pd.DataFrame(
        meta_maps
    )
    m.to_csv(m_filename, index = False, sep="\t")
        
    
def fl_mapping(heatmap, f_map):
    # thres = [0.25, 0.1, 0.25, 0.15, 0.4, 0.1, 0.1, 0.4, 0.3, 0.1] # tscam + attn
    thres = [0.275, 0.2, 0.20, 0.23, 0.48, 0.20, 0.18, 0.51, 0.39, 0.14]  # tscam
    # thres = [0.28, 0.20, 0.05, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.05] # pann
    output = np.zeros((len(heatmap), len(f_map)))
    f_maps = np.concatenate(f_map)
    for hidx, d in enumerate(heatmap):
        didx = np.where(d > min(thres))[0]
        for dd in didx:
            for fidx, ff_map in enumerate(f_map):
                if dd in ff_map and (d[dd] > thres[fidx]):
                    output[hidx, fidx] = 1
                    break
    return output

def draw_timeline(heatmap, f_class):
    output = []
    step = config.hop_size / config.sample_rate * 1000 / 1024 # add ratio
    for fidx, cls in enumerate(f_class):
        sta = 0
        prev = heatmap[sta, fidx]
        for i in range(1, len(heatmap)):
            if heatmap[i, fidx] != prev:
                if prev != 0:
                    output.append([sta * step, i * step, cls])
                prev = heatmap[i, fidx]
                sta = i
        if prev != 0:
            output.append([sta * step, len(heatmap) * step, cls])
    return output



def main():
    # default settings
    # t = np.load("label_flipping_setting.npy",allow_pickle=True).item()
    print("Load File Group:", config.test_file)
    print("Class:", config.fl_class_num)
    process(config.heatmap_dir, config.test_file, config.fl_class_num, config.fl_audioset_mapping)
    # audioset_process(config.heatmap_dir, config.test_file)

if __name__ == '__main__':
    main()
