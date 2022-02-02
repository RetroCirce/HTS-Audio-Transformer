# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# The main code for training and evaluating HTSAT
import os
from re import A, S
import sys
import librosa
import numpy as np
import argparse
import h5py
import math
import time
import logging
import pickle
import random
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torch.utils.data.distributed import DistributedSampler

from utils import create_folder, dump_config, process_idc, prepprocess_audio, init_hier_head

import config
from sed_model import SEDWrapper, Ensemble_SEDWrapper
from models import Cnn14_DecisionLevelMax
from data_generator import SEDDataset, DESED_Dataset, ESC_Dataset, SCV2_Dataset


from model.htsat import HTSAT_Swin_Transformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings



warnings.filterwarnings("ignore")

class data_prep(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, device_num):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device_num = device_num

    def train_dataloader(self):
        train_sampler = DistributedSampler(self.train_dataset, shuffle = False) if self.device_num > 1 else None
        train_loader = DataLoader(
            dataset = self.train_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size // self.device_num,
            shuffle = False,
            sampler = train_sampler
        )
        return train_loader
    def val_dataloader(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle = False) if self.device_num > 1 else None
        eval_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size // self.device_num,
            shuffle = False,
            sampler = eval_sampler
        )
        return eval_loader
    def test_dataloader(self):
        test_sampler = DistributedSampler(self.eval_dataset, shuffle = False) if self.device_num > 1 else None
        test_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size // self.device_num,
            shuffle = False,
            sampler = test_sampler
        )
        return test_loader
    

def save_idc():
    train_index_path = os.path.join(config.dataset_path, "hdf5s", "indexes", config.index_type + ".h5")
    eval_index_path = os.path.join(config.dataset_path,"hdf5s", "indexes", "eval.h5")
    process_idc(train_index_path, config.classes_num,  config.index_type + "_idc.npy")
    process_idc(eval_index_path, config.classes_num, "eval_idc.npy")

def weight_average():
    model_ckpt = []
    model_files = os.listdir(config.wa_folder)
    wa_ckpt = {
        "state_dict": {}
    }

    for model_file in model_files:
        model_file = os.path.join(config.wa_folder, model_file)
        model_ckpt.append(torch.load(model_file, map_location="cpu")["state_dict"])
    keys = model_ckpt[0].keys()
    for key in keys:
        model_ckpt_key = torch.cat([d[key].float().unsqueeze(0) for d in model_ckpt])
        model_ckpt_key = torch.mean(model_ckpt_key, dim = 0)
        assert model_ckpt_key.shape == model_ckpt[0][key].shape, "the shape is unmatched " + model_ckpt_key.shape + " " + model_ckpt[0][key].shape
        wa_ckpt["state_dict"][key] = model_ckpt_key
    torch.save(wa_ckpt, config.wa_model_path)

def esm_test():
    device_num = torch.cuda.device_count()
    print("each batch size:", config.batch_size // device_num)
    if config.fl_local:
        fl_npy = np.load(config.fl_dataset, allow_pickle = True)
        # import dataset SEDDataset
        eval_dataset = DESED_Dataset(
            dataset = fl_npy,
            config = config
        )
    else:
        # dataset file pathes
        eval_index_path = os.path.join(config.dataset_path,"hdf5s", "indexes", "eval.h5")
        eval_idc = np.load("eval_idc.npy", allow_pickle = True)

        # import dataset SEDDataset
        eval_dataset = SEDDataset(
            index_path=eval_index_path,
            idc = eval_idc,
            config = config,
            eval_mode = True
        )
    audioset_data = data_prep(eval_dataset, eval_dataset, device_num)
    trainer = pl.Trainer(
        deterministic=True,
        gpus = device_num, 
        max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        checkpoint_callback = False,
        accelerator = "ddp" if device_num > 1 else None,
        num_sanity_val_steps = 0,
        # resume_from_checkpoint = config.resume_checkpoint,
        replace_sampler_ddp = False,
        gradient_clip_val=1.0
    )
    sed_models = []
    for esm_model_path in config.esm_model_pathes:
        sed_model = HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config = config,
            depths = config.htsat_depth,
            embed_dim = config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head
        )
        sed_wrapper = SEDWrapper(
            sed_model = sed_model, 
            config = config,
            dataset = eval_dataset
        )
        ckpt = torch.load(esm_model_path, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")
        sed_wrapper.load_state_dict(ckpt["state_dict"], strict=False)
        sed_models.append(sed_wrapper)
    
    model = Ensemble_SEDWrapper(
        sed_models = sed_models, 
        config = config,
        dataset = eval_dataset
    )
    trainer.test(model, datamodule=audioset_data)


def test():
    device_num = torch.cuda.device_count()
    print("each batch size:", config.batch_size // device_num)
    # dataset file pathes
    if config.fl_local:
        fl_npy = np.load(config.fl_dataset, allow_pickle = True)
        # import dataset SEDDataset
        eval_dataset = DESED_Dataset(
            dataset = fl_npy,
            config = config
        )
    else:
        if config.dataset_type == "audioset":
            eval_index_path = os.path.join(config.dataset_path,"hdf5s", "indexes", "eval.h5")
            eval_idc = np.load("eval_idc.npy", allow_pickle = True)
            eval_dataset = SEDDataset(
                index_path=eval_index_path,
                idc = eval_idc,
                config = config,
                eval_mode = True
            )
        elif config.dataset_type == "esc-50":
            full_dataset = np.load(os.path.join(config.dataset_path, "esc-50-data.npy"), allow_pickle = True)
            eval_dataset = ESC_Dataset(
                dataset = full_dataset,
                config = config,
                eval_mode = True
            )
        elif config.dataset_type == "scv2":
            test_set = np.load(os.path.join(config.dataset_path, "scv2_test.npy"), allow_pickle = True)
            eval_dataset = SCV2_Dataset(
                dataset = test_set,
                config = config,
                eval_mode = True
            )
        # import dataset SEDDataset
        
    audioset_data = data_prep(eval_dataset, eval_dataset, device_num)
    trainer = pl.Trainer(
        deterministic=True,
        gpus = device_num, 
        max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        checkpoint_callback = False,
        accelerator = "ddp" if device_num > 1 else None,
        num_sanity_val_steps = 0,
        # resume_from_checkpoint = config.resume_checkpoint,
        replace_sampler_ddp = False,
        gradient_clip_val=1.0
    )
    sed_model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config = config,
        depths = config.htsat_depth,
        embed_dim = config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )
    
    model = SEDWrapper(
        sed_model = sed_model, 
        config = config,
        dataset = eval_dataset
    )
    if config.resume_checkpoint is not None:
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    trainer.test(model, datamodule=audioset_data)

    

def train():
    device_num = torch.cuda.device_count()
    print("each batch size:", config.batch_size // device_num)
    
    # dataset file pathes
    if config.dataset_type == "audioset":
        train_index_path = os.path.join(config.dataset_path, "hdf5s","indexes", config.index_type + ".h5")
        eval_index_path = os.path.join(config.dataset_path,"hdf5s", "indexes", "eval.h5")
        train_idc = np.load(config.index_type + "_idc.npy", allow_pickle = True)
        eval_idc = np.load("eval_idc.npy", allow_pickle = True)
    elif config.dataset_type == "esc-50":
        full_dataset = np.load(os.path.join(config.dataset_path, "esc-50-data.npy"), allow_pickle = True)
    elif config.dataset_type == "scv2":
        train_set = np.load(os.path.join(config.dataset_path, "scv2_train.npy"), allow_pickle = True)
        test_set = np.load(os.path.join(config.dataset_path, "scv2_test.npy"), allow_pickle = True)
   
    # set exp folder
    exp_dir = os.path.join(config.workspace, "results", config.exp_name)
    checkpoint_dir = os.path.join(config.workspace, "results", config.exp_name, "checkpoint")
    if not config.debug:
        create_folder(os.path.join(config.workspace, "results"))
        create_folder(exp_dir)
        create_folder(checkpoint_dir)
        dump_config(config, os.path.join(exp_dir, config.exp_name), False)

    # import dataset SEDDataset
    if config.dataset_type == "audioset":
        print("Using Audioset")
        dataset = SEDDataset(
            index_path=train_index_path,
            idc = train_idc,
            config = config
        )
        eval_dataset = SEDDataset(
            index_path=eval_index_path,
            idc = eval_idc,
            config = config,
            eval_mode = True
        )
    elif config.dataset_type == "esc-50":
        print("Using ESC")
        dataset = ESC_Dataset(
            dataset = full_dataset,
            config = config,
            eval_mode = False
        )
        eval_dataset = ESC_Dataset(
            dataset = full_dataset,
            config = config,
            eval_mode = True
        )
    elif config.dataset_type == "scv2":
        print("Using SCV2")
        dataset = SCV2_Dataset(
            dataset = train_set,
            config = config,
            eval_mode = False
        )
        eval_dataset = SCV2_Dataset(
            dataset = test_set,
            config = config,
            eval_mode = True
        )

    audioset_data = data_prep(dataset, eval_dataset, device_num)
    if config.dataset_type == "audioset":
        checkpoint_callback = ModelCheckpoint(
            monitor = "mAP",
            filename='l-{epoch:d}-{mAP:.3f}-{mAUC:.3f}',
            save_top_k = 20,
            mode = "max"
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor = "acc",
            filename='l-{epoch:d}-{acc:.3f}',
            save_top_k = 20,
            mode = "max"
        )
    trainer = pl.Trainer(
        deterministic=True,
        default_root_dir = checkpoint_dir,
        gpus = device_num, 
        val_check_interval = 0.1,
        max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        callbacks = [checkpoint_callback],
        accelerator = "ddp" if device_num > 1 else None,
        num_sanity_val_steps = 0,
        resume_from_checkpoint = None, 
        replace_sampler_ddp = False,
        gradient_clip_val=1.0
    )
    sed_model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config = config,
        depths = config.htsat_depth,
        embed_dim = config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )
    
    model = SEDWrapper(
        sed_model = sed_model, 
        config = config,
        dataset = dataset
    )
    if config.resume_checkpoint is not None:
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")
        # finetune on the esc and spv2 dataset
        ckpt["state_dict"].pop("sed_model.tscam_conv.weight")
        ckpt["state_dict"].pop("sed_model.tscam_conv.bias")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    elif config.swin_pretrain_path is not None: # train with pretrained model
        ckpt = torch.load(config.swin_pretrain_path, map_location="cpu")
        # load pretrain model
        ckpt = ckpt["model"]
        found_parameters = []
        unfound_parameters = []
        model_params = dict(model.state_dict())

        for key in model_params:
            m_key = key.replace("sed_model.", "")
            if m_key in ckpt:
                if m_key == "patch_embed.proj.weight":
                    ckpt[m_key] = torch.mean(ckpt[m_key], dim = 1, keepdim = True)
                if m_key == "head.weight" or m_key == "head.bias":
                    ckpt.pop(m_key)
                    unfound_parameters.append(key)
                    continue
                assert model_params[key].shape==ckpt[m_key].shape, "%s is not match, %s vs. %s" %(key, str(model_params[key].shape), str(ckpt[m_key].shape))
                found_parameters.append(key)
                ckpt[key] = ckpt.pop(m_key)
            else:
                unfound_parameters.append(key)
        print("pretrain param num: %d \t wrapper param num: %d"%(len(found_parameters), len(ckpt.keys())))
        print("unfound parameters: ", unfound_parameters)
        model.load_state_dict(ckpt, strict = False)
        model_params = dict(model.named_parameters())
    trainer.fit(model, audioset_data)



def main():
    parser = argparse.ArgumentParser(description="HTS-AT")
    subparsers = parser.add_subparsers(dest = "mode")
    parser_train = subparsers.add_parser("train")
    parser_test = subparsers.add_parser("test")
    parser_esm_test = subparsers.add_parser("esm_test")
    parser_saveidc = subparsers.add_parser("save_idc")
    parser_wa = subparsers.add_parser("weight_average")
    args = parser.parse_args()
    # default settings
    logging.basicConfig(level=logging.INFO) 
    pl.utilities.seed.seed_everything(seed = config.random_seed)

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "esm_test":
        esm_test()
    elif args.mode == "save_idc":
        save_idc()
    elif args.mode == "weight_average":
        weight_average()
    else:
        raise Exception("Error Mode!")
    

if __name__ == '__main__':
    main()

