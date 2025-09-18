from typing import Any
from cog import BasePredictor, Input, Path
from model.htsat import HTSAT_Swin_Transformer
import torch
import librosa
import numpy as np
import pandas as pd

import config

SAMPLE_RATE = 16000


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # preprocess the class_label_indice
        df = pd.read_csv("class_label_indice.csv", sep=",")
        self.idx_2_label = {}
        for row in df.iterrows():
            idx, _, label = row[1]
            self.idx_2_label[idx] = label

        # load model
        checkpoints = torch.load(
            "./HTSAT_AudioSet_Saved_6.ckpt", map_location=torch.device("cpu")
        )
        new_checkpoints = {"state_dict": {}}
        for old_key in checkpoints["state_dict"].keys():
            new_key = old_key.replace("sed_model.", "")
            new_checkpoints["state_dict"][new_key] = checkpoints["state_dict"][old_key]
        self.sed_model = HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config=config,
            depths=config.htsat_depth,
            embed_dim=config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head,
        )
        self.sed_model.load_state_dict(new_checkpoints["state_dict"])
        self.sed_model.eval()

    # Define the arguments and types the model takes as input
    def predict(self, audio: Path = Input(description="Audio to classify")) -> Any:
        """Run a single prediction on the model"""
        # Preprocess the audio
        waveform, sr = librosa.load(audio, sr=SAMPLE_RATE)

        with torch.no_grad():
            x = torch.from_numpy(waveform).float()
            output_dict = self.sed_model(x[None, :], None, True)
            pred = output_dict["clipwise_output"]
            pred_post = torch.nn.functional.sigmoid(pred)
            pred_post = pred_post[0].detach().cpu().numpy()

            pred_labels = np.argsort(pred_post)
            pred_labels = pred_labels[-3:][::-1]
            # print("pred probability sum: {:.2f}".format(np.sum(pred_post)))
        return [
            [pred_label, self.idx_2_label[pred_label], pred_post[pred_label]]
            for pred_label in pred_labels
        ]
