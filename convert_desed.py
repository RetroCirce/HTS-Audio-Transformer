# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# Convert the DESED dataset

import numpy as np
import os
import librosa
import config
from utils import float32_to_int16
import soundfile as sf
def main():
    desed_folder = os.path.join(config.desed_folder, "audio", "eval", "public")
    fl_files = os.listdir(desed_folder)
    output_dir = os.path.join(config.desed_folder, "audio", "eval", "resample")
    output_dict = []
    for f in fl_files:
        y, sr = librosa.load(os.path.join(desed_folder, f), sr = config.sample_rate)
        sf.write(os.path.join(output_dir, f), y, sr)
        print(f, sr, float32_to_int16(y))
        temp_dict = {
            "audio_name": f,
            "waveform": float32_to_int16(y)
        }
        output_dict.append(temp_dict)
    npy_file = os.path.join(config.desed_folder, "eval.npy")
    np.save(npy_file, output_dict)



if __name__ == '__main__':
    main()
