# Hierarchical Token Semantic Audio Transformer


## Introduction

The Code Repository for  "[HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection](https://arxiv.org/abs/2202.00874)", in ICASSP 2022.

In this paper, we devise a model, HTS-AT, by combining a [swin transformer](https://github.com/microsoft/Swin-Transformer) with a token-semantic module and adapt it in to **audio classification** and **sound event detection tasks**. HTS-AT is an efficient and light-weight audio transformer with a hierarchical structure and has only 30 million parameters. It achieves new state-of-the-art (SOTA) results on AudioSet and ESC-50, and equals the SOTA on Speech Command V2. It also achieves better performance in event localization than the previous CNN-based models. 

![HTS-AT Architecture](fig/arch.png)

## Classification Results on AudioSet, ESC-50, and Speech Command V2 (mAP)

<p align="center">
<img src="fig/ac_result.png" align="center" alt="HTS-AT ClS Result" width="50%"/>
</p>


## Localization/Detection Results on DESED dataset (F1-Score)

![HTS-AT Localization Result](fig/local_result.png)


Below is the updated README in English with added instructions for preparing the `htsat_esc_training.ipynb` file:

---

# HTS-Audio-Transformer

Installation is performed via Conda and Pip, and configuration is managed in the `config.py` file.

> **Note:** This repository is a refactored version of [RetroCirce/HTS-Audio-Transformer](https://github.com/RetroCirce/HTS-Audio-Transformer) with a streamlined organization for easier installation and usage.

---

## Getting Started

### Installation

#### 1. Create the Conda Environment

Create an environment (replace `your_env_name` with your desired environment name) with all the necessary dependencies:

```bash
conda create -n your_env_name -c pytorch -c nvidia -c conda-forge pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 sox ffmpeg h5py=3.6.0 librosa==0.8.1 matplotlib==3.5.1 numpy==1.22 pandas==1.4.0 scikit-learn==1.0.2 scipy==1.7.3 tensorboard==2.8.0 pytorch-lightning==1.5.9
```

Activate the environment:

```bash
conda activate your_env_name
```

#### 2. Install Additional Packages

Install extra packages using Pip:

```bash
pip install museval==0.4.0 torchcontrib==0.0.2 torchlibrosa==0.0.9 tqdm==4.62.3 wget notebook ipywidgets gdown
```

#### 3. System Dependencies (Linux)

If you are running on Linux, ensure that **SOX** and **ffmpeg** are installed. Although these packages are included in the Conda command, you may also install them manually if needed:

```bash
sudo apt install sox
conda install -c conda-forge ffmpeg
```

---

## Downloading and Processing Datasets

Before starting training or evaluation, you must prepare your datasets. Edit the `config.py` file to modify the following variables:

- **`dataset_path`**: Path to your processed dataset folder.
- **`desed_folder`**: Path to your DESED folder (if applicable).
- **`classes_num`**: Number of classes (e.g., 527 for AudioSet).

### AudioSet

1. **Index the Data:**

   Adjust the paths in the `./create_index.sh` script if necessary, then run:

   ```bash
   ./create_index.sh
   ```

2. **Save Class Information:**

   Count the number of samples per class and save the information to `.npy` files:

   ```bash
   python main.py save_idc
   ```

### ESC-50

Open the notebook `esc-50/prep_esc50.ipynb` and run the cells to process the dataset.

### Speech Command V2

Open the notebook `scv2/prep_scv2.ipynb` and run the cells to process the dataset.

### DESED Dataset

Generate the `.npy` data files from the DESED dataset by running:

```bash
python conver_desed.py
```

---

## HTS-ESC Training Notebook

This repository also includes the `htsat_esc_training.ipynb` notebook, which is specifically designed for training the model on the ESC-50 dataset. To prepare and use this notebook:

1. **Configure for ESC-50:**  
   Open `config.py` and set the following parameters for ESC-50:
   ```python
   dataset_path   = "path/to/your/processed/esc50"
   dataset_type   = "esc-50"
   loss_type      = "clip_ce"
   sample_rate    = 32000
   hop_size       = 320
   classes_num    = 50
   ```
2. **Open the Notebook:**  
   Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   and open the `htsat_esc_training.ipynb` file located in the repository root.
3. **Run the Cells:**  
   Execute each cell sequentially. The notebook handles data preprocessing, model initialization, and training specific to the ESC-50 dataset. Follow the inline comments for detailed guidance.

---

## Configuration (`config.py`)

The `config.py` file contains all the configuration settings required to run the code. Read the introductory comments in the file and adjust the settings according to your needs.

> **IMPORTANT:**  
> Like many Transformer-based models, HTS-AT requires a warm-up phase to prevent underfitting at the beginning of training. The default settings are tuned for the full AudioSet (2.2M samples). If your dataset size differs (e.g., 100K, 1M, 10M samples, etc.), you might need to adjust the warm-up steps or epochs accordingly.

### Example Configurations

- **AudioSet:**

  ```python
  dataset_path   = "path/to/your/processed/audioset"
  dataset_type   = "audioset"
  balanced_data  = True
  loss_type      = "clip_bce"
  sample_rate    = 32000
  hop_size       = 320
  classes_num    = 527
  ```

- **ESC-50:**

  ```python
  dataset_path   = "path/to/your/processed/esc50"
  dataset_type   = "esc-50"
  loss_type      = "clip_ce"
  sample_rate    = 32000
  hop_size       = 320
  classes_num    = 50
  ```

- **Speech Command V2:**

  ```python
  dataset_path   = "path/to/your/processed/scv2"
  dataset_type   = "scv2"
  loss_type      = "clip_bce"
  sample_rate    = 16000
  hop_size       = 160
  classes_num    = 35
  ```

- **DESED:**

  ```python
  resume_checkpoint = "path/to/your/audioset_checkpoint"
  heatmap_dir       = "directory_for_localization_results"
  test_file         = "heatmap_output_filename"
  fl_local          = True
  fl_dataset        = "path/to/your/desed_npy_file"
  ```

---

## Training and Evaluation

> **Note:** The model currently supports single GPU training/testing.

All scripts are executed via `main.py`.

- **Training:**

  ```bash
  CUDA_VISIBLE_DEVICES=0 python main.py train
  ```

- **Testing:**

  ```bash
  CUDA_VISIBLE_DEVICES=0 python main.py test
  ```

- **Ensemble Testing:**

  ```bash
  CUDA_VISIBLE_DEVICES=0 python main.py esm_test
  ```

  (Check the ensemble settings in `config.py`.)

- **Weight Averaging:**

  ```bash
  python main.py weight_average
  ```

---

## Localization on DESED

To perform localization on the DESED dataset:

1. Ensure `fl_local=True` in `config.py`.
2. Run the test:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python main.py test
   ```

3. Organize and gather the localization results:

   ```bash
   python fl_evaluate.py
   ```

4. You can also use the notebook `fl_evaluate_f1.ipynb` to produce the final localization results.

---

## Model Checkpoints

Pre-trained model checkpoints for AudioSet, ESC-50, Speech Command V2, and DESED are provided. Feel free to download and test these checkpoints.

---

## Citing

If you use this work in your research, please cite:

```bibtex
@inproceedings{htsat-ke2022,
  author = {Ke Chen and Xingjian Du and Bilei Zhu and Zejun Ma and Taylor Berg-Kirkpatrick and Shlomo Dubnov},
  title = {HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection},
  booktitle = {{ICASSP} 2022}
}
```

---

Our work is based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer), which is a famous image classification transformer model.
