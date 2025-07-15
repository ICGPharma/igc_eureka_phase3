# Solution - Eureka Dementia Audio Classification Challenge

Username: <IGCPHARMA-Team>

## Summary

This project provides a pipeline to classify audio samples into three diagnostic categories: Control, Mild Cognitive Impairment (MCI), and Alzheimer's Disease (ADRD). The approach uses the Distil-Whisper model (distil-large-v3) from Hugging Face as the backbone.

### Steps:
- **Data Preprocessing**:
  - Noise reduction and normalization using custom audio preprocessing scripts.
  - Audio processing includes resampling, noise gating, filtering, and loudness normalization.

- **Modeling**:
  - Pretrained Hugging Face model: `distil-whisper/distil-large-v3`
  - Fine-tuned with additional classification layers specifically for dementia diagnosis.

- **Training Strategy**:
  - Stage 1: Trained for 4 epochs using a 90% train and 10% validation split.
  - Stage 2: Additional training for 2 epochs using the entire dataset.

- **Inference**:
  - Direct inference using trained model weights to generate submission predictions.

## Setup
To setup the python environment navigate to igc_code_submission and follow these instructions. 
### 1. Python Environment

- **Python version**: 3.12.5

- Recommended to use a conda environment:

```bash

conda create -n audio_ad python=3.12.5
conda activate audio_ad
```

### 2. Install Required Python Packages

Install packages from provided `requirements.txt` and install accelerate:

```bash
pip install -r requirements.txt
pip install --upgrade transformers accelerate
```

### 3. Additional Tools

Install `ffmpeg` if not available:

```bash
sudo apt update
sudo apt install ffmpeg
```
### 4. Hugging Face Authentication

Authenticate Hugging Face CLI to access the pretrained models:

```bash
huggingface-cli login
```

You will be prompted to enter your Hugging Face token, which you can obtain [here](https://huggingface.co/settings/tokens).

### Project Structure

Ensure your project has the following structure before running:
The files bad_audios.txt, Metadata.csv, Train_labels.csv are neede since we performed a manual filtering of the low quiality files. The raw audios must be placed inside: ```/src/data/raw_data/train_audios``` and ```/src/data/raw_data/test_audios``` 

```
./src/
├── data/
│   ├── post_data/
│   │   ├── Metadata.csv
│   │   ├── Train_labels.csv
│   │   ├── Train_Features.csv
│   │   ├── train_audios/
│   │   └── test_audios/
│   ├── raw_data/
│   │   ├── train_audios/
│   │   └── test_audios/
│   ├── bad_audios.txt
│   └── Test_Features.csv
├── models/
│   └── final_model_checkpoint.yaml
├── utils/
│   ├── ad_dataset.py
│   ├── train.py
│   └── utils.py
├── audio_preprocessing.py
├── run_inference.py
├── run_train.py
├── requirements.txt
├── README.md
├── checkpoints/
└── results/
```

## Hardware

- **GPU**: NVIDIA RTX8000 (48GB VRAM) X 4
- **Training time**:
  - Stage 1 (~2 hours for 4 epochs)
  - Stage 2 (~1 hours for 2 epochs)

- **Inference time**:
  - ~15 minutes

## Run Training

### Audio Processing Pipeline
To train from scratch it is required to process the complete set of audio files with the provided pipeline, for easier reproducibility we are sharing
the preprocess audios in the folder data/post_data:

```bash
cd src
python audio_preprocessing.py --input_dir ../data/raw_data/train_audios --output_dir ../data/post_data/train_audios
python audio_preprocessing.py --input_dir ../data/raw_data/test_audios --output_dir ../data/post_data/test_audios
```

If using the talkbank dataset with full audios process with:
```bash
cd src
python audio_preprocessing.py --csv_file ../data/audios_final_partition.csv --train --output_dir ../data/post_data/train_audios/ --dataset_type talkbank-parallel --no-segment
python audio_preprocessing.py --csv_file ../data/audios_final_partition.csv --no-train --output_dir ../data/post_data/test_audios/ --dataset_type talkbank-parallel --no-segment
```

If using the talkbank dataset with split audios process with:
```bash
cd src
python audio_preprocessing.py --csv_file ../data/split_audios_final_partition.csv --train --output_dir ../data/post_data/train_audios/ --dataset_type talkbank-parallel --segment
python audio_preprocessing.py --csv_file ../data/split_audios_final_partition.csv --no-train --output_dir ../data/post_data/test_audios/ --dataset_type talkbank-parallel --segment
```
### Run Training

In the config.yaml file you can modify the paths if needed.
```bash
cd src
python run_train.py
```

### Data Requirements
- Training and validation audio data (`.wav` or `.mp3`).
- Metadata and label CSV files as specified in `config.yaml`.

### Storage Requirements
- Model weights ~3GB saved in `checkpoints` folder.
- Intermediate training metrics and plots saved under `results`.

### Trained Model Weights
- The final model weights (`final_model.pth`) are stored under `checkpoints` after training.

### Network Requirements
- Internet access required to download pretrained Hugging Face models.

## Run Inference

To generate predictions:

```bash
cd src
python run_inference.py
```

### Data Requirements
- Raw test audio data (`.wav` or `.mp3`).
- Metadata CSV file.

### Output
- Generates a submission file (`submission_whisper_final.csv`) with predicted probabilities, based on the pretrained model at models/final_model_checkpoint.pth. To perform inference with a different model change the path at config.yaml - paths.final_model 

