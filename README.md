# GSE-Network

This repository contains an implementation of the **Guided Speech Enhancement Network (GSENet)** as described in [Yang et al., "Guided Speech Enhancement Network," ICASSP 2023](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096763&isnumber=10094560). GSENet is a U-Net–like architecture designed for multi-microphone speech enhancement that leverages both beamformed outputs and raw microphone signals to produce high-quality enhanced speech.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Local Training](#local-training)
  - [Google Cloud Training](#google-cloud-training)
- [Dataset](#dataset)
- [References](#references)
- [License](#license)

## Features

- **Multi-Input Enhancement:**  
  Combines both beamformed outputs and raw reference microphone signals.
- **U-Net–like Architecture:**  
  Incorporates encoder–decoder blocks with skip connections, time dilation, and a custom STFT-based loss.
- **Custom Data Loading:**  
  Supports a directory-based dataset where each recording is a folder containing multiple WAV files.
- **PyTorch Lightning:**  
  Uses PyTorch Lightning for structured training, checkpointing, and logging.
- **Google Cloud Integration:**  
  Includes instructions and support for training on Google Cloud Platform (GCP) using Compute Engine or Vertex AI.

## Architecture

The GSENet architecture consists of:

1. **Initial Convolution:**  
   A 2D convolution fuses the real and imaginary parts of both the beamformed and reference signals.
2. **Encoder Blocks (EBlocks):**  
   Downsample and extract features from the input.
3. **Bottleneck:**  
   A convolution and a Time Dilation Block (TDBlock) process the compressed representation.
4. **Decoder Blocks (DBlocks):**  
   Upsample and combine features using skip connections. To ensure that the skip connection tensors match the upsampled tensor dimensions, a center-cropping utility is applied if needed.
5. **Final Convolution and iSTFT:**  
   Produce a 2-channel complex output that is converted back to the time domain via an inverse STFT.

See [model.py](model.py) and [layers.py](layers.py) for full implementation details.

## Installation

### Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [torchaudio](https://pytorch.org/audio/)
- [google-cloud-storage](https://pypi.org/project/google-cloud-storage/) (optional, for uploading checkpoints to GCS)
- Other dependencies as listed in `requirements.txt` (if provided)

### Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your_username/gsenet.git
cd gsenet
pip install -r requirements.txt
```


## Training


Dataset Structure:

```
data/
  train/
    recording_000010001.WAV/ # directory for one sample
      channel_1.wav       # Beamformed output
      channel_2.wav       # Reference microphone signal
      ...                 # Other reference microphone channels
      ground_truth.wav    # Clean target speech
    recording_000010002.WAV/
      ...
  val/
    recording_000010xxx.WAV/
      channel_1.wav
      channel_2.wav
      ...
      ground_truth.wav
    ...
  test/
    recording_000010xxx.WAV/
      channel_1.wav
      channel_2.wav
      ...
      ground_truth.wav
    ...
```

### Local Training

*Run Training:*
Execute the training script with desired parameters. For example:

```bash
    python train.py --data_path "data" --batch_size 16 --epochs 50 --checkpoint_dir "./checkpoints"
```
This command:
- Loads training and validation data using the custom dataset loader.
- Trains the model for 50 epochs.
- Saves checkpoints locally (and optionally uploads them to GCS if the --gcs_bucket flag is provided)

### Google Cloud Training

Follow these steps to train on GCP:

Upload Your Data to Google Cloud Storage (GCS):
1. Create a GCS bucket (e.g., my-gsenet-data) in the Google Cloud Console.
2. Upload your local data folder:
3. console: `gsutil -m cp -r /local/path/to/data gs://my-gsenet-data/`

Configure Environment:

Option A: Compute Engine / Deep Learning VM
1. Create a VM instance with GPU support (or use a Deep Learning VM image).
2. SSH into the instance, clone your repository, and install dependencies.
3. Mount GCS bucket using GCSFuse or download the data locally.

Option B: Vertex AI Custom Training
1. Containerize your training code using a Dockerfile.
2. Push your container to Google Container Registry.
3. Submit a custom training job via the Vertex AI dashboard.

For the training of this model, I went with option A to train ~30 hours of data.

*Run Training on GCP:*

Update the training command to point to your GCS data path and checkpoint directory. For example:

```bash
python train.py --data_path "/mnt/gcs/data" --batch_size 16 --epochs 50 --checkpoint_dir "./checkpoints" --gcs_bucket "my-gsenet-data"
```

Monitor logs and training progress via SSH (Compute Engine) or the Vertex AI dashboard.

Post-Training:

- Retrieve your model checkpoints from the specified checkpoint directory or GCS bucket.
- Clean up resources to avoid extra costs.

# Reference 

Y. Yang et al., "Guided Speech Enhancement Network," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10096763.Abstract: High quality speech capture has been widely studied for both voice communication and human computer interface reasons. To improve the capture performance, we can often find multi-microphone speech enhancement techniques deployed on various devices. Multi-microphone speech enhancement problem is often decomposed into two decoupled steps: a beamformer that provides spatial filtering and a single-channel speech enhancement model that cleans up the beamformer output. In this work, we propose a speech enhancement solution that takes both the raw microphone and beamformer outputs as the input for an ML model. We devise a simple yet effective training scheme that allows the model to learn from the cues of the beamformer by contrasting the two inputs and greatly boost its capability in spatial rejection, while conducting the general tasks of denoising and dereverberation. The proposed solution takes advantage of classical spatial filtering algorithms instead of competing with them. By design, the beamformer module then could be selected separately and does not require a large amount of data to be optimized for a given form factor, and the network model can be considered as a standalone module which is highly transferable independently from the microphone array. We name the ML module in our solution as GSENet, short for Guided Speech Enhancement Network. We demonstrate its effectiveness on real world data collected on multi-microphone devices in terms of the suppression of noise and interfering speech. keywords: {Training;Performance evaluation;Noise reduction;Speech enhancement;Signal processing;Filtering algorithms;Real-time systems;multi-microphone speech enhancement;speech denoising;neural spatial filtering;beamforming},URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096763&isnumber=10094560


