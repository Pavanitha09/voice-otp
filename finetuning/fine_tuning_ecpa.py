import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from omegaconf import OmegaConf
import nemo
import nemo.collections.asr as nemo_asr
from glob import glob
import os 
import re
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import librosa

class CustomDataset(Dataset):
    def __init__(self, audio_filepaths, labels, featurizer):
        self.audio_filepaths = audio_filepaths
        self.labels = labels
        self.featurizer = featurizer

    def __len__(self):
        return len(self.audio_filepaths)

    def __getitem__(self, idx):
        audio_filepath = self.audio_filepaths[idx]
        label = self.labels[idx]
        # Load and preprocess audio data using your featurizer
        audio, sr = librosa.load(audio_filepath, sr=None)
        features = self.featurizer.get_features(audio,length=1024)
        return features, label

train_audio_files = glob( "D:\VOICE OTP\DataFolder\*\*[A,B]\*.wav" )
train_audio_files.sort()
eval_audio_files = glob("D:\VOICE OTP\DataFolder\*\*[C]\*.wav" )
eval_audio_files.sort()
train_samples, test_samples = [], []
train_labels, test_labels = [], []

for audio_file in train_audio_files:
    train_samples.append(audio_file)
    label = re.split(r'[\\/]',audio_file)[-3]
    train_labels.append(label)

for audio_file in eval_audio_files:
    test_samples.append(audio_file)
    label = re.split(r'[\\/]',audio_file)[-3]
    test_labels.append(label)

Path("checkpoints").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)


featurizer_config = {
    'dither': 0.0,
    'pad_to': 2.0,
    'window': 'hann',
    'window_size': 0.02,
    'window_stride': 0.01,
    'preemph': 0.97,
    'n_fft': 400,
    'sample_rate': 16000,
}

# Define the featurizer
featurizer = nemo_asr.modules.AudioToMelSpectrogramPreprocessor(**featurizer_config)

# Instantiate your custom datasets
train_dataset = CustomDataset(
    audio_filepaths=train_samples,
    labels=train_labels,
    featurizer=featurizer
)

test_dataset = CustomDataset(
    audio_filepaths=test_samples,
    labels=test_labels,
    featurizer=featurizer
)

train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the configuration file
config_path = 'ecapa_tdnn.yaml'
config = OmegaConf.load(config_path)

print(config)
# Set random seed for reproducibility
seed_everything(42)

# Create the model and trainer
model = EncDecSpeakerLabelModel.from_pretrained(model_name="ecapa_tdnn")

# Initialize the NeMo Lightning trainer
trainer = pl.Trainer(
    accelerator='cpu',
    max_epochs=config.trainer.max_epochs,
    accumulate_grad_batches=config.trainer.accumulate_grad_batches,
    gradient_clip_val=config.trainer.gradient_clip_val,
    log_every_n_steps=config.trainer.log_every_n_steps,
    precision=config.trainer.precision,
    fast_dev_run=True,
)

# Fit the model to the data
trainer.fit(model, train_data_loader, test_data_loader)

# Save the fine-tuned model
model.save_to(config.model.save_path)
