import os
import torch
import torchaudio
from torch.utils.data import Dataset

class GSEDataLoader(Dataset):
    """
    Updated dataset loader for GSENet with the following assumptions:
      - Each sample is a directory (e.g., "recording_000010001.WAV") containing:
          * channel_1.WAV        --> Beamformed output signal
          * channel_2.WAV        --> Reference microphone signal (you can change this choice)
          * channel_3.WAV        --> (Optional; not used here)
          * channel_4.WAV        --> (Optional; not used here)
          * channel_5.WAV        --> (Optional; not used here)
          * ground_truth.WAV     --> Clean target signal
    """
    def __init__(self, data_path, split="train"):
        super().__init__()
        self.data_path = os.path.join(data_path, split)
        # Only include directories (each representing one recording)
        self.recordings = sorted([
            d for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ])

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        rec_dir = os.path.join(self.data_path, self.recordings[idx])
        
        # Define file paths based on your description:
        beam_path = os.path.join(rec_dir, "channel_1.wav")
        # Here we choose channel_2 as the reference mic; adjust if desired.
        ref_path = os.path.join(rec_dir, "channel_2.wav")
        clean_path = os.path.join(rec_dir, "ground_truth.wav")
        
        # Load audio (torchaudio.load returns a tuple (waveform, sample_rate))
        beam_waveform, sr_beam = torchaudio.load(beam_path)
        ref_waveform, sr_ref = torchaudio.load(ref_path)
        clean_waveform, sr_clean = torchaudio.load(clean_path)
        
        # Optionally, check that the sample rates are consistent
        # e.g., assert sr_beam == sr_ref == sr_clean
        
        # Typically, the audio is loaded with shape [channels, time]. 
        # If the files are mono (channel dimension 1), squeeze the extra dimension.
        beam_waveform = beam_waveform.squeeze(0)
        ref_waveform = ref_waveform.squeeze(0)
        clean_waveform = clean_waveform.squeeze(0)
        
        return ref_waveform, beam_waveform, clean_waveform
