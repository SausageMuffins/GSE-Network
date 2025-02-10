import os
import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F

class GSEDataLoader(Dataset):
    """
    Updated dataset loader for GSENet with the following assumptions:
      - Each sample is a directory (e.g., "recording_000010001.WAV") containing:
          * channel_1.wav        --> Beamformed output signal
          * channel_2.wav        --> Reference microphone signal
          * channel_3.wav        --> (Optional; not used here)
          * channel_4.wav        --> (Optional; not used here)
          * channel_5.wav        --> (Optional; not used here)
          * ground_truth.wav     --> Clean target signal
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
        
        # Define file paths (ensure filenames match your actual files: lower/upper-case matters)
        beam_path = os.path.join(rec_dir, "channel_1.wav")
        ref_path  = os.path.join(rec_dir, "channel_2.wav")
        clean_path = os.path.join(rec_dir, "ground_truth.wav")
        
        # Load audio (torchaudio.load returns a tuple (waveform, sample_rate))
        beam_waveform, sr_beam = torchaudio.load(beam_path)
        ref_waveform, sr_ref = torchaudio.load(ref_path)
        clean_waveform, sr_clean = torchaudio.load(clean_path)
        
        # Squeeze the channel dimension if mono (shape becomes [time])
        beam_waveform = beam_waveform.squeeze(0)
        ref_waveform = ref_waveform.squeeze(0)
        clean_waveform = clean_waveform.squeeze(0)
        
        # Find the maximum length among the three signals
        max_len = max(beam_waveform.size(0), ref_waveform.size(0), clean_waveform.size(0))
        
        # Pad each signal to the same length
        beam_waveform  = F.pad(beam_waveform,  (0, max_len - beam_waveform.size(0)))
        ref_waveform   = F.pad(ref_waveform,   (0, max_len - ref_waveform.size(0)))
        clean_waveform = F.pad(clean_waveform, (0, max_len - clean_waveform.size(0)))
        
        return ref_waveform, beam_waveform, clean_waveform


def pad_collate_fn(batch):
    """
    Custom collate function that pads each tensor in the batch
    to the length of the longest tensor.
    Assumes each item in the batch is a tuple of (ref, beam, clean).
    """
    refs, beams, cleans = zip(*batch)
    
    # Find the maximum length in this batch for each waveform type
    max_len_ref = max([t.size(0) for t in refs])
    max_len_beam = max([t.size(0) for t in beams])
    max_len_clean = max([t.size(0) for t in cleans])
    
    # You can choose to pad all to a common max length or pad each separately.
    # Here, we pad each according to its own max length:
    refs_padded = torch.stack([F.pad(t, (0, max_len_ref - t.size(0))) for t in refs])
    beams_padded = torch.stack([F.pad(t, (0, max_len_beam - t.size(0))) for t in beams])
    cleans_padded = torch.stack([F.pad(t, (0, max_len_clean - t.size(0))) for t in cleans])
    
    return refs_padded, beams_padded, cleans_padded
