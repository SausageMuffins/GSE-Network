import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import EBlock, TDBlock, DBlock

class GSENet(nn.Module):
    """
    A U-Net architecture for speech enhancement using causal convolutions.

    Args:
        None
    """
    def __init__(self):
        super(GSENet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(7, 7), padding=(3, 3)) # maintain causality, maintain freq dimensions

        self.eblock1 = EBlock(Cin=16, Cout=32, Stime=1, Sfreq=2, Dtime=False)
        self.eblock2 = EBlock(Cin=32, Cout=48, Stime=2, Sfreq=2, Dtime=False)
        self.eblock3 = EBlock(Cin=48, Cout=48, Stime=1, Sfreq=2, Dtime=True)
        self.eblock4 = EBlock(Cin=48, Cout=96, Stime=1, Sfreq=2, Dtime=True)
        self.eblock5 = EBlock(Cin=96, Cout=96, Stime=1, Sfreq=2, Dtime=True)
        print("Downsampled")
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=48, kernel_size=(3, 3), padding=(1, 1))
        self.tdblock = TDBlock(Cout=48)

        self.up_conv1 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3, 3), padding=(1, 1))
        self.dblock1 = DBlock(Cin=96, Cout=96, Stime=1, Sfreq=2, Dtime=True)
        self.dblock2 = DBlock(Cin=96, Cout=48, Stime=1, Sfreq=2, Dtime=True)
        self.dblock3 = DBlock(Cin=48, Cout=48, Stime=1, Sfreq=2, Dtime=True)
        self.dblock4 = DBlock(Cin=48, Cout=32, Stime=2, Sfreq=2, Dtime=False)
        self.dblock5 = DBlock(Cin=32, Cout=16, Stime=1, Sfreq=2, Dtime=False)
        self.up_conv2 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=(7, 7), padding=(3, 3))

    def forward(self, ref_mic, beamformed):
        """
        Forward pass for GSENet.

        Args:
            ref_mic (torch.Tensor): Raw reference microphone input, shape [batch, samples].
            beamformed (torch.Tensor): Beamformer output, shape [batch, samples].

        Returns:
            torch.Tensor: Enhanced speech output in time domain, shape [batch, ~samples].
        """
        device = ref_mic.device

        # Compute STFT for both inputs using a 1024-point FFT and 256 hop - this is used in the training process, mentioned in the paper.
        # This yields ~513 frequency bins (0..512).
        win_size = 1024
        hop_len = 256
        window = torch.hann_window(win_size, device=device)

        ref_stft = torch.stft(
            ref_mic, 
            n_fft=win_size, 
            hop_length=hop_len,
            win_length=win_size,
            window=window,
            return_complex=True
        )  # shape: [batch, freq, time]

        beam_stft = torch.stft(
            beamformed,
            n_fft=win_size,
            hop_length=hop_len,
            win_length=win_size,
            window=window,
            return_complex=True
        )

        # Combine real/imag parts into 2 channels each.
        # shape for each: [batch, 2, freq, time]
        ref_real_imag = torch.stack([ref_stft.real, ref_stft.imag], dim=1)
        beam_real_imag = torch.stack([beam_stft.real, beam_stft.imag], dim=1)

        # Concatenate along channel dim => shape: [batch, 4, freq, time]
        x = torch.cat([ref_real_imag, beam_real_imag], dim=1) 
        
        # Initial conv
        x = F.leaky_relu(self.conv1(x), negative_slope=0.3)
        skip1 = x  # Save skip connection
        
        # Encoder blocks
        skip2 = self.eblock1(x)
        skip3 = self.eblock2(skip2)
        skip4 = self.eblock3(skip3)
        skip5 = self.eblock4(skip4)
        skip6 = self.eblock5(skip5)

        print(f"skip1 shape: {skip1.shape}")
        print(f"skip2 shape: {skip2.shape}")
        print(f"skip3 shape: {skip3.shape}")
        print(f"skip4 shape: {skip4.shape}")
        print(f"skip5 shape: {skip5.shape}")
        print(f"skip6 shape: {skip6.shape}")
        
        # Bottleneck conv + TD Block
        x = F.leaky_relu(self.conv2(skip6), negative_slope=0.3)
        x = self.tdblock(x)

        # Upsampling
        x = F.leaky_relu(self.up_conv1(x), negative_slope=0.3)
        x = x + skip6

        x = self.dblock1(x) + skip5
        x = self.dblock2(x) + skip4
        x = self.dblock3(x) + skip3
        x = self.dblock4(x) + skip2
        x = self.dblock5(x) + skip1

        x = self.up_conv2(x)  # shape: [batch, 2, freq, time]

        # Permute to [batch, freq, time, 2] for istft
        x = x.permute(0, 2, 3, 1)

        print(x.shape)
        # Inverse STFT
        x = torch.istft(
            x,
            n_fft=win_size,
            hop_length=hop_len,
            win_length=win_size,
            window=window,
            return_complex=False
        )

        return x