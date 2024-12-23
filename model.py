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

        # Initial Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(7, 7), padding=(3, 3))

        # Downsampling Segment
        self.eblock1 = EBlock(Cin=16, Cout=32, Stime=1, Sfreq=2, Dtime=False)
        self.eblock2 = EBlock(Cin=32, Cout=48, Stime=2, Sfreq=2, Dtime=False)
        self.eblock3 = EBlock(Cin=48, Cout=48, Stime=1, Sfreq=2, Dtime=True)
        self.eblock4 = EBlock(Cin=48, Cout=96, Stime=1, Sfreq=2, Dtime=True)
        self.eblock5 = EBlock(Cin=96, Cout=96, Stime=1, Sfreq=2, Dtime=True)

        # Post-downsampling layers
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=48, kernel_size=(3, 3), padding=(1, 1))
        self.tdblock = TDBlock(Cout=48)

        # Upsampling Segment
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
            ref_mic (torch.Tensor): Raw reference microphone input.
            beamformed (torch.Tensor): Beamformer output.

        Returns:
            torch.Tensor: Enhanced speech output.
        """
        # Compute STFT for both inputs
        ref_stft = torch.stft(ref_mic, n_fft=320, hop_length=150, win_length=320, window=torch.hann_window(320), return_complex=True)  # Shape: [batch, freq, time]
        beam_stft = torch.stft(beamformed, n_fft=320, hop_length=150, win_length=320, window=torch.hann_window(320), return_complex=True)

        # Combine real and imaginary components into 2-channel representation
        ref_real_imag = torch.stack([ref_stft.real, ref_stft.imag], dim=1)  # Shape: [batch, 2, freq, time]
        beam_real_imag = torch.stack([beam_stft.real, beam_stft.imag], dim=1)  # Shape: [batch, 2, freq, time]

        # Concatenate STFT inputs along channel dimension
        x = torch.cat([ref_real_imag, beam_real_imag], dim=1)  # Shape: [batch, 4, freq, time]

        # Apply initial convolution
        x = F.leaky_relu(self.conv1(x), negative_slope=0.3)
        skip1 = x  # Save skip connection after conv1

        # Sequential EBlocks
        skip2 = self.eblock1(x)
        skip3 = self.eblock2(skip2)
        skip4 = self.eblock3(skip3)
        skip5 = self.eblock4(skip4)
        skip6 = self.eblock5(skip5)

        # Post-downsampling conv and Time Dilation Block
        x = F.leaky_relu(self.conv2(skip6), negative_slope=0.3)
        x = self.tdblock(x)

        # Upsampling Segment
        x = F.leaky_relu(self.up_conv1(x), negative_slope=0.3)
        x = x + skip6  # Add skip connection from eblock5

        x = self.dblock1(x) + skip5  # Add skip connection from eblock4
        x = self.dblock2(x) + skip4  # Add skip connection from eblock3
        x = self.dblock3(x) + skip3  # Add skip connection from eblock2
        x = self.dblock4(x) + skip2  # Add skip connection from eblock1

        x = self.dblock5(x) + skip1  # Add skip connection from initial conv
        x = self.up_conv2(x)  # Final output

        # Inverse STFT
        x = torch.istft(
            x, n_fft=320, hop_length=150, win_length=320, window=torch.hann_window(320), return_complex=False
        )

        return x
