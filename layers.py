import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE:
# The inputs are spectrograms with time (horizontal) and frequency (vertical) dimensions.
# Padding: 
    # - there is padding to account for causality in the temporal dimension - ensuring convolutions over t-2, t-1 and t only.
    # - there is also padding to maintain frequency dimensions in the convolutions.

class TDBlock(nn.Module):
    """
    Time Dilation Block (TDblock):
    Sequential block containing two convolutional layers with different dilation factors.
    
    Args:
        Cout (int): Number of output channels of the previous layer.
    """
    def __init__(self, Cout):
        super(TDBlock, self).__init__()
        self.dilated_conv1 = nn.Conv2d(in_channels=Cout, out_channels=Cout, kernel_size=(3, 3), dilation=(3, 1), padding=(2, 1))  # Causal padding
        self.dilated_conv2 = nn.Conv2d(in_channels=Cout, out_channels=Cout, kernel_size=(3, 3), dilation=(9, 1), padding=(8, 1))  # Causal padding

    def forward(self, x):
        x = F.leaky_relu(self.dilated_conv1(x), negative_slope=0.3)
        x = F.leaky_relu(self.dilated_conv2(x), negative_slope=0.3)
        return x

class EBlock(nn.Module):
    """
    Encoder Block (EBlock):
    A block with a skip connection and optional time dilation block.

    Args:
        Cin (int): Number of input channels.
        Cout (int): Number of output channels.
        Stime (int): Temporal stride.
        Sfreq (int): Frequency stride.
        Dtime (bool): Whether to include the Time Dilation Block.
    """
    def __init__(self, Cin, Cout, Stime, Sfreq, Dtime):
        super(EBlock, self).__init__()
        self.use_tdb = Dtime

        # First causal convolutional layer
        self.conv1 = nn.Conv2d(in_channels=Cin, out_channels=Cin, kernel_size=(3, 3), padding=(2, 1))  # Causal padding

        # Time Dilation Block
        if self.use_tdb:
            self.time_dilation = TDBlock(Cin)

        # Second causal convolutional layer with stride
        self.conv2 = nn.Conv2d(
            in_channels=Cin,
            out_channels=Cout,
            kernel_size=(max(3, 2 * Stime), max(3, 2 * Sfreq)),
            stride=(Stime, Sfreq),
            padding=(max(3, 2 * Stime) - 1, max(3, 2 * Sfreq) - 1),  # Causal padding
        )

    def forward(self, x):
        # Skip connection input
        skip = x

        # First convolution
        x = F.leaky_relu(self.conv1(x), negative_slope=0.3)

        # Time Dilation Block
        if self.use_tdb:
            x = self.time_dilation(x)

        # Second convolution with stride
        x = self.conv2(x)

        # Add skip connection
        x += skip

        return x

class DBlock(nn.Module):
    """
    Decoder Block (DBlock):
    A block symmetric to the Encoder Block (EBlock) with a skip connection and optional time dilation block.

    Args:
        Cin (int): Number of input channels.
        Cout (int): Number of output channels.
        Stime (int): Temporal stride.
        Sfreq (int): Frequency stride.
        Dtime (bool): Whether to include the Time Dilation Block.
    """
    def __init__(self, Cin, Cout, Stime, Sfreq, Dtime):
        super(DBlock, self).__init__()
        self.use_tdb = Dtime

        # First causal convolutional layer with stride
        self.conv1 = nn.Conv2d(
            in_channels=Cin,
            out_channels=Cout,
            kernel_size=(max(3, 2 * Stime), max(3, 2 * Sfreq)),
            stride=(Stime, Sfreq),
            padding=(max(3, 2 * Stime) - 1, max(3, 2 * Sfreq) - 1),  # Causal padding - maintain causality (current and past time frames) + dimensionality (freq)
        )

        # Time Dilation Block
        if self.use_tdb:
            self.time_dilation = TDBlock(Cout)

        # Second causal convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=Cout, out_channels=Cout, kernel_size=(3, 3), padding=(2, 1))  # Causal padding

    def forward(self, x):
        # Skip connection input
        skip = x

        # First convolution with stride
        x = F.leaky_relu(self.conv1(x), negative_slope=0.3)

        # Time Dilation Block
        if self.use_tdb:
            x = self.time_dilation(x)

        # Second convolution
        x = self.conv2(x)

        # Add skip connection
        x += skip

        return x