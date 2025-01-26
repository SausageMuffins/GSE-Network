import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE:
# The inputs are spectrograms with time (horizontal) and frequency (vertical) dimensions.
# Padding: 
#   - There is padding to account for causality in the temporal dimension,
#     ensuring convolutions over t-2, t-1, and t only.
#   - There is also padding to maintain frequency dimensions in the convolutions.

class TDBlock(nn.Module):
    """
    Time Dilation Block (TDblock):
    Sequential block containing two convolutional layers with different dilation factors.
    
    Args:
        Cout (int): Number of output channels of the previous layer.
    """
    def __init__(self, Cout):
        super(TDBlock, self).__init__()
        self.dilated_conv1 = nn.Conv2d(
            in_channels=Cout,
            out_channels=Cout,
            kernel_size=(3, 3),
            dilation=(3, 1),
            padding=(3, 1)  # Adjust for causal coverage in time
        )
        
        self.dilated_conv2 = nn.Conv2d(
            in_channels=Cout,
            out_channels=Cout,
            kernel_size=(3, 3),
            dilation=(9, 1),
            padding=(9, 1)  # Adjust for causal coverage in time
        )

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
        Stime (int): Stride in the time dimension.
        Sfreq (int): Stride in the frequency dimension.
        Dtime (bool): Whether to use the Time Dilation Block.
    """
    def __init__(self, Cin, Cout, Stime, Sfreq, Dtime):
        super(EBlock, self).__init__()
        self.use_tdb = Dtime

        # First causal convolutional layer
        # padding=(1,1) -> keeps shape in both time and frequency
        self.conv1e = nn.Conv2d(
            in_channels=Cin,
            out_channels=Cin,
            kernel_size=(3, 3),
            padding=(1, 1)  # for causal-like padding
        )

        # Optional Time Dilation Block
        if self.use_tdb:
            self.time_dilation = TDBlock(Cin)

        # Second convolution with stride
        # Use (3,3) for simpler "same-like" shape (common in U-Net).
        kernel_size = (3, 3)
        stride = (Stime, Sfreq)
        padding = (1, 1)

        self.conv2e = nn.Conv2d(
            in_channels=Cin,
            out_channels=Cout,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # Skip-conv with the same stride => ensures output shape matches the main path
        self.skip_conv = None
        if (Stime != 1 or Sfreq != 1) or (Cin != Cout):
            self.skip_conv = nn.Conv2d(
                in_channels=Cin,
                out_channels=Cout,
                kernel_size=1,        # reduce channels only
                stride=stride,        # same stride
                padding=0
            )

    def forward(self, x):
        skip = x

        # First convolution
        x = F.leaky_relu(self.conv1e(x), negative_slope=0.3)

        # Optional Time Dilation
        if self.use_tdb:
            x = self.time_dilation(x)

        # Second convolution with stride
        x = self.conv2e(x)

        # Skip path
        if self.skip_conv is not None:
            skip = self.skip_conv(skip) # 1x1 conv to adjust shapes to match x

        # Add skip connection
        x = x + skip
        return x

class DBlock(nn.Module):
    """
    Decoder Block (DBlock):
    A block symmetric to the Encoder Block (EBlock) with a skip connection and 
    optional time dilation block.

    Args:
        Cin (int): Number of input channels.
        Cout (int): Number of output channels.
        Stime (int): Temporal stride (used to upsample time dimension).
        Sfreq (int): Frequency stride (used to upsample frequency dimension).
        Dtime (bool): Whether to include the Time Dilation Block.
    """
    def __init__(self, Cin, Cout, Stime, Sfreq, Dtime):
        super(DBlock, self).__init__()
        self.use_tdb = Dtime
        # First "causal" convolution
        self.conv1d = nn.Conv2d(
            in_channels=Cin,
            out_channels=Cin,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        # Optional time dilation
        if self.use_tdb:
            from layers import TDBlock
            self.time_dilation = TDBlock(Cin)

        # Main path: transposed convolution for upsampling
        kernel_size = (3, 3)
        stride = (Stime, Sfreq)
        padding = (1, 1)

        t_pad = 0
        f_pad = Sfreq - 1 if Sfreq > 1 else 0
        
        self.output_padding_main = (t_pad, f_pad)

        self.deconv2d = nn.ConvTranspose2d(
            in_channels=Cin,
            out_channels=Cout,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=self.output_padding_main
        )

        # Skip path. If we need upsampling or channel fix, we do a transposed conv:
        self.skip_conv = None
        if (Stime != 1 or Sfreq != 1) or (Cin != Cout):
            self.output_padding_skip = (t_pad, f_pad)
            self.skip_conv = nn.ConvTranspose2d(
                in_channels=Cin,
                out_channels=Cout,
                kernel_size=1,
                stride=stride,
                padding=0,
                output_padding=self.output_padding_skip
            )

    def forward(self, x):
        skip = x
        x = F.leaky_relu(self.conv1d(x), negative_slope=0.3)

        if self.use_tdb:
            x = self.time_dilation(x)

        # Upsample main path
        x = self.deconv2d(x)

        # Upsample skip if needed
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)

        # Add skip
        x = x + skip
        print("DBlock output shape:", x.shape)
        return x
