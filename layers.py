# layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TDBlock(nn.Module):
    """
    Time Dilation Block (TDBlock):
    Applies two causal convolutions with increasing dilation in the *time* dimension.
    (We assume the tensor shape is [B, C, Freq, Time], so dilation should affect
    the last dimension if we want to expand the temporal receptive field.)
    
    Args:
        Cout (int): Number of output channels.
    """
    def __init__(self, Cout):
        super(TDBlock, self).__init__()
        
        # We set dilation so that the second dimension (Freq) remains 1
        # and the third dimension (Time) uses 3, then 9.
        # Similarly, padding is adjusted to preserve causal coverage in time.
        # The kernel_size=(3, 3) means: (Freq=3, Time=3).
        
        self.dilated_conv1 = nn.Conv2d(
            in_channels=Cout,
            out_channels=Cout,
            kernel_size=(3, 3),     # (Freq=3, Time=3)
            dilation=(1, 3),        # Dilate only the time dimension
            padding=(1, 3)          # Enough padding in freq=1, time=3 to keep size
        )
        
        self.dilated_conv2 = nn.Conv2d(
            in_channels=Cout,
            out_channels=Cout,
            kernel_size=(3, 3),
            dilation=(1, 9),        # Larger dilation in the time dimension
            padding=(1, 9)
        )

    def forward(self, x):
        x = F.leaky_relu(self.dilated_conv1(x), negative_slope=0.3)
        x = F.leaky_relu(self.dilated_conv2(x), negative_slope=0.3)
        return x


class EBlock(nn.Module):
    """
    Encoder Block (EBlock):
    A block with a skip connection and optional TDBlock.
    The shape is [B, C, Freq, Time]. We map:
        - Sfreq -> stride in the freq dimension (H)
        - Stime -> stride in the time dimension (W)

    Args:
        Cin (int):  Number of input channels.
        Cout (int): Number of output channels.
        Stime (int): Stride factor in the time dimension.
        Sfreq (int): Stride factor in the freq dimension.
        Dtime (bool): Whether to use the TDBlock.
    """
    def __init__(self, Cin, Cout, Stime, Sfreq, Dtime):
        super(EBlock, self).__init__()
        self.use_tdb = Dtime

        # First causal conv
        # kernel_size=(3,3), padding=(1,1) => "same-like" for freq & time
        self.conv1e = nn.Conv2d(
            in_channels=Cin,
            out_channels=Cin,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        # Optional Time Dilation Block
        if self.use_tdb:
            self.time_dilation = TDBlock(Cin)

        # Second convolution with stride
        # IMPORTANT: If our shape is [B, C, Freq, Time], we want:
        #            stride=(Sfreq, Stime)
        kernel_size = (3, 3)
        stride = (Sfreq, Stime)
        padding = (1, 1)

        self.conv2e = nn.Conv2d(
            in_channels=Cin,
            out_channels=Cout,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # Skip conv to match shapes if needed
        self.skip_conv = None
        if (Stime != 1 or Sfreq != 1) or (Cin != Cout):
            self.skip_conv = nn.Conv2d(
                in_channels=Cin,
                out_channels=Cout,
                kernel_size=1,
                stride=stride,
                padding=0
            )

    def forward(self, x):
        skip = x

        # First conv
        x = F.leaky_relu(self.conv1e(x), negative_slope=0.3)

        # Optional TDBlock
        if self.use_tdb:
            x = self.time_dilation(x)

        # Second conv with stride
        x = self.conv2e(x)

        # Skip path
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)

        # Combine
        x = x + skip
        return x

# layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DBlock(nn.Module):
    """
    Decoder Block (DBlock):
    Symmetric to the EBlock, but uses transposed convolutions to upsample
    [B, C, freq, time]. For stride=(Sfreq, Stime):
        - Sfreq upscales the freq dimension,
        - Stime upscales the time dimension.

    Args:
        Cin (int):  Number of input channels.
        Cout (int): Number of output channels.
        Stime (int): Upsampling factor in the time dimension.
        Sfreq (int): Upsampling factor in the freq dimension.
        Dtime (bool): Whether to include the Time Dilation Block.
    """
    def __init__(self, Cin, Cout, Stime, Sfreq, Dtime=False):
        super(DBlock, self).__init__()
        self.use_tdb = Dtime
        
        # First "causal" convolution (no stride), analogous to conv1e in EBlock.
        self.conv1d = nn.Conv2d(
            in_channels=Cin,
            out_channels=Cin,
            kernel_size=(3, 3),
            padding=(1, 1)   # Keep freq/time dimension the same size
        )

        # Optional time dilation
        if self.use_tdb:
            self.time_dilation = TDBlock(Cin)  # TDBlock is assumed defined in this file

        # Transposed conv for main path upsampling
        # We interpret stride=(Sfreq, Stime) since dimension order is [B, C, freq, time].
        kernel_size = (3, 3)
        stride = (Sfreq, Stime)
        padding = (1, 1)        # So that shape roughly doubles when stride=2
        output_padding = (0, 0) # Often zero if we want exact doubling from EBlock

        self.deconv2d = nn.ConvTranspose2d(
            in_channels=Cin,
            out_channels=Cout,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )

        # Skip path: if shape or channel count differs, we match them via a 1×1 transposed conv
        self.skip_conv = None
        if (Cin != Cout) or (Sfreq != 1 or Stime != 1):
            self.skip_conv = nn.ConvTranspose2d(
                in_channels=Cin,
                out_channels=Cout,
                kernel_size=1,
                stride=stride,
                padding=0,
                output_padding=output_padding
            )

    def forward(self, x):
        # Save the skip tensor before transformations
        skip = x
        
        # First causal conv
        x = F.leaky_relu(self.conv1d(x), negative_slope=0.3)

        # Optional TDBlock
        if self.use_tdb:
            x = self.time_dilation(x)

        # Main path upsampling
        x = self.deconv2d(x)

        # Skip path, if needed
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)

        # Combine
        x = x + skip
        print("DBlock output shape:", x.shape)
        return x
