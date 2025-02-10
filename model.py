import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import EBlock, TDBlock, DBlock

def center_crop(tensor, target_shape):
    _, _, h, w = tensor.shape
    target_h, target_w = target_shape
    dh = (h - target_h) // 2
    dw = (w - target_w) // 2
    return tensor[:, :, dh:dh+target_h, dw:dw+target_w]

class GSENet(nn.Module):
    """
    A U-Net-like architecture for speech enhancement using causal convolutions,
    based on the structure described in the paper. The network uses:
      - An initial 2D conv to combine real+imag parts of ref & beam signals
      - A stack of EBlock modules (downsampling in freq/time) with optional TDBlock
      - A bottleneck TDBlock
      - A stack of DBlock modules (upsampling in freq/time) symmetric to the encoder
      - A final 2D conv to produce 2-channel real+imag output
    """
    def __init__(self):
        super(GSENet, self).__init__()

        # Initial conv: in_channels=4 (real+imag of ref + beam = 2+2), out_channels=16
        # kernel_size=(7,7) => "same-like" padding=(3,3) for both freq/time
        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=(7, 7),
            padding=(3, 3)   # keep freq/time size the same
        )

        # ----- Encoder path -----
        # Each EBlock => downsampling primarily in the frequency dimension,
        # plus optional TDBlock. The arguments are (Cin, Cout, Stime, Sfreq, Dtime).
        # Because EBlock internally does stride=(Sfreq, Stime),
        # we interpret Sfreq => freq stride, Stime => time stride from the paper.
        
        self.eblock1 = EBlock(Cin=16,  Cout=32, Stime=1, Sfreq=2, Dtime=False)
        self.eblock2 = EBlock(Cin=32,  Cout=48, Stime=2, Sfreq=2, Dtime=False)
        self.eblock3 = EBlock(Cin=48,  Cout=48, Stime=1, Sfreq=2, Dtime=True)
        self.eblock4 = EBlock(Cin=48,  Cout=96, Stime=1, Sfreq=2, Dtime=True)
        self.eblock5 = EBlock(Cin=96,  Cout=96, Stime=1, Sfreq=2, Dtime=True)

        # print("Downsampled")

        # Bottleneck: reduce channels to 48 before a TDBlock
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=48,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.tdblock = TDBlock(Cout=48)

        # ----- Decoder path -----
        # We first do a normal conv to get back to 96 channels,
        # then pass through a sequence of DBlocks that each upsamples
        # freq/time in the same pattern as the EBlocks.
        
        self.up_conv1 = nn.Conv2d(
            in_channels=48,
            out_channels=96,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        self.dblock1 = DBlock(Cin=96, Cout=96, Stime=1, Sfreq=2, Dtime=True)
        self.dblock2 = DBlock(Cin=96, Cout=48, Stime=1, Sfreq=2, Dtime=True)
        self.dblock3 = DBlock(Cin=48, Cout=48, Stime=1, Sfreq=2, Dtime=True)
        self.dblock4 = DBlock(Cin=48, Cout=32, Stime=2, Sfreq=2, Dtime=False)
        self.dblock5 = DBlock(Cin=32, Cout=16, Stime=1, Sfreq=2, Dtime=False)

        # Final conv: produce 2 channels (real/imag) from the last upsample
        self.up_conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=2,
            kernel_size=(7, 7),
            padding=(3, 3)
        )

    def forward(self, ref_mic, beamformed):
        """
        Forward pass for GSENet.

        Args:
            ref_mic (torch.Tensor): [batch, samples] reference mic waveform
            beamformed (torch.Tensor): [batch, samples] beamformer output waveform

        Returns:
            torch.Tensor: Enhanced speech in time domain, shape [batch, ~samples].
        """
        device = ref_mic.device

        # ----- 1) STFT -----
        # Paper: 1024-pt FFT (20 ms window), hop=256 (10 ms)
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
        )  # [batch, freq, time]

        beam_stft = torch.stft(
            beamformed,
            n_fft=win_size,
            hop_length=hop_len,
            win_length=win_size,
            window=window,
            return_complex=True
        )

        # ----- 2) Create 4-channel input [B,4,Freq,Time] -----
        # Real/imag of ref => 2 channels, real/imag of beam => 2 channels
        ref_real_imag = torch.stack([ref_stft.real, ref_stft.imag], dim=1)
        beam_real_imag = torch.stack([beam_stft.real, beam_stft.imag], dim=1)
        x = torch.cat([ref_real_imag, beam_real_imag], dim=1)  # [B,4,Freq,Time]

        # ----- 3) Encoder -----
        x = F.leaky_relu(self.conv1(x), negative_slope=0.3)
        skip1 = x  # for the last skip connection

        skip2 = self.eblock1(x)
        skip3 = self.eblock2(skip2)
        skip4 = self.eblock3(skip3)
        skip5 = self.eblock4(skip4)
        skip6 = self.eblock5(skip5)

        # Debug shapes if needed
        # print(f"skip1: {skip1.shape}")
        # print(f"skip2: {skip2.shape}")
        # print(f"skip3: {skip3.shape}")
        # print(f"skip4: {skip4.shape}")
        # print(f"skip5: {skip5.shape}")
        # print(f"skip6: {skip6.shape}")

        # ----- 4) Bottleneck -----
        x = F.leaky_relu(self.conv2(skip6), negative_slope=0.3)
        x = self.tdblock(x)

        # ----- 5) Decoder -----
        x = F.leaky_relu(self.up_conv1(x), negative_slope=0.3)
        # Ensure shapes match for addition with skip6
        if x.shape[2:] != skip6.shape[2:]:
            x = center_crop(x, skip6.shape[2:])
        x = x + skip6

        x = self.dblock1(x)
        if x.shape[2:] != skip5.shape[2:]:
            x = center_crop(x, skip5.shape[2:])
        x = x + skip5

        x = self.dblock2(x)
        if x.shape[2:] != skip4.shape[2:]:
            x = center_crop(x, skip4.shape[2:])
        x = x + skip4

        x = self.dblock3(x)
        if x.shape[2:] != skip3.shape[2:]:
            x = center_crop(x, skip3.shape[2:])
        x = x + skip3

        x = self.dblock4(x)
        if x.shape[2:] != skip2.shape[2:]:
            x = center_crop(x, skip2.shape[2:])
        x = x + skip2

        x = self.dblock5(x)
        if x.shape[2:] != skip1.shape[2:]:
            x = center_crop(x, skip1.shape[2:])
        x = x + skip1


        # Final 2D conv => [B,2,Freq,Time]
        x = self.up_conv2(x)

        # ----- 6) iSTFT -----
        # Permute to [B, Freq, Time, 2] so torch.istft can interpret last dim as real/imag
        x = x.permute(0, 2, 3, 1).contiguous() # contiguous() needed for the last dimension
        x = torch.view_as_complex(x) # change to complex tensor for istft

        # print(f"Final STFT shape before iSTFT: {x.shape}")

        x = torch.istft(
            x,
            n_fft=win_size,
            hop_length=hop_len,
            win_length=win_size,
            window=window,
            return_complex=False
        )
        # x => [B, samples] in time domain

        return x
