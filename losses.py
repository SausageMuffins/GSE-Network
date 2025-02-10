import torch
import torch.nn as nn

class GSENetSTFTLoss(nn.Module):
    """
    STFT Loss for GSENet:
    - Computes loss in the STFT domain.
    - Uses magnitude loss (L1) and spectral convergence loss.
    - Normalizes STFT magnitudes to avoid gain mismatch issues.
    
    Note: This loss now accepts time-domain waveforms for both the enhanced
    and clean signals, computing their STFTs internally.
    """

    def __init__(self, fft_size=1024, hop_size=256, win_size=1024):
        super(GSENetSTFTLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        # Register the Hann window as a buffer to ensure device compatibility.
        self.register_buffer("window", torch.hann_window(win_size))

    def stft(self, x):
        """
        Computes the STFT of a waveform.
        
        Args:
            x (torch.Tensor): Time-domain waveform of shape [B, samples].
        
        Returns:
            mag (torch.Tensor): Magnitude of the STFT [B, Freq, Time].
            stft (torch.Tensor): Complex STFT tensor [B, Freq, Time].
        """
        # Ensure the window is on the same device as the input
        window = self.window.to(x.device)
        stft = torch.stft(
            x, n_fft=self.fft_size, hop_length=self.hop_size,
            win_length=self.win_size, window=window,
            return_complex=True
        )
        mag = torch.abs(stft)
        return mag, stft

    def forward(self, enhanced_wav, clean):
        """
        Compute the STFT loss between the enhanced waveform and clean speech.
        
        Args:
            enhanced_wav (torch.Tensor): Time-domain enhanced waveform [B, samples].
            clean (torch.Tensor): Time-domain clean speech [B, samples].
        
        Returns:
            torch.Tensor: The combined magnitude (L1) and spectral convergence loss.
        """
        # Compute STFT magnitudes for both signals
        mag_enhanced, _ = self.stft(enhanced_wav)
        mag_clean, _ = self.stft(clean)

        # Normalize magnitudes to prevent gain mismatch issues
        mag_clean_norm = mag_clean / (torch.norm(mag_clean, dim=(1, 2), keepdim=True) + 1e-8)
        mag_enhanced_norm = mag_enhanced / (torch.norm(mag_enhanced, dim=(1, 2), keepdim=True) + 1e-8)

        # Magnitude Loss (L1)
        mag_loss = torch.mean(torch.abs(mag_enhanced_norm - mag_clean_norm))

        # Spectral Convergence Loss
        spec_loss = torch.norm(mag_enhanced - mag_clean, 'fro') / (torch.norm(mag_clean, 'fro') + 1e-8)

        return mag_loss + spec_loss
