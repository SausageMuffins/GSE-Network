import torch
from model import GSENet

def test_gsenet_forward():
    # 1. Create synthetic input data
    #    Suppose we have batch_size=2 and an input length of 16000 samples (1 second at 16kHz, for example)
    batch_size = 2
    sample_length = 16000
    
    ref_mic = torch.randn(batch_size, sample_length)
    beamformed = torch.randn(batch_size, sample_length)
    
    # 2. Instantiate the model
    model = GSENet()
    
    # 3. Run the model forward pass
    output = model(ref_mic, beamformed)
    
    # 4. Print results (e.g., shape) to verify correctness
    print(f"Output shape: {output.shape}")

# Run the test
if __name__ == "__main__":
    test_gsenet_forward()
