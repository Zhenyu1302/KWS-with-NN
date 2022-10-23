import torch
import torchaudio

# Take waveform out and transform into fbank feature map
# Then pack them back to batch
def feature(data):
    # Extract fbank feature and feature maps
    # Could combine the function into collate_fn
    fbank = []
    for i in range(data.shape[0]):
        fbank_map = torchaudio.compliance.kaldi.fbank(data[i:i+1,:],frame_length=25.0,frame_shift=10.0,num_mel_bins=40)
        fbank += [fbank_map]
    fbank = torch.stack(tuple(fbank))
    # Add an extra dimension to compate with CNN input (batch_size, channel, width, height)
    fbank = fbank[None,:]
    fbank = torch.permute(fbank,(1,0,2,3))
    return fbank
