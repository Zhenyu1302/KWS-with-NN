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

# Stack frames to form feature maps
# Feature extraction method compatible with RNN-related or phonetic based non-streaming model
def feature_stack(data):
    # Extract fbank feature and feature maps
    # Stack adjacent 40 frames
    stacked_tuple = []
    for i in range(data.shape[0]):
        fbank_map = torchaudio.compliance.kaldi.fbank(data[i:i+1,:],frame_length=25.0,frame_shift=10.0,num_mel_bins=40)
        small_frame_list = []
        for j in range(len(fbank_map)-40+1):
          small_frame = fbank_map[j:j+40]
          small_frame_list += [small_frame]       
        stacked_tuple += [torch.stack(tuple(small_frame_list))]
    fbank = torch.stack(tuple(stacked_tuple))
    return fbank
