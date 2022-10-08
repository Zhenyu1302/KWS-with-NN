import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import matplotlib.pyplot as plt
import numpy as np
from torchaudio.datasets import SPEECHCOMMANDS
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

# Download and create training and testing split of the data. 
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

# Extract the keyword labels
labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))

def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch

def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]
    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    # Concatenate label indexes along a new dimension
    targets = torch.stack(targets)
    return tensors, targets

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
  
batch_size = 256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 64,
                               kernel_size = (8,20),
                               dtype=torch.float)
        self.pool = nn.MaxPool2d((3,1))
        self.conv2 = nn.Conv2d(in_channels = 64,
                               out_channels = 64,
                               kernel_size = (4,10),
                               dtype=torch.float)
        self.fc1 = nn.Linear(64*27*12,32) # The output of CNN layer is flattened before being fed in FC layer
        self.fc2 = nn.Linear(32,128)
        self.fc3 = nn.Linear(128,35) # The num of output equals to the num of classes

    def forward(self,x):
        # Each iteration process a stacked frame and return the result
        # Whole-word based could use the whole frame without using iterations
        # Phonetic based requires CTC or other methods to enable mapping
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1,64*27*12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = CNN()
print(model.parameters)
n = count_parameters(model)
print("Number of parameters: %s" % n)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply features and model on whole batch directly on device
        data = torch.squeeze(data)
        fbank_features = feature(data)
        output = model(fbank_features)
        #print(output)
        #print(target)
        #print(target.shape)

        # Loss and backward propagation
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

       # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            
def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = torch.squeeze(data)
        fbank_features = feature(data)
        output = model(fbank_features)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        #pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")
    error_rate = correct / len(test_loader.dataset)
    return error_rate
  
log_interval = 20
n_epoch = 30

# The transform needs to live on the same device as the model and the data.
#transform = transform.to(device)
#with tqdm(total=n_epoch) as pbar:
Rate = []
for epoch in range(1, n_epoch + 1):
    train(model, epoch, log_interval)
    error_rate = test(model, epoch)
    Rate.append(error_rate)
x = np.linspace(1,30,30)
plt.plot(x,Rate)
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.title('Accuracy of CNN')
plt.show()
