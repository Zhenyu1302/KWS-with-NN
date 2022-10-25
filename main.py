#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

from data_loader import SubsetSC, collate_fn
from model import CiC1D_s, CiC3D, CNN_S
from train import train, test

# Determine the device type
# It could run on CPU and GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Obtain data and split into two sets
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

# Obatin all the labels within the dataset
labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

# Pack data into batches
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
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

# Push model to device
# Include all the models in benchmark
MODELS = nn.ModuleList([CNN_S(),CiC1D_s(),CiC3D()])
for model in MODELS:
  model.to(device)
  
# Train the model
log_interval = 20
n_epoch = 10
for epoch in range(1, n_epoch + 1):
    train(MODELS, epoch, log_interval,train_loader)
    error_rate = test(MODELS, epoch,test_loader)
x = np.linspace(1,n_epoch,n_epoch)
plt.plot(x,error_rate[0],label='CNN')
plt.plot(x,error_rate[1],label='CiC1D')
plt.plot(x,error_rate[2],label='CiC3D')
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.title('Accuracy of CiC')
plt.legend()
plt.show()
plt.savefig('/srv/scratch/z5282382/KWS_benchmark.png')
