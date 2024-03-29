#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import time

from data_loader import SubsetSC
from model import CiC1D_s, CiC3D, CNN_S
from train import train, test, collate_fn

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

# Prepare dataloder
# drop_last is set to true to ensure each batch has same number of samples
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

# Push model to device
# Include all the models in benchmark
# Re-configurate optimizer affect CNN performance, so we configurate all and store them in a list
optimizer = []
MODELS = nn.ModuleList([CNN_S(),CiC1D_s(),CiC3D()])
for model in MODELS:
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
  model.to(device)
  optimizer.append(optim.Adam(model.parameters(), lr=0.001, weight_decay=0))
  
# Train the model
log_interval = 20 # Define how often to print training statistics
n_epoch = 10 # Define the number of epochs
accuracy = np.zeros((len(MODELS),n_epoch))
for epoch in range(1, n_epoch + 1):
    train(MODELS, epoch, log_interval,train_loader,labels,device,optimizer)
    error_rate = test(MODELS, epoch,test_loader,labels,device)
    accuracy[:,epoch-1] = error_rate
x = np.linspace(1,n_epoch,n_epoch)
plt.plot(x,accuracy[0],label='CNN')
plt.plot(x,accuracy[1],label='CiC1D')
plt.plot(x,accuracy[2],label='CiC3D')
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.title('Accuracy of CiC')
plt.legend()
plt.show()
plt.savefig('/srv/scratch/z5282382/KWS_benchmark_'+time.strftime("%Y%m%d%H%M%S")+'.png')
