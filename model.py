import torch
import torch.nn as nn
import torch.nn.functional as F

# Normal CNN architecture with small kernel
# It outperforms the one with significantly larger kernel
class CNN_S(nn.Module):
    def __init__(self):
        super(CNN_S,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 64,
                               kernel_size = (20,8),
                               dtype=torch.float)
        self.pool = nn.MaxPool2d((3,1))
        self.conv2 = nn.Conv2d(in_channels = 64,
                               out_channels = 64,
                               kernel_size = (10,4),
                               dtype=torch.float)
        self.fc1 = nn.Linear(64*17*30,32) # The output of CNN layer is flattened before being fed in FC layer
        self.fc2 = nn.Linear(32,128)
        self.fc3 = nn.Linear(128,35) # The num of output equals to the num of classes

    def forward(self,x):
        # Each iteration process a stacked frame and return the result
        # Whole-word based could use the whole frame without using iterations
        # Phonetic based requires CTC or other methods to enable mapping
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1,64*17*30)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Normal CNN architecture with large kernel
class CNN_L(nn.Module):
    def __init__(self):
        super(CNN_L,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 64,
                               kernel_size = (65,8),
                               dtype=torch.float)
        self.pool = nn.MaxPool2d((3,1))
        self.conv2 = nn.Conv2d(in_channels = 64,
                               out_channels = 64,
                               kernel_size = (10,4),
                               dtype=torch.float)
        self.fc1 = nn.Linear(64*2*30,32) # The output of CNN layer is flattened before being fed in FC layer
        self.fc2 = nn.Linear(32,128)
        self.fc3 = nn.Linear(128,35) # The num of output equals to the num of classes

    def forward(self,x):
        # Each iteration process a stacked frame and return the result
        # Whole-word based could use the whole frame without using iterations
        # Phonetic based requires CTC or other methods to enable mapping
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = x.view(-1,64*2*30)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# CiC1D model with small kernel in the 1st layer of each block
# In trial run with 10 epochs, it outperforms the one with larger kernel
class CiC1D_s(nn.Module):
    def __init__(self):
        super(CiC1D_s,self).__init__()
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(20,8),
                                              stride=1),
                                    nn.BatchNorm2d(64))
        self.conv12 = nn.ModuleList([nn.Conv3d(in_channels=1,
                                out_channels=1,
                                kernel_size=(3,1,1),
                                stride=1) for i in range(62)])
        self.batch12 = nn.BatchNorm3d(1)
        self.conv13 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(62,1,1),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.conv21 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(64,15,8),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.batch22 = nn.BatchNorm3d(1)
        self.conv22 = nn.ModuleList([nn.Conv3d(in_channels=1,
                                out_channels=1,
                                kernel_size=(3,1,1),
                                stride=1) for i in range(62)])
        self.conv23 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(62,1,1),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.conv31 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(64,10,8),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.conv32 = nn.ModuleList([nn.Conv3d(in_channels=1,
                                out_channels=1,
                                kernel_size=(3,1,1),
                                stride=1) for i in range(62)])
        self.batch32 = nn.BatchNorm3d(1)
        self.conv33 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(62,1,1),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.lin = nn.Sequential(nn.Linear(64*56*19,64),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Linear(64,35)) 

    def forward(self,x):
        # Block 1
        x = F.relu((self.conv11(x)))
        x = x[None,:]
        x = torch.permute(x,(1,0,2,3,4)) # Take the output channel as an additional dimension
        x_1 = []
        for i in range(62):
          x_1 += [self.conv12[i](x[:,:,i:i+3,:])] # Take adjacent 3 channels for pointwise convolution, or sparse connection and to avoid extra dimension
        x = torch.squeeze(torch.stack(tuple(x_1)),dim=3)
        x = F.relu(self.batch12(x.permute((1,2,0,3,4)))) # Take the channels as an additional dimension
        x = self.conv13(x)
        # Block 2
        x = F.relu(self.conv21(x.permute((0,2,1,3,4))))
        x = torch.permute(x,(0,2,1,3,4))
        x_2 = []
        for j in range(62):
           x_2 += [self.conv22[j](x[:,:,j:j+3,:])]
        x = torch.squeeze(torch.stack(tuple(x_2)),dim=3)
        x = F.relu(self.batch22(x.permute((1,2,0,3,4))))
        x = self.conv23(x)
        # Block 3
        x = F.relu(self.conv31(x.permute((0,2,1,3,4))))
        x = torch.permute(x,(0,2,1,3,4))
        x_3 = []
        for k in range(62):
           x_3 += [self.conv32[k](x[:,:,k:k+3,:])]
        x = torch.squeeze(torch.stack(tuple(x_3)),dim=3)
        x = F.relu(self.batch32(x.permute((1,2,0,3,4))))
        x = self.conv33(x)
        x = torch.squeeze(x)
        x = x.view(-1,64*56*19)
        x = self.lin(x)
        return x

# CiC1D with larger kernel which covers 2/3 of input in time
class CiC1D_l(nn.Module):
    def __init__(self):
        super(CiC1D_l,self).__init__()
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(65,8),
                                              stride=1),
                                    nn.BatchNorm2d(64))
        self.conv12 = nn.ModuleList([nn.Conv3d(in_channels=1,
                                out_channels=1,
                                kernel_size=(3,1,1),
                                stride=1) for i in range(62)])
        self.batch12 = nn.BatchNorm3d(1)
        self.conv13 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(62,1,1),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.conv21 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(64,20,8),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.batch22 = nn.BatchNorm3d(1)
        self.conv22 = nn.ModuleList([nn.Conv3d(in_channels=1,
                                out_channels=1,
                                kernel_size=(3,1,1),
                                stride=1) for i in range(62)])
        self.conv23 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(62,1,1),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.conv31 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(64,10,8),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.conv32 = nn.ModuleList([nn.Conv3d(in_channels=1,
                                out_channels=1,
                                kernel_size=(3,1,1),
                                stride=1) for i in range(62)])
        self.batch32 = nn.BatchNorm3d(1)
        self.conv33 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(62,1,1),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.lin = nn.Sequential(nn.Linear(64*6*19,64),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Linear(64,35)) 

    def forward(self,x):
        # Block 1
        x = F.relu((self.conv11(x)))
        x = x[None,:]
        x = torch.permute(x,(1,0,2,3,4))
        x_1 = []
        for i in range(62):
          x_1 += [self.conv12[i](x[:,:,i:i+3,:])]
        x = torch.squeeze(torch.stack(tuple(x_1)),dim=3)
        x = F.relu(self.batch12(x.permute((1,2,0,3,4))))
        x = self.conv13(x)
        # Block 2
        x = F.relu(self.conv21(x.permute((0,2,1,3,4))))
        x = torch.permute(x,(0,2,1,3,4))
        x_2 = []
        for j in range(62):
           x_2 += [self.conv22[j](x[:,:,j:j+3,:])]
        x = torch.squeeze(torch.stack(tuple(x_2)),dim=3)
        x = F.relu(self.batch22(x.permute((1,2,0,3,4))))
        x = self.conv23(x)
        # Block 3
        x = F.relu(self.conv31(x.permute((0,2,1,3,4))))
        x = torch.permute(x,(0,2,1,3,4))
        x_3 = []
        for k in range(62):
           x_3 += [self.conv32[k](x[:,:,k:k+3,:])]
        x = torch.squeeze(torch.stack(tuple(x_3)),dim=3)
        x = F.relu(self.batch32(x.permute((1,2,0,3,4))))
        x = self.conv33(x)
        x = torch.squeeze(x)
        x = x.view(-1,64*6*19)
        x = self.lin(x)
        return x
   
# CiC3D model with the 2nd layer in each block changed into a 3D kernel
# The 3D kernel still act similar to sparse connection in CiC1D
# The 1st layer of each block follows the same configuration of CiC1D_s
class CiC3D(nn.Module):
    def __init__(self):
        super(CiC3D,self).__init__()
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(20,8),
                                              stride=1),
                                    nn.BatchNorm2d(64))
        self.conv12 = nn.ModuleList([nn.Conv3d(in_channels=1,
                                out_channels=1,
                                kernel_size=(3,1,1),
                                stride=1) for i in range(62)])
        self.batch12 = nn.BatchNorm3d(1)
        self.conv13 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(62,5,5),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.conv21 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(64,15,8),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.batch22 = nn.BatchNorm3d(1)
        self.conv22 = nn.ModuleList([nn.Conv3d(in_channels=1,
                                out_channels=1,
                                kernel_size=(3,1,1),
                                stride=1) for i in range(62)])
        self.conv23 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(62,5,5),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.conv31 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(64,10,8),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.conv32 = nn.ModuleList([nn.Conv3d(in_channels=1,
                                out_channels=1,
                                kernel_size=(3,1,1),
                                stride=1) for i in range(62)])
        self.batch32 = nn.BatchNorm3d(1)
        self.conv33 = nn.Sequential(nn.Conv3d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=(62,5,5),
                                              stride=1),
                                    nn.BatchNorm3d(64))
        self.lin = nn.Sequential(nn.Linear(64*44*7,64),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Linear(64,35)) 

    def forward(self,x):
        # Block 1
        x = F.relu((self.conv11(x)))
        x = x[None,:]
        x = torch.permute(x,(1,0,2,3,4))
        x_1 = []
        for i in range(62):
          x_1 += [self.conv12[i](x[:,:,i:i+3,:])]
        x = torch.squeeze(torch.stack(tuple(x_1)),dim=3)
        x = F.relu(self.batch12(x.permute((1,2,0,3,4))))
        x = self.conv13(x)
        # Block 2
        x = F.relu(self.conv21(x.permute((0,2,1,3,4))))
        x = torch.permute(x,(0,2,1,3,4))
        x_2 = []
        for j in range(62):
           x_2 += [self.conv22[j](x[:,:,j:j+3,:])]
        x = torch.squeeze(torch.stack(tuple(x_2)),dim=3)
        x = F.relu(self.batch22(x.permute((1,2,0,3,4))))
        x = self.conv23(x)
        # Block 3
        x = F.relu(self.conv31(x.permute((0,2,1,3,4))))
        x = torch.permute(x,(0,2,1,3,4))
        x_3 = []
        for k in range(62):
           x_3 += [self.conv32[k](x[:,:,k:k+3,:])]
        x = torch.squeeze(torch.stack(tuple(x_3)),dim=3)
        x = F.relu(self.batch32(x.permute((1,2,0,3,4))))
        x = self.conv33(x)
        x = torch.squeeze(x)
        x = x.view(-1,64*44*7)
        x = self.lin(x)
        return x
