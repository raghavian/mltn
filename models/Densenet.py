### Adapted from https://github.com/bamos/densenet.pytorch

import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math
import pdb

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        self.nChannels = nChannels
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
#        pdb.set_trace()
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = out.view(x.shape[0],self.nChannels,-1).mean(2)
        out = torch.sigmoid(self.fc(out))
        return out.squeeze()


class BaselineMLP(nn.Module):
    def __init__(self, inCh=128**3, nhid=8, nClasses=1,bn=False):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(inCh,32*nhid)
        self.fc2 = nn.Linear(32*nhid,16*nhid)
        self.fc3 = nn.Linear(16*nhid,8*nhid)
#           self.fc4 = nn.Linear(8*nhid,4*nhid)
#           self.fc5 = nn.Linear(4*nhid,2*nhid)
#           self.fc6 = nn.Linear(2*nhid,nhid)
        self.fc7 = nn.Linear(8*nhid,1)
        self.bn = bn

        if self.bn:
            self.bn1 = nn.BatchNorm1d(32*nhid)
            self.bn2 = nn.BatchNorm1d(16*nhid)
            self.bn3 = nn.BatchNorm1d(8*nhid)
            self.bn4 = nn.BatchNorm1d(4*nhid)
            self.bn5 = nn.BatchNorm1d(2*nhid)
            self.bn6 = nn.BatchNorm1d(nhid)

    def forward(self,x):
        b = x.shape[0]
#           pdb.set_trace()
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)

        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn2(x)

        x = F.relu(self.fc3(x))
        if self.bn:
            x = self.bn3(x)

#           x = F.relu(self.fc4(x))
#           if self.bn:
#               x = self.bn4(x)

#           x = F.relu(self.fc5(x))
#           if self.bn:
#               x = self.bn5(x)

#           x = F.relu(self.fc6(x))
#           if self.bn:
#               x = self.bn6(x)

        x = (self.fc7(x))

        return x.view(b)

class BaselineCNN(nn.Module):
    def __init__(self, inCh=1, nhid=8, nClasses=1,kernel=3,bn=False):
        super(BaselineCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(inCh,nhid,kernel_size=kernel,padding=1) 
        self.conv2 = nn.Conv3d(nhid,2*nhid,kernel_size=kernel,padding=1) 
        self.conv3 = nn.Conv3d(2*nhid,4*nhid,kernel_size=kernel,padding=1) 
        self.conv4 = nn.Conv3d(4*nhid,8*nhid,kernel_size=kernel,padding=1) 
        self.conv5 = nn.Conv3d(8*nhid,16*nhid,kernel_size=kernel,padding=1) 

        self.bn = bn 

        if self.bn:
            self.bn1 = nn.BatchNorm3d(nhid)
            self.bn2 = nn.BatchNorm3d(2*nhid)
            self.bn3 = nn.BatchNorm3d(4*nhid)
            self.bn4 = nn.BatchNorm3d(8*nhid)
            self.bn5 = nn.BatchNorm3d(16*nhid)
            self.bn6 = nn.BatchNorm1d(1)
            self.bn7 = nn.BatchNorm1d(1)


        self.pool = nn.MaxPool3d(2, stride=2)   

        self.fc1 = nn.Linear(64*16*nhid,50)
        self.fc2 = nn.Linear(50,10)
        self.fc3 = nn.Linear(10,1)

    def forward(self,x):
        b = x.shape[0]

#           pdb.set_trace() 
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        if self.bn:
            x = self.bn1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        if self.bn:
            x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        if self.bn:
            x = self.bn3(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)
        if self.bn:
            x = self.bn4(x)

        x = F.relu(self.conv5(x))
        x = self.pool(x)
        if self.bn:
            x = self.bn5(x)

        x = x.view(b,1,-1)
        
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn6(x)
        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn7(x)
        x = (self.fc3(x))

        return x.view(b)
