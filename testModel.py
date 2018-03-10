from __future__ import division
from __future__ import print_function

import sys
import os
import torch
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torchvision import datasets, transforms
from loadModelMNIST import *


def get_test_acc(model,test_loader):
    correct = 0
    total = 0
    model.eval()
    for data in test_loader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    return correct / total


if __name__ == "__main__":
    BATCH_SIZE = 32

    model = loadModel().cuda()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]);
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)
    test_acc = get_test_acc(model,test_loader)     
    print('test_acc %.5f' % (test_acc))
    
