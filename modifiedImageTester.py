from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import numpy as np
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
import pickle
import PIL
#import matplotlib.pyplot as plt
from ProjectImageHandler import *
from pickleDatasetHandler import *

classes = [0,1,2,3,4,5,6,7,8,9]


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
        correct += (predicted == labels.data.long()).sum()
    return correct / total


if __name__ == "__main__":

    if (len(sys.argv) < 2):
        print("USAGE: %s <pickle file>" % (sys.argv[0]))
        exit()
    else:
        pickle_file = sys.argv[1]

    BATCH_SIZE = 128

    model = loadModel().cuda()
        
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]);

    #The array in the pickle_file is assumed to be of the form:
    # [ [ [train_img][train_labels] ] [ [test_img][test_labels] ] ]
    pd = pickleDataset(pickle_file,train=False,transform=transform)

    loader = torch.utils.data.DataLoader(pd, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=4)
    
    test_acc = get_test_acc(model,loader)
    
    print('accuracy = %.5f' % (test_acc))
