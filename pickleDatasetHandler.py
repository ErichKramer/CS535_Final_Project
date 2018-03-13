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

class pickleDataset(torch.utils.data.Dataset):
    
    def __init__(self,pickle_file,root_dir="./data",train=False,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        pickle_data = pickle.load(open(pickle_file))
        self.validate_correct_data(pickle_data)
        self.pDataset = []
        i = int(not train)  #index 0 = Train 1 = test
        for j in range(0,len(pickle_data[i])):
            img = PIL.Image.fromarray(pickle_data[i][0][j])
            label = pickle_data[i][1][j]
            self.pDataset.append([img,label])

    def validate_correct_data(self,pickle_data):
        a = np.copy(pickle_data[0][0][0])
        image = PIL.Image.fromarray(pickle_data[0][0][0])
        ih = ProjectImageHandler()
        tensorImage = transforms.ToTensor()(image)
        pilImage = transforms.ToPILImage()(tensorImage)
        b = np.array(pilImage)
        if not (a==b).all():
            print("Error: image data may have a different encoding or something")
            exit()

    def __len__(self):
        return len(self.pDataset)

    def __getitem__(self, idx):
        img,label = self.pDataset[idx]
        if self.transform:
            img = self.transform(img)
        return [img,label]

    
        
if __name__ == "__main__":

    pickle_file  = "data/mnist_blurred_2.p"

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]);

    #The array in the pickle_file is assumed to be of the form:
    # [ [ [train_img][train_labels] ] [ [test_img][test_labels] ] ]
    pd = pickleDataset(pickle_file,train=False,transform=transform)

    sample = pd[0]
    print(np.array(pd.pDataset[0][0]))
    print(pd.pDataset[0][1])
    print(sample[0])
    print(sample[1])

