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
import matplotlib.pyplot as plt


class ProjectImageHandler:
    """ this function calls methods from pyTorch and other modules
    to manipulate images"""
    
    def __init__(self,mean=(0.1307,),std=(0.3081,),maxDegrees=25):
        #see http://pytorch.org/docs/master/torchvision/transforms.html for examples
        self.mean = mean
        self.std = std
        self.ToPiLImage = torchvision.transforms.ToPILImage(mode=None)  #object that handles conversion from tensor to pil
        self.ToTensor = torchvision.transforms.ToTensor() #object that converts numpy array or pil to a tensor
        self.Normalize = torchvision.transforms.Normalize(mean,std)  #normalize tensor image with mean and standard deviation
        self.RandomRotation = torchvision.transforms.RandomRotation(maxDegrees,resample=False, expand=False, center=None)
        self.ColorJitter = torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
         
    def convertToPilImage(self,tensorImage):
        pilImage = self.ToPiLImage(tensorImage)
        return pilImage

    def convertToTensorImage(self,pilImage):
        tensorImage = self.ToTensor(pilImage)
        return tensorImage

    def normalizeTensor(self,tensorImage):
        tensorImage = self.Normalize(tensorImage)
        return tensorImage

    def unnormalizeTensor(self,tensorImage):
        for t, m, s in zip(tensorImage, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensorImage

    def saveTensorImage(self,image,fileName):
        torchvision.utils.save_image(image,fileName)

    def savePilImage(self,pilImage,fileName):
        tensorImage = self.convertToTensorImage(pilImage)
        self.saveTensorImage(tensorImage,fileName)

    def randomRotatePil(self,pilImage):
        newPilImage = self.RandomRotation(pilImage)
        return newPilImage

    def randomRotateTensor(self,tensorImage):
        pilImage = self.convertToPilImage(tensorImage)
        pilImage = self.randomRotatePil(pilImage)
        newTensorImage = self.convertToTensorImage(pilImage)
        return newTensorImage

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

    #tensorImage,label = testset[0]
    #ih = ProjectImageHandler()
    #pilImage = ih.convertToPilImage(ih.unnormalizeTensor(tensorImage))
    #print(tensorImage)
    #ih.saveTensorImage(tensorImage,"test1.png")
    #tensorImage2 = ih.ToTensor(pilImage)
    ##tensorImage2 = ih.normalizeTensor(tensorImage2)
    #pilImage2 = ih.randomRotatePil(pilImage)
    #ih.savePilImage(pilImage,"test2.png")
    #ih.savePilImage(pilImage2,"test3.png")
    #print(ih.randomRotateTensor(tensorImage))
