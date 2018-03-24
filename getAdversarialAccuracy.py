#python2.7, requires torch
from __future__ import division
from __future__ import print_function
import loadModelMNIST 
import testModel
import random
import pdb
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pickle

from ProjectImageHandler import ProjectImageHandler as PM

def getAdversarialAccuracy(dataSet,noise,advLabels):
    classes = testModel.classes
    correct = 0
    advCorrect = 0
    total = len(dataSet)
    model.eval()
    for i in (range(len(dataSet))):
        image,label = dataSet[i]
        advImg = image + noise[i]
        advVar = Variable(advImg).cuda()
        output = model(advVar)
        _,idx = torch.max(output,1)
        classifiedlabel = classes[idx.data[0]]
        if str(classifiedlabel) == str(label):
            correct += 1
        elif str(classifiedlabel) == str(advLabels[i]):
            advCorrect += 1
    accuracy = (correct / float(total)) * 100
    advAcc = (advCorrect / float(total)) * 100
    return(accuracy,advAcc)

def readNoiseFile(noiseFile):
    noise,labels = pickle.load(open(noiseFile))
    return noise,labels

if __name__ == "__main__":

    model = loadModelMNIST.loadRandom().cuda()
    model.eval()

    transform = transforms.Compose( [transforms.ToTensor(), 
            transforms.Normalize( (0.1307,), (0.3081,)) ] )

    MNIST = torchvision.datasets.MNIST( root='./data', train=False, 
                    download=True, transform=transform)

    noiseFile = "adversary_SGD_Testset"
    noise,advLabels = readNoiseFile(noiseFile)
    accuracy,advAcc = getAdversarialAccuracy(MNIST,noise,advLabels)

    print("accuracy = %.3f%%\nadversarial label accuracy = %.3f%%" % (accuracy,advAcc))
