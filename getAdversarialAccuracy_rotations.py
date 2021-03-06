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
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage.interpolation
import matplotlib.pyplot as plt

import pickle

from ProjectImageHandler import ProjectImageHandler as PM

def get_accuracy(dataSet):
    classes = testModel.classes
    correct = 0
    total = len(dataSet)
    pm = PM(mean=(0.1307,),std=(0.3081,))
    model.eval()
    for i in (range(len(dataSet))):
        image,label = dataSet[i]
        imgVar = Variable(image).cuda()
        try:
            output = model(imgVar)
        except:
            print(image)
            exit()
        _,idx = torch.max(output,1)
        classifiedlabel = classes[idx.data[0]]
        if str(classifiedlabel) == str(label):
            correct += 1
    accuracy = (correct / float(total)) * 100
    return(accuracy)

def readNoiseFile(noiseFile):
    noise,labels = pickle.load(open(noiseFile))
    return noise,labels


def add_adversarial_noise(dataSet,noise):
    newDataSet = []
    total = len(dataSet)
    for i in (range(len(dataSet))):
        image,label = dataSet[i]
        advImg = image + noise[i]
        newDataSet.append([advImg,label])
    return newDataSet

def unnormalize_dataset(dataSet,mean=(0.1307,),std=(0.3081,)):
    newDataSet = []
    pm = PM(mean=mean,std=std)
    for i in (range(len(dataSet))):
        image,label = dataSet[i]
        unnormalizedAdvImg = pm.unnormalizeTensor(image)
        newDataSet.append([unnormalizedAdvImg,label])
    return newDataSet

def normalize_dataset(dataSet,mean=(0.1307,),std=(0.3081,)):
    newDataSet = []
    pm = PM(mean=mean,std=std)
    for i in (range(len(dataSet))):
        image,label = dataSet[i]
        unnormalizedAdvImg = pm.normalizeTensor(image)
        newDataSet.append([unnormalizedAdvImg,label])
    return newDataSet


def predictImageLabel(model,image):
    #Note: image = Variable(tensorImage.cuda())
    model.eval()
    output = model(image)
    _,idx = torch.max(output,1)
    label = classes[idx.data[0]]
    return label    

def gaussianBlur_dataset(dataSet,sig=0.5):
    newDataSet = []
    pm = PM()
    for i in (range(len(dataSet))):
        image,label = dataSet[i]
        newImage = torch.FloatTensor(gaussian_filter(image.numpy(), sigma=sig))
        newDataSet.append([newImage,label])
    return newDataSet

def rotate_dataset(dataSet,angle=90):
    newDataSet = []
    pm = PM()
    for i in (range(len(dataSet))):
        image,label = dataSet[i]
        newImage = torch.FloatTensor(scipy.ndimage.interpolation.rotate(image.numpy(), angle=angle,axes=(1,2), reshape=False))
        newDataSet.append([newImage,label])
    return newDataSet


if __name__ == "__main__":

    noiseFile = "adversary_SGD_Testset"
    noise,advLabels = readNoiseFile(noiseFile)

    outFile = "adversarial_rotate.txt"

    model = loadModelMNIST.loadModel().cuda()
    model.eval()

    transform = transforms.Compose( [transforms.ToTensor(), 
            transforms.Normalize( (0.1307,), (0.3081,)) ] )

    angles = []
    accuracies = []

    for i in range(0,361):
        angle = i

        MNIST = torchvision.datasets.MNIST( root='./data', train=False, 
                                            download=True,transform=transform)

        #MNIST = add_adversarial_noise(MNIST,noise)
        MNIST = unnormalize_dataset(MNIST,mean=(0.1307,),std=(0.3081,))
        MNIST = rotate_dataset(MNIST,angle=angle)
        MNIST = normalize_dataset(MNIST,mean=(0.1307,),std=(0.3081,))
        accuracy = get_accuracy(MNIST)

        angles.append(angle)
        accuracies.append(accuracy)
        print("angle = %.1f\taccuracy = %.3f%%" % (angle,accuracy))

    adv_angles = []
    adv_accuracies = []

    for i in range(0,361):
        angle = i

        MNIST = torchvision.datasets.MNIST( root='./data', train=False, 
                                            download=True,transform=transform)

        MNIST = add_adversarial_noise(MNIST,noise)
        MNIST = unnormalize_dataset(MNIST,mean=(0.1307,),std=(0.3081,))
        MNIST = rotate_dataset(MNIST,angle=angle)
        MNIST = normalize_dataset(MNIST,mean=(0.1307,),std=(0.3081,))
        accuracy = get_accuracy(MNIST)

        adv_angles.append(angle)
        adv_accuracies.append(accuracy)
        print("angle = %.1f\taccuracy = %.3f%%" % (angle,accuracy))


    plt.title("Accuracy with Rotations applied to Test and Adversarial Images")
    plt.xlabel("degrees")
    plt.ylabel("accuracy")
    test_handle, = plt.plot(angles,accuracies,label="test")
    adv_handle, = plt.plot(adv_angles,adv_accuracies,label="adversarial")
    plt.legend(handles=[test_handle,adv_handle],loc=1)
    plt.savefig("adversarial_rotate.png")

    f = open(outFile,"w")
    for i in range(len(angles)):
        if angles[i] == adv_angles[i]:
            f.write("%.2f\t%.5f\t%.5f\n" % (angles[i],accuracies[i],adv_accuracies[i]))
        else:
            print("Error: sigma's not equal between regular and adversarial test sets")
    f.close()

