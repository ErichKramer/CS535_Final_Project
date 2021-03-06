from __future__ import division
from __future__ import print_function

import sys
import pdb
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
import cPickle as pickle

model_base_url = "http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/"  #prebuilt models
mnist_model = "mnist-b07bb66b.pth"
model_dir = "./"

#The following class for the pretrained MNIST model was borrowed from
#https://github.com/aaron-xichen/pytorch-playground/blob/master/mnist/model.py
class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model= nn.Sequential(layers)

    def gaussNoise(self, image, stddev=.5):
        #generate random noise to overlay on image of dims
        #pdb.set_trace()
        return Variable( torch.randn( image.size()) *stddev) 

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)

def loadModel():
    input_dims = 784
    n_hiddens=[256, 256]
    n_class=10
    model = MLP(input_dims, n_hiddens, n_class)
    state_dict = model_zoo.load_url(model_base_url + mnist_model,model_dir)
    model.load_state_dict(state_dict)
    return model

#function to retrain network after random noise
def retrain(model, epochs=5):
    #disgustingly bake everything in
    model.train()
    opt = optim.SGD(model.parameters(), lr=.01, weight_decay=.0001, momentum=.9)
    transform = transforms.Compose( [transforms.ToTensor(),
                transforms.Normalize( (0.1307,), (0.3081,)) ] )

    data = torchvision.datasets.MNIST( root='./data', train=True, download=True, 
            transform=transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False, num_workers=2)
    for epoch in range(epochs):
        for idx, (data,target) in enumerate(data_loader):
            data, target = Variable(data), Variable(target)
            opt.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            opt.step()

    model.eval()
    return


def loadRandom():
    return pickle.load(open("retrained.pt", "rb") )

    input_dims = 784
    n_hiddens=[256, 256]
    n_class=10

    #use gauss
    def randomForward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        input = input + self.gaussNoise(input)
        return self.model.forward(input)

    MLP.forward = randomForward
    model = MLP(input_dims, n_hiddens, n_class)
    state_dict = model_zoo.load_url(model_base_url + mnist_model,model_dir)
    model.load_state_dict(state_dict)

    retrain(model)
    #model is now loaded training values + random noise, not yet trained

    pdb.set_trace()
    return model






if __name__ == "__main__":
    model = loadModel()
