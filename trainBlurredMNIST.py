#!/usr/bin/python

from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from ProjectImageHandler import ProjectImageHandler as PM
import pickle

class MLP(nn.Module):

    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(28 * 28,256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256,256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)        
        self.out = nn.Linear(256,10)

    def forward(self,x):
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.out(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def eval_net(dataloader,model):
    correct = 0
    total = 0
    total_loss = 0
    model.eval() 
    criterion = nn.CrossEntropyLoss(size_average=False)
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.data[0]
    model.train() 
    return total_loss / total, correct / total


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

def gaussianBlur_dataset(dataSet,sig=0.5):
    newDataSet = []
    pm = PM()
    for i in (range(len(dataSet))):
        image,label = dataSet[i]
        newImage = torch.FloatTensor(gaussian_filter(image.numpy(), sigma=sig))
        newDataSet.append([newImage,label])
    return newDataSet

def add_adversarial_noise(dataSet,noise):
    newDataSet = []
    total = len(dataSet)
    for i in (range(len(dataSet))):
        image,label = dataSet[i]
        advImg = image + noise[i]
        newDataSet.append([advImg,label])
    return newDataSet

def readNoiseFile(noiseFile):
    noise,labels = pickle.load(open(noiseFile))
    return noise,labels

def initialize_with_pretrained_model(model,pretrained_model):
    pretrained_dict = torch.load(pretrained_model)
    model_dict = model.state_dict()
    for k in pretrained_dict.keys():
        if k in model_dict and str(model_dict[k].size()) == str(pretrained_dict[k].size()):
            model_dict[k] = pretrained_dict[k]
    model.load_state_dict(model_dict)

def loadModel():
    model = MLP().cuda()
    model.train()
    initialize_with_pretrained_model(model,"bInitial.pth")
    print(model)
    return model

def initial_training_run(noiseFile="adversary_SGD_Testset",BATCH_SIZE=32,MAX_EPOCH=100,learning_rate=0.01,weight_decay=0.0001,momentum=0.9,
                         outFile="blurredMNINT_initial.txt",outImg="blurredMNIST_inital_loss.png",outImg2="blurredMNIST_inital_acc.png",outModel="bInitial.pth"):
    #this function trains the model and tests the adversarial set without applying any gaussian noise

    #the noise will be added to a duplicate test set as a second measure of accuracy
    f = open(outFile,mode="w")
    
    noise,advLabels = readNoiseFile(noiseFile)

    model = MLP().cuda()

    transform = transforms.Compose( [transforms.ToTensor(),
                                    transforms.Normalize( (0.1307,), (0.3081,)) ] )   

    MNIST_TrainSet = torchvision.datasets.MNIST( root='./data', train=True, 
                                        download=True,transform=transform)
    MNIST_TrainLoader = torch.utils.data.DataLoader(MNIST_TrainSet, batch_size=BATCH_SIZE,
                                                    shuffle=True, num_workers=2)    


    MNIST_TestSet = torchvision.datasets.MNIST( root='./data', train=False, 
                                        download=True,transform=transform)
    MNIST_TestLoader = torch.utils.data.DataLoader(MNIST_TestSet, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)

    MNIST_TestSet2 = torchvision.datasets.MNIST( root='./data', train=False, 
                                        download=True,transform=transform)
    MNIST_TestSet2 = add_adversarial_noise(MNIST_TestSet2,noise)
    MNIST_TestLoader2 = torch.utils.data.DataLoader(MNIST_TestSet2, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=weight_decay, momentum=momentum)

    tn_loss = []
    tst_loss = []
    a_loss = []
    tn_acc = []
    tst_acc = []
    a_acc = []

    for epoch in range(MAX_EPOCH):
        
        running_loss = 0.0
        for i, data in enumerate(MNIST_TrainLoader, 0):
            inputs,labels = data
            inputs,labels = Variable(inputs).cuda(),Variable(labels).cuda()
            optimizer.zero_grad()

            #forward + backwards + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 500 == 499:
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(MNIST_TrainLoader,model)
        test_loss, test_acc = eval_net(MNIST_TestLoader,model)
        adv_loss, adv_acc = eval_net(MNIST_TestLoader2,model)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f adv_test_loss: %.5f adv_test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc, adv_loss, adv_acc))
        f.write("%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" % (epoch+1, train_loss, train_acc, test_loss, test_acc, adv_loss, adv_acc))
        tn_loss.append(float(train_loss))
        tst_loss.append(float(test_loss))
        a_loss.append(float(adv_loss))
        tn_acc.append(float(train_acc))
        tst_acc.append(float(test_acc))
        a_acc.append(float(adv_acc))

    epochs = range(1,MAX_EPOCH+1)
    torch.save(model.state_dict(), outModel)
    plt.title("Train and Test Loss of New MNIST Model")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    train_hdl1, = plt.plot(epochs,tn_loss,label="train")
    test_hdl1, = plt.plot(epochs,tst_loss,label="test")
    adv_hdl1, = plt.plot(epochs,a_loss,label="adversarial")
    plt.legend(handles=[train_hdl1,test_hdl1,adv_hdl1],loc=1)
    plt.savefig(outImg)
    plt.clf()

    torch.save(model.state_dict(), outModel)
    plt.title("Train and Test Accuracies of New MNIST Model")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    train_hdl2, = plt.plot(epochs,tn_acc,label="train")
    test_hdl2, = plt.plot(epochs,tst_acc,label="test")
    adv_hdl2, = plt.plot(epochs,a_acc,label="adversarial")
    plt.legend(handles=[train_hdl2,test_hdl2,adv_hdl2],loc=1)
    plt.savefig(outImg2)
    plt.close()
    
    f.close()


def gaussian_training_run(noiseFile="adversary_SGD_Testset",BATCH_SIZE=32,MAX_EPOCH=20,learning_rate=0.01,weight_decay=0.0001,momentum=0.9,
                          outPrefix="blurredMNINT_gaussian",max_sigma=5):

    sigmaTestFile = "%s_gaussData.txt" % (outPrefix)
    sigmaTestLossImg = "%s_sigVsLoss" % (outPrefix)
    sigmaTestAccImg = "%s_sigVsAcc" % (outPrefix)
    tn_sigLoss = []
    tst_sigLoss = []
    a_sigLoss = []
    tn_sigAcc = []
    tst_sigAcc = []
    a_sigAcc = []
    sig_arr = []

    fSig = open(sigmaTestFile,mode="w")
    
    noise,advLabels = readNoiseFile(noiseFile)


    transform = transforms.Compose( [transforms.ToTensor() ] ) 

    transform_adv = transforms.Compose( [transforms.ToTensor(),
                                        transforms.Normalize( (0.1307,), (0.3081,)) ] )

    for i in range(max_sigma*10+1):
        sigma = i * 0.1
        sig_arr.append(sigma)

        tn_loss = []
        tst_loss = []
        a_loss = []
        tn_acc = []
        tst_acc = []
        a_acc = []

        print("SIGMA =" + str(sigma))

        outFile = "%s_%d.txt" % (outPrefix,i)
        outImgLoss = "%s_%d_loss.png" % (outPrefix,i)
        outImgAcc = "%s_%d_acc.png" % (outPrefix,i)
        f = open(outFile,mode="w")
        
        model = MLP().cuda()
    
        MNIST_TrainSet = torchvision.datasets.MNIST( root='./data', train=True, 
                                                     download=True,transform=transform)
        MNIST_TrainSet = gaussianBlur_dataset(MNIST_TrainSet,sig=sigma)
        MNIST_TrainSet = normalize_dataset(MNIST_TrainSet,mean=(0.1307,),std=(0.3081,))
        MNIST_TrainLoader = torch.utils.data.DataLoader(MNIST_TrainSet, batch_size=BATCH_SIZE,
                                                    shuffle=True, num_workers=2)    


        MNIST_TestSet = torchvision.datasets.MNIST( root='./data', train=False, 
                                                    download=True,transform=transform)
        MNIST_TestSet = gaussianBlur_dataset(MNIST_TestSet,sig=sigma)
        MNIST_TestSet = normalize_dataset(MNIST_TestSet,mean=(0.1307,),std=(0.3081,))
        MNIST_TestLoader = torch.utils.data.DataLoader(MNIST_TestSet, batch_size=BATCH_SIZE,
                                                    shuffle=False, num_workers=2)


        MNIST_TestSet2 = torchvision.datasets.MNIST( root='./data', train=False, 
                                                     download=True,transform=transform_adv)
        MNIST_TestSet2 = add_adversarial_noise(MNIST_TestSet2,noise) #noise was created on normalized set
        MNIST_TestSet2 = unnormalize_dataset(MNIST_TestSet2,mean=(0.1307,),std=(0.3081,))
        MNIST_TestSet2 = gaussianBlur_dataset(MNIST_TestSet2,sig=sigma)
        MNIST_TestSet2 = normalize_dataset(MNIST_TestSet2,mean=(0.1307,),std=(0.3081,))
        MNIST_TestLoader2 = torch.utils.data.DataLoader(MNIST_TestSet2, batch_size=BATCH_SIZE,
                                                    shuffle=False, num_workers=2)


        model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=weight_decay, momentum=momentum)
 
        for epoch in range(MAX_EPOCH):
            running_loss = 0.0
            for i, data in enumerate(MNIST_TrainLoader, 0):
                inputs,labels = data
                inputs,labels = Variable(inputs).cuda(),Variable(labels).cuda()
                optimizer.zero_grad()

                #forward + backwards + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]
                if i % 500 == 499:
                    print('    Step: %5d avg_batch_loss: %.5f' %
                          (i + 1, running_loss / 500))
                    running_loss = 0.0
            print('    Finish training this EPOCH, start evaluating...')
            train_loss, train_acc = eval_net(MNIST_TrainLoader,model)
            test_loss, test_acc = eval_net(MNIST_TestLoader,model)
            adv_loss, adv_acc = eval_net(MNIST_TestLoader2,model)
            print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f adv_test_loss: %.5f adv_test_acc %.5f' %
                      (epoch+1, train_loss, train_acc, test_loss, test_acc, adv_loss, adv_acc))       
            f.write("%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" % (epoch+1, train_loss, train_acc, test_loss, test_acc, adv_loss, adv_acc))
            tn_loss.append(float(train_loss))
            tst_loss.append(float(test_loss))
            a_loss.append(float(adv_loss))
            tn_acc.append(float(train_acc))
            tst_acc.append(float(test_acc))
            a_acc.append(float(adv_acc))
            
        tn_sigLoss.append(tn_loss[MAX_EPOCH - 1])
        tst_sigLoss.append(tst_loss[MAX_EPOCH - 1])
        a_sigLoss.append(a_loss[MAX_EPOCH - 1])
        tn_sigAcc.append(tn_acc[MAX_EPOCH - 1])
        tst_sigAcc.append(tst_acc[MAX_EPOCH - 1])
        a_sigAcc.append(a_acc[MAX_EPOCH - 1])

        fSig.write("%.2f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" % (sigma, tn_loss[MAX_EPOCH - 1], tn_acc[MAX_EPOCH - 1],
                                                            tst_loss[MAX_EPOCH - 1], tst_acc[MAX_EPOCH - 1],
                                                            a_loss[MAX_EPOCH - 1], a_acc[MAX_EPOCH - 1]))

        epochs = range(1,MAX_EPOCH+1)
        plt.title("Train and Test Loss of New MNIST Model")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        train_hdl1, = plt.plot(epochs,tn_loss,label="train")
        test_hdl1, = plt.plot(epochs,tst_loss,label="test")
        adv_hdl1, = plt.plot(epochs,a_loss,label="adversarial")
        plt.legend(handles=[train_hdl1,test_hdl1,adv_hdl1],loc=1)
        plt.savefig(outImgLoss)
        plt.clf()

        plt.title("Train and Test Accuracies of New MNIST Model")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        train_hdl2, = plt.plot(epochs,tn_acc,label="train")
        test_hdl2, = plt.plot(epochs,tst_acc,label="test")
        adv_hdl2, = plt.plot(epochs,a_acc,label="adversarial")
        plt.legend(handles=[train_hdl2,test_hdl2,adv_hdl2],loc=1)
        plt.savefig(outImgAcc)
        plt.clf()

    plt.title("Gaussian Effects on Loss of New MNIST Model")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    train_hdl1, = plt.plot(sig_arr,tn_sigLoss,label="train")
    test_hdl1, = plt.plot(sig_arr,tst_sigLoss,label="test")
    adv_hdl1, = plt.plot(sig_arr,a_sigLoss,label="adversarial")
    plt.legend(handles=[train_hdl1,test_hdl1,adv_hdl1],loc=1)
    plt.savefig(sigmaTestLossImg)
    plt.clf()
    
    plt.title("Gaussian Effects on Accuracies of New MNIST Model")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    train_hdl2, = plt.plot(sig_arr,tn_sigAcc,label="train")
    test_hdl2, = plt.plot(sig_arr,tst_sigAcc,label="test")
    adv_hdl2, = plt.plot(sig_arr,a_sigAcc,label="adversarial")
    plt.legend(handles=[train_hdl2,test_hdl2,adv_hdl2],loc=1)
    plt.savefig(sigmaTestAccImg)
    plt.clf()
    plt.close()
        
    fSig.close()

def final_gaussian_run(noiseFile="adversary_SGD_Testset",BATCH_SIZE=32,MAX_EPOCH=100,learning_rate=0.01,weight_decay=0.0001,momentum=0.9,
                         outFile="blurredMNINT_final.txt",outImg="blurredMNIST_final_loss.png",outImg2="blurredMNIST_final_acc.png",outModel="bFinal.pth",sigma=1.0):

    f = open(outFile,mode="w")
    
    noise,advLabels = readNoiseFile(noiseFile)

    model = MLP().cuda()

    transform = transforms.Compose( [transforms.ToTensor() ] ) 

    transform_adv = transforms.Compose( [transforms.ToTensor(),
                                        transforms.Normalize( (0.1307,), (0.3081,)) ] )


    MNIST_TrainSet = torchvision.datasets.MNIST( root='./data', train=True, 
                                                 download=True,transform=transform)
    MNIST_TrainSet = gaussianBlur_dataset(MNIST_TrainSet,sig=sigma)
    MNIST_TrainSet = normalize_dataset(MNIST_TrainSet,mean=(0.1307,),std=(0.3081,))
    MNIST_TrainLoader = torch.utils.data.DataLoader(MNIST_TrainSet, batch_size=BATCH_SIZE,
                                                    shuffle=True, num_workers=2)    


    MNIST_TestSet = torchvision.datasets.MNIST( root='./data', train=False, 
                                                download=True,transform=transform)
    MNIST_TestSet = gaussianBlur_dataset(MNIST_TestSet,sig=sigma)
    MNIST_TestSet = normalize_dataset(MNIST_TestSet,mean=(0.1307,),std=(0.3081,))
    MNIST_TestLoader = torch.utils.data.DataLoader(MNIST_TestSet, batch_size=BATCH_SIZE,
                                                   shuffle=False, num_workers=2)


    MNIST_TestSet2 = torchvision.datasets.MNIST( root='./data', train=False, 
                                                 download=True,transform=transform_adv)
    MNIST_TestSet2 = add_adversarial_noise(MNIST_TestSet2,noise) #noise was created on normalized set
    MNIST_TestSet2 = unnormalize_dataset(MNIST_TestSet2,mean=(0.1307,),std=(0.3081,))
    MNIST_TestSet2 = gaussianBlur_dataset(MNIST_TestSet2,sig=sigma)
    MNIST_TestSet2 = normalize_dataset(MNIST_TestSet2,mean=(0.1307,),std=(0.3081,))
    MNIST_TestLoader2 = torch.utils.data.DataLoader(MNIST_TestSet2, batch_size=BATCH_SIZE,
                                                    shuffle=False, num_workers=2)    
    
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=weight_decay, momentum=momentum)

    tn_loss = []
    tst_loss = []
    a_loss = []
    tn_acc = []
    tst_acc = []
    a_acc = []

    for epoch in range(MAX_EPOCH):
        
        running_loss = 0.0
        for i, data in enumerate(MNIST_TrainLoader, 0):
            inputs,labels = data
            inputs,labels = Variable(inputs).cuda(),Variable(labels).cuda()
            optimizer.zero_grad()

            #forward + backwards + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 500 == 499:
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(MNIST_TrainLoader,model)
        test_loss, test_acc = eval_net(MNIST_TestLoader,model)
        adv_loss, adv_acc = eval_net(MNIST_TestLoader2,model)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f adv_test_loss: %.5f adv_test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc, adv_loss, adv_acc))
        f.write("%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" % (epoch+1, train_loss, train_acc, test_loss, test_acc, adv_loss, adv_acc))
        tn_loss.append(float(train_loss))
        tst_loss.append(float(test_loss))
        a_loss.append(float(adv_loss))
        tn_acc.append(float(train_acc))
        tst_acc.append(float(test_acc))
        a_acc.append(float(adv_acc))

    epochs = range(1,MAX_EPOCH+1)
    torch.save(model.state_dict(), outModel)
    plt.title("Train, Test, and Adversarial Test Loss (Sigma = 1.0)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    train_hdl1, = plt.plot(epochs,tn_loss,label="train")
    test_hdl1, = plt.plot(epochs,tst_loss,label="test")
    adv_hdl1, = plt.plot(epochs,a_loss,label="adversarial")
    plt.legend(handles=[train_hdl1,test_hdl1,adv_hdl1],loc=1)
    plt.savefig(outImg)
    plt.clf()

    torch.save(model.state_dict(), outModel)
    plt.title("Train, Test, and Adversarial Test Accuracy (Sigma = 1.0)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    train_hdl2, = plt.plot(epochs,tn_acc,label="train")
    test_hdl2, = plt.plot(epochs,tst_acc,label="test")
    adv_hdl2, = plt.plot(epochs,a_acc,label="adversarial")
    plt.legend(handles=[train_hdl2,test_hdl2,adv_hdl2],loc=1)
    plt.savefig(outImg2)
    plt.close()
    
    f.close()


def final_gaussian_run_wRotation(noiseFile="adversary_SGD_Testset",BATCH_SIZE=32,MAX_EPOCH=100,learning_rate=0.01,weight_decay=0.0001,momentum=0.9,
                                 outFile="blurredMNINT_final_wRot.txt",outImg="blurredMNIST_final_wRot_loss.png",outImg2="blurredMNIST_final_wRot_acc.png",
                                 outModel="bFinal_wRot.pth",sigma=1.0,deg=25):

    f = open(outFile,mode="w")
    
    noise,advLabels = readNoiseFile(noiseFile)

    model = MLP().cuda()

    transform = transforms.Compose( [transforms.ToTensor() ] ) 

    transform_adv = transforms.Compose( [transforms.ToTensor(),
                                        transforms.Normalize( (0.1307,), (0.3081,)) ] )


    MNIST_TestSet = torchvision.datasets.MNIST( root='./data', train=False, 
                                                download=True,transform=transform)
    MNIST_TestSet = gaussianBlur_dataset(MNIST_TestSet,sig=sigma)
    MNIST_TestSet = normalize_dataset(MNIST_TestSet,mean=(0.1307,),std=(0.3081,))
    MNIST_TestLoader = torch.utils.data.DataLoader(MNIST_TestSet, batch_size=BATCH_SIZE,
                                                   shuffle=False, num_workers=2)


    MNIST_TestSet2 = torchvision.datasets.MNIST( root='./data', train=False, 
                                                 download=True,transform=transform_adv)
    MNIST_TestSet2 = add_adversarial_noise(MNIST_TestSet2,noise) #noise was created on normalized set
    MNIST_TestSet2 = unnormalize_dataset(MNIST_TestSet2,mean=(0.1307,),std=(0.3081,))
    MNIST_TestSet2 = gaussianBlur_dataset(MNIST_TestSet2,sig=sigma)
    MNIST_TestSet2 = normalize_dataset(MNIST_TestSet2,mean=(0.1307,),std=(0.3081,))
    MNIST_TestLoader2 = torch.utils.data.DataLoader(MNIST_TestSet2, batch_size=BATCH_SIZE,
                                                    shuffle=False, num_workers=2)    
    
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=weight_decay, momentum=momentum)

    tn_loss = []
    tst_loss = []
    a_loss = []
    tn_acc = []
    tst_acc = []
    a_acc = []

    for epoch in range(MAX_EPOCH):
        transform_train = transforms.Compose( [transforms.RandomRotation(degrees=deg),transforms.ToTensor() ] )

        MNIST_TrainSet = torchvision.datasets.MNIST( root='./data', train=True, 
                                                     download=True,transform=transform_train)
        MNIST_TrainSet = gaussianBlur_dataset(MNIST_TrainSet,sig=sigma)
        MNIST_TrainSet = normalize_dataset(MNIST_TrainSet,mean=(0.1307,),std=(0.3081,))
        MNIST_TrainLoader = torch.utils.data.DataLoader(MNIST_TrainSet, batch_size=BATCH_SIZE,
                                                        shuffle=True, num_workers=2)    
        
        running_loss = 0.0
        for i, data in enumerate(MNIST_TrainLoader, 0):
            inputs,labels = data
            inputs,labels = Variable(inputs).cuda(),Variable(labels).cuda()
            optimizer.zero_grad()

            #forward + backwards + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 500 == 499:
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(MNIST_TrainLoader,model)
        test_loss, test_acc = eval_net(MNIST_TestLoader,model)
        adv_loss, adv_acc = eval_net(MNIST_TestLoader2,model)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f adv_test_loss: %.5f adv_test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc, adv_loss, adv_acc))
        f.write("%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" % (epoch+1, train_loss, train_acc, test_loss, test_acc, adv_loss, adv_acc))
        tn_loss.append(float(train_loss))
        tst_loss.append(float(test_loss))
        a_loss.append(float(adv_loss))
        tn_acc.append(float(train_acc))
        tst_acc.append(float(test_acc))
        a_acc.append(float(adv_acc))

    epochs = range(1,MAX_EPOCH+1)
    torch.save(model.state_dict(), outModel)
    plt.title("Train, Test, and Adversarial Test Loss (Sigma = 1.0)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    train_hdl1, = plt.plot(epochs,tn_loss,label="train")
    test_hdl1, = plt.plot(epochs,tst_loss,label="test")
    adv_hdl1, = plt.plot(epochs,a_loss,label="adversarial")
    plt.legend(handles=[train_hdl1,test_hdl1,adv_hdl1],loc=1)
    plt.savefig(outImg)
    plt.clf()

    torch.save(model.state_dict(), outModel)
    plt.title("Train, Test, and Adversarial Test Accuracy (Sigma = 1.0)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    train_hdl2, = plt.plot(epochs,tn_acc,label="train")
    test_hdl2, = plt.plot(epochs,tst_acc,label="test")
    adv_hdl2, = plt.plot(epochs,a_acc,label="adversarial")
    plt.legend(handles=[train_hdl2,test_hdl2,adv_hdl2],loc=1)
    plt.savefig(outImg2)
    plt.close()
    
    f.close()

    

if __name__ == "__main__":
    BATCH_SIZE = 32 
    MAX_EPOCH = 30

    learning_rate = 0.01
    weight_decay = 0.0001
    momentum = 0.9
    sigma = 0

    #initial_training_run(noiseFile="adversary_SGD_Testset",BATCH_SIZE=32,MAX_EPOCH=100,learning_rate=0.01,weight_decay=0.0001,momentum=0.9,
    #                    outFile="blurredMNINT_initial.txt",outImg="blurredMNIST_inital_loss.png",outImg2="blurredMNIST_inital_acc.png",outModel="bInitial.pth");


    #gaussian_training_run(noiseFile="adversary_SGD_Testset",BATCH_SIZE=32,MAX_EPOCH=20,learning_rate=0.01,weight_decay=0.0001,momentum=0.9,
    #                      outPrefix="blurredMNINT_gaussian",max_sigma=5)

    #final_gaussian_run(noiseFile="adversary_SGD_Testset",BATCH_SIZE=32,MAX_EPOCH=100,learning_rate=0.01,weight_decay=0.0001,momentum=0.9,
    #                   outFile="blurredMNINT_final.txt",outImg="blurredMNIST_final_loss.png",outImg2="blurredMNIST_final_acc.png",outModel="bFinal.pth",sigma=1.0)

    final_gaussian_run_wRotation(noiseFile="adversary_SGD_Testset",BATCH_SIZE=32,MAX_EPOCH=100,learning_rate=0.01,weight_decay=0.0001,momentum=0.9,
                                 outFile="blurredMNINT_final_wRot.txt",outImg="blurredMNIST_final_wRot_loss.png",outImg2="blurredMNIST_final_wRot_acc.png",
                                 outModel="bFinal_wRot.pth",sigma=1.0,deg=25)



#    exit()

