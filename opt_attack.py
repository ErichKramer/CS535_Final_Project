#python2.7, requires torch
#Written by Erich Kramer

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
import cPickle
from ProjectImageHandler import ProjectImageHandler as PM

#need to load in test data


class optAttacker:

    def __init__(self, net=None, debug=False):

        #default net to MNIST, allow one to be passed in
        if net == None:
            self.net = loadModelMNIST.loadModel().cuda()#?
        else: 
            self.net = net
        
        self.cross = nn.CrossEntropyLoss()
        self.debug = False
        self.count = 0
        #can putting optim here speed things up?
        #self.r = nn.Parameter( data=torch.zeros(1,784), requires_grad=True)
        #self.optim  = optim.SGD(params=[self.r], lr=.008)#params=[self.r?]



    #mode is the possible classifications, e.g. output layer width as list
    #expects true_t
    def filterFit(self, x, true_y, goal_y=None, mode=range(10), force=False, iters=1000 ):
        #get goal, confirm its not predicted before running
        if goal_y == None:
            mode.pop(true_y)
            goal_y = random.choice( mode )#list of classes\actual class

        r = nn.Parameter( data=torch.zeros(1,28,28).cuda(), requires_grad=True)
        x_var = Variable(x.cuda()) #assumes x is torch tensor
        y_tensor_target = Variable( torch.LongTensor( [ goal_y]).cuda() )

        #return 0 filter if you ask it to fit to current prediction
        pred = torch.max( self.net( x_var.cuda() ).data, 1)
        if pred == goal_y:
            if self.debug:
                print("Warning: Goal is equivalent to predicted. This is not expected")
            return r


        opt = optim.SGD( params=[r], lr=.008)
        self.optimizer = opt

        for i in range(iters):
            opt.zero_grad()
            tmp = x_var+r
            #tmp = tmp.cuda()

            outputs = self.net( tmp ) #net.forward if doesnt work
            #loss is equal to the actual loss against the desired, and the magnitude of r
            adv_loss = self.cross(outputs, y_tensor_target) + torch.mean(torch.pow( r, 2) )
            adv_loss.backward()
            self.optimizer.step()

            #check to see if we should stop now
            predicted = torch.max( self.net( x_var ).data, 1)
            if (not force) and (int(predicted[1]) == goal_y):#grab index and cast to int
                if self.debug:
                    print("Filter optimized for {} iterations".format(i) )
                break
        return r.data.cpu()
        
    #data must be of form images,labels
    def attackSet( self, data, mapTo=None ):

        #create list of goals if nothing is given
        if mapTo == None:
            mapTo = []
            #get list of all possible classes, remove repeats using set
            classes = list(set(d[1] for d in data ))#ew ew ew ew ew
            #pull a random from a list containing all but real value
            for _,label in data:
                classes.remove(label)
                mapTo.append( random.choice(classes) )
                classes.append(label)

        #begin to build a list of adversarial filters
        adversarial_filters = []
        for (image,label),goal in zip(data, mapTo):
            adversarial_filters.append( self.filterFit( image, label, goal ) )
            #pdb.set_trace()
        #return mapTo incase it was generated here
        return adversarial_filters, mapTo

    #only for MNIST currently
    def displayImage(self, example, r=None, f=None):
        image,_ = example#example is tuple of image,label
        if r == None:
            adversary = image
        else:
            adversary = image+r


        #manip.unnormalizeTensor(adversary)
        if f == None:
            plt.imshow( adversary.numpy()[0], cmap='gray')
            plt.show()
        else:
            plt.imsave( f, adversary.numpy()[0], cmap='gray')
            self.count +=1

if __name__ == "__main__":


    stochastic = loadModelMNIST.loadRandom().cuda()

    atk = optAttacker(net=stochastic) 

    BATCH_SIZE=32
    #normalize to original mean/ std. dev
    transform = transforms.Compose( [transforms.ToTensor(), 
                    transforms.Normalize( (0.1307,), (0.3081,)) ] )

    MNIST = torchvision.datasets.MNIST( root='./data', train=False, 
                    download=True, transform=transform) 

    MNIST_LOADER = torch.utils.data.DataLoader(MNIST, batch_size=BATCH_SIZE, 
                    shuffle=False, num_workers=2)
    
    image,label = MNIST[100]#pull random
    
    r_set, new_class= atk.attackSet(MNIST)

    pdb.set_trace()
    pickle.dump( (r_set, new_class), open("adversary_SGD_Testset", "wb"))


    #r = atk.filterFit(image, label, 1)

    atk.displayImage( MNIST[0], r)
    
    #numpy array is (1,28,28), imshow wants 2D
    #pdb.set_trace()




