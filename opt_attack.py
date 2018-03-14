#python2.7, requires torch
from __future__ import division
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import loadModelMNIST 
import testModel
import random
import pdb


#need to load in test data

class optAttacker:
    def __init__(self, net=None):

        #default net to MNIST, allow one to be passed in
        if net == None:
            self.net = loadModelMNIST.loadModel().cuda()#?
        else: 
            self.net = net
        
        self.cross = nn.CrossEntropyLoss()
        
        #can putting optim here speed things up?
        #self.r = nn.Parameter( data=torch.zeros(1,784), requires_grad=True)
        #self.optim  = optim.SGD(params=[self.r], lr=.008)#params=[self.r?]


    #expects true_t
    def filterFit(self, x, true_y, goal_y=None, mode=range(10) ):
        #mode is the possible classifications, e.g. output layer width as list
        #init an empty filter
        

        r = nn.Parameter( data=torch.zeros(1,28,28), requires_grad=True)
        x_var = Variable(x) #assumes x is torch tensor
        

        pdb.set_trace()
        pred = torch.max( self.net( x_var.cuda() ).data, 1)
        #return 0 filter if you ask it to fit to current prediction
        if pred == goal_y:
            print("Warning: Goal is equivalent to predicted. This is not expected")
            return r

        if goal_y == None:
            mode.pop(true_y)
            goal_y = random.choice( mode )#list of classes\actual class
        y_tensor_target = Variable( torch.LongTensor( [ goal_y]) )
        

        opt = optim.SGD( params=[r], lr=.008)


        for i in range(1000):
            print("Beginning to optimize filter, iter ", i)
            opt.zero_grad()
            tmp = x_var+r
            tmp = tmp.cuda()
            pdb.set_trace()

            outputs = self.net( tmp[0] ) #net.forward if doesnt work
            #loss is equal to the actual loss against the desired, and the magnitude of r`
            adv_loss = self.cross(outputs, y_tensor_target) + torch.mean(torch.pow( r, 2) )
            adv_loss.backward()
            self.optimizer.step()
            
            pdb.set_trace()
            predicted = torch.max( outputs.data, 1)

            if int(predicted[1]) == goal_y:#grab index and cast to int
                break
        #before return
        pdb.set_trace()
        
        return r

if __name__ == "__main__":

    atk = optAttacker() 

    BATCH_SIZE=32
    #normalize to original mean/ std. dev
    transform = transforms.Compose( [transforms.ToTensor(), 
                    transforms.Normalize( (0.1307,), (0.3081,)) ] )

    MNIST = torchvision.datasets.MNIST( root='./data', train=False, 
                    download=True, transform=transform) 

    MNIST_LOADER = torch.utils.data.DataLoader(MNIST, batch_size=BATCH_SIZE, 
                    shuffle=False, num_workers=2)
    
    test_acc = testModel.get_test_acc( atk.net, MNIST_LOADER)
    print('test_acc %.5f' % (test_acc))

    image,label = MNIST[0]
    
    r = atk.filterFit(image, label)

    pdb.set_trace()


