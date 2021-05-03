import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
from matplotlib import pyplot as plt

class Model():
    def __init__(self, path = './cifar_net.pth'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = Net().to(self.device)
        self.PATH = path
        self.f = None
    
    def setF(self,f):
        self.f = f
    
    def loadData(self, bs=4):
        self.bs = bs
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.dataset = torchvision.datasets.CIFAR10(root='./data',
                                                download=True, transform=transform)

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.bs,
                                                  shuffle=True)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=False, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.bs,
                                                 shuffle=False)

        self.classes = tuple(self.dataset.classes)
        
    def trainModel(self, epochs=2):
        print(self.device)
        self.model.to(self.device)
        self.history_train = []
        self.history_test= []
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            l=0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                a = loss.item()
                running_loss += a
                l += a
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            self.history_train.append(l/len(self.trainloader))
            self.history_test.append(self.error_test())
        print('Finished Training')
    
    def trainModelAdversarial(self,attack,epochs=2,alpha=0.5):
        #WARNING: works only with bs1
        print(self.device)
        self.model.to(self.device)
        self.history_train = []
        self.history_test= []
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            l=0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                adv_inputs, adv_outputs = attack.perform_single(inputs,labels) #untargeted attack
                outputs = self.model(inputs)
                loss = alpha*self.criterion(outputs, labels)+(1-alpha)*self.criterion(adv_outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                a = loss.item()
                running_loss += a
                l += a
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            self.history_train.append(l/len(self.trainloader))
            self.history_test.append(self.error_test())
        print('Finished Adversarial Training')
        
    def saveModel(self):
        torch.save(self.model.state_dict(), self.PATH)
        
    def loadModel(self):
        self.model.load_state_dict(torch.load(self.PATH,map_location = self.device))
        self.criterion = nn.CrossEntropyLoss()
        
    def error_test(self):
        print(self.device)
        running_loss = 0.0
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs,labels)
                running_loss += loss.item()
        return running_loss/len(self.testloader)
    
    def testModel(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                if self.f is not None:
                    data = self.f(data)
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        
    def identify(self,x):
        with torch.no_grad():
            x = x[None, ...]
            x = x.to(self.device)
            outputs = self.model(x)
            _, predicted = torch.max(outputs.data, 1)
            return predicted[0]
        

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.fc1 = nn.Linear(128*4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.apool = nn.AvgPool2d((2,2))
        
        self.layers=[]

    def forward(self, x):
        self.layers.clear()
        
        x = self.pool(F.relu(self.conv1(x)))
        self.layers.append(x)                        #took only pooling layers' output for the moment, should probably check the convolution outputs 
        x = self.pool(F.relu(self.conv2(x)))
        self.layers.append(x)
        x = self.pool(F.relu(self.conv3(x)))
        self.layers.append(x)
        x = self.apool(x)
        self.layers.append(x)
        x = x.view(-1, 128*2*2)
        self.layers.append(x)
        x = F.relu(self.fc1(x))
        self.layers.append(x)
        x=self.fc2(x)
        self.layers.append(x)
        return x
    