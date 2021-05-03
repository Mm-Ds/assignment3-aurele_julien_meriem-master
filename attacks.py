from pathlib import Path
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod

from model import Model


class Attack(object):

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def generate_adv(self,x,y,**kwargs):   
       pass
    
    def evaluate_model_under_attack(self, **kwargs):
      """
      rerturns accuracy of the model (ratio of correctly classified examples) on test set under attack
      (when the model is given adversarial examples generated from test set)

      **kwargs: dictionary of parameters to pass to 'generate_adv()', depending on the chosen attack.
      (check generate_adv() docstring in child classes.)
      """
      # Accuracy counter

      correct = 0
      # Loop over all examples in test set
      for data in self.model.testloader:
          if self.model.f is not None:
                data = self.model.f(data)
          data, label = data
          data = data.to(self.model.device)
          label = label.to(self.model.device)

          output = self.model.model(data)

          init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

          # If the initial prediction is wrong, dont bother attacking, just move on
          if init_pred.item() != label.item():
              continue
          
          y = torch.tensor([init_pred.item()]).to(self.model.device) #for the moment only on fixed target class
          x_adv= self.generate_adv(data,y, **kwargs)
          pred = self.model.model(x_adv).max(1, keepdim=True)[1] # get the index of the max log-probability
          if pred.item() == label.item():
              correct += 1

      # Calculate final accuracy 
      acc = correct/float(len(self.model.testloader))
      print("Parameters: {}\tTest Accuracy = {} / {} = {}".format(kwargs, correct, len(self.model.testloader), acc))

      # Return the accuracy and an adversarial example
      return acc

    
    def build_confusion_matrix(self, **kwargs):
      '''
      This function aims at understanding if some classes j are easier to fool the network when 
      performing an attack taking them as target starting from a class i
      returns:
        - C :a matrix of size i, j where:
          C[i,j] = confidence of the fooled network in class j when attacked with target j starting from 
          i prediction

      **kwargs: dictionary of parameters to pass to 'generate_adv()' depending on the chosen attack. 
       (check generate_adv() docstring for the different attacks.)
      '''
      C = np.zeros((10,10))
      counter = np.zeros((10,))
      # Loop over all examples in test set
      for i, data in enumerate(self.model.testloader, 0):
          if i%100==0: print(i, ' examples have been processed.')
          data, label= data[0].to(self.model.device), data[1].to(self.model.device)

          output = self.model.model(data)

          init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
          init_pred = init_pred.item()
          # If the initial prediction is wrong, dont bother attacking, just move on
          if  init_pred!= label.item():
              continue
          counter[init_pred]+=1
          C[init_pred,init_pred] += nn.Softmax(dim=1)(output)[0,init_pred]

          for t in range(10):
              if t!=init_pred:
                  init_confidence_t = nn.Softmax(dim=1)(output)[0,t]
                  target = torch.tensor([t]).to(self.model.device)
                  x_adv = self.generate_adv(data,target, **kwargs)
                  pred = self.model.model(x_adv)
                  final_confidence_t = nn.Softmax(dim=1)(pred)[0,t]
                  C[init_pred,t]+=final_confidence_t
      
      #normalize the matrix with respect to the number of investigated examples
      C=C/counter[:,None]
      return C


class FGSM_Attack(Attack):

    def __init__(self, model):
        super().__init__(model)

    def generate_adv(self, x, y, targeted=False, **kwargs):
        '''
        takes an image x and generates adversarial image using FGSM meth. on x 
        if targeted ==True performs targeted generation, y is targete label
        if targeted ==False performs untargeted generation, y is the true label of x
        **kwargs must be a dictionary of one element: eps
        returns:
            -adversarial example
        '''
        assert len(kwargs)==1
        eps= kwargs['eps']


        x_adv = Variable(x.data, requires_grad=True).to(self.model.device)

        h_adv = self.model.model(x_adv)         # prediction for initial (original) x
        cost = self.model.criterion(h_adv, y)

        self.model.model.zero_grad()
        cost.backward()

        if targeted:
          x_adv = x_adv - eps*x_adv.grad.sign_()
        else:
          x_adv = x_adv + eps*x_adv.grad.sign_()

        x_adv = torch.clamp(x_adv, -1, 1)

        return x_adv
    

class PGD_Attack(Attack):

    def __init__(self, model):
        super().__init__(model)

    def generate_adv(self, x, y, targeted=False, **kwargs):
        '''                                                                                               
        takes an image x and its label y, and generates adversarial image using PGD meth. on x 

        - targeted:True if targeted generation is willed , False else.
        **kwargs : must be a dictionary of:
            -n_iterations: number of iteraions for CW meth.
            -step: stepsize of the PGD
            -norm: 'inf' or 2 for the infinity and L2 norm resp. 
        returns:
            -adversarial example
        '''
        #can we do eps, step, norm, n_iter, targeted= kwargs
        assert len(kwargs)==4

        n_iterations=kwargs['n_iterations']
        eps= kwargs['eps']
        step=kwargs['step']
        norm=kwargs['norm']       

        assert norm in ["inf",1,2]

        x_adv = x.data.to(self.model.device)
        y = y.to(self.model.device)

        for i in range(n_iterations):

            x_adv.requires_grad = True                               
            pred = self.model.model(x_adv)  
            self.model.model.zero_grad()                             
            loss = self.model.criterion(pred, y)                   
            loss.backward()                                                                        
          
            if targeted :                                         #  if targeted attack, perform gradient DESCENT on the loss (the latter was in this case computed using the adversarial target)
                x_adv = x_adv - step*x_adv.grad.sign() 
          
            else:                                                 #  if untargeted attack, perform gradient ASCENT on the loss (was computed using the real target)
                x_adv = x_adv + step*x_adv.grad.sign()         

            #Project back into the L"norm" ball of radius "eps"      
            delta= x_adv - x

            if norm=="inf":
            
                delta = torch.clamp(delta, min=-eps, max=eps)
          
            elif norm==2:
            
                # if delta.norm(norm, dim=(1,2,3)) <= eps:
                #   scaling_factor=eps
                # else:
                #   scaling_factor = delta.norm(norm, dim=1)
                # delta*= eps /scaling_factor 
     
                delta_norm= delta.norm(norm, dim=(1,2,3))
                scaling_factor = min(1., eps/delta_norm)       
                delta *= scaling_factor
          
            else: 
                raise NotImplementedError
          
            x_adv= x+delta
            #Rescale the perturbation values back to [0,1] if necessary
            x_adv = torch.clamp(x_adv,min=-1, max=1).detach()
   
        return x_adv.detach()


class CW_Attack(Attack):

    def __init__(self, model):
        super().__init__(model)
    
    def generate_adv(self, x, y,targeted=False, **kwargs): #  n_iterations=1000, lr=0.001, targeted=False, verbose=True,  #Pas très sure qu'il est corrct de passer les arguments comme cela(entre autres, plus de valeurs par défaut)
        '''                                                                                               
        takes an image x and its label y, and generates adversarial image using Carlini Wagner meth. on x 

        - targeted:True if targeted generation is willed , False else.
        **kwargs : must be a dictionary of:
            -n_iterations: number of iteraions for CW meth.
            -lr: learning rate
            -verbose: True for progress prints, False else.
        returns:
            -adversarial example
        '''
        assert len(kwargs)==3      

        n_iterations= kwargs['n_iterations']
        lr=  kwargs['lr']
        verbose=  kwargs['verbose']

        #shape=tuple(x.size())
        #number of steps for binary search (c value)
        binary_number=9

        #c values
        maxc=1e10                               
        minc=0
        c=1e-3 
        #loss values
        min_loss=1000000                         
        min_loss_img=x                         
        k=0 
        #box constraint
        b_min = -1                               
        b_max = 1
        b_mul=(b_max-b_min)/2.0
        b_plus=(b_min+b_max)/2.0          
        
        x=x.cpu() 
        tlab=Variable(torch.from_numpy(np.eye(10)[y]).cuda().float())
        for binary_index in range(binary_number):
            if verbose:
                print("------------Start {} search, current c is {}------------".format(binary_index,c))

            w = Variable(torch.from_numpy(np.arctanh((x.numpy()-b_plus)/b_mul*0.99999)).float()).cuda()
            w_pert=Variable(torch.zeros_like(w).cuda().float())
            w_pert.requires_grad = True
            optimizer = optim.Adam([w_pert], lr=lr)
            isSuccessfulAttack=False

            for iteration_index in range(1,n_iterations+1):
                optimizer.zero_grad()

                img_new=torch.tanh(w_pert+w)*b_mul+b_plus  
                loss_1= self.loss1_func(w,img_new,b_mul,b_plus)  
                self.model.model.eval()
                output= self.model.model(img_new)                      
                loss_2=c*self.f(output,tlab,targeted)              
                loss=loss_1+loss_2                         
                loss.backward(retain_graph=True)
                optimizer.step() 
                pred_result=output.argmax(1, keepdim=True).item()

                if(targeted):
                    if(min_loss>loss_1 and pred_result==y):
                        flag=False
                        for i in range(20):
                            output= self.model.model(img_new)
                            pred_result=output.argmax(1, keepdim=True).item()
                            if(pred_result!=y):
                                flag=True 
                                break
                        if(flag):
                            continue
                        min_loss=loss_1
                        min_loss_img=img_new
                        if verbose:
                            print('success when loss: {}, pred: {}'.format(min_loss,pred_result))
                        isSuccessfulAttack=True
                else:
                    if(min_loss>loss_1 and pred_result!=y):
                        flag=False
                        for i in range(50):
                            output= self.model.model(img_new)
                            pred_result=output.argmax(1, keepdim=True).item()
                            if(pred_result==y):
                                flag=True  
                                break
                        if(flag):
                            continue
                        min_loss=loss_1
                        min_loss_img=img_new
                        if verbose:
                            print('success when loss: {}, pred: {}'.format(min_loss,pred_result))
                        isSuccessfulAttack=True
            if(isSuccessfulAttack):
                maxc=min(maxc,c)
                if maxc<1e9: 
                    c=(minc+maxc)/2
            else:
                minc=max(minc,c)
                if(maxc<1e9):
                    c=(maxc+minc)/2
                else:
                    c=c*10
        return min_loss_img
    
    
    def loss1_func(self,w,x,d,c):
        return torch.dist(x,(torch.tanh(w)*d+c),p=2)
            
    def f(self,output,tlab,targeted,k=0):
        real= torch.max(output*tlab)
        second=torch.max((1-tlab)*output)
        if(targeted):
            # If targeted, optimize for making the class most likely 
            return torch.max(second - real, torch.Tensor([-k]).cuda())
        else:
            # If untargeted, optimize for making the other classes most likely 
            return torch.max(real - second, torch.Tensor([-k]).cuda())