import numpy as np
from model import Model
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch.utils
import random
from skimage import io
from sklearn.cluster import KMeans

class Defence(object):

    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def randomCrop(self,x,cs=4):
        c, w, h = x.shape
        xr = random.randint(0,cs)
        yr = random.randint(0,cs)
        crop = x[:,xr:w+xr-cs,yr:w+yr-cs]
        return crop
    
    
    def cropF(self,batch,minval=-1,maxval=1):
        ts = []
        for index,img in enumerate(batch[0]):
            crop = self.randomCrop(img.numpy())
            crop = np.stack([np.pad(crop[c,:,:],((2,2),(2,2)),mode='constant', 
                                            constant_values=random.uniform(-1, 1)) for c in range(3)], axis=0)
            ts.append(crop)
        t = torch.tensor(ts, dtype=torch.float)
        return [t,batch[1]]
    
    def cropR(self,batch):
        ts = []
        for index,img in enumerate(batch[0]):
            crop = self.randomCrop(img.numpy())
            crop = resize(crop, (3,32,32))
            ts.append(crop)
        t = torch.tensor(ts, dtype=torch.float)
        return [t,batch[1]]
    
    def quant(self,batch,n_colors = 15):
        ts = []
        for index,img in enumerate(batch[0]):
            npimg = (img / 2 + 0.5).numpy()
            original = np.transpose(500*npimg/2+0.5, (1, 2, 0))
            arr = original.reshape((-1, 3))
            kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            lc = centers[labels].reshape(original.shape).astype('uint8')
            lc = lc/255
            lc = np.transpose((lc-0.5)*2, (2, 0, 1))
            ts.append(lc)
        t = torch.tensor(ts, dtype=torch.float)
        return [t,batch[1]]
    
    def testModelNoise(self,testset,ntype='Gaussian',nitteration=100,sigma=0.5):
        correct = 0
        for data,label in testset:
            results = [0 for i in range(10)]
            for i in range(nitteration):
                if ntype == 'Gaussian':
                    noise = np.random.normal(0,sigma,(3,32,32)).reshape(3,32,32)
                elif ntype == 'Laplace':
                    noise = np.random.laplace(0,sigma,(3,32,32)).reshape(3,32,32)
                elif ntype == 'Uniform':
                    noise = np.random.uniform(-1,1,(3,32,32)).reshape(3,32,32)
    
                img = data.numpy()
                noisy = img + noise
                nt = torch.tensor(noisy, dtype=torch.float)
                results[self.model.identify(nt)] += 1
            if np.argmax(results) == label:
                correct += 1
        return 100 * correct / len(testset)
                    
    
    def testModel(self,testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))