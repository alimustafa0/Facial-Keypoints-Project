## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## output size = (W-F)/S +1 = (224-3)/1 +1 = 222
        # the output Tensor for one image, will have the dimensions: (32, 222, 222)
        # after one pool layer, this becomes (64, 111, 111)
        self.conv1 = nn.Conv2d(1, 64, 3)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout 
        # or batch normalization) to avoid overfitting
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 32 inputs, 64 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (111-3)/1 +1 = 109
        # the output tensor will have dimensions: (128, 109, 109)
        # after another pool layer this becomes (128, 54, 54);
        self.conv2 = nn.Conv2d(64, 128, 3)
        
        # third conv layer: 64 inputs, 128 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output tensor will have dimensions: (256, 52, 52)
        # after another pool layer this becomes (256, 26, 26);
        self.conv3 = nn.Conv2d(128, 256, 3)
        
        # 4th conv layer: 128 inputs, 256 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (26-3)/1 +1 = 24
        # the output tensor will have dimensions: (512, 24, 24)
        # after another pool layer this becomes (512, 12, 12);
        self.conv4 = nn.Conv2d(256, 512, 3)
        
        # 5th conv layer: 256 inputs, 512 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (12-3)/1 +1 = 10
        # the output tensor will have dimensions: (1024, 10, 10)
        # after another pool layer this becomes (1024, 5, 5);
        self.conv5 = nn.Conv2d(512, 1024, 3)
        
        
        # 6th conv layer: 512 inputs, 1024 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (5-3)/1 +1 = 3
        # the output tensor will have dimensions: (2048, 3, 3)
        # after another pool layer this becomes (2048, 1, 1);
        self.conv6 = nn.Conv2d(1024, 2048, 3)
        
        
        # 1024 outputs * the 1*1 map size
        # class torch.nn.Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(2048*1*1, 1000)  # 2048*1*1 = 2048
        
        # class torch.nn.Linear(in_features, out_features, bias=True)
        self.fc2 = nn.Linear(1000, 1000)
        
        # dropout with p=0.3
        # class torch.nn.Dropout(p=0.5, inplace=False)
        # p (float, optional) – probability of an element to be zero-ed.
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # finally, create 136 output channels (for the 136 keypoint x,y coord.)
        self.fc3 = nn.Linear(1000, 136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
