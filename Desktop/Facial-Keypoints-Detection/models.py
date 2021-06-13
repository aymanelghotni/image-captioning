## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

        
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1=nn.Conv2d(1,32,4)
        self.conv2=nn.Conv2d(32,64,3)
        self.conv3=nn.Conv2d(64,128,2)
        self.conv4=nn.Conv2d(128,256,1)
        self.batchnorm1=nn.BatchNorm2d(32)
        self.batchnorm2=nn.BatchNorm2d(64)
        self.batchnorm3=nn.BatchNorm2d(128)
        self.batchnorm4=nn.BatchNorm2d(256)
        self.batchnorm5=nn.BatchNorm1d(1024)
        self.batchnorm6=nn.BatchNorm1d(1024)
        self.maxpool=nn.MaxPool2d(2,2)
        self.dense1=nn.Linear(256*13*13,1024)
        self.dense2=nn.Linear(1024,136)
        self.dropout1=nn.Dropout(0.1)
        self.dropout2=nn.Dropout(0.2)
        self.dropout3=nn.Dropout(0.3)
        self.dropout4=nn.Dropout(0.4)
        self.dropout5=nn.Dropout(0.5)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x=self.batchnorm1(self.dropout1(self.maxpool(F.relu(self.conv1(x)))))
        x=self.batchnorm2(self.dropout2(self.maxpool(F.relu(self.conv2(x)))))
        x=self.batchnorm3(self.dropout3(self.maxpool(F.relu(self.conv3(x)))))
        x=self.batchnorm4(self.dropout4(self.maxpool(F.relu(self.conv4(x)))))
        x=x.view(x.size(0), -1)
        x=self.batchnorm5(self.dropout5(F.relu(self.dense1(x))))
        x=self.dense2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
