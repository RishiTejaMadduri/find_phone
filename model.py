#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import sys
from torch.nn.init import xavier_uniform_, zeros_

# In[5]:


class MyModel(nn.Module):
    def __init__(self, output_size=None, in_channels=3, pretrained=False):
        super(MyModel, self).__init__()

        #Downloading a ResNet 
        pretrained_model = torchvision.models.resnet18(pretrained=False) 
#         print(pretrained_model)

        #Extracting rest of the modules
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        
        self.layer0 = nn.Sequential(
            pretrained_model._modules['conv1'],
            pretrained_model._modules['bn1'],
            self.relu,
            self.maxpool
        )
        # clear memory
        del pretrained_model
        
        #Final convolution to give class dimension output
        self.final = nn.Linear(90112, 2)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
                    
    def forward(self, inputs):
        x = self.layer0(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], -1)
        x = self.final(x)
        return x

