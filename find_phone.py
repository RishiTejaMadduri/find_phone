#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import model
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


def load_data(name, device):
    image = np.asarray(Image.open(name))
    image = np.transpose(image, (2,0,1))
    image = torch.tensor(image).type(torch.FloatTensor).to(device)
    image = image.unsqueeze(0)
    
    return image


# In[4]:


def normalize(output, shape):
    
    col_i = shape[2]
    row_i = shape[3]
    
    x1 = output[0][0]
    x1 = x1/(col_i - 1.)
    
    y1 = output[0][1]
    y1 = y1/(row_i - 1.)
    
    x = np.array(torch.tensor(x1).cpu())
    y = np.array(torch.tensor(y1).cpu())
    
    print(x, y)

# In[5]:


def main():
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    Model = model.MyModel().to(device)
    state_dict = torch.load( "./best_model.pth")
    Model.load_state_dict(state_dict)
    
    name = sys.argv[1]
    
    image = load_data(name, device)
    output = Model(image)
    
    normalize(output, image.shape)

if __name__ == '__main__':
    main()