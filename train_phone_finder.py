#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

from logger import AverageMeter
from dataloader import MyDataLoader
import model


# In[14]:

#A Custom loss function that normalizes network output and applies MSE loss to X and Y coordinates individually
def MyLoss(output, target, img_shape):
    
    loss = nn.MSELoss()
    
    #Normalizing logic
    col_i = img_shape[2]
    row_i = img_shape[3]
    
    x1 = output[0][0]
    x1 = x1/(col_i - 1.)
    
    y1 = output[0][1]
    y1 = y1/(row_i - 1.)
    
    x2 = target[0]
    y2 = target[1]
    
    #Error
    error_x = loss(x1, x2)
    error_y = loss(y1, y2)
    
    return error_x, error_y


def main():
    epochs = 20
    lr = 0.001
    momentum = 0.9
    beta = 0.999
    weight_decay = 0.0
    data = sys.argv[1]
    train_set = MyDataLoader(data)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 1, shuffle = False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    Model = model.MyModel().to(device)
    print(Model)
    Model.init_weights()
    
    optim_params = [{'params': Model.parameters(), 'lr': lr}]
    
    optimizer = torch.optim.Adam(optim_params, betas = (momentum, beta), weight_decay = weight_decay)
    
    #class to store the value of loss
    loss = AverageMeter(precision=4)
    best_loss = torch.tensor(np.inf)
    
    #Training loop
    for epoch in range(epochs):
        for i, (image, label) in enumerate(train_loader):
            
            image = image.type(torch.FloatTensor).to(device)
            label = (torch.tensor(label)).type(torch.FloatTensor).to(device)
            output = Model(image)
            loss_x, loss_y = MyLoss(output, label, image.shape)

            curr_loss = (loss_x + loss_y)/2
            loss.update(curr_loss.item(), 1)
            optimizer.zero_grad()
            curr_loss.backward()
            optimizer.step()
            
            print("Epoch: ", epoch, "Loss_X: ", loss_x.item(), "Loss_Y: ", loss_y.item(), "Avg_loss: ", loss.avg)
            
            if epoch%5 == 0:
                if curr_loss<best_loss:
                    best_loss = curr_loss
                    path = "./best_model.pth"
                    torch.save(Model.state_dict(), path)
                    


# In[18]:


if __name__ == '__main__':
    main()

