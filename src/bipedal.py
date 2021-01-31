# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 02:45:35 2021

@author: deepu
"""

#read libraries
import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


#create environement
env = gym.make('BipedalWalker-v3')

#start the environment
state = env.reset()

#visualize robot
env.render()

#take a random step
state, reward, game_over, _ = env.step(np.random.random(env.action_space.shape))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.kernel_size_1 = 4
        self.out_channels1 = 9
        
        self.kernel_size_2 = 4
        self.out_channels2 = 2
        
        self.padding       = 2
        self.n_hidden      = 256
        
        
        self.conv1 =  nn.Conv1d(in_channels     = 9,
                                out_channels    = self.out_channels1,
                                kernel_size     = self.kernel_size_1,
                                padding         = self.padding)

        self.conv2 = nn.Conv1d(in_channels  = self.out_channels1,
                               out_channels = self.out_channels2,
                               kernel_size  = self.kernel_size_2,
                               padding      = self.padding)
        
        self.conv2_drop = nn.Dropout()
        self.fc1 = nn.Linear(64, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, 6)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    


network = Net()
network.cuda()

gpu_available = "GPU available?:      " + str(torch.cuda.is_available())
using_cuda    = "Network using cuda?: " + str(next(network.parameters()).is_cuda)

print(gpu_available)
print(using_cuda)

