import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from numpy import genfromtxt
from torch.autograd import Variable
from torch.nn import Embedding
from torch import zeros, from_numpy, Tensor, LongTensor, FloatTensor
from argparse import ArgumentParser
from helper import *

"""
Embedding_Combination_Layer:
1) __init__ function: creates object of class HighwayCombination
   INPUT for __init__: config
2) forward function:
   INPUTS: word_embeddings,character_embeddings (ie. output of Embedding_Layer_1)
   character_embeddings : ( N x W x Dim1) where N is batch size, W : sequence length
   word_embeddings:  ( N x W x Dim2) where N is batch size, m : sequence length 
   OUTPUT: combined tensor of word_embeddings and character_embeddings : ( N x W x Dim1+Dim2) where N is batch size, m : sequence length 

"""

class Embedding_Combination_Layer(nn.Module):
    def __init__(self,config):
        super(Embedding_Combination_Layer,self).__init__()
        self.config = config

        self.highway_combination = HighwayCombination(self.config)

    def forward(self,word_embeddings, char_embeddings):

        embedding_combination = self.highway_combination(word_embeddings, char_embeddings)
        return embedding_combination


class HighwayCombination(nn.Module):

    def __init__(self, config):
        super(HighwayCombination, self).__init__()

        self.highwaynet = HighwayNet(config.depth,config.size)

    def forward(self, word_embeddings, char_embeddings):
        x = torch.cat((word_embeddings, char_embeddings), dim=2)
        return self.highwaynet(x)
