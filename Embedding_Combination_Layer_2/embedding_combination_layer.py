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
from Embedding_Combination_Layer_2.helper import *

"""
Embedding_Combination_Layer:
1) __init__ function: creates object of class HighwayCombination
   INPUT for __init__: config
2) forward function:
   INPUTS: word_embeddings,character_embeddings (ie. output of Embedding_Layer_1)
   character_embeddings : ( N x m x Dim1) where N is batch size, m : sequence length
   word_embeddings:  ( N x m x Dim2) where N is batch size, m : sequence length
   OUTPUT: combined tensor of word_embeddings and character_embeddings: (N x m x Dim1+Dim2)

"""

class Embedding_Combination_Layer(nn.Module):
    def __init__(self,config):
        super(Embedding_Combination_Layer,self).__init__()
        self.config = config

        self.highway_combination = _HighwayCombination(self.config)

    def forward(self,word_embeddings, char_embeddings):

        embedding_combination = self.highway_combination(word_embeddings, char_embeddings)
        return embedding_combination


class _HighwayCombination(nn.Module):

    def __init__(self, config):
        super(_HighwayCombination, self).__init__()
        self.config = config
        self.use_char_emb = self.config.use_char_emb
        self.use_word_emb = self.config.use_word_emb
        self.data_dir = self.config.data_dir
        if(self.use_char_emb == True and self.use_word_emb == True):
            self.emb_combination_size = self.config.word_emb_size + self.config.char_emb_out_size
        elif(self.use_char_emb == False and self.use_word_emb == True):
            self.emb_combination_size = self.config.word_emb_size
        elif(self.use_char_emb == True and self.use_word_emb == False):
            self.emb_combination_size =  self.config.char_emb_out_size

        # self.emb_size = self.config.word_emb_size + self.config.char_emb_out_size
        self.highwaynet = HighwayNet(self.config.depth,self.emb_combination_size)

    def forward(self, word_embeddings, char_embeddings):
        if(self.use_char_emb == True and self.use_word_emb == True):
            x = torch.cat((word_embeddings, char_embeddings), dim=2) # (N x m x Dim1+Dim2)
        elif(self.use_char_emb == False and self.use_word_emb == True):
            x = word_embeddings # (N x m x Dim1)
        elif(self.use_char_emb == True and self.use_word_emb == False):
            x = char_embeddings # (N x m x Dim2)


        return self.highwaynet(x) # (N x m x Dim1+Dim2)
