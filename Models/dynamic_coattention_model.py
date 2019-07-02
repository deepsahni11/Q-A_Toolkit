import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.nn import Embedding
from argparse import ArgumentParser
import numpy as np
import logging
import code
import pickle
import os

torch.manual_seed(4)
np.random.seed(4)

from Encoding_Layer_3.word_level_encoding import *
from Cross_Interaction_Layer_4.coattention import *
from Output_Layer_10.decoder import *



class DCN_Model(nn.Module):

    def __init__(self, hidden_dim, embedding_matrix, dropout_ratio, maxout_pool_size, max_number_of_iterations):
        super(DCN_Model, self).__init__()

        self.encoder = Word_Level_Encoder(hidden_dim, embedding_matrix, dropout_ratio)
        self.coattention_encoder = Coattention_Encoder(hidden_dim, maxout_pool_size, embedding_matrix, max_number_of_iterations, dropout_ratio)
        self.decoder = Dynamic_Decoder(hidden_dim, maxout_pool_size, max_number_of_iterations, dropout_ratio)

    def forward(self, context_word_indexes, context_word_mask, question_word_indexes, question_word_mask,span_tensor):
        passage_representation = self.encoder.forward(context_word_indexes, context_word_mask)

        question_representation = self.encoder.forward(question_word_indexes, question_word_mask)


        U_matrix = self.coattention_encoder.forward(question_representation, passage_representation,context_word_mask)

        loss, index_start, index_end = self.decoder.forward(U_matrix, context_word_mask, span_tensor)

        return loss, index_start, index_end
