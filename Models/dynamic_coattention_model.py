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

from Encoding_Layer_3.encoding_layer import *
from Cross_Interaction_Layer_4.cross_interaction import *
from Self_Interaction_Layer_5.self_interaction import *
from Output_Layer_10.decoder import *



class DCN_Model(nn.Module):

    def __init__(self, config, embedding_matrix):
        super(DCN_Model, self).__init__()
        self.config = config
        self.encoder = Encoding_Layer(self.config, embedding_matrix)
        self.cross_interaction = Cross_Interaction(self.config)
        self.self_interaction = Self_Interaction(self.config)
        self.decoder = Decoder(self.config)

    def forward(self, context_word_indexes, context_word_mask, question_word_indexes, question_word_mask,span_tensor):
        passage_representation = self.encoder(context_word_indexes, context_word_mask)

        question_representation = self.encoder(question_word_indexes, question_word_mask)


        # A_Q_matrix,A_D_matrix,A_Q_vector,A_D_vector = query_attention_matrix,document_attention_matrix,query_attention_vector,document_attention_vector
        query_attention_matrix,document_attention_matrix,query_attention_vector,document_attention_vector = self.cross_interaction(question_representation, passage_representation)

        # A_Q_matrix,A_D_matrix,A_Q_vector,A_D_vector,question_representation, context_representation,document_word_sequence_mask
        U_matrix = self.self_interaction(query_attention_matrix,document_attention_matrix,query_attention_vector,document_attention_vector,question_representation, passage_representation,context_word_mask)

        loss, index_start, index_end = self.decoder(U_matrix, context_word_mask, span_tensor)

        return loss, index_start, index_end
