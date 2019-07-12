import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import os
import numpy as np

from helper import *
from Encoding_Layer_3.encoding_layer import *
from Cross_Interaction_Layer_4.embedding_combination_layer import *

torch.manual_seed(4)
np.random.seed(4)

class Gated_attention_multihop(nn.Module):
    def __init__(self,config):
        self.config = config
        self.bi_rnn = Bi_RNN(self.config)
        self.gated_attention_reader = Gated_attention_cross_interaction(self.config)

    def forward(self,question_word_sequence_packed,document_representation):
        # m = max length of an instance in document
        # n = max length of an instance in question
        D = document_representation # B x m x l

        for _ in range(self.config.gated_attention_layers - 1):
            # Q = question representation
            Q_new = self.bi_rnn(question_word_sequence_packed) # B x n x l
            Q = Q_new
            D_new = self.gated_attention_reader(Q,D) # B x m x l
            D = self.bi_rnn(D_new) # B x m x l

        Q_new = self.bi_rnn(question_word_sequence_packed)
        Q = Q_new

        return Q,D
