import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import os
import numpy as np

from helper import *
torch.manual_seed(4)
np.random.seed(4)

class Bi_RNN(nn.Module):
    def __init__(self,config):
        self.config = config
        if(config.encoder_type = "bi-lstm"):
            self.encoder = nn.LSTM(input_size = self.config.input_size,hidden_size = self.config.hidden_size,num_layers = self.config.num_layers,bidirectional = True,dropout = self.config.dropout,batch_first = True)
        elif(config.encoder_type = "bi-gru"):
            self.encoder = nn.GRU(input_size = self.config.input_size,hidden_size = self.config.hidden_size,num_layers = self.config.num_layers,bidirectional = True,dropout = self.config.dropout,batch_first = True)

    def initHidden(self,batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_size), requires_grad = False) # Initial hidden state
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_size), requires_grad = False) # Initial cell state
        return h0, c0

    def forward(self,word_sequence_packed):
        # word_sequence_packed is a tensor of dimension of B x m x l
        batch_size = word_sequence_packed.size()[0]

        initial_hidden_states = self.initHidden(batch_size)

        output, hidden_state_final = self.encoder(word_sequence_packed,initial_hidden_states)
        output_padded, _ = pad_packed_sequence(output, batch_first=True)

        return output
