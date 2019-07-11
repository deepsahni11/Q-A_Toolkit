import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import os
import numpy as np
torch.manual_seed(4)
np.random.seed(4)


class Fusion_BiLSTM(nn.Module):
    def __init__(self, hidden_dim,dropout_ratio):
        super(Fusion_BiLSTM, self).__init__()


        self.hidden_dim = hidden_dim
        self.dropout_ratio= dropout_ratio
         # batch_first = True
        # Input: has a dimension of B * m * embedding_dim
        # Function parameters: input_size, hidden_size, num_layers_of_LSTM = 1(here)
        self.fusion_bilstm = nn.LSTM(3 * self.hidden_dim, self.hidden_dim, 1, batch_first=True,
                                     bidirectional=True, dropout=self.dropout_ratio)

    def initHidden(self,batch_size):
        h0 = Variable(torch.zeros(2, batch_size, self.hidden_dim), requires_grad = False) # Initial hidden state
        c0 = Variable(torch.zeros(2, batch_size, self.hidden_dim), requires_grad = False) # Initial cell state
        return h0, c0

    def forward(self, word_sequence_embeddings, word_sequence_mask):

        # stores length of per instance for context/question
        length_per_instance = torch.sum(word_sequence_mask, 1)

        initial_hidden_states = self.initHidden(len(length_per_instance))

        # All RNN modules accept packed sequences as inputs.
        # Input: word_sequence_embeddings has a dimension of B x m+1 x 3l (l is the size of the glove_embedding/ pre-trained embedding/embedding_dim)
        packed_word_sequence_embeddings = pack_padded_sequence(word_sequence_embeddings, length_per_instance, batch_first=True,enforce_sorted=False)

        # since the input was a packed sequence, the output will also be a packed sequence
        output, _ = self.fusion_bilstm(packed_word_sequence_embeddings,initial_hidden_states)

        # Pads a packed batch of variable length sequences.
        # It is an inverse operation to pack_padded_sequence().
        # dimension:  B x m x 2l
        output_to_BiLSTM_padded, _ = pad_packed_sequence(output, batch_first=True)


        return output_to_BiLSTM_padded
