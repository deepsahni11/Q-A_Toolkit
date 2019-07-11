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

"""
Self-Interaction:
1) init function: creates onject of class Fusion_BiLSTM
   INPUTS: config
2) forward function:
    m = max length of instances in one batch of document
    n = max length of instances in one batch of question
   INPUTS: A_Q_matrix,A_D_matrix,A_Q_vector,A_D_vector (OUTPUTS to Co-Attention Layer)
           A_Q_matrix : B x (m + 1) x (n + 1): representation of each question using max words of document
           A_D_matrix : B x (n + 1) x (m + 1):: representation of each documnet using max words of question
           A_Q_vector : B  x (n + 1) x 1:
           A_D_vector : B  x (m + 1) x 1

           question_representation(Q: B x (n + 1) x l ), context_representation(D: B x (m + 1) x l)( OUTPUTS to EncodingLayer)
           document_word_sequence_mask
   OUTPUTS: U (final document representation after passing though bi-lstm)
"""

class Self_Interaction(nn.Module):

    def __init__(self,config):
        super(Self_Interaction, self).__init__()

        self.config = config
        self.fusion_bilstm = Fusion_BiLSTM(self.config.hidden_dim, self.config.dropout_ratio)

    def forward(self,A_Q_matrix,A_D_matrix,A_Q_vector,A_D_vector,question_representation, context_representation,document_word_sequence_mask):

        Q = question_representation
        # transpose(tensor, first_dimension to be transposed, second_dimension to be transposed)
        Q_transpose = torch.transpose(Q_non_linearity, 1, 2) #dimension: B x l x (n + 1)

        D = context_representation

        A_Q = A_Q_matrix
        A_D = A_D_matrix

        D_transpose = torch.transpose(D, 1, 2) #dimension: B x l x (m + 1)
        C_Q = torch.bmm(D_transpose, A_Q) # (B x l x (m + 1)) x (B x (m + 1) x (n + 1)) => B x l x (n + 1)

        L = torch.bmm(D, Q_transpose)
        L_tranpose = torch.transpose(L,1,2)



        # concatenation along dimension=1:(B x l x (n + 1) ; B x l x (n + 1)  -----> B x 2l x (n + 1) ) x (B x (n + 1) x (m + 1)) ====> B x 2l x (m + 1)
        C_D = torch.bmm(torch.cat((Q_transpose, C_Q), 1), A_D) # B x 2l x (m + 1)
        C_D_transpose = torch.transpose(C_D, 1, 2)  # B x (m + 1) x 2l



        #fusion BiLSTM
        # concatenation along dimension = 2:  (B x (m + 1) x 2l ; B x (m + 1) x l  -----> B x (m + 1) x 3l )
        bi_lstm_input = torch.cat((C_D_transpose, D), 2) # B x (m + 1) x 3l

        U = self.fusion_bilstm(bi_lstm_input, document_word_sequence_mask) # B x m x 2l


        return U
