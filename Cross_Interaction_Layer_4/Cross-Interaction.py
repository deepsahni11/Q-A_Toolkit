
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

"""
Cross_interaction:
1) init function:
   INPUTS: Config
2) forward function:
   INPUTS: question_representation(Q: B x (n + 1) x l ), context_representation(D: B x (m + 1) x l)
           m = max length of instances in one batch of document
           n = max length of instances in one batch of question
   OUTPUTS:
   A_Q_matrix : B x (m + 1) x (n + 1): representation of each question using max words of document
   A_D_matrix : B x (n + 1) x (m + 1):: representation of each documnet using max words of question
   A_Q_vector : B  x 1 x (n + 1):
   A_D_vector : B  x 1 x (m + 1)

"""


class Cross_interaction(nn.Module):
    def __init__(self,config):
    #hidden_dim, maxout_pool_size, embedding_matrix, max_number_of_iterations, dropout_ratio):
        super(DCN_Coattention_Encoder, self).__init__()

        self.config = config

        ## nn.Linear(input_dim, output_dim)
        # Affine mapping from l ==> l
        self.question_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)


    def forward(self, question_representation, context_representation):


        ############## m = max length of instances in one batch of document ;  n= max length of instances in one batch of question ############################33
        Q = question_representation # B x (n + 1) x l
        D = context_representation  # B x (m + 1) x l

        # view function is meant to reshape the tensor.(Similar to reshape function in numpy)
        # view( row_size = -1 ,means that number of rows are unknown, column_size)
        # pass the Q tensor through a non-linearity
        Q_non_linearity = torch.tanh(self.question_proj(Q.view(-1, self.config.hidden_dim))).view(Q.size()) #B x (n + 1) x l


        # transpose(tensor, first_dimension to be transposed, second_dimension to be transposed)
        Q_transpose = torch.transpose(Q_non_linearity, 1, 2) #dimension: B x l x (n + 1)

        # Performs a batch matrix-matrix product of matrices stored in batch1 and batch2.
        # batch1 and batch2 must be 3-D tensors each containing the same number of matrices.
        L = torch.bmm(D, Q_transpose) # dimension of L : B x (m + 1) x (n + 1)

        A_Q_matrix = F.softmax(L, dim=2) # B x (m + 1) x (n + 1)
        A_D_matrix = F.softmax(L_tranpose, dim=2)  # B x (n + 1) x (m + 1)

        A_Q_vector = torch.mean(A_Q_matrix,1)# B  x 1 x (n + 1)
        A_D_vector = torch.mean(A_D_matrix,1) # B  x 1 x (m + 1) 

        return A_Q_matrix,A_D_matrix,A_Q_vector,A_D_vector
