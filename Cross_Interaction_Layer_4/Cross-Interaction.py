
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
class Bidaf_cross_interaction(nn.Module):
    def __init__(self,config):
        super(Bidaf_cross_interaction, self).__init__()
        self.config = config
        self.softmax_f = torch.nn.Softmax()
        self.bilinear_similarity_matrix = bidaf_bilinear(self.config.hidden_dim, self.config.hidden_dim)

    def forward(self, question_representation, context_representation):
        Q = question_representation # B x (n) x l
        D = context_representation  # B x (m) x l

        # view function is meant to reshape the tensor.(Similar to reshape function in numpy)
        # view( row_size = -1 ,means that number of rows are unknown, column_size)
        # pass the Q tensor through a non-linearity
        ## L = cosine similarity matrix
        # self.config.batch_size, passage_encodings, query_encodings, passage_combine_embeddings.size()[1], query_combine_embeddings.size()[1]
        S = self.bilinear_similarity_matrix(self.config.batch_size,D,Q,D.size()[1],Q.size()[1])
        S_max_col_query, _ = torch.max(S, dim = 2) # B x m(or T)
        S_max_col_query_resized = torch.unsqueeze(S_max_col_query, 2) # B x m(or T) x 1
        b_attention_query_vector = softmax(S_max_col_query_resized, axis = 1, f = self.softmax_f)

        S_softmax_document = softmax(S, axis=1, f= self.softmax_f)
        S_attention_document = S_softmax_document.permute(0,2,1)
        return None,S_attention_document,b_attention_query_vector,None



class DCN_cross_interaction(nn.Module):
    def __init__(self,config):
    #hidden_dim, maxout_pool_size, embedding_matrix, max_number_of_iterations, dropout_ratio):
        super(DCN_Coattention_Encoder, self).__init__()
        self.config = config
        ## nn.Linear(input_dim, output_dim)
        # Affine mapping from l ==> l
        self.similarity_matrix = Cosine_Similarity(self.config)



    def forward(self, question_representation, context_representation):


        ############## m = max length of instances in one batch of document ;  n= max length of instances in one batch of question ############################33
        Q = question_representation # B x (n + 1) x l
        D = context_representation  # B x (m + 1) x l

        # view function is meant to reshape the tensor.(Similar to reshape function in numpy)
        # view( row_size = -1 ,means that number of rows are unknown, column_size)
        # pass the Q tensor through a non-linearity
        ## L = cosine similarity matrix
        L = self.similarity_matrix(Q,D)


        A_Q_matrix = F.softmax(L, dim=2) # B x (m + 1) x (n + 1)
        A_D_matrix = F.softmax(L_tranpose, dim=2)  # B x (n + 1) x (m + 1)

        # A_Q and A_D are attention weights

        A_Q_vector = torch.mean(A_Q_matrix,1)# B  x 1 x (n + 1)
        A_D_vector = torch.mean(A_D_matrix,1) # B  x 1 x (m + 1)

        return A_Q_matrix,A_D_matrix,A_Q_vector,A_D_vector
