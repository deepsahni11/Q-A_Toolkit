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



def softmax(input, axis=1, f=None):
    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    soft_max_2d = f(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size)-1)

class bidaf_bilinear(nn.Module): # similarity matrix for BiDAF paper

    def __init__(self, passage_vec_size, query_vec_size):
        super(bidaf_bilinear, self).__init__()

        self.w_concat = autograd.Variable(torch.randn(1,1,2*passage_vec_size).type(torch.FloatTensor)) #1 x 1 x 2*l l=> hidden size
        
        self.w_query = autograd.Variable(torch.randn(1,1,2*query_vec_size).type(torch.FloatTensor))  #1 x 1 x 2*l
        self.w_passage = autograd.Variable(torch.randn(1,1,2*passage_vec_size).type(torch.FloatTensor)) #1 x 1 x 2*l
        self.passage_vec_size = passage_vec_size 
        self.query_vec_size = query_vec_size

        self.w_concat = torch.nn.init.xavier_normal_(self.w_concat)
        self.w_query  = torch.nn.init.xavier_normal_(self.w_query)
        self.w_passage = torch.nn.init.xavier_normal_(self.w_passage)


    def forward(self, batch_size, passage_vectors, query_vectors, passage_length, query_length):
        
        temp_w_passage = passage_vectors * self.w_concat #B x p x 2*l  p=>passage length
        
        
        query_vector_permute = query_vectors.permute(0,2,1)  #B x 2*l x q q=> query length
        
        w_concat_m = torch.bmm (temp_w_passage, query_vector_permute)  #B x p x q
    
        w_passage = self.w_passage * passage_vectors #B x p x 2*l
     
        w_passage = torch.sum(w_passage, dim=2) #B x p
        
        w_passage = torch.unsqueeze(w_passage, dim=2).repeat(1,1, query_length) #B x p x q
    
        w_query = self.w_query * query_vectors #B x q x 2*l
        
        w_query = torch.sum(w_query, dim=2) #B x q
        w_query = torch.unsqueeze(w_query, dim=1).repeat(1, passage_length, 1) #B x p x q
        dot_product = w_passage + w_query + w_concat_m #B x p x q

        return dot_product 

class Cosine_Similarity(nn.Module): # similarity matrix for DCN paper
    def __init__(self,config):
        super(Cosine_Similarity, self).__init__()
        self.config = config
        self.question_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)

    def forward(self,question_representation,document_representation):
        Q = question_representation
        D = document_representation

        if(self.config.query_non_linearity == True):

            Q_non_linearity = torch.tanh(self.question_proj(Q.view(-1, self.config.hidden_dim))).view(Q.size()) #B x (n) x l
            Q = Q_non_linearity
        # transpose(tensor, first_dimension to be transposed, second_dimension to be transposed)
        Q_transpose = torch.transpose(Q, 1, 2) #dimension: B x l x (n)

        # Performs a batch matrix-matrix product of matrices stored in batch1 and batch2.
        # batch1 and batch2 must be 3-D tensors each containing the same number of matrices.
        Cosine_similarity_matrix = torch.bmm(D, Q_transpose) # dimension of L : B x (m) x (n)

        return Cosine_similarity_matrix
