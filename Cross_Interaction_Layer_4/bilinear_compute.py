import torch
from torch import nn
from torch import autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence





class bidaf_bilinear(nn.Module):

    def __init__(self, passage_vec_size, query_vec_size):
        super(bidaf_bilinear, self).__init__()

        self.w_concat = autograd.Variable(torch.randn(1,1,2*passage_vec_size).type(torch.FloatTensor))
        self.w_query = autograd.Variable(torch.randn(1,1,2*query_vec_size).type(torch.FloatTensor))
        self.w_passage = autograd.Variable(torch.randn(1,1,2*passage_vec_size).type(torch.FloatTensor))
        self.passage_vec_size = passage_vec_size
        self.query_vec_size = query_vec_size

        self.w_concat = torch.nn.init.xavier_normal_(self.w_concat)
        self.w_query  = torch.nn.init.xavier_normal_(self.w_query)
        self.w_passage = torch.nn.init.xavier_normal_(self.w_passage)


    def forward(self, batch_size, passage_vectors, query_vectors, passage_length, query_length):
        temp_w_passage = passage_vectors * self.w_concat

        query_vector_permute = query_vectors.permute(0,2,1)
        w_concat_m = torch.bmm (temp_w_passage, query_vector_permute) 
    
        w_passage = self.w_passage * passage_vectors
        w_passage = torch.sum(w_passage, dim=2)
        w_passage = torch.unsqueeze(w_passage, dim=2).repeat(1,1, query_length)

        w_query = self.w_query * query_vectors
        w_query = torch.sum(w_query, dim=2)
        w_query = torch.unsqueeze(w_query, dim=1).repeat(1, passage_length, 1)
        dot_product = w_passage + w_query + w_concat_m
    

        return dot_product 
