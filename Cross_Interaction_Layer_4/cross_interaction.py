
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import os
import numpy as np

from Cross_Interaction_Layer_4.helper import *
torch.manual_seed(4)
np.random.seed(4)

"""
DCN_cross_interaction:
1) init function:
   INPUTS: Config
2) forward function:
   INPUTS: question_representation(Q: B x (n + 1) x l ), context_representation(D: B x (m + 1) x l)
           m = max length of instances in one batch of document
           n = max length of instances in one batch of question
   OUTPUTS:
   A_Q_matrix : B x (m + 1) x (n + 1): representation of each question using max words of document
   A_D_matrix : B x (n + 1) x (m + 1): representation of each documnet using max words of question
   A_Q_vector : B  x 1 x (n + 1):
   A_D_vector : B  x 1 x (m + 1)

"""
class Cross_Interaction(nn.Module):
    def __init__(self,config):
        super(Cross_Interaction, self).__init__()
        self.config = config
        if(self.config.cross_interaction_type == "dcn"):
            self.cross_interaction = DCN_cross_interaction(config)
        elif(self.config.cross_interaction_type == "bidaf"):
            self.cross_interaction = Bidaf_cross_interaction(config)
        elif(self.config.cross_interaction_type == "gated_attention"):
            self.cross_interaction = Gated_attention_cross_interaction(config)

    def forward(self,question_representation, context_representation):
        if(self.config.cross_interaction_type == "gated_attention"):
            updated_document_matrix = self.cross_interaction(question_representation, context_representation)
            return updated_document_matrix
        else:
            query_attention_matrix,document_attention_matrix,query_attention_vector,document_attention_vector = self.cross_interaction(question_representation, context_representation)
            return query_attention_matrix,document_attention_matrix,query_attention_vector,document_attention_vector

class Gated_attention_cross_interaction(nn.Module):
    def __init__(self,config):
        super(Gated_attention_cross_interaction, self).__init__()
        self.config = config
        self.similarity_matrix = Cosine_Similarity(config)
        self.softmax = torch.nn.Softmax()

    def forward(self,question_representation,document_representation):
        Q = question_representation # B x n x 2l
        D = document_representation # B x m x 2l
        # S = similarity matrix of Gated_attention_reader paper(cosine similarity)
        S = self.similarity_matrix(Q,D) # B x m x n
        S_softmax = self.softmax(S,dim = 2) # B x m x n

        q_matrix = torch.bmm(S_softmax, Q) # B x n x 2l
        document_x_matrix = q_matrix + D  # B x m x 2l

        # returns updated document matrix because gated attention reader has no self interaction layer
        return document_x_matrix
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
        super(DCN_cross_interaction, self).__init__()
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
        L_transpose = torch.transpose(L,1,2)


        A_Q_matrix = F.softmax(L, dim=2) # B x (m + 1) x (n + 1)
        A_D_matrix = F.softmax(L_transpose, dim=2)  # B x (n + 1) x (m + 1)

        # A_Q and A_D are attention weights

        A_Q_vector = torch.mean(A_Q_matrix,1)# B  x 1 x (n + 1)
        A_D_vector = torch.mean(A_D_matrix,1) # B  x 1 x (m + 1)

        return A_Q_matrix,A_D_matrix,A_Q_vector,A_D_vector
