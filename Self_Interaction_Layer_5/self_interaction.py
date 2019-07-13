import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import os
import numpy as np

from Self_Interaction_Layer_5.helper import *

torch.manual_seed(4)
np.random.seed(4)

"""
DCN_self_interaction:
1) init function: creates onject of class Fusion_BiLSTM
   INPUTS: config
2) forward function:
    m = max length of instances in one batch of document
    n = max length of instances in one batch of question
   INPUTS: A_Q_matrix,A_D_matrix,A_Q_vector,A_D_vector (OUTPUTS to Co-Attention Layer)
           A_Q_matrix : B x (m + 1) x (n + 1): representation of each question using max words of document
           A_D_matrix : B x (n + 1) x (m + 1):: representation of each documnet using max words of question
           A_Q_vector : B  x 1 x (n + 1):
           A_D_vector : B  x 1 x (m + 1)

           question_representation(Q: B x (n + 1) x l ), context_representation(D: B x (m + 1) x l)( OUTPUTS to EncodingLayer)
           document_word_sequence_mask
   OUTPUTS: U (final document representation after passing though bi-lstm)
"""
class Self_Interaction(nn.Module):
    def __init__(self,config):
        super(Self_Interaction, self).__init__()
        self.config = config
        if(self.config.self_interaction_type == "dcn"):
            self.self_interaction = DCN_self_interaction(config)
        elif(self.config.self_interaction_type == "bidaf"):
            self.self_interaction = Bidaf_self_interaction(config)

    def forward(self,query_attention_matrix,document_attention_matrix,query_attention_vector,document_attention_vector,question_representation, context_representation,document_word_sequence_mask):
        self_interaction_output = self.self_interaction(query_attention_matrix,document_attention_matrix,query_attention_vector,document_attention_vector,question_representation, context_representation,document_word_sequence_mask)
        return self_interaction_output
class Bidaf_self_interaction(nn.Module):

    def __init__(self,config):
        super(Bidaf_self_interaction, self).__init__()

        self.config = config
        self.self_match = bidaf_self_match(self.config.input_size, self.config.hidden_size, num_layers=1, bidirectional=True, dropout=0, batch_first=True)

        self.document_aware_query = documentAwareQuery(self.config.daq_rep)
        self.query_aware_document = queryAwareDocument(self.config.qad_rep)

    def forward(self,passage_representation, question_represenation, b_attention_query_vector,S_attention_document):
        document_aware_query_rep_matrix = self.document_aware_query(passage_representation, question_represenation, b_attention_query_vector)
        query_aware_document_rep_vector = self.query_aware_document(passage_representation, question_represenation, S_attention_document)
# passage_vectors, query_vectors, query_aware_passage_rep ,query_aware_passage_mat, passage_aware_query_rep, passage_aware_query_mat
        self_match_representation = self.self_match(passage_representation,question_represenation,query_aware_document_rep_vector,None,None,document_aware_query_rep_matrix)

        return self_match_representation



class DCN_self_interaction(nn.Module):

    def __init__(self,config):
        super(DCN_self_interaction, self).__init__()

        self.config = config
        self.question_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.fusion_bilstm = Fusion_BiLSTM(self.config.hidden_dim, self.config.dropout_ratio)

    def forward(self,A_Q_matrix,A_D_matrix,A_Q_vector,A_D_vector,question_representation, context_representation,document_word_sequence_mask):

        Q = question_representation
        # transpose(tensor, first_dimension to be transposed, second_dimension to be transposed)
        Q_non_linearity = torch.tanh(self.question_proj(Q.view(-1, self.config.hidden_dim))).view(Q.size())
        Q_transpose = torch.transpose(Q_non_linearity, 1, 2) #dimension: B x l x (n + 1)

        D = context_representation

        A_Q = A_Q_matrix
        A_D = A_D_matrix

        D_transpose = torch.transpose(D, 1, 2) #dimension: B x l x (m + 1)

        # C_Q = query aware document representation
        C_Q = torch.bmm(D_transpose, A_Q) # (B x l x (m + 1)) x (B x (m + 1) x (n + 1)) => B x l x (n + 1)

        L = torch.bmm(D, Q_transpose)
        L_transpose = torch.transpose(L,1,2)



        # concatenation along dimension=1:(B x l x (n + 1) ; B x l x (n + 1)  -----> B x 2l x (n + 1) ) x (B x (n + 1) x (m + 1)) ====> B x 2l x (m + 1)
        # C_D = document aware query representation
        C_D = torch.bmm(torch.cat((Q_transpose, C_Q), 1), A_D) # B x 2l x (m + 1)
        C_D_transpose = torch.transpose(C_D, 1, 2)  # B x (m + 1) x 2l



        #fusion BiLSTM
        # concatenation along dimension = 2:  (B x (m + 1) x 2l ; B x (m + 1) x l  -----> B x (m + 1) x 3l )
        bi_lstm_input = torch.cat((C_D_transpose, D), 2) # B x (m + 1) x 3l

        U = self.fusion_bilstm(bi_lstm_input, document_word_sequence_mask) # B x m x 2l


        return U
