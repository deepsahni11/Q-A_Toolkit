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




class queryAwareDocument(nn.Module):
    def __init__(self, qad_rep='method_1'):
        super(queryAwareDocument, self).__init__()
        self.qad_rep = qad_rep
        self.softmax_f = torch.nn.Softmax()

    def forward(self, passage_outputs, question_outputs, b_attention_query_vector):
        # if self.qad_rep == 'method_1':
        #     bilinear_comp = softmax(bilinear_comp, axis=1, f= self.softmax_f)
        #     bilinear_comp = bilinear_comp.permute(0,2,1)
        #
        #     qad_rep = torch.bmm(bilinear_comp, passage_outputs)
        #
        #     return None,qad_rep
        #
        # if self.qad_rep== 'method_2':
        #     softmax_col = softmax(bilinear_comp, axis = 2, f = self.softmax_f)
        #     max_sofmax_col, _ = torch.max(softmax_col, dim = 2)
        passage_outputs_list = torch.unbind(passage_outputs, 0)
        max_softmax_col_list = torch.unbind(b_attention_query_vector, 0)

        temp_rep = []
        for i in range(len(max_softmax_col_list)):
            temp_softmax = torch.unsqueeze(max_softmax_col_list[i], dim=0)
            temp_rep.append(torch.mm(temp_softmax, passage_outputs_list[i]))


        qad_rep_vec = torch.stack(temp_rep, 0)
        qad_rep_vec = torch.squeeze(qad_rep_vec)
        return qad_rep_vec, None


class documentAwareQuery(nn.Module):
    def __init__(self, daq_rep='method1'):
        super(documentAwareQuery, self).__init__()
        self.daq_rep = daq_rep
        self.softmax_f = torch.nn.Softmax()
    def forward(self, passage_outputs, question_outputs, S_attention_document):
        # if self.daq_rep == 'method1':
        new_bilinear_comp = softmax(bilinear_comp, axis=2, f = self.softmax_f)

        daq_rep = torch.bmm(new_bilinear_comp, question_outputs)
        return None, daq_rep

        # if self.daq_rep== 'method2':
        #     max_sofmax_col, _ = torch.max(softmax_col, dim = 1)
        #     question_outputs_list = torch.unbind(question_outputs, 0)
        #     max_softmax_col_list = torch.unbind(max_sofmax_col, 0)
        #     temp_rep = []
        #     for i in range(len(max_softmax_col_list)):
        #         temp_softmax = torch.unsqueeze(max_softmax_col_list[i], dim=0)
        #         temp_rep.append(torch.mm(temp_softmax, question_outputs_list[i]))
        #     daq_rep_vec = torch.stack(temp_rep, 0)
        #     daq_rep_vec = torch.squeeze(daq_rep_vec)
        #     return daq_rep_vec, None


class bidaf_self_match(nn.Module):

    #Concat the context_aware query_representation along with passage vectors
    #Use the query_aware context representation along with passage vector representation.

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, dropout=0, batch_first=True):
        super(bidaf_selfmatch, self).__init__()

        self.bidirectional_encoder_1 = torch.nn.GRU(input_size = 8*input_size, hidden_size = hidden_size,
                                    num_layers = num_layers,
                                    bidirectional = bidirectional,
                                    dropout = dropout,
                                    batch_first = True)

        self.bidirectional_encoder_2 = torch.nn.GRU(input_size = 2*hidden_size,
                                   hidden_size = hidden_size,
                                   num_layers = num_layers,
                                   bidirectional = bidirectional,
                                   dropout = dropout,
                                   batch_first = True)


    def forward(self, passage_vectors, query_vectors, query_aware_passage_rep ,query_aware_passage_mat, passage_aware_query_rep, passage_aware_query_mat):

        passage_dot_qap_rep = passage_vectors * passage_aware_query_mat

        temp = torch.unsqueeze(query_aware_passage_rep, dim=1)
        #temp = temp.repeat(1, passage_vectors.size()[1], 1)

        passage_dot_paq_rep = passage_vectors * temp

        G = torch.cat((passage_vectors, passage_aware_query_mat,
                         passage_dot_qap_rep, passage_dot_paq_rep), dim = -1)

        batch_size = passage_vectors.size()[0]
        hidden_size = 100
        h_0 = torch.autograd.Variable(torch.zeros(2, batch_size, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)
        h_0_2 = torch.autograd.Variable(torch.zeros(2, batch_size, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)
        #c_0 = Variable(torch.zeros(2, batch_size, hidden_size), requires_grad=False)
        temp_outputs, _ = self.bidirectional_encoder_1(G, h_0)
        self_match_outputs, _ = self.bidirectional_encoder_2(temp_outputs, h_0_2)
        del h_0, h_0_2
        return [G, self_match_outputs]


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
