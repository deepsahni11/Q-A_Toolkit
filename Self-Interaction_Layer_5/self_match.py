import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from embedding import RNN

class bidaf_selfmatch(nn.Module):

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