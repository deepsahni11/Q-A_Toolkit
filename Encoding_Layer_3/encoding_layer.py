import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#from embedding import RNN


class Encoding_Layer(nn.Module):
    def __init__(self,config):
        super(EncodingLayerRNN, self).__init__()
        self.config = config

        self.encoder = nn.LSTM(input_size = self.config.input_size,hidden_size = self.config.hidden_size,num_layers = self.config.num_layers,bidirectional = self.config.bidirectional,dropout = self.config.dropout,batch_first = True)
        # self.passage_encoder = nn.LSTM(input_size = self.config.input_size,hidden_size = self.config.hidden_size, num_layers = self.config.num_layers,bidirectional = self.config.bidirectional,dropout = self.config.dropout, batch_first = True)

        self.hidden_size = config.hidden_size

    def initHidden(self,batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_size), requires_grad = False) # Initial hidden state
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_size), requires_grad = False) # Initial cell state
        return h0, c0

    def forward(self, word_sequence_packed):

        batch_size = word_sequence_packed.size()[0]

        initial_hidden_states = self.initHidden(batch_size)
        # h_q = torch.autograd.Variable(torch.zeros(2, batch_size, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)
        # h_p = torch.autograd.Variable(torch.zeros(2, batch_size, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)
        output, hidden_state_final = self.question_encoder(word_sequence_packed,initial_hidden_states)
        # passage_outputs, hidden_passage_state = self.passage_encoder(context_pack, h_p)
        # del h_q, h_p
        #passage_outputs = passage_outputs.data
        #question_outputs = question_outputs.data
        #hidden_question_state = hidden_question_state
        #hidden_passage_state = hidden_passage_state
        return output, hidden_state_final


# class EncodingLayerSameRNN(nn.Module):
#
#     def __init__(self, input_size = 300, hidden_size = 512, num_layers = 1, bidirectional=True, dropout = 0, batch_first=True):
#         super(EncodingLayerRNN, self).__init__()
#
#         self.common_encoder = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers= num_layers,
#                                             bidirectional=bidirectional, dropout = dropout, batch_first= True)
#
#         self.linear_query = nn.Linear(2*hidden_size, 2*hidden_size, bias=True)
#
#         self.hidden_size =  hidden_size
#     def forward(self, context_pack, question_pack):
#
#         batch_size = context_pack.size()[0]
#         hidden_size = self.hidden_size
#
#         h_p = torch.autograd.Variable(torch.zeros(2,batch_size, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)
#         c_p = torch.autograd.Variable(torch.zeros(2, batch_size, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)
#
#         h_q = torch.autograd.Variable(torch.zeros(2, batch_first, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)
#         c_q = torch.autograd.Variable(torch.zeros(2, batch_first, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)
#
#         passage_outputs, hidden_passage_state = self.common_encoder(context_pack, (h_p, c_p))
#         temp_query_outputs, temp_query_state = self.common_encoder(question_pack, (h_q, c_q))
#
#         temp_query_outputs = self.linear_query(temp_query_outputs.view(-1, temp_query_outputs.size(2)))
#         temp_query_state   = self.linear_query(temp_query_state.view(-1, temp_query_state.size(2)))
#
#         return passage_outputs, hidden_passage_state, temp_query_outputs, temp_query_state
