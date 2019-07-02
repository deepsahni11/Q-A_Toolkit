import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from embedding import RNN


class EncodingLayerRNN(nn.Module):
    def __init__(self,input_size = 300,  hidden_size = 512, num_layers=1, bidirectional=True, dropout=0, batch_first=True):
        super(EncodingLayerRNN, self).__init__()

        self.question_encoder = torch.nn.GRU(input_size = input_size,
                                    hidden_size = hidden_size,
                                    num_layers = num_layers,
                                    bidirectional = bidirectional,
                                    dropout = dropout,
                                    batch_first = True)

        self.passage_encoder = torch.nn.GRU(input_size = input_size, 
                                   hidden_size = hidden_size, 
                                   num_layers = num_layers,
                                   bidirectional = bidirectional,
                                   dropout = dropout, 
                                   batch_first = True)

        self.hidden_size = hidden_size
    def forward(self, context_pack, question_pack):

        batch_size = context_pack.size()[0]
        hidden_size = self.hidden_size

        h_q = torch.autograd.Variable(torch.zeros(2, batch_size, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)
        h_p = torch.autograd.Variable(torch.zeros(2, batch_size, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)
        question_outputs, hidden_question_state = self.question_encoder(question_pack, h_q)
        passage_outputs, hidden_passage_state = self.passage_encoder(context_pack, h_p)
        del h_q, h_p
        #passage_outputs = passage_outputs.data
        #question_outputs = question_outputs.data
        #hidden_question_state = hidden_question_state
        #hidden_passage_state = hidden_passage_state
        return passage_outputs, hidden_passage_state, question_outputs, hidden_question_state


class EncodingLayerSameRNN(nn.Module):

    def __init__(self, input_size = 300, hidden_size = 512, num_layers = 1, bidirectional=True, dropout = 0, batch_first=True):
        super(EncodingLayerRNN, self).__init__()

        self.common_encoder = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers= num_layers,
                                            bidirectional=bidirectional, dropout = dropout, batch_first= True)

        self.linear_query = nn.Linear(2*hidden_size, 2*hidden_size, bias=True)

        self.hidden_size =  hidden_size
    def forward(self, context_pack, question_pack):

        batch_size = context_pack.size()[0]
        hidden_size = self.hidden_size

        h_p = torch.autograd.Variable(torch.zeros(2,batch_size, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)
        c_p = torch.autograd.Variable(torch.zeros(2, batch_size, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)

        h_q = torch.autograd.Variable(torch.zeros(2, batch_first, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)
        c_q = torch.autograd.Variable(torch.zeros(2, batch_first, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)

        passage_outputs, hidden_passage_state = self.common_encoder(context_pack, (h_p, c_p))
        temp_query_outputs, temp_query_state = self.common_encoder(question_pack, (h_q, c_q))

        temp_query_outputs = self.linear_query(temp_query_outputs.view(-1, temp_query_outputs.size(2)))
        temp_query_state   = self.linear_query(temp_query_state.view(-1, temp_query_state.size(2)))

        return passage_outputs, hidden_passage_state, temp_query_outputs, temp_query_state
