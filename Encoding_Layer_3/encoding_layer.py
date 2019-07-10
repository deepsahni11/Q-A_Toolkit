import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
Encoding_Layer:
1) __init__ function: creates object of class LSTM/GRU as and when called
   INPUT to __init__: config

2) forward function:
   INPUTS: word_sequence_packed(for question/context):  tensor of dimension of B x m x l( B: batch_size, m= maximum length of sequence in that batch, l= hidden dimension)
   OUTPUTS: output( dimension of B x m x l )

"""

class Encoding_Layer(nn.Module):
    def __init__(self,config):
        super(Encoding_Layer, self).__init__()
        self.config = config

        if(config.encoder_type = "lstm"):
            self.encoder = nn.LSTM(input_size = self.config.input_size,hidden_size = self.config.hidden_size,num_layers = self.config.num_layers,bidirectional = self.config.bidirectional,dropout = self.config.dropout,batch_first = True)
        elif(config.encoder_type = "gru"):
            self.encoder = nn.GRU(input_size = self.config.input_size,hidden_size = self.config.hidden_size,num_layers = self.config.num_layers,bidirectional = self.config.bidirectional,dropout = self.config.dropout,batch_first = True)

        # self.passage_encoder = nn.LSTM(input_size = self.config.input_size,hidden_size = self.config.hidden_size, num_layers = self.config.num_layers,bidirectional = self.config.bidirectional,dropout = self.config.dropout, batch_first = True)

        self.hidden_size = config.hidden_size

    def initHidden(self,batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_size), requires_grad = False) # Initial hidden state
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_size), requires_grad = False) # Initial cell state
        return h0, c0

    def forward(self, word_sequence_packed):

        # word_sequence_packed is a tensor of dimension of B x m x l
        batch_size = word_sequence_packed.size()[0]

        initial_hidden_states = self.initHidden(batch_size)

        output, hidden_state_final = self.encoder(word_sequence_packed,initial_hidden_states)
        output_padded, _ = pad_packed_sequence(output, batch_first=True)

        return output
