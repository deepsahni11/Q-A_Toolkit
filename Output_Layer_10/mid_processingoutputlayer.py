import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from embedding import RNN

class mid_processing_unit(nn.Module):

    def __init__(self, input_size, bidirectional, dropout, mid_processing):

        super(mid_processing_unit, self).__init__()
        self.init_variables(input_size, bidirectional, dropout, mid_processing)


    def init_variables(self, input_size, bidirectional, dropout, mid_processing='bidaf'):
        if mid_processing == 'bidaf':
            self.encode_layer = torch.nn.GRU(2*input_size, hidden_size = input_size, num_layers=1,
                                    bidirectional=bidirectional,
                                    dropout=dropout,
                                    batch_first = True)
            self.hidden_size = input_size 

    def forward(self, *args):

        list_of_rep = []

        for arg in args:
            list_of_rep.append(arg)


          
        concat_rep = torch.cat(list_of_rep, dim=len(list_of_rep[0].size()) - 1)

        batch_size = concat_rep.size()[0]
        h_p = torch.autograd.Variable(torch.zeros(2, batch_size, self.hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)

        encoded_objs, _ = self.encode_layer(concat_rep, h_p)
        del h_p, batch_size
        
        return encoded_objs 