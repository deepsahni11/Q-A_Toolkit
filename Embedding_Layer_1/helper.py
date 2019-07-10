from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn.functional as F

def get_rnn(rnn_type):
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        network = torch.nn.GRU
        print (rnn_type)
    elif rnn_type == "lstm":
        network = torch.nn.LSTM
    else:
        raise ValueError("Invalid RNN type %s" % rnn_type)

    return network

class RNN(nn.Module):
    """ RNN Module """

    def __init__(self, input_size, hidden_size,
                 output_projection_size=None, num_layers=1,
                 bidirectional=True, cell_type="gru", dropout=0,
                 pack=False, batch_first=True):
        super(RNN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)

        if output_projection_size is not None:
            self.output_layer = nn.Linear(hidden_size * 2 if bidirectional else hidden_size,
                                          output_projection_size)
        network = get_rnn(cell_type)
        self.network = network(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, bidirectional=bidirectional,
                               dropout=dropout, batch_first=batch_first)

    def forward(self, input_variable):
        outputs, hidden = self.network(input_variable)
        """
        if self.pack:
            padded_outputs, lengths = pad_packed_sequence(outputs)
            if hasattr(self, "output_layer"):
                outputs = pack_padded_sequence(self.output_layer(padded_outputs), lengths)
        """
        if hasattr(self, "output_layer"):
                outputs = self.output_layer(outputs)

        return outputs, hidden


class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, \
                    filter_height, filter_width, is_train=None, \
                    keep_prob=1.0, padding=0):
        super(Conv1D, self).__init__()

        self.is_train = is_train
        self.dropout_ = nn.Dropout(1. - keep_prob)
        self.keep_prob = keep_prob
        kernel_size = (filter_height, filter_width)
        self.conv2d_ = nn.Conv2d(in_channels, out_channels, kernel_size, \
                                    bias=True, padding=padding)



    def forward(self, in_):
        if self.is_train is not None and self.keep_prob < 1.0:
            self.dropout_(in_)
        '''
        tf: input tensor of shape [batch, in_height, in_width, in_channels]
        pt: input tensor of shape [batch, in_channels, in_height, in_width]
        '''
        t_in = in_.permute(0, 3, 1, 2)
        xxc = self.conv2d_(t_in)
        out, argmax_out = torch.max(F.relu(xxc), -1)
        return out


class MultiConv1D(nn.Module):
    def __init__(self, is_train, keep_prob, kernel_dim, kernel_sizes, embedding_dim):
        super(MultiConv1D, self).__init__()

        self.is_train = is_train
        self.keep_prob = keep_prob
        self.conv1d_list = nn.ModuleList([Conv1D(embedding_dim, kernel_dim, 1, K, keep_prob = keep_prob, padding=0) for K in kernel_sizes])

    def forward(self, in_):
        outs = []
        for conv1d_layer in self.conv1d_list:
            out = conv1d_layer(in_)
            outs.append(out)

        concat_out = torch.cat(outs, 1)
        concat_out = concat_out.permute(0,2,1)
        return concat_out
