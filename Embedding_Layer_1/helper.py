from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn.functional as F




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
