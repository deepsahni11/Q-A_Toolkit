from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn.functional as F

# class WordEmbedding:
#     def ()
# return , input
#
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


class CharLevelEmbeddingCNN(nn.Module):

    def __init__(self, is_train, keep_prob, kernel_dim , kernel_size, embedding_dim):
        super(CharLevelEmbeddingCNN, self).__init__()

        self.char_embeddings = MultiConv1D(is_train, keep_prob, kernel_dim, kernel_size, embedding_dim)

    def forward(self,char_tensor):
        char_embeds = self.char_embeddings(char_tensor)
        return char_embeds


class CharLevelEmbeddingRNN(nn.Module):
    def __init__(self, vocab_size, char_embedding_tensor=None, char_embedding_dim=300, output_dim=300,
                 padding_idx=None, bidirectional=True, cell_type="gru", num_layers=1):
        super(CharLevelEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=char_embedding_dim, padding_idx=padding_idx)
        if char_embedding_tensor is not None:
            self.embedding.weight.data.copy_(char_embedding_tensor)
        self.network = RNN(char_embedding_dim, output_dim, bidirectional=bidirectional,
                           cell_type=cell_type, num_layers=num_layers, pack=True, batch_first=True)

        if bidirectional:
            self.projection_layer = nn.Linear(output_dim * 2, output_dim)

    def forward(self, words_tensor, lengths):
        """
        :param words_tensor: tuple of (words_tensor (B x T), lengths)
        :return:
        """
        embed = self.embedding(words_tensor)
        embed_pack = pack_padded_sequence(embed, lengths, batch_first=True)
        outputs, hidden = self.network(embed_pack)

        if hasattr(self, "projection_layer"):
            batch_size = len(words_tensor)
            hidden = self.projection_layer(hidden.view(batch_size, -1))

        return hidden

class WordLevelEmbedding():

    def __init__(self, vocab_size, word_embedding_size = 300, initial_word_embedding = None,
                 padding_idx = None, fine_tune=False):

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim = word_embedding_size)
        self.word_embedding.weight.data.copy(initial_word_embedding)

        if (fine_tune == False):
            self.word_embedding.requires_grad = False
            
    def get_embedding(self, word_tensor):
        ans = self.word_embedding(word_tensor)
        return ans
