import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from Encoding_Layer_3.helper import *
torch.manual_seed(4)
np.random.seed(4)

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
        self.use_char_emb = self.config.use_char_emb
        self.use_word_emb = self.config.use_word_emb
        self.data_dir = self.config.data_dir

        if(self.use_char_emb == True and self.use_word_emb == True):
            self.emb_combination_size = self.config.word_emb_size + self.config.char_emb_out_size
        elif(self.use_char_emb == False and self.use_word_emb == True):
            self.emb_combination_size = self.config.word_emb_size
        if(self.use_char_emb == True and self.use_word_emb == False):
            self.emb_combination_size =  self.config.char_emb_out_size

        if(self.config.encoder_type == "bi-lstm"):
            self.encoder = nn.LSTM(input_size = self.emb_combination_size,hidden_size = self.config.hidden_dim,num_layers = self.config.num_layers,bidirectional = True,dropout = self.config.dropout,batch_first = True)
        elif(self.config.encoder_type == "lstm"):
            self.encoder = nn.LSTM(input_size = self.emb_combination_size,hidden_size = self.config.hidden_dim,num_layers = self.config.num_layers,bidirectional = False,dropout = self.config.dropout,batch_first = True)
        elif(self.config.encoder_type == "bi-gru"):
            self.encoder = nn.GRU(input_size = self.emb_combination_size,hidden_size = self.config.hidden_dim,num_layers = self.config.num_layers,bidirectional = True,dropout = self.config.dropout,batch_first = True)
        elif(self.config.encoder_type == "gru"):
            self.encoder = nn.GRU(input_size = self.emb_combination_size,hidden_size = self.config.hidden_dim,num_layers = self.config.num_layers,bidirectional = False,dropout = self.config.dropout,batch_first = True)

        self.sentinel = nn.Parameter(torch.rand(self.config.hidden_dim))
        self.hidden_dim = self.config.hidden_dim

    def initHidden(self,batch_size):
        if(self.config.encoder_type == "bi-lstm"):
            h0 = Variable(torch.zeros(2, batch_size, self.hidden_dim), requires_grad = False) # Initial hidden state
            c0 = Variable(torch.zeros(2, batch_size, self.hidden_dim), requires_grad = False) # Initial cell state
        elif(self.config.encoder_type == "lstm"):
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad = False) # Initial hidden state
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad = False) # Initial cell state
        elif(self.config.encoder_type == "bi-gru"):
            h0 = Variable(torch.zeros(2, batch_size, self.hidden_dim), requires_grad = False) # Initial hidden state
            c0 = Variable(torch.zeros(2, batch_size, self.hidden_dim), requires_grad = False) # Initial cell state
        elif(self.config.encoder_type == "gru"):
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad = False) # Initial hidden state
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad = False) # Initial cell state
        # h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad = False) # Initial hidden state
        # c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad = False) # Initial cell state
        return h0, c0

    def forward(self, embedding_combination,word_sequence_mask):
        #word_sequence_indexes, word_sequence_mask

        # word_sequence_packed is a tensor of dimension of B x m x l
        batch_size = embedding_combination.size()[0]
        #
        # initial_hidden_states = self.initHidden(batch_size)
        #
        # output, hidden_state_final = self.encoder(word_sequence_packed,initial_hidden_states)
        # output_padded, _ = pad_packed_sequence(output, batch_first=True)

        length_per_instance = torch.sum(word_sequence_mask, 1)


        initial_hidden_states = self.initHidden(batch_size)
        # returns the word_sequences_embeddings with the embeddings for each token/word from word_sequence_indexes
        # word_sequence_embeddings is a tensor of dimension of B x m x l
        # word_sequence_embeddings = self.embedding(word_sequence_indexes)

        # All RNN modules accept packed sequences as inputs.
        # Input: word_sequence_embeddings has a dimension of B x m x l (l is the size of the glove_embedding/ pre-trained embedding/embedding_dim)
        packed_word_sequence_embeddings = pack_padded_sequence(embedding_combination,length_per_instance,batch_first=True,enforce_sorted=False)



        # nn.LSTM encoder gets an input of pack_padded_sequence of dimensions
        # since the input was a packed sequence, the output will also be a packed sequence
        output, _ = self.encoder(packed_word_sequence_embeddings,initial_hidden_states)


        # Pads a packed batch of variable length sequences.
        # It is an inverse operation to pack_padded_sequence().
        # dimension:  B x m x l
        output_to_LSTM_padded, _ = pad_packed_sequence(output, batch_first=True)

        if(self.config.sentinel == False):
            return output_to_LSTM_padded
        else:
            # list() creates a list of elements if an iterable is passed
            # batch_size is a scalar which stores the value of batch size. (batch_size = B)
            # batch_size, _ = list(word_sequence_mask.size())


            # dimension of sentinel matrix =  B x 1 x l (replicates or expands along given dimension)
            length_per_instance_new_dim = length_per_instance.unsqueeze(1).expand(batch_size, self.hidden_dim).unsqueeze(1)


            # sentinel to be concatenated to the data
            # dimension of sentinel_zero =  B x 1 x l
            sentinel_zero = torch.zeros(batch_size, 1, self.hidden_dim)

            # copy sentinel vector at the end
            # dimension of output_to_LSTM_padded_with_sentinel =  B x (m + 1) x l
            output_to_LSTM_padded_with_sentinel = torch.cat([output_to_LSTM_padded, sentinel_zero], 1)



            return output_to_LSTM_padded_with_sentinel
