
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
torch.manual_seed(4)
np.random.seed(4)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import os


with open(r"E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\glove_word_embeddings.pkl", "rb") as input_file:
    emb_matrix = pickle.load(input_file)
#
names = ["validation_context","train_context","validation_question","train_question"]
# data_dir = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\"



def get_pretrained_embedding(embedding_matrix):
    embedding = nn.Embedding(*embedding_matrix.shape)
    embedding.weight = nn.Parameter(torch.from_numpy(embedding_matrix).float())
    embedding.weight.requires_grad = False
    return embedding


class Word_Level_Encoder(nn.Module):

    def __init__(self, hidden_dim, embedding_matrix, dropout_ratio):
        super(Word_Level_Encoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = get_pretrained_embedding(embedding_matrix)
        self.embedding_dim = self.embedding.embedding_dim

        # batch_first = True
        # Input: has a dimension of B x m x embedding_dim
        # Function parameters: input_size, hidden_size, num_layers_of_LSTM = 1(here)
        self.encoder = nn.LSTM(self.embedding_dim, self.hidden_dim, 1, batch_first=True,
                              bidirectional=False, dropout=dropout_ratio)

#         self.dropout_emb = nn.Dropout(p=dropout_ratio)

        # creates a random vector with size= hidden_dim
        self.sentinel = nn.Parameter(torch.rand(hidden_dim,))

    def initHidden(self,batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad = False) # Initial hidden state
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad = False) # Initial cell state
        return h0, c0


    def forward(self, word_sequence_indexes, word_sequence_mask):

        # stores length of per instance for context/question
        # tensor of size = B
        length_per_instance = torch.sum(word_sequence_mask, 1)


        initial_hidden_states = self.initHidden(len(length_per_instance))
        # returns the word_sequences_embeddings with the embeddings for each token/word from word_sequence_indexes
        # word_sequence_embeddings is a tensor of dimension of B x m x l
        word_sequence_embeddings = self.embedding(word_sequence_indexes)

        # All RNN modules accept packed sequences as inputs.
        # Input: word_sequence_embeddings has a dimension of B x m x l (l is the size of the glove_embedding/ pre-trained embedding/embedding_dim)
        packed_word_sequence_embeddings = pack_padded_sequence(word_sequence_embeddings,length_per_instance,batch_first=True,enforce_sorted=False)



        # nn.LSTM encoder gets an input of pack_padded_sequence of dimensions
        # since the input was a packed sequence, the output will also be a packed sequence
        output, _ = self.encoder(packed_word_sequence_embeddings,initial_hidden_states)


        # Pads a packed batch of variable length sequences.
        # It is an inverse operation to pack_padded_sequence().
        # dimension:  B x m x l
        output_to_LSTM_padded, _ = pad_packed_sequence(output, batch_first=True)



        # list() creates a list of elements if an iterable is passed
        # batch_size is a scalar which stores the value of batch size. (batch_size = B)
        batch_size, _ = list(word_sequence_mask.size())


        # dimension of sentinel matrix =  B x 1 x l (replicates or expands along given dimension)
        length_per_instance_new_dim = length_per_instance.unsqueeze(1).expand(batch_size, self.hidden_dim).unsqueeze(1)


        # sentinel to be concatenated to the data
        # dimension of sentinel_zero =  B x 1 x l
        sentinel_zero = torch.zeros(batch_size, 1, self.hidden_dim)

        # copy sentinel vector at the end
        # dimension of output_to_LSTM_padded_with_sentinel =  B x (m + 1) x l
        output_to_LSTM_padded_with_sentinel = torch.cat([output_to_LSTM_padded, sentinel_zero], 1)



        return output_to_LSTM_padded_with_sentinel
