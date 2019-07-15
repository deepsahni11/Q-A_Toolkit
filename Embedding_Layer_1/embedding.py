from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn.functional as F
from Embedding_Layer_1.helper import *
import pickle
import os
"""
Embedding(it is a super class that creates objects of other embedding classes):
1) __init__ function: creates objects of classes a) CharLevelEmbeddingCNN b) WordLevelEmbedding
  INPUT for __init__: config

2) forward function:
  INPUTS for forward: word_tensor, char_tensor
  OUTPUTS from forward: return word_level_embedding, character_level_cnn embedding ( N x W x Dim) where N is batch size, W : sequence length
  a) CharLevelEmbeddingCNN:
     INPUTS to __init__(from config): is_train(boolean),keep_prob(1- drop_out_probability), kernel_dim(output channel size) , kernel_size(tuple containing (filter_height ,filter_width) ) , embedding_dim(input channel size), initial_char_embedding
     INPUTS for forward ()  : char_tensor
     OUTPUT from forward ():  return  character_level_cnn embedding ( N x W x Dim) where N is batch size, W : sequence length
  b) WordLevelEmbedding:
     INPUTS to __init__(from config): vocab_size(int), word_embedding_size, initial_word_embedding ,padding_idx , fine_tune
     INPUTS for forward() : word_tensor
     OUTPUTS for forward: word level embedding

"""




class Embedding_layer(nn.Module):
    def __init__(self, config):
        super(Embedding_layer,self).__init__()

        self.config = config
        self.use_char_emb = self.config.use_char_emb
        self.use_word_emb = self.config.use_word_emb
        self.data_dir = self.config.data_dir


        if(self.use_char_emb == True and self.use_word_emb == True):
            self.initial_word_embedding = pickle.load(open(os.path.join(self.data_dir, "glove_word_embeddings.pkl"), "rb"))
            self.initial_char_embedding = pickle.load(open(os.path.join(self.data_dir, "char_embeddings.pkl"), "rb"))
            self.char_embedding_cnn = _CharLevelEmbeddingCNN(self.config,self.initial_char_embedding)
            self.word_level_embedding = _WordLevelEmbedding(self.config,self.initial_word_embedding)

        elif(self.use_char_emb == False and self.use_word_emb == True):
            self.initial_word_embedding = pickle.load(open(os.path.join(self.data_dir, "glove_word_embeddings.pkl" ), "rb"))
            self.word_level_embedding = _WordLevelEmbedding(self.config,self.initial_word_embedding)

        elif(self.use_char_emb == True and self.use_word_emb == False):
            self.initial_char_embedding = pickle.load(open(os.path.join(self.data_dir, "char_embeddings.pkl" ), "rb"))
            self.char_embedding_cnn = _CharLevelEmbeddingCNN(self.config,self.initial_char_embedding)

    def forward(self,batch_word_indexes,batch_char_indexes):

        if(self.use_char_emb == True and self.use_word_emb == True):
            char_embedding = self.char_embedding_cnn(batch_char_indexes)
            word_embedding = self.word_level_embedding(batch_word_indexes)
            return word_embedding,char_embedding

        elif(self.use_char_emb == False and self.use_word_emb == True):
            word_embedding = self.word_level_embedding(batch_word_indexes)
            return word_embedding,None

        elif(self.use_char_emb == True and self.use_word_emb == False):
            char_embedding = self.char_embedding_cnn(batch_char_indexes)
            return None,char_embedding


class _CharLevelEmbeddingCNN(nn.Module):

    def __init__(self,config,initial_char_embedding):
        super(_CharLevelEmbeddingCNN, self).__init__()

        self.char_embed_mat_initial = nn.Embedding(config.char_vocab_size, embedding_dim=config.char_emb_size, padding_idx = config.padding_idx)
        self.char_embed_mat_initial.weight.data.copy_(torch.FloatTensor(initial_char_embedding))
        self.char_embeddings = MultiConv1D(config.is_train, config.keep_prob, config.kernel_dim, config.kernel_size,config.embedding_dim)

    def forward(self,batch_char_indexes):
        # char_tensor = temp1 ( batch_size X length )

        # temp1 = content_batches["tokenchar_"]
        batch_char_indexes_size = batch_char_indexes.size()
        batch_char_indexes_resized = batch_char_indexes_resized.view(self.config.batch_size, -1)
        batch_char_indexes_embedded = self.char_embed_mat_initial(batch_char_indexes_resized)
        batch_char_indexes_embedded_resized = batch_char_indexes_embedded.view(batch_char_indexes_size[0], batch_char_indexes_size[1], batch_char_indexes_size[2], -1)


        batch_char_indexes_embedded_resized_after_cnn = self.char_embeddings(batch_char_indexes_embedded_resized)
        # encoded_query_char   = self.char_embed(encoded_query_char)

        # batch_char_indexes_embedded = self.char_embed_mat_initial(batch_char_indexes)
        # batch_char_indexes_embedded_after_cnn = self.char_embeddings(batch_char_indexes_embedded)
        # char_embeds = self.char_embeddings(char_tensor)
        return batch_char_indexes_embedded_resized_after_cnn

class _WordLevelEmbedding(nn.Module):

    def __init__(self,config, initial_word_embedding):

        super(_WordLevelEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(config.vocab_size, embedding_dim = config.word_embedding_size)
        self.word_embedding.weight.data.copy_(torch.FloatTensor(initial_word_embedding))

        if (config.fine_tune == False):
            self.word_embedding.requires_grad = False

    def forward(self,batch_word_indexes):
        batch_word_indexes_embedded = self.word_embedding(batch_word_indexes)
        return batch_word_indexes_embedded
