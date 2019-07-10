from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn.functional as F
from helper import *


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
        super(Embedding,self).__init__()

        self.config = config


        if(self.use_char_emb == True && self.use_word_emb == True):
            self.initial_word_embedding = pickle.load(open(os.path.join(emb_dir, "word_embeddings" + str(config.word_emb_size) + ".pkl")))
            self.initial_char_embedding = pickle.load(open(os.path.join(self.emb_dir, "char_embeddings" + str(config.char_emb_size) +  ".pkl")))
            self.char_embedding_cnn = CharLevelEmbeddingCNN(self.config,initial_char_embedding)
            self.word_level_embedding = WordLevelEmbedding(self.config,initial_word_embedding)

        elif(self.use_char_emb == False && self.use_word_emb == True):
            self.initial_word_embedding = pickle.load(open(os.path.join(self.emb_dir, "word_embeddings" + str(config.word_emb_size) + ".pkl")))
            self.word_level_embedding = WordLevelEmbedding(self.config,initial_word_embedding)

        elif(self.use_char_emb == True && self.use_word_emb == False):
            self.initial_char_embedding = pickle.load(open(os.path.join(self.emb_dir, "char_embeddings" + str(config.char_emb_size) +  ".pkl")))
            self.char_embedding_cnn = CharLevelEmbeddingCNN(self.config,initial_char_embedding)

    def forward(self,char_tensor,word_tensor):

        if(self.use_char_emb == True && self.use_word_emb == True):
            char_embedding = self.char_embedding_cnn(char_tensor)
            word_embedding = self.word_level_embedding(word_tensor)
            return word_embedding,char_embedding

        elif(self.use_char_emb == False && self.use_word_emb == True):
            word_embedding = self.word_level_embedding(word_tensor)
            return word_embedding,None

        elif(self.use_char_emb == True && self.use_word_emb == False):
            char_embedding = self.char_embedding_cnn(char_tensor)
            return None,char_embedding


class _CharLevelEmbeddingCNN(nn.Module):

    def __init__(self,config,initial_char_embedding):
        super(CharLevelEmbeddingCNN, self).__init__()


        self.char_embed_mat_initial = nn.Embedding(config.char_vocab_size, embedding_dim=config.char_emb_size, padding_idx = config.padding_idx)
        self.char_embed_mat_initial.weight.data.copy_(torch.FloatTensor(initial_char_embedding))
        self.char_embeddings = MultiConv1D(config.is_train, config.keep_prob, config.kernel_dim, config.kernel_size,config.embedding_dim)
.
    def forward(self,char_tensor):
        # char_tensor = temp1 ( batch_size X length )
        encoded_content_char = self.char_embed_mat_initial(char_tensor)
        encoded_content_char_after_cnn = self.char_embeddings(encoded_content_char)
        # char_embeds = self.char_embeddings(char_tensor)
        return encoded_content_char_after_cnn

class _WordLevelEmbedding(nn.Module):

    def __init__(self,config, initial_word_embedding):

        self.word_embedding = nn.Embedding(config.vocab_size, embedding_dim = config.word_embedding_size)
        self.word_embedding.weight.data.copy(initial_word_embedding)

        if (config.fine_tune == False):
            self.word_embedding.requires_grad = False

    def get_embedding(self,word_tensor):
        word_embedding_final = self.word_embedding(word_tensor)
        return word_embedding_final
