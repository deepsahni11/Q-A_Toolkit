from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn.functional as F
from helper import *
"""
Embedding(it is a super class that creates objects of other embedding classes):
1) __init__ function: creates objects of classes a) CharLevelEmbeddingCNN b) WordLevelEmbedding
  a) CharLevelEmbeddingCNN:
     INPUTS: is_train(boolean),keep_prob(1- drop_out_probability), kernel_dim(output channel size) , kernel_size(tuple containing (filter_height ,filter_width) ) , embedding_dim(input channel size), initial_char_embedding
     OUTPUT: character_level_cnn embedding
  b) WordLevelEmbedding:
     INPUTS: vocab_size(int), word_embedding_size, initial_word_embedding ,padding_idx , fine_tune
     OUTPUTS: word level embedding

"""




class Embedding_layer(nn.Module):
    def __init__(self,use_char_emb, use_word_emb,char_embed_type, is_train, keep_prob, kernel_dim , kernel_size, embedding_dim,vocab_size,
                 padding_idx=None, word_embedding_size = 300, fine_tune=False):
        super(Embedding,self).__init__()

        self.use_char_emb = use_char_emb
        self.use_word_emb = use_word_emb
        self.char_embed_type = char_embed_type
        self.char_tensor = char_tensor
        self.word_tensor = word_tensor


        if(self.use_char_emb == True && self.use_word_emb == True):
            self.initial_word_embedding = pickle.load(open(os.path.join(self.emb_dir, "word_embeddings" + str(config.word_emb_size) + ".pkl")))
            self.initial_char_embedding = pickle.load(open(os.path.join(self.emb_dir, "char_embeddings" + str(config.char_emb_size) +  ".pkl")))
            self.char_embedding_cnn = CharLevelEmbeddingCNN(is_train, keep_prob, kernel_dim , kernel_size, embedding_dim,initial_char_embedding)
            self.word_level_embedding = WordLevelEmbedding(vocab_size, word_embedding_size , initial_word_embedding,padding_idx, fine_tune)

        elif(self.use_char_emb == False && self.use_word_emb == True):
            self.initial_word_embedding = pickle.load(open(os.path.join(self.emb_dir, "word_embeddings" + str(config.word_emb_size) + ".pkl")))
            self.word_level_embedding = WordLevelEmbedding(vocab_size, word_embedding_size, initial_word_embedding,padding_idx, fine_tune)

        elif(self.use_char_emb == True && self.use_word_emb == False):
            self.initial_char_embedding = pickle.load(open(os.path.join(self.emb_dir, "char_embeddings" + str(config.char_emb_size) +  ".pkl")))
            self.char_embedding_cnn = CharLevelEmbeddingCNN(is_train, keep_prob, kernel_dim , kernel_size, embedding_dim,initial_char_embedding)

    def forward(self,char_tensor,word_tensor):

        if(self.use_char_emb == True && self.use_word_emb == True):
            char_embedding = self.char_embedding_cnn(char_tensor)
            word_embedding = self.word_level_embedding(word_tensor)
            return char_embedding,word_embedding

        elif(self.use_char_emb == False && self.use_word_emb == True):
            word_embedding = self.word_level_embedding(word_tensor)
            return word_embedding

        elif(self.use_char_emb == True && self.use_word_emb == False):
            char_embedding = self.char_embedding_cnn(char_tensor)
            return char_embedding


class CharLevelEmbeddingCNN(nn.Module):

    def __init__(self, is_train, keep_prob, kernel_dim , kernel_size, embedding_dim,initial_char_embedding):
        super(CharLevelEmbeddingCNN, self).__init__()


        self.char_embed_mat_initial = nn.Embedding(self.char_vocab_size, embedding_dim=config.char_emb_size, padding_idx = padding_idx)
        self.char_embed_mat_initial.weight.data.copy_(torch.FloatTensor(initial_char_embedding))
        self.char_embeddings = MultiConv1D(is_train, keep_prob, kernel_dim, kernel_size, embedding_dim)

    def forward(self,char_tensor):
        # char_tensor = temp1 ( batch_size X length )
        encoded_content_char = self.char_embed_mat_initial(char_tensor)
        encoded_content_char_after_cnn = self.char_embeddings(encoded_content_char)
        # char_embeds = self.char_embeddings(char_tensor)
        return encoded_content_char_after_cnn

class WordLevelEmbedding(nn.Module):

    def __init__(self, vocab_size, word_embedding_size = 300, initial_word_embedding = None,
                 padding_idx = None, fine_tune=False):

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim = word_embedding_size)
        self.word_embedding.weight.data.copy(initial_word_embedding)

        if (fine_tune == False):
            self.word_embedding.requires_grad = False

    def get_embedding(self,word_tensor):
        word_embedding_final = self.word_embedding(word_tensor)
        return word_embedding_final
