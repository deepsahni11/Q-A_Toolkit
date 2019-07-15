import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.nn import Embedding
from argparse import ArgumentParser
import numpy as np
import logging
import code
import pickle
import os

torch.manual_seed(4)
np.random.seed(4)
from Embedding_Layer_1.embedding import *
from Embedding_Combination_Layer_2.embedding_combination_layer import *
from Encoding_Layer_3.encoding_layer import *
from Cross_Interaction_Layer_4.cross_interaction import *
from Self_Interaction_Layer_5.self_interaction import *
from Output_Layer_10.decoder import *


class Bidaf_Model(nn.Module):

    def __init__(self, config):
        super(Bidaf_Model, self).__init__()
        self.config = config
        self.embedding = Embedding_layer(self.config)
        self.embedding_combination = Embedding_Combination_Layer(self.config)
        self.encoder = Encoding_Layer(self.config)
        self.cross_interaction = Cross_Interaction(self.config)
        self.self_interaction = Self_Interaction(self.config)
        self.decoder = Decoder(self.config)



    def forward(self, context_batch_word_indexes,context_batch_char_indexes,context_batch_word_mask,question_batch_word_indexes,question_batch_char_indexes,question_batch_word_mask,span_tensor_batch):
    # , context_word_indexes, context_word_mask, question_word_indexes, question_word_mask,span_tensor):
    # ,context_batch_word_indexes,context_batch_word_mask
    # ,question_batch_word_indexes,question_batch_word_mask
        context_word_embedding,context_char_embedding = self.embedding(context_batch_word_indexes,context_batch_char_indexes)
        context_embedding_combination = self.embedding_combination(context_word_embedding,context_char_embedding)

        question_word_embedding,question_char_embedding = self.embedding(question_batch_word_indexes,question_batch_char_indexes)
        question_embedding_combination = self.embedding_combination(question_word_embedding,question_char_embedding)


        passage_representation = self.encoder(context_embedding_combination,context_batch_word_mask)

        question_representation = self.encoder(question_embedding_combination,question_batch_word_mask)


        # A_Q_matrix,A_D_matrix,A_Q_vector,A_D_vector = query_attention_matrix,document_attention_matrix,query_attention_vector,document_attention_vector
        query_attention_matrix,document_attention_matrix,query_attention_vector,document_attention_vector = self.cross_interaction(question_representation, passage_representation)

        # A_Q_matrix,A_D_matrix,A_Q_vector,A_D_vector,question_representation, context_representation,document_word_sequence_mask
        self_match_representation = self.self_interaction(query_attention_matrix,document_attention_matrix,query_attention_vector,document_attention_vector,question_representation, passage_representation,context_batch_word_mask)
        # self_match_representation = U_matrix
        loss, index_start, index_end = self.decoder(self_match_representation, context_batch_word_mask, span_tensor_batch)


        return loss, index_start, index_end
