import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import code
import pickle
import os
from torch import autograd
import embedding 
import encoding_layer 
import bilinear_compute
import encoding_combination_layer
import query_aware_document_representation
import document_aware_query_representation
import self_match
import dataset_iterator_squad
import predict_start
import predict_end
import mid_processingoutputlayer
import torch.nn.functional as F
from numpy import genfromtxt
from torch.autograd import Variable
from torch.nn import Embedding
from argparse import ArgumentParser
import gc

dtype = torch.cuda.FloatTensor
ldtype = torch.cuda.LongTensor


PADDING = 'VALID'
torch.cuda.manual_seed_all(4)
torch.manual_seed(4) 

class BiDAF(nn.Module):
    def __init__(self, config):
        super(BiDAF, self).__init__()
        self.config = config
        self.logits = None
        self.yp = None



        #self.encoding_layer_type = config.encoding_layer_type
        #self.bilinear_compute    = config.bilinear_compute_type
        #self.encoding_combination_layer = config.encoding_combination_layer_type
        #self.query_aware_document_representation = config.query_aware_document_representation_type
        #self.document_aware_query_representation_type = config.document_aware_query_representation_type
        
        #self.self_match_representation_type =  config.self_match_representation_type

        #self.predict_start_type = config.predict_start_type
        #self.mid_processingoutputlayer_type = config.mid_processingoutputlayer_type
        #self.predict_end_type = config.predict_end_type

        self.char_embed_type = config.char_embed_type

        self.char_embedding_input_size  = config.char_emb_size
        self.word_embedding_size        = config.word_emb_size
        self.char_embedding_output_size = config.char_out_size

        self.batch_size          = config.batch_size
        #self.max_doc_length_word = config.max_num_sent_words
        #self.max_query_words     = config.max_query_words
        self.char_vocab_size     = config.char_vocab_size

        self.hidden_size   = config.hidden_size
        self.max_word_size = config.max_word_size
        self.num_layers = config.num_layers

        self.data_dir = config.data_dir
        self.emb_dir  = config.emb_dir

        self.char_embed_type = config.char_embed_type
        self.use_char_emb = config.use_char_emb

        self.dictionaries = pickle.load(open(os.path.join(self.emb_dir, "dictionaries.pkl")))

        self.word_vocab_size     = len(self.dictionaries["word"])
        self.dropout = 0 #config.dropout
        self.rnn_bidirectional = config.rnn_bidirectional

        padding_idx = 0
        word_embedding_tensor = pickle.load(open(os.path.join(self.emb_dir, "word_embeddings" + str(config.word_emb_size) + ".pkl")))
        self.word_embed_mat = nn.Embedding(self.word_vocab_size, embedding_dim=config.word_emb_size, padding_idx=padding_idx)
        self.word_embed_mat.weight.data.copy_(torch.FloatTensor(word_embedding_tensor))
	self.word_embed_mat.weight.requires_grad = config.finetune
        #self.word_embed_mat.cpu()

        char_embedding_tensor = pickle.load(open(os.path.join(self.emb_dir, "char_embeddings" + str(config.char_emb_size) +  ".pkl")))
        self.char_embed_mat = nn.Embedding(self.char_vocab_size, embedding_dim=config.char_emb_size, padding_idx = padding_idx)
	self.char_embed_mat.weight.data.copy_(torch.FloatTensor(char_embedding_tensor))
        #self.word_embed = embedding.WordLevelEmbedding(self.word_vocab_size, config.word_emb_size,
        #                                               initial_word_embedding=self.dictionaries["word"])
        if (self.char_embed_type == 'RNN'):
            self.char_embed = embedding.CharLevelEmbeddingRNN(config.char_vocab_size,
                                           config.char_emb_size)

        elif (self.char_embed_type == "CNN"):
            # char-level convs
            filter_sizes = list(map(int, config.out_channel_dims.split(',')))
            heights = list(map(int, config.filter_heights.split(',')))
            self.filter_sizes = filter_sizes
            self.heights = heights
            self.char_embed = embedding.CharLevelEmbeddingCNN(config.is_train, config.keep_prob,  kernel_dim=filter_sizes[0], kernel_size=heights, embedding_dim=config.char_emb_size)
            #self.char_embedding_query = embedding.CharLevelEmbeddingCNN(config.is_train, config.keep_prob)
            self.emb_size = self.config.word_emb_size + self.config.char_emb_out_size 

        self.encoding_comb_layer = encoding_combination_layer.HighwayCombination(2, self.emb_size)
        self.encoding_layer = encoding_layer.EncodingLayerRNN(input_size = self.emb_size, 
                                                              hidden_size = self.hidden_size, 
                                                              num_layers = self.num_layers,
                                                              bidirectional=self.rnn_bidirectional, 
                                                              dropout = self.dropout)

        self.bilinear_matrix = bilinear_compute.bidaf_bilinear(passage_vec_size = self.hidden_size,
                                                                 query_vec_size = self.hidden_size)

        
        self.document_aware_query_representation = document_aware_query_representation.documentAwareQuery(config.documentAwareQuery_method)
        self.query_aware_document_representation  = query_aware_document_representation.queryAwareDocument(config.queryAwareDocument_method)
        
        self.self_match = self_match.bidaf_selfmatch(input_size = self.hidden_size, num_layers=self.num_layers, bidirectional=self.rnn_bidirectional,
                                                     dropout=self.dropout, hidden_size=self.hidden_size)

        self.predict_start_logits = predict_start.predict_start_bidaf(self.hidden_size)

        self.mid_processingoutputlayer = mid_processingoutputlayer.mid_processing_unit(input_size=self.hidden_size,
                                          bidirectional=True, dropout = self.dropout, mid_processing='bidaf')

        self.predict_end_logits = predict_start.predict_start_bidaf(self.hidden_size)

    def forward(self, content_batches, query_batches):

        temp_batches = content_batches
        """
        for i,v in content_batches.iteritems():
            temp_batches[i] = Variable(torch.cuda.LongTensor(v))
            temp_batches[i].requires_grad = False
        content_batches = temp_batches
        temp_batches1 = query_batches
        for i,v in query_batches.iteritems():
            temp_batches1[i] = Variable(torch.cuda.LongTensor(v)) 
            temp_batches1[i].requires_grad = False
        query_batches = temp_batches1
        """

        encoded_content = self.word_embed_mat(content_batches["token"])
        encoded_query   = self.word_embed_mat(query_batches["token"])

        temp1 = content_batches["tokenchar_"]
        temp1_size = content_batches["token_char"].size()
        temp1 = temp1.view(self.config.batch_size, -1)
        encoded_content_char = self.char_embed_mat(temp1)
        encoded_content_char = encoded_content_char.view(temp1_size[0], temp1_size[1], temp1_size[2], -1)

        temp1 = query_batches["tokenchar_"]
        temp1_size = query_batches["tokenchar_"].size()
        temp1 = temp1.view(self.config.batch_size, -1)
        encoded_query_char = self.char_embed_mat(temp1)
        encoded_query_char = encoded_query_char.view(temp1_size[0], temp1_size[1], temp1_size[2], -1)


        if (self.use_char_emb):
            encoded_content_char = self.char_embed(encoded_content_char)
            encoded_query_char   = self.char_embed(encoded_query_char)

        #print (encoded_content_char.size(), encoded_query_char.size(), encoded_content.size(), encoded_query.size())
        if (self.use_char_emb):

            passage_combine_embeddings = self.encoding_comb_layer(encoded_content, encoded_content_char)
            query_combine_embeddings   = self.encoding_comb_layer(encoded_query, encoded_query_char)

        #passage_combine_embeddings, query_combine_embeddings = encoded_content.repeat(1,1,1), encoded_query.repeat(1,1,1)
        passage_encodings, passage_hidden_state, query_encodings, query_hidden_state = self.encoding_layer(passage_combine_embeddings, query_combine_embeddings)
        #passage_encodings, query_encodings = passage_combine_embeddings.repeat(1,1,1), query_combine_embeddings.repeat(1,1,1)
        #print (passage_encodings.size(), query_encodings.size())
        bilinear_mat      = self.bilinear_matrix(self.config.batch_size, passage_encodings, query_encodings, passage_combine_embeddings.size()[1], query_combine_embeddings.size()[1])

        #print (bilinear_mat)
        qad_vector, qad_matrix = self.query_aware_document_representation(passage_encodings, query_encodings, bilinear_mat)
        daq_vector, daq_matrix = self.document_aware_query_representation(passage_encodings, query_encodings, bilinear_mat)


        self_match_representation = self.self_match(passage_encodings, query_encodings, qad_vector,qad_matrix,daq_vector,
                                                    daq_matrix)
        #self_match_representation = []
        #self_match_representation.append(passage_combine_embeddings.repeat(1,1,4))
        #self_match_representation.append(passage_encodings)
        start_logits = self.predict_start_logits(self_match_representation[0], self_match_representation[1])

        mid_process  = self.mid_processingoutputlayer(self_match_representation[1])
        end_logits   = self.predict_end_logits(self_match_representation[0], mid_process)
        gc.collect()
        return start_logits, end_logits
