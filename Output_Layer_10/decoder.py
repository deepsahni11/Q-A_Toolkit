
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import os
import numpy as np
from Output_Layer_10.helper import *
torch.manual_seed(4)
np.random.seed(4)

"""
Dynamic_Decoder:
1) init function: creates object of Highway Maxout Highway_Maxout_Network
   INPUTS: config

2) forward function:
   INPUTS: U (final document representation after passing though bi-lstm): B x m x 2l
            document_word_sequence_mask: B x m
            span_tensor: B x 2
   OUTPUTS: loss, index_start, index_end
"""
class Decoder(nn.Module):
    def __init__(self,config):
        super(Decoder, self).__init__()
        self.config = config

        if(self.config.decoder_type == "bidaf"):
            self.decoder = BIDAF_decoder(self.config)
        elif(self.config.decoder_type == "dcn"):
            self.decoder = DCN_Dynamic_Decoder(self.config)

    def forward(self,output_to_self_interaction,document_word_sequence_mask,span_tensor):
        if(self.config.decoder_type == "bidaf"):
            loss,index_start,index_end = self.decoder(output_to_self_interaction,span_tensor)
            return loss,index_start,index_end
        elif(self.config.decoder_type == "dcn"):
            loss,index_start,index_end = self.decoder(output_to_self_interaction,document_word_sequence_mask,span_tensor)
            return loss,index_start,index_end

class BIDAF_decoder(nn.Module):
    def __init__(self,config):
        super(BIDAF_decoder, self).__init__()
        self.config = config
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.answer_start_logits = predict_start_bidaf(config)
        self.mid_processing_layer = mid_processing_unit(config)
        self.answer_end_logits = predict_end_bidaf(config)
    def forward(self,self_match_representation,span_tensor):
        start_logits = self.answer_start_logits(self_match_representation[0], self_match_representation[1])
        mid_processing = self.mid_processing_layer(self_match_representation[1])
        end_logits = self.answer_end_logits(self_match_representation[0], mid_processing)

        _, index_start = torch.max(start_logits, 1)
        _, index_end   = torch.max(end_logits,  1)

        answer_start_batch = span_tensor[:,0]
        answer_end_batch = span_tensor[:,1]


        step_loss = self.loss_function(start_logits, answer_start_batch)
        step_loss += self.loss_function(end_logits, answer_end_batch)

        return step_loss,index_start, index_end


class DCN_Dynamic_Decoder(nn.Module):
    def __init__(self, config):
        super(DCN_Dynamic_Decoder, self).__init__()
        self.config = config
        self.max_number_of_iterations = self.config.max_number_of_iterations

        self.hidden_dim = self.config.hidden_dim
        # batch_first = True
        # Input: has a dimension of B * m * embedding_dim
        # Function parameters: input_size, hidden_size, num_layers_of_LSTM = 1(here)
        self.decoder = nn.LSTM(4 * self.config.hidden_dim, self.config.hidden_dim, 1, batch_first=True, bidirectional=False)

        self.maxout_start = Highway_Maxout_Network(self.config.hidden_dim, self.config.maxout_pool_size, self.config.dropout_ratio)
        self.maxout_end = Highway_Maxout_Network(self.config.hidden_dim, self.config.maxout_pool_size, self.config.dropout_ratio)

    def initHidden(self,batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad = False) # Initial hidden state
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad = False) # Initial cell state
        return h0, c0


    def forward(self, U, document_word_sequence_mask,span_tensor):
        batch_size, max_word_length, _ = list(U.size()) # U has dimension : B x m x 2l

        curr_mask_start,  curr_mask_end = None, None
        results_mask_start, results_start = [], []
        results_mask_end, results_end = [], []
        step_losses = []


        # dimension = B x m
        mask_matrix = (1.0 - document_word_sequence_mask.float()) * (-1e30)

        # dimension = B
        indices = torch.arange(0, batch_size)


        # initialize start_i_minus_1, end_i_minus_1: these are the initial values of start and end indices
        # start_i_minus_1 = the first index for the context/question
        # end_i_minus_1 = the last index for the context/question

        # dimension = B
        start_i_minus_1 = torch.zeros(batch_size).long()

        # dimension = B
        end_i_minus_1 = torch.sum(document_word_sequence_mask, 1) - 1



        # After every iteration the hidden and current state
        # at t = length of the sequence (for the one-directional lstm) will
        # be returned by the lstm
        # the hidden_state_i(h_i) will serve as an input to next lstm
        hidden_and_current_state_i = self.initHidden(batch_size)
        start_target = None
        end_target = None

        # this sets the start and end target (ie. the y_label) for an answer
        if span_tensor is not None:
            # Dimension = B
            start_target = span_tensor[:,0]


            # Dimension = B
            end_target = span_tensor[:,1]



        # this is just an initialization of u_start_i_minus_1
        # u_start_i_minus_1 is essentially u_start_zero outside the loop
        u_start_i_minus_1 = U[indices, start_i_minus_1, :]  # B x 2l

        # Why do we need an iterative procedure to predict the start and end indices for an answer ?
        # Solution: there may exist several intuitive answer spans within the document, each corresponding to a
        # local maxima. An iterative technique to select an answer span by alternating between
        # predicting the start point and predicting the end point. This iterative procedure allows the model to
        # recover from initial local maxima corresponding to incorrect answer spans.
        for _ in range(self.max_number_of_iterations):
            u_end_i_minus_1 = U[indices, end_i_minus_1, :]  # B x 2l

            # u_concatenated is fed to the lstm
            u_concatenated = torch.cat((u_start_i_minus_1, u_end_i_minus_1), 1)  # B x 4l



            # the hidden_and_current_state_i = h_i,c_i are essentially hidden and current cell states
            # for t = length of the sequence (for the one-directional lstm) after every iteration
            # u_concatenated.unsqueeze(1) has a dimension : B x 1 x 4l
            lstm_output, hidden_and_current_state_i = self.decoder(u_concatenated.unsqueeze(1), hidden_and_current_state_i)

            # h_i has dimension = 1 x B x l
            # c_i has dimension = 1 x B x l
            h_i, c_i = hidden_and_current_state_i



            # Inputs to the Highway_Maxout_Network(to find start index) are: hidden_state_i(h_i), start_i_minus_1(index), u_concatenated ==>(u_start_i_minus_1;u_end_i_minus_1)
            start_i_minus_1, curr_mask_start, step_loss_start = self.maxout_start(h_i, U, curr_mask_start, start_i_minus_1,
                                                                u_concatenated, mask_matrix, start_target)



            u_start_i_minus_1 = U[indices, start_i_minus_1, :]  # B x 2l

            u_concatenated = torch.cat((u_start_i_minus_1, u_end_i_minus_1), 1)  # b x 4l

            # Inputs to the Highway_Maxout_Network(to find end index) are: hidden_state_i(h_i), end_i_minus_1(index), u_concatenated ==>(u_start_i_minus_1;u_end_i_minus_1)
            end_i_minus_1, curr_mask_end, step_loss_end = self.maxout_end(h_i, U, curr_mask_end, end_i_minus_1,
                                                              u_concatenated, mask_matrix, end_target)

            # we minimize the cumulative softmax cross entropy of the start and end points across all iterations.
            if span_tensor is not None:
                step_loss = step_loss_start + step_loss_end
#                 print(step_loss)
                step_losses.append(step_loss)

            results_mask_start.append(curr_mask_start) # appends all the curr_mask_start ==> dimension: num_iterations x B
            results_start.append(start_i_minus_1) # appends all the start_indexes ==> dimension: num_iterations x B
            results_mask_end.append(curr_mask_end) # appends all the curr_mask_end ==> dimension: num_iterations x B
            results_end.append(end_i_minus_1) # appends all the end_indexes ==> dimension: num_iterations x B



        # Dimension = B
        result_pos_start1 = torch.sum(torch.stack(results_mask_start, 1), 1).long()
        result_pos_start = result_pos_start1 - 1

        # Dimension = B
        index_start = torch.gather(torch.stack(results_start, 1), 1, result_pos_start.unsqueeze(1)).squeeze()

        # Dimension = B
        result_pos_end1 = torch.sum(torch.stack(results_mask_end, 1), 1).long()
        result_pos_end = result_pos_end1 - 1

        # Dimension = B
        index_end = torch.gather(torch.stack(results_end, 1), 1, result_pos_end.unsqueeze(1)).squeeze()

        loss = None

        if span_tensor is not None:
            # step losses has dimension = num_iterations x B
            sum_losses = sum(step_losses)
            batch_avg_loss = sum_losses / self.max_number_of_iterations
            loss = batch_avg_loss

        # X = [loss, index_start, index_end]
        # print (X, len(X))
        # print("loss")
        # print(loss)
        # print("index_start")
        # print(index_start)
        # print("index_end")
        # print(index_end)
        return loss, index_start, index_end
