
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import os

class Highway_Maxout_Network(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, dropout_ratio):
        super(Highway_Maxout_Network, self).__init__()
        self.hidden_dim = hidden_dim # l
        self.maxout_pool_size = maxout_pool_size # p

        # Affine mapping from 5l ==> l
        self.r = nn.Linear(5 * hidden_dim, hidden_dim, bias=False) 
       

        # Affine mapping from 3*l ==> l*p
        self.max_out_layer1 = nn.Linear(3 * hidden_dim, hidden_dim*maxout_pool_size)
        
        # Affine mapping from l ==> l*p
        self.max_out_layer2 = nn.Linear(hidden_dim, hidden_dim*maxout_pool_size)
       
        # Affine mapping from 2*l ==> p
        self.max_out_layer3 = nn.Linear(2 * hidden_dim, maxout_pool_size)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, h_i, U, curr_mask_vector, index_i_minus_1, u_concatenated, mask_matrix, target=None):
        batch_size, max_word_length , _ = list(U.size())

        # concatenation of ( h_i of dimension = b x l ; u_concatenated of dimension = b x 4l ) along dimension 1 = gives b x 5l
        # self.r(b x 5l) ====> b x l (change of vector space)
        r = torch.tanh(self.r(torch.cat((h_i.view(-1, self.hidden_dim), u_concatenated), 1)))  # b x 5l => b x l
       

        # hidden_dim = l
        r_expanded = r.unsqueeze(1).expand(batch_size, max_word_length, self.hidden_dim).contiguous()  # b x m x l

        m_t1_input = torch.cat((U, r_expanded), 2).view(-1, 3*self.hidden_dim)  # b*m x 3l

        m_t1_output = self.max_out_layer1(m_t1_input)  # b*m x p*l
        
        m_t1_output_resized, _ = m_t1_output.view(-1, self.hidden_dim, self.maxout_pool_size).max(2) # b*m x l

        # m_t2_input =  m_t1_output_resized
        m_t2_output = self.max_out_layer2(m_t1_output_resized)  # b*m x l*p
        
        m_t2_output_resized, _ = m_t2_output.view(-1, self.hidden_dim, self.maxout_pool_size).max(2)  # b*m x l

        m_t3_input = torch.cat((m_t1_output_resized, m_t2_output_resized), 1)  # b*m x 2l
        alpha1 = self.max_out_layer3(m_t3_input)  # b * m x p
        alpha2, _ = alpha1.max(1)  # b*m
        alpha3 = alpha2.view(-1, max_word_length) # b x m

        alpha3 = alpha3 + mask_matrix  # b x m
        
        # alpha can be treated as probabilities that assign probability masses todifferent words in context. The word with
        # maximum weight(probability) becomes the index(start/end)
        alpha4 = F.softmax(alpha3, 1)  # b x m
        _, index_i = torch.max(alpha4, dim=1) # b

        if curr_mask_vector is None:
            curr_mask_vector = (index_i == index_i) # b
        else:
            index_i = index_i*curr_mask_vector.long()  # b
            index_i_minus_1 = index_i_minus_1*curr_mask_vector.long()  # b
            curr_mask_vector = (index_i != index_i_minus_1) # b

        step_loss = None
        
        

        target[target < 0] = 0
        
        
        ## loss is only calculated only on that the predicted index at i_th time-step which varies 
        ## from the predicted index at time-step (i-1)_th time-step
        if target is not None:
            step_loss = self.loss(alpha4, target)  # b
            step_loss1 = step_loss * curr_mask_vector.float() # b

        return index_i, curr_mask_vector, step_loss1 # all have dimension: b

class Dynamic_Decoder(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, max_number_of_iterations, dropout_ratio):
        super(Dynamic_Decoder, self).__init__()
        self.max_number_of_iterations = max_number_of_iterations
        
        # batch_first = True
        # Input: has a dimension of B * m * embedding_dim
        # Function parameters: input_size, hidden_size, num_layers_of_LSTM = 1(here)
        self.decoder = nn.LSTM(4 * hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=False)

        self.maxout_start = Highway_Maxout_Network(hidden_dim, maxout_pool_size, dropout_ratio)
        self.maxout_end = Highway_Maxout_Network(hidden_dim, maxout_pool_size, dropout_ratio)

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
        hidden_and_current_state_i = None
        start_target = None
        end_target = None
        
        # this sets the start and end target (ie. the y_label) for an answer
        if span_tensor is not None:
            # Dimension = B
            start_target = span_tensor[:,0]
            
            
            # Dimension = B
            end_target = span_tensor[:,1]
            
            
        # this is just an initialization of u_start
        # u_start_i_minus_1 is essentially u_start_zero outside the loop
        u_start_i_minus_1 = U[indices, start_i_minus_1, :]  # b x 2l
        
        # Why do we need an iterative procedure to predict the start and end indices for na answer ? 
        # Solution: there may exist several intuitive answer spans within the document, each corresponding to a
        # local maxima. An iterative technique to select an answer span by alternating between
        # predicting the start point and predicting the end point. This iterative procedure allows the model to
        # recover from initial local maxima corresponding to incorrect answer spans.
        for _ in range(self.max_number_of_iterations):
            u_end_i_minus_1 = U[indices, end_i_minus_1, :]  # b x 2l
            
            # u_concatenated is fed to the lstm
            u_concatenated = torch.cat((u_start_i_minus_1, u_end_i_minus_1), 1)  # b x 4l

            # the hidden_and_current_state_i = h_i,c_i are essentially hidden and current cell states 
            # for t = length of the sequence (for the one-directional lstm) after every iteration
            lstm_output, hidden_and_current_state_i = self.decoder(u_concatenated.unsqueeze(1), hidden_and_current_state_i)
            h_i, c_i = hidden_and_current_state_i

            # Inputs to the Highway_Maxout_Network(to find start index) are: hidden_state_i(h_i), start_i_minus_1(index), u_concatenated ==>(u_start_i_minus_1;u_end_i_minus_1) 
            start_i_minus_1, curr_mask_start, step_loss_start = self.maxout_start(h_i, U, curr_mask_start, start_i_minus_1,
                                                                u_concatenated, mask_matrix, start_target)
            u_start_i_minus_1 = U[indices, start_i_minus_1, :]  # b x 2l

            u_concatenated = torch.cat((u_start_i_minus_1, u_end_i_minus_1), 1)  # b x 4l

            # Inputs to the Highway_Maxout_Network(to find end index) are: hidden_state_i(h_i), end_i_minus_1(index), u_concatenated ==>(u_start_i_minus_1;u_end_i_minus_1) 
            end_i_minus_1, curr_mask_end, step_loss_end = self.maxout_end(h_i, U, curr_mask_end, end_i_minus_1,
                                                              u_concatenated, mask_matrix, end_target)

            # we minimize the cumulative softmax cross entropy of the start and end points across all iterations.
            if span_tensor is not None:
                step_loss = step_loss_start + step_loss_end
                step_losses.append(step_loss)

            
            results_mask_start.append(curr_mask_start) # appends all the curr_mask_start ==> dimension: b x num_iterations
            results_start.append(start_i_minus_1) # appends all the start_indexes ==> dimension: b x num_iterations
            results_mask_end.append(curr_mask_end) # appends all the curr_mask_end ==> dimension: b x num_iterations
            results_end.append(end_i_minus_1) # appends all the end_indexes ==> dimension: b x num_iterations

        
        result_pos_start1 = torch.sum(torch.stack(results_mask_start, 1), 1).long()

        result_pos_start = result_pos_start1 - 1
        index_start = torch.gather(torch.stack(results_start, 1), 1, result_pos_start.unsqueeze(1)).squeeze()

        result_pos_end1 = torch.sum(torch.stack(results_mask_end, 1), 1).long()

        result_pos_end = result_pos_end1 - 1
        index_end = torch.gather(torch.stack(results_end, 1), 1, result_pos_end.unsqueeze(1)).squeeze()

        loss = None

        if span_tensor is not None:
            sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
            batch_avg_loss = sum_losses / self.max_number_of_iterations
            loss = torch.mean(batch_avg_loss)

            
        return loss, index_start, index_end

