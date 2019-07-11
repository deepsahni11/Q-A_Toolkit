
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import os
import numpy as np
torch.manual_seed(4)
np.random.seed(4)


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



        ## loss is only calculated only on that the predicted index at i_th time-step which varies
        ## from the predicted index at time-step (i-1)_th time-step
        if target is not None:
            step_loss = self.loss(alpha3, target)  # b

        return index_i, curr_mask_vector, step_loss # all have dimension: b
