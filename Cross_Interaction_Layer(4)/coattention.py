
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import os


class Coattention_Encoder(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, embedding_matrix, max_number_of_iterations, dropout_ratio):
        super(Coattention_Encoder, self).__init__()
        self.hidden_dim = hidden_dim

        ## nn.Linear(input_dim, output_dim)
        # Affine mapping from l ==> l
        self.question_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.fusion_bilstm = Fusion_BiLSTM(hidden_dim, dropout_ratio)
#         self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, question_representation, context_representation,document_word_sequence_mask):
        
        ####### m = max length of instances in one batch of document ;  n= max length of instances in one batch of question
        Q = question_representation # B x (n + 1) x l
        D = context_representation  # B x (m + 1) x l
        
        print("question_representation.(Output to Encoder Layer) ==  " + str(Q.size()))
        print("context_representation. (Output to Encoder Layer)  ==  " + str(D.size()))

        # view function is meant to reshape the tensor.(Similar to reshape function in numpy)
        # view( row_size = -1 ,means that number of rows are unknown, column_size)
        # pass the Q tensor through a non-linearity 
        Q2 = torch.tanh(self.question_proj(Q.view(-1, self.hidden_dim))).view(Q.size()) #B x (n + 1) x l

        ##################################   Co-Attention starts here  #######################################
        
        ########################################   Step - 1  ##################################################
        # transpose(tensor, first_dimension to be transposed, second_dimension to be transposed)
        Q_transpose = torch.transpose(Q2, 1, 2) #dimension: B x l x (n + 1)
        
        # Performs a batch matrix-matrix product of matrices stored in batch1 and batch2.
        # batch1 and batch2 must be 3-D tensors each containing the same number of matrices.
        L = torch.bmm(D, Q_transpose) # dimension of L : B x (m + 1) x (n + 1)

        ####################################### Step-2 ######################################################
        A_Q = F.softmax(L, dim=2) # B x (m + 1) x (n + 1)


        D_transpose = torch.transpose(D, 1, 2) #dimension: B x l x (m + 1)
        C_Q = torch.bmm(D_transpose, A_Q) # (B x l x (m + 1)) x (B x (m + 1) x (n + 1)) => B x l x (n + 1)

        ####################################### Step-3 #######################################################
        L_tranpose = torch.transpose(L,1,2)
        A_D = F.softmax(L_tranpose, dim=2)  # B x (n + 1) x (m + 1)
        
        
        # concatenation along dimension=1:(B x l x (n + 1) ; B x l x (n + 1)  -----> B x 2l x (n + 1) ) x (B x (n + 1) x (m + 1)) ====> B x 2l x (m + 1)
        C_D = torch.bmm(torch.cat((Q_transpose, C_Q), 1), A_D) # B x 2l x (m + 1)
        C_D_transpose = torch.transpose(C_D, 1, 2)  # B x (m + 1) x 2l

        
        #######################################  Step-4 ##########################################################
        #fusion BiLSTM
        # concatenation along dimension = 2:  (B x (m + 1) x 2l ; B x (m + 1) x l  -----> B x (m + 1) x 3l )
        bi_lstm_input = torch.cat((C_D_transpose, D), 2) # B x (m + 1) x 3l
       
        U = self.fusion_bilstm(bi_lstm_input, document_word_sequence_mask) # B x m x 2l
        
        print("size of U.(U is output of Co-attention encoder) ==  " + str(U.size()))
        
        return U


class Fusion_BiLSTM(nn.Module):
    def __init__(self, hidden_dim, dropout_ratio):
        super(Fusion_BiLSTM, self).__init__()
        
         # batch_first = True
        # Input: has a dimension of B * m * embedding_dim
        # Function parameters: input_size, hidden_size, num_layers_of_LSTM = 1(here)
        self.fusion_bilstm = nn.LSTM(3 * hidden_dim, hidden_dim, 1, batch_first=True,
                                     bidirectional=True, dropout=dropout_ratio)
        
#         self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, word_sequence_embeddings, word_sequence_mask):
        
        # stores length of per instance for context/question
        length_per_instance = torch.sum(word_sequence_mask, 1)
        
      
        # All RNN modules accept packed sequences as inputs.
        # Input: word_sequence_embeddings has a dimension of B x m+1 x 3l (l is the size of the glove_embedding/ pre-trained embedding/embedding_dim)
        packed_word_sequence_embeddings = pack_padded_sequence(word_sequence_embeddings, length_per_instance, batch_first=True,enforce_sorted=False)
        
        # since the input was a packed sequence, the output will also be a packed sequence
        output, _ = self.fusion_bilstm(packed_word_sequence_embeddings)
        
        # Pads a packed batch of variable length sequences.
        # It is an inverse operation to pack_padded_sequence().
        # dimension:  B x m x 2l
        output_to_BiLSTM_padded, _ = pad_packed_sequence(output, batch_first=True)


        return output_to_BiLSTM_padded