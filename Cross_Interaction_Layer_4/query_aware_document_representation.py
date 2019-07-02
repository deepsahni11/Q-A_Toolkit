import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def softmax(input, axis=1, f=None):
    input_size = input.size()
    
    trans_input = input.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    
    soft_max_2d = f(input_2d)
    
    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size)-1)


class queryAwareDocument(nn.Module):
    def __init__(self, qad_rep='method_1'):
        super(queryAwareDocument, self).__init__()
        self.qad_rep = qad_rep
        self.softmax_f = torch.nn.Softmax()

    def forward(self, passage_outputs, question_outputs, bilinear_comp):
        if self.qad_rep == 'method_1':
            bilinear_comp = softmax(bilinear_comp, axis=1, f= self.softmax_f)
            bilinear_comp = bilinear_comp.permute(0,2,1)

            qad_rep = torch.bmm(bilinear_comp, passage_outputs)

            return None,qad_rep

        if self.qad_rep== 'method_2':
            softmax_col = softmax(bilinear_comp, axis = 2, f = self.softmax_f)
            max_sofmax_col, _ = torch.max(softmax_col, dim = 2)
            passage_outputs_list = torch.unbind(passage_outputs, 0)
            max_softmax_col_list = torch.unbind(max_sofmax_col, 0)
            
            temp_rep = []
            for i in range(len(max_softmax_col_list)):
                temp_softmax = torch.unsqueeze(max_softmax_col_list[i], dim=0)
                temp_rep.append(torch.mm(temp_softmax, passage_outputs_list[i]))

       	  
            qad_rep_vec = torch.stack(temp_rep, 0)
            qad_rep_vec = torch.squeeze(qad_rep_vec)
            return qad_rep_vec, None