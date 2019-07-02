import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class predict_end_bidaf(nn.Module):

    def __init__(self, input_size):

        super(predict_end_bidaf, self).__init__()
        self.w = torch.autograd.Variable(torch.Tensor(10*input_size, 1))
        init.xavier_normal(self.w)

    def forward(self, *args):

        list_of_rep = []

        for arg in args:
            list_of_rep.append(arg)

        concat_rep = torch.cat(list_of_rep, dim=-1)

        temp = torch.squeeze(self.w)
        temp = torch.unsqueeze(temp, dim=0)
        temp = torch.unsqueeze(temp, dim=0)
        temp = temp.repeat(concat_rep.size()[0], concat_rep.size()[1], 1)

        res = temp * concat_rep 
        res = torch.sum(res, dim=-1)
        logits = res
        return logits