import torch
from torch import nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class predict_start_bidaf(nn.Module):

    def __init__(self, input_size):

        super(predict_start_bidaf, self).__init__()
        self.w = torch.autograd.Variable(torch.Tensor(10*input_size, 1).type(torch.cuda.FloatTensor))
        init.xavier_normal(self.w)

    def forward(self, *args):

        list_of_rep = []

        for arg in args:
            list_of_rep.append(arg)

        x = list_of_rep
        concat_rep = torch.cat((x[0],x[1]), len(x[0].size()) -1 )

        temp = torch.squeeze(self.w)
        temp = torch.unsqueeze(temp, dim=0)
        temp = torch.unsqueeze(temp, dim=0)
        #temp = temp.repeat(x[0].size()[0], x[0].size()[1], 1)

        res = temp * concat_rep 
        res = torch.sum(res, dim=-1)
        logits = res
        return logits