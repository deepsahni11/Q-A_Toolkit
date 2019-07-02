import torch
from torch import nn
import query_aware_document_representation
import document_aware_query_representation

from query_aware_document_representation import softmax 

class crossInteractionLayer(nn.Module):
    def __init__(self, daq_method='method_1', qad_method='method_1', cross_int_method='method_1'):
        super(crossInteractionLayer, self).__init__()

        self.daq_layer = query_aware_document_representation.queryAwareDocument(qad_method) 
        self.qad_layer = document_aware_query_representation.documentAwarequery(daq_method)
        self.softmax_f = torch.nn.Softmax()
    def forward(self, passage_outputs, question_outputs, bilinear_comp):

        daq_vec, daq_mat = self.daq_layer(passage_outputs, question_outputs, bilinear_comp)
        qad_vec, qad_mat = self.qad_layer(passage_outputs, question_outputs, bilinear_comp)

        if (cross_int_method == 'method_1'):
            bilinear_comp = softmax(bilinear_comp, axis=2, f = self.softmax_f)
            temp = torch.bmm(bilinear_comp, qad_mat)

            cross_int_mat = torch.cat((daq_vec, temp), dim=-1)
            cross_int_vec = None
        
        return daq_vec, daq_mat, qad_vec, qad_mat, cross_int_vec, cross_int_mat
