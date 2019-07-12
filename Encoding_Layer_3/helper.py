import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def get_pretrained_embedding(embedding_matrix):
    embedding = nn.Embedding(*embedding_matrix.shape)
    embedding.weight = nn.Parameter(torch.from_numpy(embedding_matrix).float())
    embedding.weight.requires_grad = False
    return embedding
