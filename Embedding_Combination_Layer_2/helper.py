
class HighwayLayer(nn.Module):
    # TODO: We may need to add weight decay here
    def __init__(self, size, bias_init=0.0, nonlin=nn.ReLU(inplace=True), gate_nonlin=F.sigmoid):
        super(HighwayLayer, self).__init__()

        self.nonlin = nonlin
        self.gate_nonlin = gate_nonlin
        self.lin = nn.Linear(size, size)
        self.gate_lin = nn.Linear(size, size)
        self.gate_lin.bias.data.fill_(bias_init)

    def forward(self, x):
        trans = self.nonlin(self.lin(x))
        gate = self.gate_nonlin(self.gate_lin(x))
        return torch.add(torch.mul(gate, trans), torch.mul((1 - gate), x))


class HighwayNet(nn.Module):
    def __init__(self, depth, size):
        super(HighwayNet, self).__init__()

        layers = [HighwayLayer(size) for _ in range(depth)]
        self.main_ = nn.Sequential(*layers)

    def forward(self, x):
        return self.main_(x)

class concatEmbeddings(nn.Module):

    def __init__(self):

        super(concatEmbeddings, self).__init__()

    def forward(word_embeddings, char_embeddings):

        res = torch.concat([word_embeddings, char_embeddings], dim=2)
        return res


class identityModule(nn.Module):

    def __init__(self):
        super(identityModule, self).__init__()


    def forward(word_embeddings):

        return word_embeddings
