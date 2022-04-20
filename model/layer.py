import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1)))


def masked_normalization(logits, mask):
    scores = F.softmax(logits, dim=-1)

    # apply the mask - zero out masked timesteps
    print("scores:{}".format(scores.size()))
    print("mask:{}".format(mask.size()))
    masked_scores = scores * mask.float()

    # re-normalize the masked scores
    normed_scores = masked_scores.div(masked_scores.sum(-1, keepdim=True))

    return normed_scores
class SelfAttention(nn.Module):
    def __init__(self, attention_size,
                 baseline=False,
                 batch_first=True,
                 layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        if baseline:
            layers = 2
        modules = []
        #一层attention
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))
        # modules.append(activation)
        # modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequence, lengths):
        # sequence size: batch_size x length x rnn size

        energies = self.attention(sequence).squeeze()

        # construct a mask, based on sentence lengths
        mask = sequence_mask(lengths, energies.size(1))

        # scores = masked_normalization_inf(energies, mask)
        scores = masked_normalization(energies, mask)
        # scores size: batch_size x length
        contexts = (sequence * scores.unsqueeze(-1)).sum(1)

        return contexts, scores
