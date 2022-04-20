import torch
from torch import nn

from model.layer import SelfAttention

class AffectiveAttention(nn.Module):
    def __init__(self,feature_size,attention_size):
        super(AffectiveAttention, self).__init__()

       

        #data
       # self.attention_type == "gate"
        self.feature_size=feature_size
        self.attention_size=attention_size

        self.gate = nn.Linear(self.feature_size, self.attention_size)
        self.sigmoid = nn.Sigmoid()
        self.attention = SelfAttention(attention_size=self.attention_size,
                                       dropout=0.0)

    def forward(self, drug_vec, patient_vec, lengths=None):
        """

        Args:
            src: token indices. 2D shape: (batch, max_len)
            features: additional features. 3D shape: (batch, max_len, feat_dim)
            lengths: actual length of each sample in batch. 1D shape: (batch)
        Returns:

        """
        # step 1: embed the sentences；50*300
        features=drug_vec
        attention_input=patient_vec
       # print("features:{}".format(features.size()))
       # print("attention_input:{}".format(attention_input.size()))
        temp=self.gate(features)
        #print("temp:{}".format(temp.size()))
       # print(attention_input.size())
        c = self.sigmoid(self.gate(features)) * attention_input
        # step 4: attend over the features
        representations, attentions = self.attention(c, lengths)

        representations = (attention_input * attentions.unsqueeze(-1)).sum(1)

        return representations