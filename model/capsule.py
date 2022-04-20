import logging
import torch
import torch.nn as nn


class Capsule(nn.Module):
    def __init__(self, opt):
        super(Capsule, self).__init__()

        #capsule input_dim应该和cnn out_put dim是一致的
        self.input_dim_capsule = opt['capsule_input_dim']
        self.dim_capsule = opt['capsule_out_dim']
        self.num_capsule = opt['num_capsule']
        self.batch_size = opt['batch_size']
        self.share_weights =opt['share_weights']
        self.num_iterations = opt['num_iterations']

        #share_weights ：True或者false区别在哪里
    
        if self.share_weights:
            W = torch.zeros(1, self.input_dim_capsule, self.num_capsule * self.dim_capsule)
        else:
            W = torch.zeros(self.batch_size, self.input_dim_capsule, self.num_capsule * self.dim_capsule)

        W = nn.init.xavier_normal_(W)
        self.W = nn.Parameter(W)

    def forward(self, x):
        """
        x: [B, L, H]      # 从 CNN / RNN 得到的结果
            L 作为 input_num_capsules, H 作为 input_dim_capsule
        """
        
        B, I, _ = x.size()  # I 是 input_num_capsules:(L+1)*C
        O, F = self.num_capsule, self.dim_capsule
       # if self.share_weights:
        #    W = torch.zeros(1, self.input_dim_capsule, self.num_capsule * self.dim_capsule)
        #else:
        #W = torch.zeros(B, self.input_dim_capsule, self.num_capsule * self.dim_capsule)
       # W = torch.zeros(B, Hs, self.num_capsule * self.dim_capsule)

        #W = nn.init.xavier_normal_(W)
        #self.W = nn.Parameter(W).cuda()
        #print("W:{}".format(W.size()))
        #print("x:{}".format(x.size()))
        #print("w:{}".format(self.W.size()))
        u = torch.matmul(x, self.W)
        #u（i->j)
        u = u.view(B, I, O, F).transpose(1, 2)  # [B, O, I, F] 
        #暂时不使用W
        #w:1*d_int*num*d_out
        #u = torch.matmul(x, self.W)

        b = torch.zeros_like(u[:, :, :, 0]).to(device=u.device)  # [B, O, I]
        for i in range(self.num_iterations):
            c = torch.softmax(b, dim=1)  # [B, O_s, I]
            v = torch.einsum('boi,boif->bof', [c, u])  # [B, O, F]
            v = squash(v)
            b = torch.einsum('bof,boif->boi', [v, u])  # [B, O, I]

        return v  # [B, O, F] [B, num_capsule, dim_capsule]

def squash(x):
    x_norm = x.norm(p=2, dim=-1, keepdim=True)
    mag = x_norm**2
    out = x / x_norm * mag / (1 + mag)

    return out