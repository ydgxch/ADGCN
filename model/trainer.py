"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.aggcn import GCNClassifier
from utils import torch_utils


class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def to_one_hot(x,length):
    """
    :param x:        [B]    一般是 target 的值
    :param length:    L     一般是关系种类树
    :return:         [B, L]  每一行，只有对应位置为1，其余为0
    """
    B = x.size(0)
    x_one_hot = torch.zeros(B, length)
    for i in range(B):
        x_one_hot[i, x[i]] = 1.0

    return x_one_hot.to(device=x.device)

def unpack_batch(batch, cuda):
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch[:10]]
        labels = Variable(batch[10].cuda())
    else:
        inputs = [Variable(b) for b in batch[:10]]
        labels = Variable(batch[10])
    tokens = batch[0]
    head = batch[5]
    subj_pos = batch[6]
    obj_pos = batch[7]
    #从batch中读出
    '''
    subj_vec=batch[10]
    obj_vec=batch[11]
    subj_pvec=batch[12]
    obj_pvec=batch[13]
    '''
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, tokens, head, subj_pos, obj_pos, lens


class Myloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.autocriterion=nn.MSELoss()
    def forward(self,x,y):
        loss_main=self.criterion(x,y)
        return loss_main

def MarginLoss(predict, target):
        m_plus, m_minus, loss_lambda = 0.9, 0.1, 0.5

        target = to_one_hot(target,5)
        max_l = (torch.relu(m_plus - predict))**2
        max_r = (torch.relu(predict - m_minus))**2
       
        loss = target * max_l + loss_lambda * (1 - target) * max_r
        loss = torch.sum(loss, dim=-1)

        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, class_num=5, alpha=None, gamma=1, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
       
        class_mask = inputs.data.new(N, C).fill_(0)
       
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
     

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P*class_mask).sum(1).view(-1,1)+1e-5

        log_p = probs.log()
      
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
       
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
    
        self.criterion=FocalLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])
        
        # step forward
        self.model.train()
        self.optimizer.zero_grad()

        logits, pooling_output = self.model(inputs)
        
        loss = self.criterion(logits, labels)
        '''
        loss_c=MarginLoss(capsule_predict,labels)
        beta=0
        loss=loss+beta*loss_c
        '''


        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.model.conv_l2() * self.opt['conv_l2']
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[11]
        # forward
        self.model.eval()
        logits,_= self.model(inputs)
        loss= self.criterion(logits, labels)

        '''
        loss_c=MarginLoss(capsule_predict,labels)
        beta=0.01
        loss=loss+loss_c*beta
        '''

        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, probs, loss.item()
