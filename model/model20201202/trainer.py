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
        inputs = [Variable(b.cuda()) for b in batch[:14]]
        labels = Variable(batch[14].cuda())
    else:
        inputs = [Variable(b) for b in batch[:14]]
        labels = Variable(batch[14])
    tokens = batch[0]
    head = batch[5]
    subj_pos = batch[6]
    obj_pos = batch[7]
    #从batch中读出
    subj_vec=batch[10]
    obj_vec=batch[11]
    subj_pvec=batch[12]
    obj_pvec=batch[13]
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, tokens, head, subj_pos, obj_pos, lens,subj_vec,obj_vec,subj_pvec,obj_pvec
class Myloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.autocriterion=nn.MSELoss()
    def forward(self,x,y):
        loss_main=self.criterion(x,y)
        #loss_auto_s=self.autocriterion(sx,sy)
       # loss_auto_o=self.autocriterion(ox,oy)
      #  alpha=0.0001
        #return torch.add(loss_main,alpha*loss_auto_o+alpha*loss_auto_s).cuda()
        return loss_main

def MarginLoss(predict, target):
        m_plus, m_minus, loss_lambda = 0.9, 0.1, 0.5

        target = to_one_hot(target,5)
        max_l = (torch.relu(m_plus - predict))**2
        max_r = (torch.relu(predict - m_minus))**2
        #print("max_l:{}".format(max_l.size()))
        #print("max_r:{}".format(max_r.size()))
        #print("target:{}".format(target.size()))
        loss = target * max_l + loss_lambda * (1 - target) * max_r
        loss = torch.sum(loss, dim=-1)

       # if reduction == 'sum':
        #    return loss.sum()
        #else:
            # 默认情况为求平均
        return loss.mean()
class FocalLoss(nn.Module):
    def __init__(self, class_num=5, alpha=None, gamma=1, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            #No_relation,mechain,effect,advise,int
            #alpha_initia=torch.FloatTensor([0.01,0.1,0.08,0.15,0.67])
            #alpha_initia=torch.FloatTensor([1.0,1.0,1.0,1.0,1.0])
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
        #print('input:{}'.format(inputs))
        #print('p:{}'.format(P))

        class_mask = inputs.data.new(N, C).fill_(0)
        #print('class_max:{}'.format(class_mask))
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
       # print('class_max_a:{}'.format(class_mask))


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        #print('alpha:{}'.format(alpha))
        #这个概率的计算probs
        probs = (P*class_mask).sum(1).view(-1,1)+1e-5

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print('probs:{}'.format(probs))

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)
        #tensor is inf
        #batch_lossinf=torch.isinf(batch_loss)
        #tensor is nan
        #batch_lossnan=torch.isnan(batch_loss)
        #combination
        #batch_loss_mask=torch.add(batch_lossinf.float(),batch_lossnan.float())
        #替换tensor元素
        
        #batch_loss_zero=torch.zeros(batch_loss.size()).cuda()

        #batch_loss_new=torch.where(batch_loss_mask==0,batch_loss,batch_loss_zero)

        #batch_loss=batch_loss_new

        








        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        #print('loss:{}'.format(loss))
        return loss

class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
       # self.criterion = nn.CrossEntropyLoss()
       # self.autocriterion=nn.MSELoss()
        #self.criterion=Myloss()
       # self.criterion=MarginLoss()
        #self.criterion=nn.CrossEntropyLoss()
        self.criterion=FocalLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            #self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        inputs, labels, tokens, head, subj_pos, obj_pos, lens,subj_vec,obj_vec,subj_pvec,obj_pvec = unpack_batch(batch, self.opt['cuda'])
        
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, pooling_output,capsule_predict = self.model(inputs)
        #print("logits:{}".format(logits))
        #print("labels:{}".format(labels))
        loss = self.criterion(logits, labels)
        loss_c=MarginLoss(capsule_predict,labels)
        beta=0
        loss=loss+beta*loss_c
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
        inputs, labels, tokens, head, subj_pos, obj_pos, lens,subj_vec,obj_vec,subj_pvec,obj_pvec = unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[15]
        # forward
        self.model.eval()
        logits,_,capsule_predict= self.model(inputs)
        loss= self.criterion(logits, labels)
        loss_c=MarginLoss(capsule_predict,labels)
        beta=0.01
        loss=loss+loss_c*beta
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, probs, loss.item()
