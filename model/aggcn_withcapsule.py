"""
GCN model for relation extraction.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model.single_attention import AffectiveAttention
from model.tree import head_to_tree, tree_to_adj
from utils import constant, torch_utils
from model.capsule import Capsule

class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """

    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def forward(self, inputs):
        outputs, pooling_output,capsule_predict = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output,capsule_predict
class SelfAttentionWide(nn.Module):

    def __init__(self, emb, heads = 8, mask = False):

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask 

        self.tokeys = nn.Linear(emb, emb * heads, bias = False)
        self.toqueries = nn.Linear(emb, emb * heads, bias = False)
        self.tovalues = nn.Linear(emb, emb * heads, bias = False)

    
    def forward(self, x):

        b, t, e = x.size()#这个就是inputs
        h = self.heads#是要多少个头 
        assert e == self.emb, f'Input embedding dimension {{e}} should match layer embedding dim {{self.emb}}'

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention

        # folding heads to batch dimensions

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))



        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim = 2)

        out = torch.bmm(dot, values).view(b, h, t, e)

        out = out.transpose(1, 2).contiguous().view(b, t, h * e)#这里应该换掉 换成bath,len,dim,h

        return out


class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.init_embeddings()
        
        self.gcn = AGGCN(opt, embeddings)
        
        self.CapsuleNetwork=CapsuleNetwork(opt)

        # mlp output layer
        in_dim = opt['hidden_dim']
        self.mem_dim = opt['hidden_dim'] 
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)
        self.emb_dim = opt["emb_dim"]
        self.linear_up=nn.Linear(self.mem_dim,self.emb_dim)
    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type,subj_vec,obj_vec,subj_pvec,obj_pvec = inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        def inputs_to_tree_reps(head, l):
            trees = [head_to_tree(head[i], l[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)
        def inputs_to_weight_reps(head,dep,l,max_length):
            adj=[]
            for i in range(len(l)):
                head_,dep_=head[i],dep[i]
                head_=head_.tolist()
                dep_=dep_.tolist()
                adj_= np.random.randint(0, 1, (max_length+1, max_length+1), dtype=int)
            
                for j,dep_relation in enumerate(dep_):
                    if j >= max_length:
                        break
                    #-1表示到自己
                    token1_id, token2_id = int(head_[j]),j+1
                    if token2_id == -1 or token2_id >= max_length:
                        token2_id = token1_id
                    adj_[token1_id, token2_id], adj_[token2_id, token1_id] = int(dep_relation), int(dep_relation)
                
                adj_=adj_[1:,1:]
                adj.append(adj_)
            
           
            adj=np.array(adj)
            
            adj = torch.from_numpy(adj).float()
            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        adj = inputs_to_tree_reps(head.data, l)
       
        weight_adj=inputs_to_weight_reps(head.data,deprel.data,l,maxlen)
        
        h, pool_mask = self.gcn(weight_adj,adj,inputs)

        # pooling
    
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)

        
        subj_out = pool(h, subj_mask, type="max")
        obj_out = pool(h, obj_mask, type="max")
       
        #obj_out=self.linear_up(obj_out)
        #subj_out=self.linear_up(subj_out)

        capsule_vec=torch.cat((subj_out.unsqueeze(1),subj_vec.unsqueeze(1),subj_pvec,obj_out.unsqueeze(1),obj_vec.unsqueeze(1),obj_pvec),dim=1)
        capsule_predic=self.CapsuleNetwork(capsule_vec)
        subj=subj_out
        obj=obj_out
        
        outputs=h_out
        outputs = self.out_mlp(outputs)

        return outputs,h_out,capsule_predic


class AGGCN(nn.Module):
    def __init__(self, opt, embeddings):
        super().__init__()
        self.opt = opt
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
        self.emb, self.pos_emb, self.ner_emb = embeddings
        self.use_cuda = opt['cuda']
        self.mem_dim = opt['hidden_dim']

        # rnn layer
        if self.opt.get('rnn', False):
            self.input_W_R = nn.Linear(self.in_dim, opt['rnn_hidden'])
            self.rnn = nn.LSTM(opt['rnn_hidden'], opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                               dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output
        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)


        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.num_layers = opt['num_layers']
        
        self.layers = nn.ModuleList()

        self.heads = opt['heads']
        self.sublayer_first = opt['sublayer_first']
        self.sublayer_second = opt['sublayer_second']
        self.emb_dim = opt["emb_dim"]
        self.dep_size = opt['dep_size']
        self.dep_embed_dim = opt['dep_embed_dim']
        self.edge_embeddings = nn.Embedding(num_embeddings=self.dep_size,
                                            embedding_dim=self.dep_embed_dim,
                                            padding_idx=0)

        # gcn layer
        for i in range(self.num_layers):
            if i == 0:
                #self.layers.append(GraphConvLayer(opt, self.mem_dim,self.dep_embed_dim, self.sublayer_first))
                #self.layers.append(GraphConvLayer(opt, self.mem_dim, self.dep_embed_dim,self.sublayer_second))
                
               # self.layers.append(graphConvLayer(opt, self.mem_dim, self.sublayer_first))
                self.layers.append(graphConvLayer(opt, self.mem_dim, self.sublayer_second))
            else:
                self.layers.append(MultiGraphConvLayer(opt, self.mem_dim,self.dep_embed_dim, self.sublayer_first, self.heads))
                self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, self.dep_embed_dim,self.sublayer_second, self.heads))

        self.aggregate_W = nn.Linear(len(self.layers) * self.mem_dim, self.mem_dim)

        self.attn = MultiHeadAttention(self.heads, self.mem_dim)

      
        self.linear_map=nn.Linear(self.dep_embed_dim,self.dep_embed_dim)
        self.action=nn.Tanh()
        

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):

        
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
       
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        #print("rnniputs:{}".format(rnn_inputs.size()))
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
       
        return rnn_outputs
    def MultiDim_attention(self,adj,attn_adj_list):
        
        batch,seq,seq,dim_edge=adj.shape
        
        #attn_adj_list = [torch.add(attn_adj.unsqueeze(3).expand(batch,seq,seq,dim_edge),adj) for attn_adj in  attn_adj_list]
        attn_adj_list_query = [self.linear_map(attn_adj.unsqueeze(3).expand(batch,seq,seq,dim_edge)) for attn_adj in  attn_adj_list]
       
        func_adj_attn_part1=[torch.add(self.linear_map(attn_adj),self.linear_map(adj)) for attn_adj in  attn_adj_list_query]
        func_adj_attn_part2=[self.action(adj_) for adj_ in func_adj_attn_part1]
        func_adj_attn_part3=[self.linear_map(adj_) for adj_ in func_adj_attn_part2]

        func_attn =[F.softmax(scores, dim=-1) for scores in func_adj_attn_part3]
       
        func_attn_list=[adj_.mul(adj) for adj_ in func_attn]
       
        return func_attn_list




    def single_attention(e,U):
        d_k = 300.0
        e_1=torch.matmul(F.softmax(torch.matmul(e,U.permute(0,2,1)/scale)),U)
        e_u=torch.add(e,e_1)
        return e_u
    
    def forward(self, adj,adj_mask,inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type,subj_vec,obj_vec,subj_pvec,obj_pvec = inputs # unpack
        #掩码获得
        src_mask = (words != constant.PAD_ID).unsqueeze(-2)
        word_embs = self.emb(words)
        
        embs=[word_embs]

        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
       
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)
        
        if self.opt.get('rnn', False):
            embs = self.input_W_R(embs)
            #修改words.size()[0]
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0]))
            

        else:
            gcn_inputs = embs
       
        gcn_inputs = self.input_W_G(gcn_inputs)
      
        
        adj = self.edge_embeddings(adj.long().contiguous())
        

        layer_list = []
        outputs = gcn_inputs
        mask = (adj_mask.sum(2) + adj_mask.sum(1)).eq(0).unsqueeze(2)
        for i in range(len(self.layers)):
            if i < 1:
                #这里将adj->替换为了adj_mask 也就是原来的adj
                outputs = self.layers[i](adj_mask,outputs)
                layer_list.append(outputs)
            else:
                attn_tensor = self.attn(outputs, outputs, src_mask)
                attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                batch,seq,seq,dim_edge=adj.shape
                #多维度attention

                #attn_adj_list = [torch.add(attn_adj.unsqueeze(3).expand(batch,seq,seq,dim_edge),adj) for attn_adj in  attn_adj_list]
               # attn_adj_list_query = [self.linear_expand(attn_adj) for attn_adj in  attn_adj_list]
                attn_adj_list=self.MultiDim_attention(adj,attn_adj_list)
                outputs = self.layers[i](attn_adj_list, outputs)
                layer_list.append(outputs)

        aggregate_out = torch.cat(layer_list, dim=2)
        
        dcgcn_output = self.aggregate_W(aggregate_out)

        return dcgcn_output, mask
#之前的gcn
class graphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, opt, mem_dim, layers):
        super(graphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()

    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)

        return out
class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, opt, mem_dim,dep_embed_dim, layers):
        super(GraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])
        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))
        
        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()
      
        self.dep_embed_dim = dep_embed_dim
        self.highway = Edgeupdate(self.head_dim, self.dep_embed_dim, dropout_ratio=0.5)
        self.transform_output = nn.ModuleList()
        for i in range(self.layers):
            self.transform_output.append(nn.Linear((self.mem_dim + self.head_dim * i),self.mem_dim))
        
    def forward(self, adj,gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            
            batch,seq,dim = outputs.shape

            adj=adj.permute(0,3,1,2) #batch*dim_edge_seq_seq
            outputs=outputs.unsqueeze(1).expand(batch,self.dep_embed_dim,seq,dim)
           
            Ax = torch.matmul(adj,outputs)
            Ax = Ax.mean(dim=1)


            AxW = self.weight_list[l](Ax)

            original_outputs = outputs.mean(dim=1)
            AxW = AxW + self.weight_list[l](original_outputs)  # self loop

            gAxW = F.relu(AxW)
        
            cache_list.append(gAxW)
           
            outputs = torch.cat(cache_list, dim=2)

            outputs_after= gAxW
            
            adj = adj.permute(0,2,3,1) #batch*seq*seq*dim_e
            batch,seq,dim = outputs_after.shape
            node_outputs1=outputs_after.unsqueeze(1).expand(batch,seq,seq,dim)
            node_outputs2=node_outputs1.permute(0,2,1,3)
          
            edge_outputs = self.highway(adj, node_outputs1, node_outputs2)
            adj=edge_outputs 
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)
        
        return out

class Edgeupdate(nn.Module):
    def __init__(self, hidden_dim, dim_e, dropout_ratio=0.5):
        super(Edgeupdate, self).__init__()
        self.hidden_dim = hidden_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.hidden_dim * 2 + self.dim_e, self.dim_e)
    def forward(self, edge, node1, node2):
        """
        :param edge: [batch, seq, seq, dim_e]
        :param node: [batch, seq, seq, dim]
        :return:
        """
        node = torch.cat([node1, node2], dim=-1) # [batch, seq, seq, dim * 2]
        edge = self.W(torch.cat([edge, node], dim=-1))
        return edge  # [batch, seq, seq, dim_e]

class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, opt, mem_dim, dep_embed_dim,layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()
        self.dep_embed_dim=dep_embed_dim
        self.highway = Edgeupdate(self.head_dim, self.dep_embed_dim, dropout_ratio=0.5)

    def forward(self, adj_list, gcn_inputs):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]
        
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []


            for l in range(self.layers):
                index = i * self.layers + l
    
                batch,seq,dim = outputs.shape
                #print("adj:{}".format(adj.size()))
                adj=adj.permute(0,3,1,2) #batch*dim_edge_seq_seq
                outputs=outputs.unsqueeze(1).expand(batch,self.dep_embed_dim,seq,dim)
                
              
                Ax = torch.matmul(adj,outputs)
                Ax = Ax.mean(dim=1)

                original_outputs = outputs.mean(dim=1)
                AxW = self.weight_list[index](Ax)
              
                AxW = AxW + self.weight_list[index](original_outputs)  # self loop
                #AxW = AxW / denom

                gAxW = F.relu(AxW)
                cache_list.append(gAxW)

                outputs_after= gAxW
                adj = adj.permute(0,2,3,1) #batch*seq*seq*dim_e
                batch,seq,dim = outputs_after.shape
                node_outputs1=outputs_after.unsqueeze(1).expand(batch,seq,seq,dim)
                node_outputs2=node_outputs1.permute(0,2,1,3)
                edge_outputs = self.highway(adj, node_outputs1, node_outputs2)
                adj=edge_outputs 

                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None: #A的作用体现在这里 但这种就是普通邻接矩阵 如果要体现这种不同的结果
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def single_attention(e,U):
        d_k = e.size(-1)**0.5
        #batch_size=50
      #  print("e:{}".format(e.size))
        e=e.view(e.size(0),1,e.size(-1))
      #  print("e:{}".format(e.shape))
        e_1=torch.bmm(F.softmax(torch.bmm(e,U.permute(0,2,1))/d_k),U)
        e_u=torch.add(e,e_1).squeeze()
        return e_u



class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
    
        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn


class CNN(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.opt=opt
        self.in_channels = opt['hidden_dim']
        self.out_channels = opt['cnn_output_dim']
        #self.kernel_sizes = opt['kernel_sizes']#这个指标需要自己设立
        self.kernel_sizes=[3,5,7,9,11]
        self.activation = 'relu'
        self.dropout = opt['cnn_dropout']
        self.keep_length = True
        for kernel_size in self.kernel_sizes:
            assert kernel_size % 2 == 1, "kernel size has to be odd numbers."
        # convolution
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=k,
                      stride=1,
                      padding=k // 2 if self.keep_length else 0,
                      dilation=1,
                      groups=1,
                      bias=False) for k in self.kernel_sizes
        ])

        # activation function
        assert self.activation in ['relu', 'lrelu', 'prelu', 'selu', 'celu', 'gelu', 'sigmoid', 'tanh'], \
            'activation function must choose from [relu, lrelu, prelu, selu, celu, gelu, sigmoid, tanh]'
        self.activations = nn.ModuleDict([
            ['relu', nn.ReLU()],
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()],
            ['selu', nn.SELU()],
            ['sigmoid', nn.Sigmoid()],
            ['tanh', nn.Tanh()],
        ])
        self.dropout = nn.Dropout(self.dropout)
    def forward(self, x):
        """
            :param x: torch.Tensor [batch_size, seq_max_length, input_size], [B, L, H] 一般是经过embedding后的值
            :param mask: [batch_size, max_len], 句长部分为0，padding部分为1。不影响卷积运算，max-pool一定不会pool到pad为0的位置
            :return:
            """
        # [B, L, H] -> [B, H, L] （注释：将 H 维度当作输入 channel 维度)
        x = torch.transpose(x, 1, 2)

        # convolution + activation  [[B, H, L], ... ]
        act_fn = self.activations[self.activation]

        x = [act_fn(conv(x)) for conv in self.convs]
        x = torch.cat(x, dim=1)

        x = x.transpose(1, 2)
        x = self.dropout(x)
        #Hs：C*d
        return x # [B, L, Hs], [B, Hs]


class CapsuleNetwork(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.opt = opt
        # capsule config
        #self.cnn = CNN(opt)
        self.capsule =Capsule(opt)

    def forward(self, x):
        #B,L,Hs
        #primary = self.cnn(x)  # 由于长度改变，无法定向mask，不mask可可以，毕竟primary capsule 就是粗粒度的信息
        #print("primary:{}".format(primary.size()))
        output = self.capsule(x)
        output = output.norm(p=2, dim=-1)  # 求得模长再返回值

        return output  # [B, N]