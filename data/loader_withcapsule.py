
from numpy import *
import json
import random
import torch
import numpy as np

from utils import constant

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):

        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID
        self.vec_dim = opt['emb_dim']
        with open(filename) as infile:
            data = json.load(infile)

        self.raw_data = data

        data = self.preprocess(data, vocab, opt)
        
        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]

        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-1]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        # 隐式调用了getitem__函数
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

        self.data = data

        print("{} batches created for {}".format(len(data), filen ame))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
      

        for d in data:

            tokens = list(d['token'])
           
            if opt['lower']:
                tokens = [t.lower() for t in tokens]

            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)

            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            
            head = [int(x) for x in d['stanford_head']]

            assert any([x == 0 for x in head])

            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            relation = self.label2id[d['relation']]

            
            #添加实体的分子向量信息
            subj_vec=d['subj_mol2v']
            #进行归一化
            subj_vec=vec_uniform(subj_vec)
            obj_vec=d['obj_mol2v']
            obj_vec=vec_uniform(obj_vec)
            subj_pvec=d['subj_pvec']
            #说明没有值
            if(sum(subj_pvec)==0):
                subj_pvec=[subj_pvec]
            else:
                subj_pvec=[vec_uniform(p) for p in subj_pvec]
            obj_pvec=d['obj_pvec']
            if(sum(obj_pvec)==0):
                obj_pvec=[obj_pvec]
            else:
                obj_pvec=[vec_uniform(p) for p in obj_pvec]
            
           
            processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type,subj_vec,obj_vec,subj_pvec,obj_pvec,relation)]
            
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 14

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        maxlen= max(lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size,maxlen)
        masks = torch.eq(words, 0)

        pos = get_long_tensor(batch[1], batch_size,maxlen)
        ner = get_long_tensor(batch[2], batch_size,maxlen)
        deprel = get_long_tensor(batch[3], batch_size,maxlen)
        head = get_long_tensor(batch[4], batch_size,maxlen)
        subj_positions = get_long_tensor(batch[5], batch_size,maxlen)
        obj_positions = get_long_tensor(batch[6], batch_size,maxlen)
        subj_type = get_long_tensor(batch[7], batch_size,maxlen)
        obj_type = get_long_tensor(batch[8], batch_size,maxlen)
       

        rels = torch.LongTensor(batch[13])


        
        subj_vec=get_float_tensor(batch[9],batch_size,self.vec_dim,maxlen)
        obj_vec=get_float_tensor(batch[10],batch_size,self.vec_dim,maxlen)
        subj_pvec=get_pvec_tensor(batch[11],batch_size,self.vec_dim,maxlen)
        obj_pvec=get_pvec_tensor(batch[12],batch_size,self.vec_dim,maxlen)
       # weight_adj=get_adj_tensor(batch[13],batch_size,maxlen)
     
        
        return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type,obj_type,subj_vec,obj_vec,subj_pvec,obj_pvec,rels,orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))


def get_long_tensor(tokens_list, batch_size,maxlen):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = maxlen
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]


def vec_uniform(vec):
    vec_max=max(vec)
    vec_min=min(vec)
    if(vec_max==vec_min):
        vec_max+=1
    for x in range(len(vec)):
        vec[x]=(float)((vec[x]-vec_min)/(vec_max-vec_min)-0.5)*2
    return vec

def get_float_tensor(tokens_list, batch_size,dim,maxlen):
    """ Convert list of list of tokens to a padded LongTensor. """
    tokens = torch.FloatTensor(batch_size, dim).fill_(0.0)
    for i, s in enumerate(tokens_list):
        #print("s:{}".format(s))
        tokens[i, :len(s)] = torch.FloatTensor(s)
    return tokens

def get_pvec_tensor(tokens_list, batch_size,dim,maxlen):
    """ Convert list of list of tokens to a padded LongTensor. """
    #token_len = maxlen
    #这里，是patient view的信息，和句子长度不一样，不可以用一个最大长度
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size,token_len,dim).fill_(0.0)
    for i, s in enumerate(tokens_list):
        for x in s:
          #  print("x:{}".format(len(s)))
          #  print("token:{}".format(tokens.size()))
            tokens[i,len(s)-1] = torch.FloatTensor(x)
    
    return tokens
def get_adj_tensor(adj,batch_size,maxlen):
    
    tokens = torch.LongTensor(batch_size, maxlen,maxlen).fill_(0)
    for i, s in enumerate(adj):
        for x in s:
            tokens[i,len(s)-1] = torch.LongTensor(x)
    return tokens
