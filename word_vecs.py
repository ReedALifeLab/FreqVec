# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:19:28 2019

@author: Ananthan, James
"""

import numpy as np
import pandas as pd
import process as sp
import os
from sklearn.preprocessing import normalize
import math


def t_n_d(file):
    """
    gets title and description from a file
    """
    s = file.read()
    des = s[s.index(','):]
    t = s[:s.index(',')]
    return t,des

class Model:
    
    def __init__(self,tags,kind,th=0.2):
        """
        :param tags:
            "nouns" or "all"
        :param kind:
            "boolean", "freq" or "tfidf"
        :param th:
            threshold for excluding words. th=0.2 means words occuring in more
            than 20% of documents are excluded
        """
        self.dict=None
        self.model_vecs=None
        self.num_docs=0
        self.tags=tags
        self.th=th
        self.kind=kind
        
    def make_dict_of_words(self,path, errorfile): # unlink files as they're found to have no words
        """
        :return: dict of words and freq in corpus 
        """
        word_dict={}
        total_docs=0
        for filename in os.listdir(path):
            file = open(path+"/"+filename,'r', encoding="utf8")
            title,des=t_n_d(file)
            try:
                if self.tags=="all":
                    des_ls = sp.tokenize_str(des)
                elif self.tags=='nouns':
                    des_ls = sp.tokenize_str_hp(des,title)
                total_docs+=1
            except:
                des_ls = []
                if errorfile is not None:
                    errorfile.write(filename + " could not be added to the dictionary\n")
                
            for word in des_ls:
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        final_dict = {k: v for k, v in word_dict.items() if v/total_docs<=self.th}
        print('made dict with tags: ',self.tags,', th: ',self.th)
        self.dict=final_dict
        self.num_docs=total_docs
        return final_dict
    
    def make_seq_freq_vec(self,seq_ls,words,words_to_index):
        """
        :param seq_ls:
            tokenized sentence as list
        :param words:
            list of words
        :param words_to_index:
            dict of word to index
        :return: 
        """
        vec=np.zeros(len(words))
        
        if self.kind == 'boolean':
            seq_set = set(seq_ls)
            for i in range(len(words)):
                if words[i] in seq_set:
                    vec[i] = 1
        elif self.kind == 'freq':
            for word in seq_ls:
                vec[words_to_index[word]] += 1
            for i in range(len(seq_ls)):
                vec[i] = 1 + math.log(vec[i]) if vec[i] > 0 else vec[i]
        elif self.kind=='tfidf':
            for word in seq_ls:
                vec[words_to_index[word]] += 1
            for i in range(len(seq_ls)):
                vec[i] = vec[i]*math.log(self.num_docs/self.dict[words[i]]) if vec[i] > 0 else vec[i]
        
        mod = np.linalg.norm(vec)
        if mod != 0:
            vec = vec / mod
        else:
            vec =vec
        
        return vec
    
    def make_vec_df(self,path,w_dict, prefix, errorfile):
        data=[]
        firms=[]
        words=list(w_dict.keys())
        words_to_index = {word : words.index(word) for word in words}
        word_dict_df=pd.DataFrame(words)
        word_dict_df.to_csv(prefix + '/freqvec_dict.csv')
        i=1
        for filename in sorted(os.listdir(path)):
            file = open(path+"/"+filename,'r', encoding="utf8")
            title,des=t_n_d(file)
            try:
                if self.tags == 'nouns':
                    des_ls = sp.tokenize_str_hp(des, title)
                elif self.kind == 'all':
                    des_ls = sp.tokenize_str(des)
                vec = self.make_seq_freq_vec(des_ls,words, words_to_index)
                data.append(vec)
                firms.append(title)
            except:
                pass
                # if errorfile is not None:
                #     errorfile.write(filename + " missed in vector-making step\n")
            i+=1
        data = np.array(data).T.tolist()
        df = pd.DataFrame(data, columns=firms)
        self.model_vecs=df
        return df
    
def make_model(tags,kind,th,errorfile=None,prefix=""):
    model = Model(tags,kind,th)
    print('making model')
    model.make_vec_df('new_data',model.make_dict_of_words('new_data', errorfile), prefix, errorfile)
    model.model_vecs.to_csv(prefix + '/'+tags+'_'+kind+'_'+str(th)+'_vectors.csv')
    print('made vectors')
    return prefix + '/'+tags+'_'+kind+'_'+str(th)+'_vectors.csv'

make_model('nouns','freq',.2)