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
import sys

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
        
    def make_dict_of_words(self,path, errorfile):
        """
        :return: dict of words and freq in corpus 
        """
        word_dict={}
        total_docs=0
        dictfails = 0
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
                    dictfails += 1
                    exc1, exc2, exc3 = sys.exc_info()
                    errorfile.write(filename + " failed in dictionary step: " + str(exc1) + " ; " + str(exc2)+ "\n")
            words_in_doc = set()
            for word in des_ls:
                if word not in word_dict and word not in words_in_doc:
                    word_dict[word]=1
                    words_in_doc.add(word)
                elif word not in words_in_doc:
                    word_dict[word]+=1
                    words_in_doc.add(word)
        final_dict = {k: v for k, v in word_dict.items() if v/total_docs<=self.th}
        if errorfile is not None:
            errorfile.write(str(dictfails) + " documents failed in dictionary step\n")
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
                if word in words_to_index:
                    vec[words_to_index[word]] += 1
            for i in range(len(words)):
                vec[i] = 1 + math.log(vec[i]) if vec[i] > 0 else vec[i]
        elif self.kind=='tfidf':
            for word in seq_ls:
                if word in words_to_index:
                    vec[words_to_index[word]] += 1
            for i in range(len(words)):
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
        vecfails = 0
        vecempty = 0
        words=list(w_dict.keys())
        words_to_index = {word : words.index(word) for word in words}
        word_dict_df=pd.DataFrame(words)
        word_dict_df.to_csv(prefix + '/' + prefix + '_dict.csv')
        i=1
        for filename in sorted(os.listdir(path)):
            file = open(path+"/"+filename,'r', encoding="utf8")
            title,des=t_n_d(file)
            try:
                if self.tags == 'nouns':
                    des_ls = sp.tokenize_str_hp(des, title)
                elif self.tags == 'all':
                    des_ls = sp.tokenize_str(des)
                vec = self.make_seq_freq_vec(des_ls,words, words_to_index)
                # data.append(vec)
                # firms.append(title)
                if np.dot(vec, vec) != 0:
                    data.append(vec)
                    firms.append(title)
                else:
                    vecempty += 1
                    if errorfile is not None:
                        errorfile.write(filename + " contained no words after preprocessing\n")
            except:
                if errorfile is not None:
                    vecfails += 1
                    exc1, exc2, exc3 = sys.exc_info()
                    errorfile.write(filename + " failed in vector step: " + str(exc1) + " ; " + str(exc2)+ "\n")
            i+=1
        data = np.array(data).T.tolist()
        df = pd.DataFrame(data, columns=firms)
        self.model_vecs=df
        if errorfile is not None:
            errorfile.write(str(vecfails) + " documents failed in vector step\n" + str(vecempty) + " documents contained no words after preprocessing\n")
        return df
    
def make_model(tags,kind,th,errorfile=None,prefix=""):
    """
    For tags, kind, th: see Model.__init__()
    :errorfile:
        string: path to text file to record errors while making the model, or None, to ignore errors.
    :prefix:
        string: path to folder containing model's output.
    """
    model = Model(tags,kind,th)
    print('making model')
    model.make_vec_df('test_data',model.make_dict_of_words('test_data', errorfile), prefix, errorfile)
    model.model_vecs.to_csv(prefix + '/'+tags+'_'+kind+'_'+str(th)+'_vectors.csv')
    print('made vectors')
    return prefix + '/'+tags+'_'+kind+'_'+str(th)+'_vectors.csv'

# make_model('nouns','freq',.2)