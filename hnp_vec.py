# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 21:18:00 2019

@author: Ananthan
"""

import numpy as np
import pandas as pd
import process as sp
import re
import os
from sklearn.preprocessing import normalize


def t_n_d(file):
    """
    gets title and description from a file
    """
    s = file.read()
    des = re.findall(r'\"([^]]*)\"', s)
    t = s[:s.index(',"')]
    return t,des
    

def make_dict_of_words(path):
    """
    :return: dict of words and freq in corpus 
    """
    word_dict={}
    total_docs=0
    for filename in os.listdir(path):
        file = open(path+"/"+filename,'r', encoding="utf8")
        print(filename)
        title,des=t_n_d(file)
        try:
            des_ls = sp.tokenize_str(des[0])
            total_docs+=1
        except:
            des_ls = sp.tokenize_str("aaa")
            print("missed ", filename)
        words_in_doc=[]
        for word in des_ls:
            if word not in word_dict and word not in words_in_doc:
                word_dict[word]=1
                words_in_doc.append(word)
            elif word not in words_in_doc:
                word_dict[word]+=1
                words_in_doc.append(word)
    final_dict = {k: v for k, v in word_dict.items() if v/total_docs<=0.2}
    print('made dict')
    return final_dict

def make_seq_freq_vec(seq_ls,words):
    """
    :param seq: 
    :param words: 
    :return: 
    """
    vec=np.zeros(len(words))
    for i in range(len(words)):
        count=seq_ls.count(words[i])
        if count>0:
            vec[i]=1
    mod = np.linalg.norm(vec)
    if mod != 0:
        vec = vec / mod
    else:
        vec =vec
    return vec

def make_vec_df(path,w_dict):
    data=[]
    firms=[]
    words=list(w_dict.keys())
    word_dict_df=pd.DataFrame(words)
    word_dict_df.to_csv('dict_hnp.csv')
    i=1
    for filename in os.listdir(path):
        print(i)
        file = open(path+"/"+filename,'r', encoding="utf8")
        title,des=t_n_d(file)
        try:
            des_ls = sp.tokenize_str_hp(des[0],title)
        except:
            des_ls = sp.tokenize_str_hp("Adam",title)
            print("missed ", filename)
        vec = make_seq_freq_vec(des_ls,words)
        data.append(vec)
        firms.append(title)
        i+=1
    data = np.array(data).T.tolist()
    df = pd.DataFrame(data, columns=firms)
    return df

out_df=make_vec_df('../test_data',make_dict_of_words('../test_data'))
out_df.to_csv('hnp_vec_tests.csv')