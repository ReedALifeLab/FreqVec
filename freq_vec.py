# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:18:11 2019

@author: Ananthan
"""
import numpy as np
import math
import pandas as pd
import process as sp
import re
import os


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
    for filename in os.listdir(path):
        file = open(path+"/"+filename,'r', encoding="utf8")
        print(filename)
        title,des=t_n_d(file)
        try:
            des_ls = sp.tokenize_str(des[0])
        except:
            des_ls = sp.tokenize_str("aaa")
            print("missed ", filename)
        for word in des_ls:
            if word not in word_dict:
                word_dict[word]=1
            else:
                word_dict[word]+=1
    print('made dict')
    return word_dict

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
            vec[i]=1+math.log(count)
    return vec

def make_vec_df(path,w_dict):
    data=[]
    firms=[]
    words=list(w_dict.keys())
    word_dict_df=pd.DataFrame(words)
    word_dict_df.to_csv('dict.csv')
    i=1
    for filename in os.listdir(path):
        print(i)
        file = open(path+"/"+filename,'r', encoding="utf8")
        title,des=t_n_d(file)
        try:
            des_ls = sp.tokenize_str(des[0])
        except:
            des_ls = sp.tokenize_str("aaa")
            print("missed ", filename)
        vec = make_seq_freq_vec(des_ls,words)
        data.append(vec)
        firms.append(title)
        i+=1
    data = np.array(data).T.tolist()
    df = pd.DataFrame(data, columns=firms)
    return df

out_df=make_vec_df('test_data',make_dict_of_words('test_data'))
out_df.to_csv('freq_vec_tests.csv')