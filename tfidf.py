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


def make_seq_tfidf_vec(seq_ls,words,w_num,n_d):
    """
    :param seq_ls: 
    :param words: 
    :param w_num:
    :param n_d:
    :return: 
    """
    vec=np.zeros(len(words))
    for i in range(len(words)):
        count=seq_ls.count(words[i])
        if count>0:
            vec[i]=count*math.log(n_d/w_num[words[i]])
    return vec

def make_vec_df(path,w_dict,n_docs):
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
        vec = make_seq_tfidf_vec(des_ls,words,w_dict,n_docs)
        data.append(vec)
        firms.append(title)
        i+=1
    data = np.array(data).T.tolist()
    df = pd.DataFrame(data, columns=firms)
    return df

wdict,num_d=make_dict_of_words('test_data')
out_df=make_vec_df('test_data',wdict,num_d)
out_df.to_csv('tfidf_vec_tests.csv')