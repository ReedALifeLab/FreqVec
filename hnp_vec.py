# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 21:18:00 2019

@author: Ananthan
"""

import numpy as np
import pandas as pd
import process as sp
# import re
import os
from sklearn.preprocessing import normalize


def t_n_d(file):
    """
    gets title and description from a file
    """
    s = file.read()
    # des = re.findall(r'\"([^]]*)\"', s)
    des = s[s.index(','):]
    t = s[:s.index(',')]
    return t,des
    

def make_dict_of_words(path, errorfile): # unlink files as they're found to have no words
    """
    :return: dict of words and freq in corpus 
    """
    word_dict={}
    total_docs=0
    for filename in os.listdir(path):
        file = open(path+"/"+filename,'r', encoding="utf8")
        # print('.', end='')
        title,des=t_n_d(file)
        try:
            # des_ls = sp.tokenize_str(des[0])
            des_ls = sp.tokenize_str(des)
            total_docs+=1
        except:
            # des_ls = sp.tokenize_str("aaa")
            des_ls = []
            if errorfile is not None:
                errorfile.write(filename + " could not be added to the dictionary\n")
            # print("missed ", filename)
        # words_in_doc = set() # does this even need to be used?
        # for word in des_ls:
        #     if word not in word_dict and word not in words_in_doc:
        #         word_dict[word]=1
        #         words_in_doc.add(word)
        #     elif word not in words_in_doc:
        #         word_dict[word]+=1
        #         words_in_doc.add(word)
        for word in des_ls:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    final_dict = {k: v for k, v in word_dict.items() if v/total_docs<=0.2}
    print('made HandP dict')
    return final_dict

def make_seq_freq_vec(seq_ls,words):
    """
    :param seq: 
    :param words: 
    :return: 
    """
    vec=np.zeros(len(words))

    # for word in seq_ls:
    #     vec[words_to_index[word]] = 1

    # for i in range(len(words)):
    #     count=seq_ls.count(words[i])
    #     if count>0:
    #         vec[i]=1
    seq_set = set(seq_ls)
    for i in range(len(words)):
        if words[i] in seq_set:
            vec[i] = 1
    mod = np.linalg.norm(vec)
    if mod != 0:
        vec = vec / mod
    else:
        vec =vec
    return vec

def make_vec_df(path,w_dict, prefix, errorfile):
    data=[]
    firms=[]
    words=list(w_dict.keys())
    # words_to_index = {word : words.index(word) for word in words}
    word_dict_df=pd.DataFrame(words)
    word_dict_df.to_csv(prefix + '/HandP_dict.csv')
    i=1
    for filename in sorted(os.listdir(path)):
        # print('.', end='')
        file = open(path+"/"+filename,'r', encoding="utf8")
        title,des=t_n_d(file)
        try:
            # des_ls = sp.tokenize_str_hp(des[0],title)
            des_ls = sp.tokenize_str_hp(des, title)
            vec = make_seq_freq_vec(des_ls,words)
            if np.dot(vec, vec) != 0:
                data.append(vec)
                firms.append(title)
            else:
                if errorfile is not None:
                    errorfile.write(filename + " had no words after hnp preprocessing\n")
        except:
            pass
            # des_ls = sp.tokenize_str_hp("Adam",title)
            # print("missed ", filename)
        i+=1
    data = np.array(data).T.tolist()
    df = pd.DataFrame(data, columns=firms)
    return df

def run_hnp_vec(prefix="", errorfile=None):
    out_df=make_vec_df('test_data',make_dict_of_words('test_data', errorfile), prefix, errorfile)
    out_df.to_csv(prefix + '/HandP_vectors.csv')
    print('made HandP vectors')
    return prefix + '/HandP_vectors.csv'