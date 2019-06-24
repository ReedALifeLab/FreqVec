# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:18:11 2019

@author: Ananthan
"""
import numpy as np
import math
import pandas as pd
import process as sp
# import re
import os


def t_n_d(file):
    """
    gets title and description from a file
    """
    s = file.read()
    # des = re.findall(r'\"([^]]*)\"', s)
    des = s[s.index(','):]
    t = s[:s.index(',')] # not all files in cleaned output had a ," combination, so just look for the comma
    return t,des
    

def make_dict_of_words(path, errorfile):
    """
    :return: dict of words and freq in corpus 
    """
    word_dict={}
    for filename in os.listdir(path):
        file = open(path+"/"+filename,'r', encoding="utf8")
        # print('.', end='')
        title,des=t_n_d(file)
        try:
            # des_ls = sp.tokenize_str(des[0])
            des_ls = sp.tokenize_str(des)
        except:
        #     des_ls = sp.tokenize_str("aaa") # previously gave bogus values as a placeholder, but now just throws out missed files entirely.
            des_ls = []
            # print("missed ", filename)
            if errorfile is not None:
                errorfile.write(filename + " could not be added to the dictionary\n")
        for word in des_ls:
            if word not in word_dict:
                word_dict[word]=1
            else:
                word_dict[word]+=1
    print('made freqvec dict')
    return word_dict

def make_seq_freq_vec(seq_ls,words, words_to_index):
    """
    :param seq: 
    :param words: 
    :return: 
    """
    vec=np.zeros(len(words))
    for word in seq_ls:
        vec[words_to_index[word]] += 1
    for i in range(len(seq_ls)):
        vec[i] = 1 + math.log(vec[i]) if vec[i] > 0 else vec[i]

    # for i in range(len(words)):
    #     count=seq_ls.count(words[i])
    #     if count>0:
    #         vec[i]=1+math.log(count)
    return vec

def make_vec_df(path,w_dict, prefix, errorfile):
    data=[]
    firms=[]
    words=list(w_dict.keys())
    words_to_index = {word : words.index(word) for word in words}
    word_dict_df=pd.DataFrame(words)
    word_dict_df.to_csv(prefix + '/freqvec_dict.csv')
    i=1
    for filename in sorted(os.listdir(path)):
        # print('.', end='')
        file = open(path+"/"+filename,'r', encoding="utf8")
        title,des=t_n_d(file)
        try:
            # des_ls = sp.tokenize_str(des[0])
            des_ls = sp.tokenize_str(des)
            vec = make_seq_freq_vec(des_ls,words, words_to_index)
            data.append(vec)
            firms.append(title)
        except:
            pass
            #des_ls = sp.tokenize_str("aaa")
            # print("missed ", filename)
            # if errorfile is not None:
            #     errorfile.write(filename + " missed in vector-making step\n")
        i+=1
    data = np.array(data).T.tolist()
    df = pd.DataFrame(data, columns=firms)
    return df

def run_freq_vec(prefix="", errorfile=None):
    out_df=make_vec_df('test_data',make_dict_of_words('test_data', errorfile), prefix, errorfile)
    out_df.to_csv(prefix + '/freqvec_vectors.csv')
    print('made freqvec vectors')
    return prefix + '/freqvec_vectors.csv'