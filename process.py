# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:47:24 2019

@author: Nick, Ananthan
"""
import string
import nltk
from gensim.parsing import (strip_multiple_whitespaces, strip_tags, strip_punctuation,
                            strip_numeric, remove_stopwords, strip_short, stem_text)


def to_lowercase(s):
    return s.lower()


def preserve_special_punct(s, spec_punct=['.', '?', '!', ':', ';']):
    punct = string.punctuation
    for c in spec_punct:
        s = s.replace(c, ' ' + c + ' ')
        punct = punct.replace(c, '')
    for c in punct:
        s = s.replace(c, ' ')
    s = strip_multiple_whitespaces(s)
    s = s.replace(' s ', 's ')
    return s

def preprocess(s):
    for filt in FILTERS:
        s = filt(s)
    return s

def tokenize_str(s):
    s=preprocess(s)
    s_ls=s.split(" ")
    return s_ls

def tokenize_str_hp(s,t):
    s_tokens = nltk.word_tokenize(s)
    tags = nltk.pos_tag(s_tokens)
    nouns = [word for word,pos in tags 
             if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    s = ' '.join(nouns).lower()
    with open("hnp_proc/"+t.replace(" ", "_")+".txt", "w") as text_file:
        text_file.write(s)
    s = preprocess(s)
    s_ls = s.split(" ")
    return s_ls


FILTERS = [to_lowercase, strip_multiple_whitespaces, strip_tags, strip_punctuation,
           preserve_special_punct, strip_numeric, remove_stopwords, strip_short, stem_text]
FILTER_KWS = [f.__name__ for f in FILTERS]
FILTER_LOOKUP = dict(zip(FILTER_KWS, FILTERS))