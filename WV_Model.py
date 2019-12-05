# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:19:28 2019

@author: Ananthan, James
"""
import psutil
import numpy as np
import pandas as pd
from scipy import spatial as spat
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

def splitfile(fromPath, toPath):
    skipped_labels = False
    line_number = 1
    with open(fromPath, 'r', encoding='utf-8') as infile:
        for text in infile:
            # print("Reading line " + str(line_number))
            line_number += 1
            if skipped_labels:
                ID = ""
                for char in text:
                    if char not in '"[],/':
                        ID += char
                    if char in ",/":
                        break
                # if ID[5:15] == '0000789019' or ID[5:15] == '0000886982' or ID[5:15] == '0000019617':
                outfile = open(toPath + "/" + ID, 'w', encoding='utf-8')
                outfile.write(text)
                outfile.close()
            else:
                skipped_labels = True

def split_by_year(fNames):
    sections = []
    vlist = sorted(fNames, key=lambda s:s[16:20])
    sect = []
    year = vlist[0][16:20]
    for f in vlist:
        if f[16:20] == year:
            sect = sect + [f]
        else:
            sections.append(sect)
            sect = [f]
            year = f[16:20]
    return sections

def make_sim_mat(inPath, outPath, diag = 0.0):
    """
    The cosine similarity of two vectors i and j is: 
                                                    (i dot j)
                                                ------------------
                                                  norm(i)*norm(j)
    """
    vecs=pd.read_csv(inPath, index_col = 0)
    vec_raw = vecs.values
    vec_raw_t = np.transpose(vec_raw)
    dots = np.dot(vec_raw_t, vec_raw) # entry i,j is the dot product of vectors i and j
    vec_norm = np.sqrt(dots.diagonal())[np.newaxis] # entry i is the norm of vector i
    sim_mat = ((dots) / np.dot(np.transpose(vec_norm), vec_norm))
    np.fill_diagonal(sim_mat, diag)
    # outname = prefix + "/" + vec_name + "_similarities" + ("" if not yearly else "_" + veccsv.index[0][16:20]) + ".csv"
    out=pd.DataFrame(sim_mat, columns = vecs.columns, index = vecs.columns)
    out.to_csv(outPath)

def t_n_d(file):
    """
    gets title and description from a file
    """
    s = file.read()
    des = s[s.index(','):]
    t = s[:s.index(',')]
    return t,des

class Model:
    def __init__(self,tags,source,kind,th=0.2, minThr=0, errorfile=None):
        """
        :param tags:
            "nouns", or "words"
        :param source:
            "sic or 10k"
        :param kind:
            "boolean", "freq" or "tfidf"
        :param th:
            threshold for excluding words. th=0.2 means words occuring in more
            than 20% of documents are excluded
        """
        self.dict=None
        self.model_vecs=[]
        self.num_docs=0
        self.tags=tags
        self.th=th
        self.minThr = minThr
        self.kind=kind
        self.source=source
        self.errorfile=errorfile

    def asString(self):
        return  self.tags+"_" + self.source+'_'+self.kind+'_'+str(self.th) + "-" + str(self.minThr)
        
    def make_dict_of_words(self,path, prefix):
        """
        :return: dict of words and freq in corpus. If tags are "sic", words are instead drawn from files in SIC_DESC_PATH, but frequency is still from corpus.
        """




        # if os.path.exists(prefix + '/' + self.tags+"(" + self.source+")"+'_'+self.kind+'_'+str(self.th)+'_dict.csv'):
        #     df = pd.read_csv(prefix + '/' + self.tags+"(" + self.source+")"+'_'+self.kind+'_'+str(self.th)+'_dict.csv', index_col=0)
        #     d = df['0'].to_dict()
        #     self.dict = {v:k for k,v in d.items()}
        #     return

        # if MEMFILENAME is not None:
        #     with open(MEMFILENAME, "a") as outfile:
        #         outfile.write("make_dict_of_words\n")
        #         outfile.write(str(psutil.virtual_memory()) + "\n")
        #         outfile.write("available mem = " + str(psutil.virtual_memory().available) + "\n")



        word_dict={}
        total_docs=0
        dictfails = 0
        SIC_DESC_PATH = "sic_descriptions"
        if self.source == "sic":
            word_dict = {}
            for filename in os.listdir(SIC_DESC_PATH):
                file = open(SIC_DESC_PATH + "/" + filename, 'r', encoding='utf8')
                text = file.read()
                des = set(sp.tokenize_str(text))
                for word in des:
                    if word not in word_dict:
                        word_dict[word] = 0
                file.close()
        for filename in os.listdir(path):
            file = open(path+"/"+filename,'r', encoding="utf8")
            title,des=t_n_d(file)
            try:
                if self.tags=="words":
                    des_ls = sp.tokenize_str(des)
                elif self.tags=='nouns':
                    des_ls = sp.tokenize_str_hp(des,title)
                total_docs+=1
            except:
                des_ls = []
                if self.errorfile is not None:
                    dictfails += 1
                    exc1, exc2, exc3 = sys.exc_info()
                    self.errorfile.write(filename + " failed in dictionary step: " + str(exc1) + " ; " + str(exc2)+ "\n")
            # words_in_doc = set()
            file.close()
            words = set(des_ls)
            for word in words:
                if self.source == "sic":
                    if word in word_dict:
                        word_dict[word] += 1
                elif self.source == "10k":
                    if word not in word_dict:
                        word_dict[word] = 1
                    else:
                        word_dict[word] += 1
            # for word in des_ls:
            #     if word not in word_dict and word not in words_in_doc:
            #         if self.tags == "sic":
            #             pass
            #         else:
            #             word_dict[word]=1
            #             words_in_doc.add(word)
            #     elif word not in words_in_doc:
            #         word_dict[word]+=1
            #         words_in_doc.add(word)
        final_dict = {}
        threshold_outcasts = {}
        cutoff_outcasts = {}
        for k,v in word_dict.items():
            if v/total_docs > self.th:
                threshold_outcasts[k] = v
            elif v <= self.minThr:
                cutoff_outcasts[k] = v
            else:
                final_dict[k] = v
        # final_dict = {k: v for k, v in word_dict.items() if v/total_docs<=self.th and v > self.minThr}
        if self.errorfile is not None:
            self.errorfile.write(str(dictfails) + " documents failed in dictionary step\n")
        print('made dict with tags: ',self.tags, ', source: ',self.source,', th: ',self.th, ' cutoff: ',self.minThr)
        self.dict=final_dict
        self.num_docs=total_docs
        word_dict_df=pd.DataFrame(list(self.dict.keys()))
        # word_dict_df.to_csv(prefix + '/' + self.tags+"_" + self.source+'_'+self.kind+'_'+str(self.th)+'_dict.csv')
        word_dict_df.to_csv(prefix + '/' + self.asString()+'_dict.csv')
        cut_dict_df = pd.DataFrame(list(cutoff_outcasts.keys()))
        cut_dict_df.to_csv(prefix + '/' + self.asString() + '_cutoffs.csv')
        thr_dict_df = pd.DataFrame(list(threshold_outcasts.keys()))
        thr_dict_df.to_csv(prefix + '/' + self.asString() + '_thresholded.csv')


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
    
    def make_vec_df(self,path, prefix):


        vecfails = 0
        vecempty = 0
        words=list(self.dict.keys())
        words_to_index = {word : words.index(word) for word in words}
        years = split_by_year(sorted(os.listdir(path)))

        for year in years:
            # yearlen = 0

            # if MEMFILENAME is not None:
            #     with open(MEMFILENAME, "a") as outfile:
            #         outfile.write("make_vec_df " + year[0][16:20] + "\n")
            #         outfile.write(str(psutil.virtual_memory()) + "\n")
            #         outfile.write("available mem = " + str(psutil.virtual_memory().available) + "\n")

            data=[]
            firms=[]
            for filename in year:
                # yearlen += 1
                # if year_size_lim is not None:
                #     if year_size_lim < yearlen:
                #         break
                file = open(path+"/"+filename,'r', encoding="utf8")
                title,des=t_n_d(file)
                try:
                    if self.tags == 'nouns':
                        des_ls = sp.tokenize_str_hp(des, title)
                    elif self.tags == 'words':
                        des_ls = sp.tokenize_str(des)
                    vec = self.make_seq_freq_vec(des_ls,words, words_to_index)
                    if np.dot(vec, vec) != 0:
                        data.append(vec)
                        firms.append(title)
                    else:
                        vecempty += 1
                        if self.errorfile is not None:
                            self.errorfile.write(filename + " contained no words after preprocessing\n")
                except:
                    if self.errorfile is not None:
                        vecfails += 1
                        exc1, exc2, exc3 = sys.exc_info()
                        self.errorfile.write(filename + " failed in vector step: " + str(exc1) + " ; " + str(exc2)+ "\n")
                file.close()
                os.unlink(path+"/"+filename)
            data = np.array(data).T.tolist()
            df = pd.DataFrame(data, columns=firms)
            outname = prefix + '/'+self.asString()+'_vectors_' + year[0][16:20] + '.csv'
            df.to_csv(outname)






            # self.model_vecs.append((outname, year[0][16:20]))
            # if MEMFILENAME is not None:
            #     with open(MEMFILENAME, "a") as outfile:
            #         outfile.write("make_sims " + year[0][16:20])
            #         outfile.write(str(psutil.virtual_memory()) + "\n")
            #         outfile.write("available mem = " + str(psutil.virtual_memory().available) + "\n")
            self.make_sims(outname, year[0][16:20], prefix, 0.0)





        if self.errorfile is not None:
            self.errorfile.write(str(vecfails) + " documents failed in vector step\n" + str(vecempty) + " documents contained no words after preprocessing\n")








    def make_sims(self,vecName, year, prefix, diag=0.0):
        outName = prefix + '/'+self.asString()+'_sims_' + year + '.csv'
        make_sim_mat(vecName, outName, diag)

    # def make_sims(self, prefix, diag=0.0):
    #     for (vecName, year) in self.model_vecs:
    #         outName = prefix + '/'+self.tags+"_"+self.source+'_'+self.kind+'_'+str(self.th)+'_sims_' + year + '.csv'
    #         make_sim_mat(vecName, outName, diag)


def make_model(tags, source,kind,th,minThr = 0, errorfile=None,prefix=""):
    """
    For tags, kind, th: see Model.__init__()
    :errorfile:
        string: path to text file to record errors while making the model, or None, to ignore errors.
    :prefix:
        string: path to folder containing model's output.
    """
    model = Model(tags,source,kind,th, minThr, errorfile)
    print('making model')
    model.make_dict_of_words('test_data', prefix)
    print('made dict')
    model.make_vec_df('test_data', prefix)
    print('made vectors')
    model.make_sims(prefix, 0.0)
    print('made sims')

# MEMFILENAME = "memtest_output.txt"
year_size_lim = None
INPUTPATH = "all_10k_raw.csv"
# INPUTPATH = "dow_test_v2_raw.csv"
# MODELS_TO_TEST = [('nouns', '10k', 'tfidf', 1.0), ('nouns', '10k', 'boolean', 1.0), ('nouns', '10k', 'boolean', 0.2)]
MODELS_TO_TEST = [('nouns', '10k', 'tfidf', 1.0, 100)]

if not os.path.exists('test_data'):
    os.mkdir('test_data')
if not os.path.exists('hnp_proc'):
    os.mkdir('hnp_proc')
for filename in os.listdir('test_data'):
    os.unlink('test_data/' + filename)

splitfile(INPUTPATH, 'test_data')

for modelspecs in MODELS_TO_TEST:
    modelname = modelspecs[0]+"_"+modelspecs[1] + "_" + modelspecs[2] + "_" + str(modelspecs[3]) + "-" + str(modelspecs[4])
    # modelname = modelspecs[0] + "_" + modelspecs[1] + "_" + str(modelspecs[2]) + "_" + INPUTPATH[:-8]
    outputpath = modelname
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)
    ERRORFILE = outputpath + "/" + modelname + "_errors.txt"
    EF = open(ERRORFILE, 'w')
    make_model(modelspecs[0], modelspecs[1], modelspecs[2], modelspecs[3], modelspecs[4], EF, outputpath)
    EF.close()
    print('finished building ' + modelname)
