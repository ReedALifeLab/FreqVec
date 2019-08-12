# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:55:47 2019

@author: Ananthan
"""
import pandas as pd
from scipy import spatial as spat
import numpy as np
import random #random module produces pseudorandom numbers, so possibly not good enough for intensive analysis.

def run_sims(vec_path, vec_name, prefix, make_shuffled_mat=False, diag_value = 0.0):
    """
    The cosine similarity of two vectors i and j is: 
                                                    (i dot j)
                                                ------------------
                                                  norm(i)*norm(j)
    """
    veccsv=pd.read_csv(vec_path, index_col = 0)
    vec_raw = veccsv.values
    vec_raw_t = np.transpose(vec_raw)
    dots = np.dot(vec_raw_t, vec_raw) # entry i,j is the dot product of vectors i and j
    vec_norm = np.sqrt(dots.diagonal())[np.newaxis] # entry i is the norm of vector i
    sim_mat = ((dots) / np.dot(np.transpose(vec_norm), vec_norm))
    np.fill_diagonal(sim_mat, diag_value)
    out=pd.DataFrame(sim_mat, columns = veccsv.columns, index = veccsv.columns)
    out.to_csv(prefix + "/" + vec_name + "_similarities.csv")
    print('made ' + vec_name + ' similarites')

    #normally, vectors are sorted according to firm. However, we can shuffle the list of firms, which breaks this ordering.
    #this has the same effect as randomly renaming the documents
    #to get a symmetrical matrix, we have to use the same reordering of the list of firms to set both the columns and rows.
    if make_shuffled_mat:
        ind = list(out.index)
        random.shuffle(ind)
        out2 = out.reindex(ind, axis='index')
        out2 = out2.reindex(ind, axis='columns')
        out2.to_csv(prefix + "/" + vec_name + "_similarities_shuffled.csv")


        
