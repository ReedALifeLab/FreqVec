# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:55:47 2019

@author: Ananthan
"""
import pandas as pd
from scipy import spatial as spat
import numpy as np
import plots as pl

vec_path='tfidf_vec_tests.csv'
veccsv=pd.read_csv(vec_path)
firm_ls = [cent for cent in list(veccsv)[1:]]
print(firm_ls)
distance_mat=[[0.0 for f2 in firm_ls] for f1 in firm_ls]
for i in range(len(firm_ls)):
    f1=firm_ls[i]
    for j in range(len(firm_ls)):
        f2=firm_ls[j]
        f1_vec = veccsv[str(f1)].tolist()
        f2_vec = veccsv[str(f2)].tolist()
        distance_mat[i][j]=spat.distance.cosine(f1_vec, f2_vec)
        if f1==f2 or i==j:
            distance_mat[i][j]=np.NaN
out=pd.DataFrame(distance_mat,columns=[i for i in range(len(firm_ls))])
out.to_csv('tfidf_vec_dist_mat_test.csv')

print(distance_mat[47][47])
min_i=0
min_j=0
minv=999999.9
th=0.325
for i in range(len(firm_ls)):
    for j in range(len(firm_ls)):
        val=distance_mat[i][j]
        if val<minv and i!=j:
            minv=val
            min_i=i
            min_j=j
            #print(minv,min_i,min_j)
print('min dist ',distance_mat[min_i][min_j])
print(min_i,min_j)
print('firms: ',firm_ls[min_i],", ",firm_ls[min_j])
pl.plot_heat_map(distance_mat,"tfidf_heat.pdf")
print(distance_mat[18][26])

        
