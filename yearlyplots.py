# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:00:48 2019

@author: Ananthan
"""

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import pandas
# PLOTPATHS = ["all_tfidf_1.0_similarities_.csv" for year in range(1994, 2019)]
M = pandas.read_csv("all_tfidf_1.0_similarities.csv", index_col=0)

#works well for 675 input files, but will give poor labels for many more or less.
def plot_heat_map(matrix,path, title):
    # height = plt.rcParams['font.size']  * matrix.shape[0] / 10
    # width = plt.rcParams['font.size'] * matrix.shape[1] / 10
    sns.set(font_scale=0.04)
    # fig, ax = plt.subplots(figsize=(2^15, 2^15))
    # n = lambda s: s[s.index("'")+1:-2]
    # matrix = matrix.rename(n, axis='columns')
    # matrix = matrix.rename(n, axis='index')
    p=sns.heatmap(matrix,vmin= 0.0, vmax = 1.0, linewidths=0.0, square=True, xticklabels=True, yticklabels=True).set_title(title)
    p.figure.savefig(path, bbox_inches='tight')
    plt.clf()

for year in range(1994, 2019):
    m = M[[i for i in M.columns if str(year) == i[16:20]]].T[[i for i in M.columns if str(year) == i[16:20]]].T
    m.to_csv("all_tfidf_1.0_similarities_" + str(year) + ".csv")
    plot_heat_map(m, "all_tfidf_1.0_similarities_heat_" + str(year) + ".pdf", "all_tfidf_1.0_similarities_" + str(year))
    print( "plotted " + str(year))