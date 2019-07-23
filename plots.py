# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:00:48 2019

@author: Ananthan
"""

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import pandas

#works well for 675 input files, but will give poor labels for many more or less.
def plot_heat_map(matrix,path, title):
    # height = plt.rcParams['font.size']  * matrix.shape[0] / 10
    # width = plt.rcParams['font.size'] * matrix.shape[1] / 10
    sns.set(font_scale=0.0035)
    fig, ax = plt.subplots(figsize=(2^15, 2^15))
    # p=sns.heatmap(matrix,vmin= 0.0, vmax = 1.0, linewidths=0.0, square=True, xticklabels=True, yticklabels=True).set_title(title)
    p=sns.heatmap(matrix,linewidths=0.0, square=True, xticklabels=True, yticklabels=True).set_title(title)
    p.figure.savefig(path, bbox_inches='tight')
    plt.clf()

def run_plots(PLOTPATHS):
    for matname in PLOTPATHS:
        m = pandas.read_csv(matname, index_col=0)
        plot_heat_map(m, matname[:-4] + "_heat.pdf", matname[:-4])