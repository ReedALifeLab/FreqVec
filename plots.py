# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:00:48 2019

@author: Ananthan
"""

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

def plot_heat_map(matrix,path):
    p=sns.heatmap(matrix,linewidths=0.1)
    p.figure.savefig(path)