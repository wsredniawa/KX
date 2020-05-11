# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:24:12 2020

@author: Wladek
"""

import pylab as py
import numpy as np
import pandas as pd 
from scipy.signal import welch, argrelmin, butter, filtfilt, spectrogram
import matplotlib.gridspec as gridspec
import spike_lib_lor as sl
from figure_properties import *

def figprof(pos, let, lowpass = 0.1, highpass = 8, title=''):
    nazwy = ['022', '027', '114', '116', '117', '127', '128', '129']
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1.05, letter= let)
    miti= -1
    csd_col = np.zeros((len(nazwy),64))
    for i,key in enumerate(nazwy):
        hist_layer = list(df[int(key)][4:36].values)
        mit_pos = sl.all_indices(hist_layer, 'mit')
        y = np.load(loaddir+'CSD'+ str(key)+'.npy')
        Fss = 1394
        [b,a] = butter(3.,[lowpass/(Fss/2.0), highpass/(Fss/2.0)] ,btype = 'bandpass')
        y = filtfilt(b,a, y)
        pol = int(len(y[0])/2)+1
        y = y[:,pol]
        y = y/np.max(abs(y+1e-10))*.3
        csd_col[i, 31-mit_pos[miti]:63-mit_pos[miti]] = y 
        y2 = csd_col[i]
        py.plot(y2+i+1, np.linspace(-30,30,64), color = 'grey')
        py.fill_betweenx(np.linspace(-30,30,64), i+1+y2, i+1, where=y2+i+1>=i+1, color='red', alpha = 0.5)
        py.fill_betweenx(np.linspace(-30,30,64), i+1+y2, i+1, where=y2+i+1<=i+1, color='blue', alpha = 0.5)
    py.yticks([-12, -5, -1, 5], ['glom.', 'EPL', 'mitral', 'grn'])  
    py.ylim(-15,15)
    py.plot([0,10], [-1,-1], color='grey', alpha=0.3, lw=10)
    py.xlabel('Rats')
    py.xlim(0.5, 8.5)
    py.title(title, fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    
loaddir = './fig4_files/saved_csd/'
df = pd.read_excel('./fig4_files/hist_table.xlsx', index_col = 0)
df_csd = pd.read_excel('./fig4_files/csd_table.xlsx', index_col = None) 

fig = py.figure(figsize = (10,12), dpi = 260)
gs = gridspec.GridSpec(4, 6, hspace=4, wspace=4)

figprof(pos=(0,6,0,2), let = 'A', lowpass = 0.3, highpass = 5, title='Delta')
figprof(pos=(0,6,2,4), let = 'B', lowpass = 80, highpass = 130, title='KX HFO')

py.savefig('/Users/Wladek/Dysk Google/Figures for HFO in olfactory bulb/pub2_paper/figs/Figure_4_figure supplement 1.png')
py.close()