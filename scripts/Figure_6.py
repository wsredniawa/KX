# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 19:19:13 2019

@author: Wladek
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:01:35 2018

@author: Wladek
"""
import numpy as np
from scipy.interpolate import splrep,sproot
import pylab as py
import pandas as pd
import os
from scipy.stats import ttest_rel
import matplotlib.image as mpimg
from scipy.signal import welch, correlate, butter, filtfilt, spectrogram
import spike_lib_lor as sl
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
from figure_properties import * 
py.close('all')
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
bs = 20
early_s = 50
early_f = 60
late_s = 100
late_f = 120
fsize = 15
    
def plot_mean_cut(po, pos, sz= 200, typ = 1, typek ='', tit = 'power'):
    global df
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1.1, letter= letters[po])
    df = pd.read_excel(saveDir+'OB_cut.xlsx')
    lista_rats = ['Rat1-bef', 'Rat1-end', 'Rat2-bef', 'Rat2-end', 'Rat3-bef', 
                  'Rat3-end', 'Rat4-bef', 'Rat4-end','Rat5-bef', 'Rat5-end',
                  'Rat7-bef', 'Rat7-end', 'Rat11-bef', 'Rat11-end', 
                  'Rat12-bef', 'Rat12-end', 'Rat13-bef', 'Rat13-end']
    lw = 7
    py.title(tit)
    hfo_befC = np.zeros(int(len(lista_rats)/2))
    hfo_endC = np.zeros(int(len(lista_rats)/2))
    hfo_befI = np.zeros(int(len(lista_rats)/2))
    hfo_endI = np.zeros(int(len(lista_rats)/2))
    
    for i in range(int(len(lista_rats)/2)):
        print(lista_rats[i*2], lista_rats[i*2+1])
        hfo_befI[i] = df[lista_rats[i*2]][typ]
        hfo_endI[i] = df[lista_rats[i*2+1]][typ]
        hfo_befC[i] = df[lista_rats[i*2]][typ+3]
        hfo_endC[i] = df[lista_rats[i*2+1]][typ+3]

        # py.plot([0,2], [hfo_befC[2*i], hfo_endC[2*i+1]], color = 'black', ls='--', lw=2)
        # py.plot([0,2], [hfo_befI[2*i], hfo_endI[2*i+1]], color = 'black', lw=2)
        # else:
            # py.plot([0,2], [hfo_c, gamma_c], color = 'black', ls='--', lw=2, label='Contralateral')
            # py.plot([0,2], [hfo, gamma], color = 'black', label='Ipsilateral', lw=2)            
    # py.legend()
    wd = 0.5
    ind1 = np.arange(0,3,2) 
    sem = np.sqrt(len(lista_rats)/2)
    ax.bar(ind1-wd/2, (hfo_befI.mean(), hfo_endI.mean()), wd, label='Ipsilateral',
           yerr = (hfo_befI.std()/sem, hfo_endI.std()/sem), color = 'black')
    ax.bar(ind1+wd/2, (hfo_befC.mean(), hfo_endC.mean()), wd, label='Contralateral',
           yerr = (hfo_befC.std()/sem, hfo_befC.std()/sem), color = 'grey')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    py.xlim(-1,3)
    py.xticks([0,2], ['bef. cut', 'after cut'])
    
    fsize = 13
    if tit=='Power': 
        py.ylim(0, 1.5e-4)
        py.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        pvalue = ttest_rel(hfo_befI, hfo_endI)[1]
        print(pvalue)
        py.text(1.5, 1*1e-4, pval(pvalue), fontsize=fsize)
        pvalue = ttest_rel(hfo_befC, hfo_endC)[1]
        print(pvalue)
        py.text(2.1, 1.1*1e-4, pval(pvalue), fontsize=fsize)
    else: 
        py.ylim(50, 200)
        pvalue = ttest_rel(hfo_befI, hfo_endI)[1]
        print(pvalue)
        py.text(1.5, 120, pval(pvalue), fontsize=fsize)
        pvalue = ttest_rel(hfo_befC, hfo_endC)[1]
        print(pvalue)
        py.text(2.1, 110, pval(pvalue), fontsize=fsize)
    if po==3:
        py.legend(loc='center', bbox_to_anchor=(1.1, 1.2), 
                  ncol=2, frameon = True, fontsize = 14)
    
def figHist(po, pos, name = '32'):
    global box3
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.05, 1.05, letter= letters[po])
    image = mpimg.imread(saveDir + 'cut_brain2.png')
    ax1.imshow(image[:,::1], extent = [0,1,0,1], aspect = 'auto')
    # py.xlim(0.1, 0.9)
#    py.ylim(0, 0.95)
    # py.title("Histology", fontsize = 20)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    py.xticks([])
    py.yticks([])
#    box3 = ax1.get_position()
#    ax1.set_position()
    
def figWave(po, pos, typ, start=400, stop = 405, title = 'Saline'):
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.05, 1.05, letter= letters[po])
    sig_loc = np.load(saveDir+'5-Cut_LFP_full'+'.npy')[typ]
    Fs=5000
    lowpass = 90 #20 beta
    highpass = 110 #50 beta
    ax1.plot(sig_loc[Fs*start:Fs*stop]*0.25, color = 'black', lw= 0.2)
    [b,a] = butter(3.,[lowpass/(Fs/2.0), highpass/(Fs/2.0)] ,btype = 'bandpass')
    sig = filtfilt(b,a, sig_loc)
    ax1.plot(sig[Fs*start:Fs*stop]+0.6, color = 'black', lw= 0.2)
    ax1.plot([2*Fs,3*Fs],[-0.35,-0.35], color = 'black')
    ax1.text(2*Fs, -0.5, '1 sec')
    ax1.text(-3000, 0, 'Raw', fontsize=15)
    ax1.text(-3000, 0.6, 'HFO', fontsize=15)
#    py.xlim(0, 1)
    py.ylim(top=0.95)
    py.title(title, y=0.9, fontsize = 18)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    py.xticks([])
    py.yticks([])
#%%
colors = ['brown', 'gray', 'olivedrab', 'darkgreen', 'royalblue', 'navy', 'darkorchid', 'red', 'y', 'teal',
          'powderblue', 'b', 'black', 'magenta']

Fss = 1394
df = pd.read_excel("/Users/Wladek/Desktop/dok/schizo/" + 'hist_table.xlsx', index_col = 0)
saveDir = './fig6_files/'
fig = py.figure(figsize = (18,8), dpi = 260)
gs = gridspec.GridSpec(7, 25, hspace=0.1, wspace=0.1)

# figspec(1, pos = (8, 14, 0, 3), typ='bef',title='Before cut')
# figspec(2, pos = (8, 14, 4, 7), typ='end',title='After cut')

figWave(1, pos = (8, 14, 0, 3), typ=2 , start=209, stop = 213, title = 'Contralateral')
figWave(2, pos = (8, 14, 4, 7), typ=3, start=209, stop = 213, title = 'Ipsilateral')

figHist(0, pos = (0, 7, 0, 7))
plot_mean_cut(3, pos = (16, 20, 1,7), typ = 1, typek='xyl', tit='Power')
plot_mean_cut(4, pos = (21, 25, 1,7), typ = 0, typek='xyl', tit='Frequency')

py.savefig('/Users/Wladek/Dysk Google/Figures for HFO in olfactory bulb/pub2_paper/figs/Figure_6.png')
py.close('all')