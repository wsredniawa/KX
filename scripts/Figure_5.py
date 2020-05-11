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
import matplotlib.image as mpimg
from scipy.signal import welch, correlate, butter, filtfilt
import spike_lib_lor as sl
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
import scipy.stats as st
from figure_properties import * 
py.close('all')
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
bs = 20
early_s = 50
early_f = 60
late_s = 100
late_f = 120
fsize = 15

def randsample(x,ile):
    ind = st.randint.rvs(0,len(x),size = ile)
    y = x[ind]
    return y

def stats(W, L):
    proc = np.arange(0,100,10)
    orig_mean = abs(W.mean() - L.mean())
    worek = np.concatenate((L, W))
    Nboots = 10000
    A=np.zeros(Nboots)
    for i in range(Nboots):
        if i*100/Nboots in list(proc): print('percent done:', i*100/Nboots)
        grupa1 = randsample(worek, W.shape[0])
        grupa2 = randsample(worek, L.shape[0])
        A[i]= abs(np.mean(grupa1) - np.mean(grupa2))>=(orig_mean)
    p_mtrx = A.sum()/Nboots
    return p_mtrx
    
def plot_mean_podf(po, sz= 200, tit ='', typ = 'HFO', typek ='bic', label='', pos=(0,4, 4,8)):
    global freq
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    set_axis(ax, -0.05, 1.1, letter= letters[po])
    if typek=='bqx':
        lista_rats = list_rats_qx
        df = pd.read_excel(saveDir+'nbqx_data.xlsx')
        link = '-'
        plot_len = 210
    elif typek=='bic': 
        df = pd.read_excel(saveDir+'bic_data.xlsx')
        lista_rats = ['6', '7', '9', '1', '2']
        link = '_'
        plot_len = 210
    elif typek=='rbo': 
        df = pd.read_excel(saveDir+'carbo_data.xlsx')
        lista_rats = ['3', '4', '5', '7']
        link = '_'
        plot_len = 210
    
    time_gh =  np.linspace(-4, int(plot_len/15-4), plot_len)
    hfo = np.zeros((len(lista_rats), plot_len))
    gamma = np.zeros((len(lista_rats), plot_len))
    ns = len(lista_rats)
    for i in range(ns):
        hfo[i] = df[lista_rats[i]+link+typ+'_cp'+typek].values[:plot_len]
        gamma[i] = df[lista_rats[i]+link+typ+'_cp2'+typek].values[:plot_len]    
    m_hfo = hfo.mean(axis=0)
    s_hfo = hfo.std(axis=0)/np.sqrt(ns)
    m_gamma = gamma.mean(axis=0)
    s_gamma = gamma.std(axis=0)/np.sqrt(ns)
    py.ylabel(tit, fontsize = fsize)
    py.plot(time_gh, m_hfo, label = label+ ' infusion', color = 'indianred')
    py.plot(time_gh, m_gamma, label = 'Saline infusion', color = 'green')
    py.fill_between(time_gh, m_hfo - s_hfo, m_hfo + s_hfo, alpha = 0.3, color = 'indianred')
    py.fill_between(time_gh, m_gamma - s_gamma, m_gamma + s_gamma, alpha = 0.3, color = 'green')
    ax.legend(loc='lower right', bbox_to_anchor=(1.1, 1), ncol=1, frameon = False, fontsize = 12)
    py.ylim(0, 2.5)
    
    if po>3: py.xlabel('Time (min)', fontsize = fsize)

def fig_spec(po, nazwa, title = 'HFO', pos=(0,4, 4,8)):
    from matplotlib.colors import LogNorm
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1.1, letter= letters[po])
    spec_mtrx = np.load(saveDir + nazwa)
    print('sec',spec_mtrx.shape)
#    freq2 =spec_mtrx[:, -1]
    if 'A16' in nazwa: backmin = 15
    else: backmin = 5 
    im= py.pcolormesh(spec_mtrx[-1,:-1]/60-backmin, spec_mtrx[:, -1], spec_mtrx[:-1, :-1],
                      cmap = 'Greens', norm = LogNorm(vmin = 1e-6, vmax=1e-4))
    py.xlim(-5, 12)
    py.ylim(1, 200)
    py.axvline(0, ls ='--', color = 'grey', lw = 3)
    py.colorbar(im)
    py.title(title, fontsize = 20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    py.ylabel('Frequency (Hz)', fontsize = fsize)
    py.xlabel('Time (min)', fontsize = fsize)

def fig_bars(po, pos, typ = 'cp', typek ='bic', ypos=2.1, okno = 60):
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1.1, letter= letters[po])
    if typek=='bqx':
        lista_rats = list_rats_qx
        df = pd.read_excel(saveDir+'nbqx_data.xlsx')
        link = '-'
        last=180
        y1, y2 = 1.5, 1.3
    elif typek=='bic': 
        df = pd.read_excel(saveDir+'bic_data.xlsx')
        lista_rats = ['6', '7', '1', '2','9']
        link = '_'
        last = 180
        y1, y2 = 1.7, 1.5
    elif typek=='rbo':
        df = pd.read_excel(saveDir+'carbo_data.xlsx')
        lista_rats = ['3', '4', '5', '7']#'9'
        link = '_'
        last = 180
        y1 , y2 = 2.4, 2

    hfo = np.zeros((2,len(lista_rats), okno))
    delta = np.zeros((2, len(lista_rats), okno))
    
    for i in range(len(lista_rats)):
        hfo[0,i] = df[lista_rats[i]+link+'HFO_cp2'+typek].values[last-okno:last]
        delta[0, i] = df[lista_rats[i]+link+'delta_cp2'+typek].values[last-okno:last]
        hfo[1,i] = df[lista_rats[i]+link+'HFO_cp'+typek].values[last-okno:last]
        delta[1, i] = df[lista_rats[i]+link+'delta_cp'+typek].values[last-okno:last]
    ind1 = np.array([0, 1.5])
    wd = 0.4
    sem = np.sqrt(len(lista_rats))
    ctrl_d_mn, ctrl_d_std = delta[0].mean(), delta[0].std()/sem
    drug_d_mn, drug_d_std = delta[1].mean(), delta[1].std()/sem
    ctrl_h_mn, ctrl_h_std = hfo[0].mean(), hfo[0].std()/sem
    drug_h_mn, drug_h_std = hfo[1].mean(), hfo[1].std()/sem
    # print(m_h_xyl, s_h_xyl)
    ax.bar(ind1[0]-wd/2, ctrl_d_mn, wd, yerr = ctrl_d_std, color = 'green')
    # ax.scatter(np.zeros(len(lista_rats))+wd, delta[1].mean(axis=1), color = 'black')
    ax.bar(ind1[1]-wd/2, ctrl_h_mn, wd, yerr = ctrl_h_std, color = 'green')
    # ax.scatter(np.zeros(len(lista_rats))+1.5+wd, hfo[1].mean(axis=1), color = 'black')
    ax.bar(ind1[0]+wd/2, drug_d_mn, wd, yerr = drug_d_std, color = 'indianred')
    # ax.scatter(np.zeros(len(lista_rats))-wd, delta[0].mean(axis=1), color = 'black')
    ax.bar(ind1[1]+wd/2, drug_h_mn, wd, yerr = drug_h_std, color = 'indianred')
    # ax.scatter(np.zeros(len(lista_rats))+1.5-wd, hfo[0].mean(axis=1), color = 'black')
    # print('shape', delta[0].mean(axis=1).shape)
    # pvalue = np.round(stats(delta[0].mean(axis=1), delta[1].mean(axis=1)), 3)
    pvalue = np.round(st.f_oneway(delta[0].mean(axis=1), delta[1].mean(axis=1))[1], 5)
    print(typek, ' delta', pvalue)
    py.text(-0.2, y1, pval(pvalue))
    # pvalue = np.round(stats(hfo[0].mean(axis=1), hfo[1].mean(axis=1)), 3)
    pvalue = np.round(st.f_oneway(hfo[0].mean(axis=1), hfo[1].mean(axis=1))[1], 5)
    print(typek, 'hfo', pvalue)
    py.text(1.5, y2, pval(pvalue))
    
    py.xticks([0, 1.5], ['Delta', 'HFO'], fontsize = fsize)
    py.xlim(-1, 2)
    py.ylabel("Norm. power", fontsize= fsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
def figWave(po, pos, name = 'RAT4.npy', start=400, stop = 405, Title = 'Saline'):
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
#    set_axis(ax1, -0.05, 1.05, letter= letters[po])
    sig =np.load(saveDir + name)
    Fs = 1395
    if 'bic' in name: Fs=5000
    if 'rbo' in name: Fs=5000
    lowpass = 80 #20 beta
    highpass = 130 #50 beta
    ax1.plot(sig[Fs*start:Fs*stop]*0.5, color = 'black', lw= 0.2)
    [b,a] = butter(3.,[lowpass/(Fs/2.0), highpass/(Fs/2.0)] ,btype = 'bandpass')
    sig = filtfilt(b,a, sig)
    ax1.plot(sig[Fs*start:Fs*stop]*2+0.7, color = 'black', lw= 0.2)
    ax1.plot([2*Fs,3*Fs],[-0.6,-0.6], color = 'black')
    ax1.text(2.5*Fs, -0.75, '1 sec')
#    py.xlim(0, 1)
    py.ylim(top=0.95)
    py.title(Title, fontsize = 20)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    py.xticks([])
    py.yticks([])
#%%
colors = ['brown', 'gray', 'olivedrab', 'darkgreen', 'royalblue', 'navy', 'darkorchid', 'red', 'y', 'teal',
          'powderblue', 'b', 'black', 'magenta']
list_rats_qx = ['2', '3', '4', '5', '6', '7', '8']

Fss = 1394
saveDir = './fig5_files/'
eles = np.linspace(1,32,32)

fig = py.figure(figsize = (18,20), dpi = 250)
gs = gridspec.GridSpec(16, 24, hspace=0.1, wspace=4)
fig_spec(6, 'Rat_A23_carbo.npy',title = 'CBX infusion', pos = (0, 7, 12, 16))
fig_spec(0, 'RAT4-KX-OB-NBQX2ug_3.npy',title = 'CNQX infusion', pos = (0, 7, 0, 4))
fig_spec(3, 'RAT_A16_03219-ok.npy',title = 'Bicuculline infusion', pos = (0, 7, 6, 10))
# fig_spec(6, 'RAT4-KX-OB-NBQX2ug_3_ctrl.npy',title = 'PFC removal', pos = (0, 7, 18, 22))

plot_mean_podf(7, tit = 'Norm. HFO power', typ = 'HFO', typek = 'rbo', label='CBX', pos = (13, 19, 12, 16))
plot_mean_podf(1, tit= 'Norm. HFO power', typ = 'HFO', typek = 'bqx',label='NBQX', pos = (13, 19, 0, 4))
plot_mean_podf(4, tit = 'Norm. HFO power', typ = 'HFO', typek = 'bic',label='Bic.', pos = (13, 19, 6, 10))

figWave(0, pos = (7, 12, 12, 16), name = 'A27rbo_LFP_full.npy', start=900, stop = 903, Title = 'After CBX')
figWave(0, pos = (7, 12, 0, 4), name = 'RAT4nbqx.npy', start=602, stop = 605, Title = 'After NBQX')
figWave(0, pos = (7, 12, 6, 10), name = 'RAT_A16_03219bic_LFP_full.npy', start=1300, stop = 1303, Title = 'After Bicuculine')
# figWave(0, pos = (17, 22, 18, 22), name = 'RAT4.npy', start=605, stop = 608, Title = 'After cut')
# fig_bars(0, pos=(20, 23, 0, 4), typ = 'cp', typek ='bqx', okno = 5)
fig_bars(8, pos=(21, 24, 12, 16), typ = 'cp', typek ='rbo', okno = 20)
fig_bars(2, pos=(21, 24, 0, 4), typ = 'cp', typek ='bqx', okno = 20)
fig_bars(5, pos=(21, 24, 6, 10), typ = 'cp', typek ='bic', okno = 20)

# figHist(7, pos = (9, 16, 18, 22))
py.savefig('/Users/Wladek/Dysk Google/Figures for HFO in olfactory bulb/pub2_paper/figs/Figure_5.png')
py.close('all')