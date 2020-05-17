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
from scipy.signal import welch, correlate, butter, filtfilt, spectrogram
import spike_lib_lor as sl
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.ticker as ticker
from figure_properties import * 
from scipy.stats import f_oneway, kruskal, ttest_rel, shapiro
py.close('all')
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
fsize = 20

def randsample(x,ile):
    ind = np.random.randint(len(x),size = ile)
    y = x[ind]
    return y

def figHist(let, pos, name = '32'):
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.05, 1.05, letter= let)
    image = mpimg.imread(loaddir+'rys1.png')
    ax1.imshow(image[:,::1], extent = [0,1,0,1], aspect = 'auto')
    # py.xlim(0.1, 0.9)
    # py.ylim(0.2, 0.9)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    py.xticks([])
    py.yticks([])

def figMI(let, pos, nazwy = ['008']):
    global orig_mean
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1.05, letter= let)
    proc = np.arange(0,100,10)
    mapa = 'Blues'
    py.title('Average modulation index', fontsize = fsize-2)
    MI_mtrx = np.zeros((8, 80, 10))
    MI_mtrx_ctrl = np.zeros((6, 80, 10))
    for i,key in enumerate(nazwy):
        y = np.load(loaddir+'MI_'+ str(key)+'.npy')
        MI_mtrx[i] = y[0].T/np.max(y[0])
    for i,key in enumerate(nazwy_ctrl):
        y = np.load(loaddir+'MI_'+ str(key)+'.npy')
        MI_mtrx_ctrl[i] = y[0].T/np.max(y[0])
    im = py.imshow(np.mean(MI_mtrx, axis=0), extent = [0.1, 8, 8, 568], cmap=cm.get_cmap(mapa), 
                   aspect = 'auto', origin='lower', vmin= 0)
    py.xlabel("Driving freq. (Hz)", fontsize = fsize-2)
    py.ylabel("Modulated freq. (Hz)", fontsize = fsize-2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    py.ylim(10,200)

    cax2 = make_axes_locatable(ax).append_axes("right", size="5%", pad=0)
    cbar = py.colorbar(im, cax = cax2)#, format=mpl.ticker.FuncFormatter(fmt))
    cbar.ax.set_title('MI', fontsize = fsize-2)
    cbar.ax.tick_params(labelsize=12)
#    aov_table = sm.stats.anova_lm(mod, typ=2)
    W = MI_mtrx
    L = MI_mtrx_ctrl
    orig_mean = (W.mean(axis=0) - L.mean(axis=0))
    worek = np.concatenate((L, W), axis=0)
    Nboots = 100000
    A=np.zeros((Nboots, L.shape[1], L.shape[2]))
    for i in range(Nboots):
        if i*100/Nboots in list(proc): print('percent done:', i*100/Nboots)
        grupa1 = randsample(worek, W.shape[0])
        grupa2 = randsample(worek, L.shape[0])
        A[i]= abs(np.mean(grupa1, axis=0) - np.mean(grupa2, axis=0))>=(orig_mean)
    p_mtrx = A.sum(axis=0)/Nboots
    ax = py.subplot(gs[pos[2]:pos[3], pos[1]:pos[1]+3])
    set_axis(ax, -0.05, 1.05, letter= 'I')
    im = py.imshow(p_mtrx, extent = [0.1, 8, 8, 568], cmap=cm.get_cmap('Reds_r'), 
                    aspect = 'auto', origin='lower', vmin=0, vmax=0.25)
    py.ylim(10,200)
    py.xlabel("Driving freq. (Hz)", fontsize = fsize-2)
    # py.ylabel("Modulated freq. [Hz]", fontsize = fsize-2)
    cax2 = make_axes_locatable(ax).append_axes("right", size="5%", pad=0)
    cbar = py.colorbar(im, cax = cax2)#, format=mpl.ticker.FuncFormatter(fmt))
    cbar.ax.set_title('p-value', fontsize = fsize-2)
    cbar.ax.tick_params(labelsize=12)

def figA(let, pos, name):
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1, letter= let)
    y = np.load(loaddir +'lfp_raw'+ name+'.npy')
    Fss = 1250
    sts = 10*Fss
    time = np.linspace(0,10, len(y[0,:sts]))
    [b_bg,a_bg] = butter(3.,[80/(Fss/2.0), 130/(Fss/2.0)] ,btype = 'bandpass')
    [b_bg2,a_bg2] = butter(3.,[0.1/(Fss/2.0), 8/(Fss/2.0)] ,btype = 'bandpass')

    py.plot(time, y[2, :sts],alpha = .5, label = 'Thermocouple', linewidth=0.8)
    sig = filtfilt(b_bg2,a_bg2, y[0,:sts])
    py.plot(time, sig - 0.07,alpha = .5, label = 'delta', linewidth=2, color = 'indianred')
    sig = filtfilt(b_bg, a_bg, y[0,:sts])
    py.plot(time, sig*2 - 0.12,alpha = .5, label = 'HFO', linewidth=0.5, color= 'black')
    py.plot([4, 5], [-0.15, -0.15], color = 'black', lw = 2)
    py.text(4.3, -0.17, '1 sec.')
#    py.legend()
    py.xlim(2, 5)
#    py.ylim(0, 0.3)
    py.xticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    py.yticks([0.02, -0.07, -0.12], ['Nasal resp.', 'Delta', 'HFO'], fontsize = fsize)
#    ax.text(2, 0.08, 'Breathig rhythm', fontsize = 10)
#    ax.text(2, -0.03, 'Local delta', fontsize = 10)
#    ax.legend(loc='center', bbox_to_anchor=(2, -0.25), ncol=2, frameon = False)

def fig_cor(let, pos):
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1.05, letter= let)
#    py.title('CSD profile')
    delta_c =[]
    breath_c =[]
    hfo_c = []
    df_loc = pd.read_excel(loaddir + 'phase_and_corr.xlsx')
    names = ['9_corr', '8_corr', '11_corr', '12_corr', '13_corr', '14_corr', '15_corr']
    for name in names:
        breath_c.append(df_loc[name].values[10]) 
        delta_c.append(df_loc[name].values[11])
        hfo_c.append(df_loc[name].values[12])
    #for point in cor_list: py.scatter(0, point, s= 100)
    py.boxplot(breath_c, positions= [0.1])
    py.boxplot(delta_c, positions= [0.6])
    py.boxplot(hfo_c, positions= [0.35])
    py.xticks([0.1, 0.6, 0.35], ['Resp./HFO', 'Resp./delta', 'Delta/HFO'], 
              rotation=0, fontsize= fsize-4)
    py.xlim(0, 0.8)
    py.ylim(-1, 1)
    py.ylabel("Correlation score", fontsize = fsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
def figC(let,pos, name):
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1, letter= let)
    y = np.load(loaddir +'lfp_raw'+ name+'.npy')
    Fss = 1250
    sts = 10*Fss
    time = np.linspace(0,10, len(y[0,:sts]))
    [b_bg,a_bg] = butter(3.,[80/(Fss/2.0), 130/(Fss/2.0)] ,btype = 'bandpass')
    [b_bg2,a_bg2] = butter(3.,[0.1/(Fss/2.0), 8/(Fss/2.0)] ,btype = 'bandpass')

    sig = filtfilt(b_bg2,a_bg2, y[0,:sts])
    py.plot(time, sig - 0.07,alpha = .5, label = 'Delta', linewidth=2, color = 'indianred')
    sig = filtfilt(b_bg, a_bg, y[0,:sts])
    py.plot(time, sig*2 - 0.12,alpha = .5, label = 'HFO', linewidth=0.5, color= 'black')
    py.xlim(3.2, 4)
#    py.ylim(0, 0.3)
    py.xticks([])
    ax.arrow(3.35, -0.1, 0.1, 0.05, head_width=0.02, head_length=0.03, shape= 'full', color = 'maroon')
    ax.arrow(3.6, -.03, 0.1, -0.05, head_width=0.02, head_length=0.03, shape= 'full', color = 'navy')
    ax.text(3.15, -0.05, 'Inhalation', fontsize = fsize-2)
    ax.text(3.65, -0.05, 'Exhalation', fontsize = fsize-2)
    py.plot([3.75, 4], [-0.15, -0.15], color = 'black', lw = 2)
    py.text(3.8, -0.16, '250 ms')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    py.yticks([-0.07, -0.12], ['Delta', 'HFO'], fontsize = fsize-2)

def fig_spec(let, pos, kan, tit):
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1.1, letter= let)
    sig = np.load(loaddir + '87xyl_LFP_full.npy')
    Fs = 1395
    freq, time_spec, spec_mtrx = spectrogram(sig[kan, Fs*2500: Fs*3000], Fs, nperseg = Fs)
    ind = np.where(freq < 200)[-1][-1]
    from matplotlib.colors import LogNorm
    im=ax.pcolormesh(time_spec/60, freq[:ind], spec_mtrx[:ind], 
                     norm = LogNorm(vmin = 1e-6, vmax=1e-4), cmap = 'Greens')
    py.colorbar(im)
    py.title(tit, fontsize = fsize)
    py.axvline(2.3, ls = '--', lw= 3, color = 'r')
    py.axvline(5.3, ls = '--', lw= 3, color = 'r')
    py.text(2.6, 170, 'Naris block', fontsize=13)
    py.ylabel('Freq. (Hz)')
    if '2' in let: py.xlabel('Time (sec)', fontsize = fsize)

def fig_statnb(let, pos):
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1.05, letter= let)
    df_loc = pd.read_excel(loaddir+'n_block.xlsx')
    bef_I = df_loc['bef_I'].values[:-1]
    dur_I = df_loc['dur_I'].values[:-1]
    aft_I = df_loc['aft_I'].values[:-1]
    bef_C = df_loc['bef_C'].values[:-1]
    dur_C = df_loc['dur_C'].values[:-1]
    aft_C = df_loc['aft_C'].values[:-1]
    sem = np.sqrt(len(bef_I))
    py.errorbar([0.05,1.05,2.05], [bef_I.mean(), dur_I.mean(), aft_I.mean()], yerr=[bef_I.std()/sem, dur_I.std()/sem, aft_I.std()/sem], 
                fmt='-o', color = 'b', label = 'Naris block')
    py.errorbar([0,1,2], [bef_C.mean(), dur_C.mean(), aft_C.mean()], yerr=[bef_C.std()/sem, dur_C.std()/sem, aft_C.std()/sem], 
                fmt='-o', ls = '--', color = 'g', label = 'Control')
    py.xticks([0,1,2], ['before', 'during' , 'after'], fontsize = fsize)
    ax.legend(loc='center', bbox_to_anchor=(1.1, 1.2), ncol=2,frameon = True, fontsize = 20)
    
    print('shap', shapiro(bef_I)[1])
    print('shap', shapiro(bef_C)[1])
    print('shap', shapiro(dur_I)[1])
    print('shap', shapiro(dur_C)[1])
    print('shap', shapiro(aft_I)[1])
    print('shap', shapiro(aft_C)[1])
    pvalue = f_oneway(bef_I, bef_C)[1]
    py.text(-.1, 2.5*1e-4, pval(pvalue))
    pvalue = f_oneway(dur_C, dur_I)[1]
    print(pvalue)
    py.text(0.95, 2.5*1e-4, pval(pvalue))
    pvalue = f_oneway(aft_I, aft_C)[1]
    py.text(1.9, 2.5*1e-4, pval(pvalue))
    py.yscale('log')
    py.ylabel('power ($mv^2$)')
    py.xlim(-0.5,2.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax = py.subplot(gs[pos[2]:pos[3], pos[1]:pos[1]+3])
    set_axis(ax, -0.05, 1.05, letter= 'C')
    for i in range(len(bef_I)):
        py.plot([bef_I[i], dur_I[i], aft_I[i]], color = 'b', label = 'Ipsilateral')
        py.plot([bef_C[i], dur_C[i], aft_C[i]], ls = '--', color = 'g', label = 'Contralateral')
    py.xticks([0,1,2], ['before', 'during' , 'after'], fontsize = fsize)
    py.xlim(-0.5,2.5)
    py.yscale('log')
    py.ylabel('power ($mv^2$)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def fig_polar(pos, num, nazwa, color):
    colors = ['navy', 'indianred', 'orange']
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]], polar=True)
    set_axis(ax, -0.3, 1.1, letter=num)
    # labels = ['Gamma', 'Ketamine HFO ~150 Hz', 'Ket-Xyl HFO 80-130 Hz']
    thetas = np.load(loaddir+nazwa)[1]
    rhos = np.load(loaddir+nazwa)[0]
    for ii in range(8):
        ax.plot([0, thetas[ii]], [0,rhos[ii]], lw = 3, color = color, alpha = 0.7)
    xT=py.xticks()[0]
    ax.set_rlabel_position(180)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',\
    r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
    py.xticks(xT, xL, fontsize = 20)
    py.yticks(fontsize=11)
    py.ylim(0, 0.8)
    
colors = ['brown', 'gray', 'olivedrab', 'darkgreen', 'royalblue', 'navy', 'darkorchid', 'red', 'y', 'teal',
          'powderblue', 'b', 'black', 'magenta']

loaddir = './fig3_files/'

fig = py.figure(figsize = (20,20), dpi = 300)
gs = gridspec.GridSpec(13, 9, hspace=10, wspace=10)

typ = ['_ph', 'delta_ph', '_pw', '_delta_pw']
typ_ind  = 3

nazwy = ['008','009','011','012','013','014','015', '016']
nazwy_ctrl = ['ctrl_007', 'ctrl_008','ctrl_012','ctrl_013','ctrl_014','ctrl_016']

labels = ['HFO 80 -180 Hz', 'Delta 0.3 - 8 Hz']

fig_spec('A1', (0,3, 0,2), kan = 0, tit = 'Ipsilateral')
fig_spec('A2', (0,3, 2,4), kan = 1, tit = 'Contralateral')
fig_statnb('B', (3,6, 0,4))

# figHist('D', (0,3, 5,9))
figA('D', (0,3, 4,9), '009')
figC('E', (3,6, 4,9), '009')
fig_polar((6, 9, 5, 8), 'F', 'ketxyl.npy', 'purple')

fig_cor('G', (0,3, 9,13))
figMI('H', (3,6, 9,13), nazwy = nazwy)

py.savefig('/Users/Wladek/Dysk Google/Figures for HFO in olfactory bulb/pub2_paper/figs/Figure_3.png')
py.close()