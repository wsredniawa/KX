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
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FuncFormatter
from figure_properties import * 
# from Figure_5_fullcsd import figCSDraw, figCSD, fig_mua
py.close('all')
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
fsize = 20

def figHist(po, name = '32', name2 = '32', pos=(0,4, 4,8)):
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.1, 1.19, letter= letters[po])
    df_32 = pd.read_excel(loaddir+'eles_'+ name+'.xlsx')
    x_ele = df_32['x_pos'].values
    y_ele = df_32['y_pos'].values
    image = mpimg.imread(loaddir+'OB_' + name2 +'.jpg')
    if name=='32':eles32 = ['97', '98', '101_L', '101_R', '102', '103']
    else: eles32 = ['114', '116', '117', '129', '122_L', '127_L', '127_R', '128_R']
#    print(len(x_ele)/2)
    ax.imshow(image, extent = [0,1,0,1], aspect = 'auto')
    py.xlim(0.08, 1)
    if name==name2:
        for i in range(int(len(x_ele)/2)):
            py.plot([x_ele[2*i], x_ele[2*i+1]], [y_ele[2*i], y_ele[2*i+1]], 
                    linewidth = 2, label = 'Rat ' + eles32[i], color=colors[i])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    py.xticks([])
    py.yticks([])
    

def figLFP(po, nazwy = ['001'], lp=0.1, hp=7, filt = True, title = 'HFO', pos=(0,4, 4,8)):
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.1, 1.25, letter= letters[po])
    for i,key in enumerate(nazwy):
        hist_layer = list(df[int(key)][4:36].values)
        mit_pos = sl.all_indices(hist_layer, 'mit')
        y = np.load(loaddir+'saved_csd/'+'lfp_'+str(key)+'.npy')
        Fs = y.shape[1]
        print('Fs', Fs)
        if filt:
            [b,a] = butter(2.,[lp/(Fs/2.0), hp/(Fs/2.0)] ,btype = 'bandpass')
            y = filtfilt(b,a, y)
        py.imshow(y[::-1], extent = [-0.5, 0.5, 0-mit_pos[miti], 32-mit_pos[miti]], cmap=cm.get_cmap(mapa),
                         vmin=-np.max(abs(y)), vmax=np.max(abs(y)), aspect = 'auto')
    py.fill_between(np.arange(0,3)-1,-1,1,facecolor='grey', alpha=0.3)
    py.xlim(-0.25, 0.25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    tiky = ['EPL', 'mitral', 'grn']
    py.xlabel('Time (sec)', labelpad = -.5, fontsize=fsize-6)
    if not filt:
        py.title(title, fontsize = fsize-2)
        if po<5: 
            py.colorbar(orientation = 'horizontal', ticks = [0.1, 0, -0.1])
            py.yticks([-3,0, 10], tiky) 
        else: 
            cbar = py.colorbar(orientation = 'horizontal', ticks = [0.5, 0, -0.5], )
            cbar.ax.set_xlabel('mV', fontsize = fsize-4)
            py.yticks([-8,0, 8], tiky) 

def fig_phase(po, pos=(0,4, 4,8)):
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.1, 1.2, letter= letters[po])
    for ll in range(2):
        py.fill_between(range(300),-2,2,facecolor='grey', alpha=0.3)
        mean_csd = np.mean(csd_col[ll], axis=0)
        std_csd = np.std(csd_col[ll], axis=0)/np.sqrt(len(nazwy))
        py.plot(mean_csd, np.linspace(-30,30,64), label = labels[ll], linewidth = 2, color = colors[ll])
        py.fill_betweenx(np.linspace(-30,30,64), mean_csd - std_csd, mean_csd + std_csd, color=colors[ll], alpha = 0.3)
        py.axvline(0, linewidth = 0.5, linestyle = '--', color = 'grey')
        if po<5: py.yticks([-3,0, 10], tiky) 
        else: py.yticks([-6,0, 6], tiky) 
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
    if ele_typ==20: py.ylim(-15, 10)
    else: py.ylim(-4, 25)
    xL=['0',r'$\frac{\pi}{2}$',r'$\pi$', r'$\frac{3\pi}{2}$']
    py.xlim(0,290)
    py.xticks([0, 90, 180, 270], xL)
    if po<5: ax.legend(loc='center', bbox_to_anchor=(-.1, 1.1), ncol=2, 
                       frameon = True, fontsize = 16)
    if po>5: py.xlabel('Relative phase', fontsize = fsize-4)

def fig_power(po, pos=(0,4, 4,8)):
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.1, 1.2, letter= letters[po])
    if po>5: py.xlabel('Power ($mV^{2}$)', fontsize = fsize-4)
    for ll in range(2,4):
        py.fill_between(range(2),-2,2,facecolor='grey', alpha=0.3)
        mean_csd = np.mean(csd_col[ll], axis=0)
        std_csd = np.std(csd_col[ll], axis=0)/np.sqrt(len(nazwy))
        py.plot(mean_csd, np.linspace(-30,30,64), label = labels[ll%2], linewidth = 2, color = colors[ll%2])
        py.fill_betweenx(np.linspace(-30,30,64), mean_csd - std_csd, mean_csd + std_csd, color=colors[ll%2], alpha = 0.3)
        py.axvline(0, linewidth = 0.5, linestyle = '--', color = 'grey')
        nzw.append('average csd')
        if po<5: py.yticks([-3,0, 10], tiky) 
        else: py.yticks([-6,0, 6], tiky) 
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if ele_typ==20: py.ylim(-15, 10)
        else: py.ylim(-4, 25)
        
def figCSDraw(pos, num, gs, df, nazwa = ['129']):
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.1, 1.1, letter=num)
    key = nazwa[0]
    hist_layer = list(df[int(key)][4:36].values)
    mit_pos = sl.all_indices(hist_layer, 'mit')
    y = np.load(loaddir+'saved_csd/CSD'+str(key)+'.npy')
    py.imshow(y[::-1], extent = [-.5, .5, 0-mit_pos[-1], 32-mit_pos[-1]], cmap=cm.get_cmap('bwr'),
                     vmin=-np.max(abs(y)), vmax=np.max(abs(y)), aspect = 'auto')
#    py.fill_between(np.arange(-1,2),-1,1,facecolor='grey', alpha=0.3)
    py.yticks([-12, -5, -1, 5], ['glom.','EPL', 'mitral', 'grn']) 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    cbar = py.colorbar(orientation = 'vertical', ticks = [10, 0, -10])
    cbar.ax.set_title('$mA/mm^2$', fontsize = fsize-2)
    py.xlabel('Time  (sec)', fontsize = fsize-4)
    py.xlim(-0.15, 0.15)
    py.ylim(-15,15)
    
def figCSD(pos, num, gs, df, comp = 0, lowpass = 80, highpass = 130):
    nazwy = ['022', '027', '114', '116', '117', '127', '128', '129']
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.1, 1.1, letter=num)
    csd_col = np.zeros((len(nazwy),64,1394))
    for i,key in enumerate(nazwy):
        hist_layer = list(df[int(key)][4:36].values)
        mit_pos = sl.all_indices(hist_layer, 'mit')
        y_load = np.load(loaddir+'saved_csd/'+'CSD'+str(key)+'.npy')
        Fss = len(y_load[0])
#        print(y.shape, Fss)
        [b,a] = butter(3.,[lowpass/(Fss/2.0), highpass/(Fss/2.0)] ,btype = 'bandpass')
        y = filtfilt(b,a, y_load)
        min_mit = mit_pos[-1]
        print(min_mit)
#        min_mit = 19
        csd_col[i, 31-min_mit:63-min_mit] = y[:, :]
    mean_csd = np.mean(csd_col, axis=0)
#    py.imshow(img1[::-1], extent = [-1, 1, -15, 10], aspect = 'auto')
    py.imshow(mean_csd[::-1], extent = [-0.5, 0.5, -25, 25], cmap=cm.get_cmap('bwr'), alpha = 0.8,
              vmin=-np.max(abs(mean_csd)), vmax=np.max(abs(mean_csd)), aspect = 'auto')
    py.xlim(-0.1,0.1)
    if highpass<20: py.xlim(-0.15,0.15)
    py.ylim(-15,15)
    py.yticks([-12, -5, -1, 5], ['glom.','EPL', 'mitral', 'grn']) 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
#    py.colorbar(orientation = 'vertical')
    py.xlabel('Time (sec)', fontsize = fsize-4)
    py.axvline(0, linewidth = 0.1, color= 'black')

def fig_mua(pos, num, gs, df, version='cor'):
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1.1, letter=num)
#    py.title('MUA average frequency/phase')
    nazwy = ['022', '027', '128', '114']#, '116'
    miti= -1
    csd_col = np.zeros((2,len(nazwy),64))
    for i,key in enumerate(nazwy):
        hist_layer = list(df[int(key)][4:36].values)
        mit_pos = sl.all_indices(hist_layer, 'mit')
        sp_mtrx = np.load(loaddir+'saved_csd/spike_cor'+ str(key)+'.npy')
        sp_freq = np.load(loaddir+'saved_csd/spike_freq'+ str(key)+'.npy')
        min_mit = mit_pos[miti]
        frqs = np.linspace(0, 558, 51)
        frq_max = frqs[list(np.argmax(sp_freq[0, :, :], axis=1))]
        if min_mit<mit_pos[miti]: min_mit = mit_pos[miti]
        csd_col[1, i, 31-min_mit:63-min_mit] = sp_mtrx[0]
        csd_col[0, i, 31-min_mit:63-min_mit] = frq_max
    i=0
    if version == 'cor':
        mean_csd = np.mean(csd_col[1], axis=0)
        std_csd = np.std(csd_col[1], axis=0)/(len(nazwy))**(1/2)
    else: 
        mean_csd = np.mean(csd_col[0], axis=0)
        std_csd = np.std(csd_col[0], axis=0)/(len(nazwy))**(1/2)
    py.plot(mean_csd+i, np.linspace(-31,32,64), color = 'black', marker = 'o')
    py.fill_betweenx(np.linspace(-31,32,64), mean_csd - std_csd, mean_csd + std_csd, color='black', alpha = 0.5)
    py.yticks([-12, -5, -1, 5], ['glom.', 'EPL', 'mitral', 'grn']) 
    py.ylim(-15, 15)
    py.axvline(0, ls ='--', color = 'grey')
    py.axvline(-0.5, ls ='--', color = 'grey')
    py.axvline(0.5, ls ='--', color = 'grey')
    if version == 'cor': 
        py.xlim(-1, 1)
        py.xlabel('MUA histogram-HFO correlation', fontsize = fsize-4)
    else: 
        py.xlim(1,200)
        py.xlabel('MUA histogram frequency', fontsize=fsize-4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def fig_pic(po, pos, name):
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.1, 1.1, letter= letters[po])
    image = mpimg.imread(loaddir+name)
    ax.imshow(image, extent = [0,1,0,1], aspect = 'auto')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    py.xlabel('Time [ms]', fontsize = fsize-4, labelpad = 10)
    py.xlim(0.2, 0.8)
    py.ylim(0.2, 0.85)
    py.xticks([0.25,0.5, 0.75], ['-25','0','25'])
    py.yticks([])
#%%    
colors = ['brown', 'gray', 'olivedrab', 'darkgreen', 'royalblue', 'navy', 'darkorchid', 'red', 'y', 'teal',
          'powderblue', 'b', 'black', 'magenta']

loaddir = './fig4_files/'
df = pd.read_excel(loaddir+ 'hist_table.xlsx', index_col = 0)
df_csd = pd.read_excel(loaddir+ 'csd_table.xlsx', index_col = None) 

eles = np.linspace(1,32,32)

lfp = 1
mapa = 'PRGn'
scal = 2
miti= 0
csd_col = np.zeros((8,64,550))

fig = py.figure(figsize = (20,20), dpi = 260)
gs = gridspec.GridSpec(25, 13, hspace=4, wspace=4)

typ = ['_ph', 'delta_ph', '_pw', '_delta_pw']
typ_ind  = 3
tiky = ['EPL', 'mitral', 'grn']

nazwy = ['097','098','099','001','101','102','103']
nzw = []
csd_col = np.zeros((4, len(nazwy),64))
df_csd = pd.read_excel(loaddir+'phase_and_corr.xlsx', index_col = None)
labels = ['KX HFO', 'Delta 0.3-8 Hz']
for ll in range(4):
    for i,key in enumerate(nazwy):
        hist_layer = list(df[int(key)][4:36].values)
        mit_pos = sl.all_indices(hist_layer, 'mit')
        y = df_csd[str(key)+typ[ll]].values
        nzw.append(str(key))
        csd_col[ll, i, 31-mit_pos[miti]:63-mit_pos[miti]] = y

ele_typ = 32
r32 = '098'
figHist(0, '32', name2 = '32', pos=(0,3,0,7))
figLFP(1, [r32], filt = False, title = '$100$ $\mu m$ spacing electrodes', pos=(3,6,0,8))
#figLFP(2, [r32],  lp=80, hp=180, title = 'HFO filtered', pos=(6,8,0,4))
#figLFP(3,  [r32], lp=0.1, hp=2, title = 'Delta filtered', pos=(6,8,4,8))
fig_power(2, pos=(6,9,0,7))
fig_phase(3, pos=(9,12,0,7))

nazwy = ['022', '027', '114', '116', '117', '127', '128', '129']
# nazwy[:2]  = ['022', '027']

nzw = []
csd_col = np.zeros((4, len(nazwy),64))
df_csd = pd.read_excel(loaddir+'phase_and_corr.xlsx', index_col = None)
labels = ['HFO 80 -180 Hz', 'Delta 0.3 - 8 Hz']
for ll in range(4):
    for i,key in enumerate(nazwy):
        hist_layer = list(df[int(key)][4:36].values)
        mit_pos = sl.all_indices(hist_layer, 'mit')
        y = df_csd[str(key)+typ[ll]].values
        nzw.append(str(key))
        csd_col[ll, i, 31-mit_pos[miti]:63-mit_pos[miti]] = y
ele_typ = 20
figHist(4, '20', name2 = '20', pos=(0,3,9,16))
figLFP(5, ['022'], filt = False, title = '$20$ $\mu m$ spacing electrodes', pos=(3,6,9,17))



fig_power(6, pos=(6,9,9,16))
fig_phase(7, pos=(9,12,9,16))
#fig_hist(pos=(0,3, 18, 25), gs= gs, num='A', name='116')
figCSDraw(pos=(0,4,18, 25), gs=gs, df=df, num='I', nazwa = ['116'])
figCSD(pos=(4,7,18, 25), gs=gs, df=df, num = 'J', lowpass = 0.3, highpass = 5)
figCSD(pos=(7,10,18, 25), gs=gs, df=df, num = 'K', comp = 1)

# fig_pic(11, pos=(1,5, 28, 34), name='mua_example.png')
# fig_mua(pos=(9, 11, 28, 34), gs=gs, df=df, num='N', version='freq')
fig_mua(pos=(10, 13, 18, 25), gs=gs, df=df, num='L')
py.savefig('/Users/Wladek/Dysk Google/Figures for HFO in olfactory bulb/pub2_paper/figs/Figure_4.png')
py.close()
