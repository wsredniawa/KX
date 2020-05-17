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
import pylab as py
import matplotlib.image as mpimg
from scipy.signal import welch, argrelmin, butter, filtfilt, spectrogram
import matplotlib.gridspec as gridspec
# from kcsd import KCSD1D
from figure_properties import *
import matplotlib as mpl
from mne.filter import notch_filter
from matplotlib.colors import LogNorm

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
 
py.close('all')
ch_order = [18,20,27,21,28,22,29,23,
            17,19,30,24,31,25,32,26,
            1,7,2,8,3,9,16,14,
            4,10,5,11,6,12,15,13]

def figHist(po, pos, name = '32'):
    global box3,image
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.05, 1.05, letter= po)
    image = mpimg.imread(loaddir2 + name)
    ax1.imshow(image, extent = [0,1,0,1], aspect = 'auto')
    py.xlim(0.1, 0.9)
#    py.ylim(0, 0.95)
    py.title(name[:4], fontsize = 15, pad=1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    py.xticks([])
    py.yticks([])
#    box3 = ax1.get_position()
#    ax1.set_position()
    
def figWave(po, pos, sig_loc, start=400, stop = 405, Title = 'Saline'):
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.05, 1.05, letter= po)
    lowpass = 80 #20 beta
    highpass = 130 #50 beta
    shift = 1.2
    sig_l = sig_loc[Fs*start:Fs*stop]
    sig3 = sig_l/abs(min(sig_l))
    ax1.plot(sig3, color = 'black', lw= 0.2)
    [b,a] = butter(3.,[lowpass/(Fs/2.0), highpass/(Fs/2.0)] ,btype = 'bandpass')
    sig3 = filtfilt(b,a, sig3)
    ax1.plot(sig3-shift, color = 'black', lw= 0.2)
    ax1.plot([10*Fs, 11*Fs],[-shift-0.3,-shift-0.3], color = 'black')
    ax1.text(10*Fs, -shift-0.5, '1 sec')
#    py.xlim(0, 1)
    ax1.text(-5000, 0, 'Raw')
    ax1.text(-5000, -1.2, 'HFO')
    py.ylim(-1.5,1)
    py.text(2*Fs, 0.95,Title, fontsize = 20)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    py.xticks([])
    py.yticks([])
    
def fig_power(po, pos, sig_loc, start=400, stop = 405, Title = '', asteriks=[90,200]):
    global freq
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.1, 1.05, letter= po)
    labels= ['Olf. bulb', 'Thalamus', 'Visual ctrx']
    
    for i in range(2,-1,-1):
        sig_loc[i] = notch_filter(sig_loc[i], Fs, np.arange(50, 450, 50))
        freq, sp = welch(sig_loc[i], Fs, nperseg = 1*Fs)
        ax1.plot(freq, sp, lw=2, label=labels[i])
    py.legend(loc=2, fontsize=15)
    ax1.text(asteriks[0], asteriks[1] , '*', fontsize=20)
    ax1.text(160, 440, '?', fontsize=20)
    py.xlim(0, 260)
    py.ylim(0,500)
    py.ylabel('power ($mv^2$)')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    py.xlabel('Frequency (Hz)')
    if po!='B3': 
        inset_axes = inset_axes(ax1, width="23%", height=1.0, loc=1)
        for n in range(3,sig_loc.shape[0]):
            sig_loc[n] = notch_filter(sig_loc[n], Fs, np.arange(50, 450, 50))
            freq, sp = welch(sig_loc[n], Fs, nperseg = 10*Fs)
            py.plot(freq, sp, lw=0.7, color='green')
        py.ylim(0,220)
        py.xlim(0, 260)
        py.xticks(fontsize=11)
        py.yticks(fontsize=11)
    # py.ylim(.1, 10e4)
    # py.yscale('log')
    # py.text(200,100, 'Olf. bulb propofol')
        py.xlabel('Frequency (Hz)')
    if po=='B1': ax1.legend(loc='lower right', bbox_to_anchor=(1.2, 1.1), 
                            ncol=3, frameon = True, fontsize = 15)
    
def fig_spec(po, pos, sig_loc, typ=0):
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.05, 1.05, letter= po)
    freq, time_spec, spec_mtrx = spectrogram(sig_loc, Fs, nperseg=Fs)
    py.pcolormesh(time_spec/60, freq, spec_mtrx, vmax=1000, cmap= 'Greens',
                  norm = LogNorm(vmin = 1e1, vmax=5e2))
    py.ylim(0,200)
    py.colorbar()
    py.xticks(fontsize=10)
    py.yticks(fontsize=10)
    times = [2.9, 3.4, 6.3, 6.8, 9.6, 10.1]
    if typ=='naris':
        for i,time in enumerate(times):
            py.axvline(time, color='grey', ls='--', lw=1)    
            if i%2==0: py.text(time, 205, 'n.b', color='black')
    else:
        py.axvline(4, color='grey', ls='--', lw=1)
        py.text(4, 205, 'KX injection', color='black')
    # py.axvline(105, color='red', ls='--', lw=2)
    # py.text(80, 210, 'Naris block', color='yellow', fontsize=20)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    py.xlabel('Time (min)')
    py.ylabel('Frequency (Hz)')
    
def fig_podf(po, pos, sig_loc, typ=0):
    global freq
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.05, 1.05, letter= po)
    freq, time_spec, spec_mtrx = spectrogram(sig_loc, Fs, nperseg=10*Fs)
    # py.pcolormesh(time_spec/60, freq, spec_mtrx, vmax=1000, cmap= 'Greens_r')
    ind_h1 = np.where(freq < 170)[-1][-1]
    ind_h2 = np.where(freq > 150)[-1][0]
    sum_high = np.sum(spec_mtrx[ind_h2:ind_h1], axis = 0)
    ind_h1 = np.where(freq < 130)[-1][-1]
    ind_h2 = np.where(freq > 80)[-1][0]
    sum_hfo = np.sum(spec_mtrx[ind_h2:ind_h1], axis = 0)
    py.plot(time_spec/60, sum_hfo/max(sum_hfo), label = 'KX HFO', color='green')
    py.plot(time_spec/60, sum_high/max(sum_high), label = 'other HFO?', color='black')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, 
               frameon = True, fontsize = 15)
    py.xticks(fontsize=10)
    py.yticks(fontsize=10)
    times = [2.9, 3.4, 6.3, 6.8, 9.6, 10.1]
    for i, time in enumerate(times):
        py.axvline(time, color='grey', ls='--', lw=1)    
        if i%2==0: py.text(time, 1.05, 'naris block', fontsize=14)
    # py.axvline(105, color='red', ls='--', lw=2)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    py.xlabel('Time (min)')
    py.ylabel('Normalized dom. freq. power')
#%%
Fs = 3750
Fs_fast = 15000
loaddir2 = './fig2_files/'

cat3_prop = np.load(loaddir2+'E2019_PRE_1_2_cat3.npy')

cat3 = np.load(loaddir2+'cat3_narisblock_fs3750.npy')

cat_OB = np.load(loaddir2+'sr_20200114-105254.npy')# cat3 3x narisblock
cat_OB_trans = np.load(loaddir2+'sr_20200114-103732.npy')# cat3 last recording
# cat_OB_trans = np.load(loaddir2+'sr_20191230-105046.npy')# cat2


cat3_order = np.zeros(cat3_prop.shape)
catr = np.zeros(cat_OB.shape)
for i in range(32): 
    cat3_order[ch_order[i]-1] = cat3_prop[i]
    catr[ch_order[i]-1] = cat_OB[i]
cat3_order[32:] = cat3_prop[32:]   
#%%
# sig_av_filt, sig_av_raw, sig_av_delta = average_hfo(cat3_filt, catr, cat3_delta)
fig = py.figure(figsize = (16,16), dpi = 250)
gs = gridspec.GridSpec(16, 16, hspace=2, wspace=1)

# figHist('A', pos=(0,5,0,8))
fig_power('A', pos=(0, 5, 0, 8), sig_loc=np.array([catr[20], cat3[40], cat3[90], cat3_order[20]]))

fig_spec('B', pos = (6, 11, 0, 8), sig_loc=cat_OB_trans[3])
fig_spec('C', pos = (12, 17, 0, 8), sig_loc=catr[20], typ='naris')
fig_podf('D', pos = (18, 24, 0, 8), sig_loc=catr[20], typ='naris')

py.savefig('/Users/Wladek/Dysk Google/Figures for HFO in olfactory bulb/pub2_paper/figs/Figure_2_figure supplement 1.png')
py.close()