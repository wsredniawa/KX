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
from kcsd import KCSD1D
from pyhht import EMD
from figure_properties import *
import matplotlib as mpl
from mne.filter import notch_filter
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pactools import Comodulogram

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
 
py.close('all')
ch_order = [18,20,27,21,28,22,29,23,
            17,19,30,24,31,25,32,26,
            1,7,2,8,3,9,16,14,
            4,10,5,11,6,12,15,13]
    

def comodulogram(po, pos, sig1, sig2, name, save=1, plot=1, vmax=0.01, mtrx=False):
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.05, 1.05, letter= po)
    estimator = Comodulogram(fs=Fs, low_fq_range= np.linspace(0.1, 4, 13), 
                             high_fq_range=np.linspace(30, 150, 25),
                             low_fq_width=0.5, method = 'tort',#method='duprelatour',
                             progress_bar=True, n_jobs=4)
    estimator.fit(sig1, sig2)
    low_fq=0.1
    high_fq=4
    low_mod=30
    high_mod=150
    if mtrx:
        im = py.imshow(mtrx, extent=[low_fq, high_fq, low_mod, high_mod], 
                       vmax=vmax, aspect='auto', origin='lower', cmap='Blues')    
    else:
        im = py.imshow(estimator.comod_.T, extent=[low_fq, high_fq, low_mod, high_mod], 
                       vmax=vmax, aspect='auto', origin='lower', cmap='Blues')

    py.ylim(30, 130)
    py.xlabel('Driving Frequency (Hz)', fontsize = 13)
    py.ylabel('Modulated high Frequency (Hz)', fontsize = 13)
    py.xticks(fontsize=14)
    py.yticks(fontsize=14) 
    
    cax2 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0)
    cbar = py.colorbar(im, cax = cax2)#, format=mpl.ticker.FuncFormatter(fmt))
    cbar.ax.set_title('MI', fontsize = 13)
    cbar.ax.tick_params(labelsize=9)
    return estimator.comod_.T
    
def figWave(po, pos, sig_loc, start=400, stop = 405, Title = 'Saline'):
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.05, 1.05, letter= po)
    lowpass = 80 #20 beta
    highpass = 130 #50 beta
    shift = 1.2
    shift2 = 0.6
    sig_l = sig_loc[Fs*start:Fs*stop]
    sig3 = sig_l/abs(min(sig_l))
    ax1.plot(sig3, color = 'black', lw= 0.2)
    [b,a] = butter(3.,[lowpass/(Fs/2.0), highpass/(Fs/2.0)] ,btype = 'bandpass')
    sig3 = filtfilt(b,a, sig3)
    ax1.plot(sig3-shift, color = 'black', lw= 0.2)
    ax1.plot([1*Fs, 2*Fs],[-shift-shift2,-shift-shift2], lw=3, color = 'black')
    ax1.text(1.1*Fs, -shift-shift2+0.1, '1 sec', fontsize=10)
    
    ax1.plot([0.95*Fs, 0.95*Fs],[-shift-shift2,-shift-shift2+.25], lw=3, color = 'black')
    ax1.text(-500, -shift-shift2+0.28, '250 $\mu$V', fontsize=10)
#    py.xlim(0, 1)
    ax1.text(-5000, 0, 'Raw', fontsize=15)
    ax1.text(-5000, -1.2, 'HFO', fontsize=15)
    py.ylim(-2,1)
    py.text(2*Fs, 1.1, Title, fontsize=20)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    py.xticks([])
    py.yticks([])
    
def fig_power(po, pos, sig_loc, start=400, stop = 405, Title = '', asteriks=[0,0]):
    global freq
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    py.title(Title, fontsize=15)
    set_axis(ax1, -0.1, 1.05, letter= po)
    labels= ['Olf. bulb', 'Thalamus', 'Visual ctrx']
    for i in range(2,-1,-1):
        sig_loc[i] = notch_filter(sig_loc[i], Fs, np.arange(50, 450, 50))
        freq, sp = welch(sig_loc[i], Fs, nperseg = 1*Fs)
        ax1.plot(freq, sp, lw=4, label=labels[i])
        # ax1.text(asteriks[0], asteriks[1] , '*', fontsize=20)
        py.arrow(asteriks[0], asteriks[1], 0, -12, length_includes_head=True, clip_on = False,
                 head_width=2, head_length=4)
    # py.ylim(.1, 10e4)
    py.ylim(0,220)
    py.xlim(0, 155)
    py.ylabel('power $mV^2$')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    py.xlabel('Frequency (Hz)')
    if po!='B1': 
        inset_axes = inset_axes(ax1, width="23%", height=1.0, loc=1)
        for n in range(3,sig_loc.shape[0]):
            sig_loc[n] = notch_filter(sig_loc[n], Fs, np.arange(50, 450, 50))
            freq, sp = welch(sig_loc[n], Fs, nperseg = 1*Fs)
            py.plot(freq, sp, lw=2, color='green')
        py.ylim(0,220)
        py.xlim(0,155)
        py.xticks(fontsize=11)
        py.yticks(fontsize=11)
    # py.ylim(.1, 10e4)
    # py.yscale('log')
    # py.text(200,100, 'Olf. bulb propofol')
        py.xlabel('Frequency (Hz)')
    if po=='B1': 
        ket = mpatches.Patch(color='green', label='Olf. bulb')
        kx = mpatches.Patch(color='orange', label='Thalamus LGN')
        gam = mpatches.Patch(color='blue', label='Visual cortex')
        ax1.legend(loc='lower right', bbox_to_anchor=(1.2, .7), handles=[ket,kx,gam],
                   ncol=1, frameon = True, fontsize = 15)
    
def fig_spec(po, pos, sig_loc, typ=0):
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.05, 1.05, letter= po)
    freq, time_spec, spec_mtrx = spectrogram(sig_loc, Fs, nperseg=2*Fs)
    im = py.pcolormesh(time_spec/60, freq, spec_mtrx, 
                       norm = LogNorm(vmin = 1e1, vmax=5e2), vmax=400, cmap= 'Greens')
    py.ylim(0,150)
    # cbar=py.colorbar()
    # cbar.ax.set_title('power ($mV^2$)', fontsize = 12)
    times = [2.9, 3.4, 6.3, 6.8]
    if typ=='naris':
        for time in times:
            py.axvline(time, color='grey', ls='--', lw=1)    
        py.text(2.6, 160, 'naris block', color='black', fontsize=15)
        py.text(6, 160, 'naris block', color='black', fontsize=15)
    else:
        py.axvline(1.45, color='grey', ls='--', lw=4)
        py.text(1.45, 170, 'KX injection', color='black', fontsize=15)
        py.arrow(1.45, 165, 0, -10, length_includes_head=True, clip_on = False,
                 head_width=0.1, head_length=4)
    # py.axvline(105, color='red', ls='--', lw=2)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    py.xlabel('Time (min)')
    py.ylabel('Frequency (Hz)')
    
    cax2 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0)
    cbar = py.colorbar(im, cax = cax2)#, format=mpl.ticker.FuncFormatter(fmt))
    cbar.ax.set_title('$mV^2$', fontsize = 12)
    cbar.ax.tick_params(labelsize=12)

def fig_gamma_to_hfo(po, pos, sig1, sig2, sig3):
    global freq, time_spec, spec_mtrx
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.05, 1.05, letter= po)
    
    tps = [[2, 5], [2, 5],[0.4, 4]]
    sigs = [sig1,sig2,sig3]   
    i=0
    dur=10
    minute = int(60/dur)
    for sig,tp in zip(sigs,tps):
        sig = notch_filter(sig, Fs, np.arange(50, 450, 50))
        freq_g, time_spec, spec_mtrx1 = spectrogram(sig, Fs, nperseg=Fs*dur, noverlap=0)
        cp = np.max(spec_mtrx1[30*dur:65*dur], axis = 0)
        cp2 = np.max(spec_mtrx1[80*dur:180*dur], axis = 0)
        if i==0:
            ax1.plot([0,1], [cp[int(tp[0]*minute)], cp[tp[1]*minute]], '-o', color = 'navy', label='Gamma 30-65 Hz')
            ax1.plot([0,1], [cp2[int(tp[0]*minute)], cp2[tp[1]*minute]],'-o', color='indianred', label= 'KX HFO')
        else:
            ax1.plot([0,1], [cp[int(tp[0]*minute)], cp[tp[1]*minute]],'-o', color = 'navy')
            ax1.plot([0,1], [cp2[int(tp[0]*minute)], cp2[tp[1]*minute]],'-o', color='indianred')
        i+=1
    # ax1.plot([hfo_power[0], hfo_power[30], hfo_power[-1]])
    ax1.legend(loc='lower right', bbox_to_anchor=(1.4, 1), ncol=1, frameon = True, fontsize = 12)
    py.xticks([0,1], ['bef. KX', 'after KX'])
    py.xlim(-0.5, 1.5)
    py.yscale('log')
    py.ylabel('power $mV^2$')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    
def pilot_plot():
    py.figure()
    import seaborn as sns
    clrs = sns.color_palette('husl', n_colors=50)
    for i in range(0,32,1):
        cat3_order[i] = notch_filter(cat3_order[i], Fs, np.arange(50, 350, 50))
        freq, sp = welch(cat3_order[i], Fs, nperseg=10*Fs)
        py.plot(freq[50:3000], sp[50:3000], label = str(i), color = clrs[i])
    py.legend()
    
#%%
Fs = 3750
Fs_fast = 15000
loaddir2 = './fig2_files/'

cat1_prop = np.load(loaddir2+'E2019_PRE_1_2_cat1.npy')
cat2_prop = np.load(loaddir2+'E2019_PRE_1_1_cat2.npy')
cat3_prop = np.load(loaddir2+'E2019_PRE_1_2_cat3.npy')


cat1 = np.load(loaddir2+'cat1_fs3750.npy')
cat1_rest = np.load(loaddir2+'cat1_rest_fs3750.npy')
cat2 = np.load(loaddir2+'cat2_narisblock_fs3750.npy')
cat3 = np.load(loaddir2+'cat3_narisblock_fs3750.npy')

cat_OB = np.load(loaddir2+'sr_20200114-105254.npy')# cat3 3x narisblock
cat3_OB_trans = np.load(loaddir2+'sr_20200114-103732.npy')[3,:6*Fs*60]# cat3 last recording
cat2_OB_trans = np.load(loaddir2+'SmartboxRecording_20191230-103826_gamma_to_HFO.npy')[3,:6*Fs*60]# cat2 gamma_hfo
cat3v2_OB_trans = np.load(loaddir2+'SmartboxRecording_20200114-114535.npy')[3,:]


cat3_order = np.zeros(cat3_prop.shape)
catr = np.zeros(cat_OB.shape)
for i in range(32): 
    cat3_order[ch_order[i]-1] = cat3_prop[i]
    catr[ch_order[i]-1] = cat_OB[i]
cat3_order[32:] = cat3_prop[32:]   
#%%
# frag1=-60*Fs
# decomposer = EMD(catr[20, frag1:])
# imfs = decomposer.decompose()
# # b,a= butter(2, [0.1/(Fs/2), 4/(Fs/2)], btype='bandpass')
# # sig1= filtfilt(b,a,cat1[20, Fs*60*1:Fs*60*3])
# b, a= butter(2, [80/(Fs/2), 130/(Fs/2)], btype='bandpass')
# sig2= filtfilt(b,a,catr[20, frag1:])
#%%
# comp = 1
# py.figure()
# py.suptitle('component: '+ str(comp))
# py.subplot(121)
# py.plot(sig2+300, label='filtered')
# py.plot(imfs[comp], label='Huang hilbert comp.')
# py.legend()
# # py.plot(imfs[1]+10)
# py.subplot(122)
# freq, sp = welch(sig2, Fs, nperseg=Fs)
# # py.plot(freq, sp)
# freq, sp = welch(imfs[comp], Fs, nperseg=Fs)
# py.plot(freq, sp, color='orange')
# py.xlim(0,1000)
#%%
# sig_av_filt, sig_av_raw, sig_av_delta = average_hfo(cat3_filt, catr, cat3_delta)
fig = py.figure(figsize = (20,22), dpi = 260)
gs = gridspec.GridSpec(20, 17, hspace=4, wspace=4)

fig_power('B1', pos=(0, 5, 5, 10), Title='Cat 1', sig_loc=np.array([cat1[5], cat1_rest[0], 
                                                     cat1_rest[32]]), asteriks=[87,35])#, cat1_prop[5]]))
fig_power('B2', pos=(6, 11, 5, 10), Title='Cat 2', sig_loc=np.array([cat2[20], cat2[40], 
                                                     cat2[90], cat2_prop[3]]), asteriks=[83,160])
fig_power('B3', pos=(12, 17, 5, 10), Title='Cat 3', sig_loc=np.array([catr[20], cat3[40], cat3[90], 
                                                      cat3_order[20]]), asteriks=[93,185])
figWave('A1', pos = (0, 5, 0, 5), sig_loc=catr[20], start=3, stop = 12, Title = 'Olfactory bulb KX')
figWave('A2', pos = (6, 11, 0, 5), sig_loc=cat3[40], start=3, stop = 12, Title = 'Thalamus KX')
# comodulogram('A2', (6, 11, 0, 5), sig1, sig2, '', vmax=0.01)
figWave('A3', pos = (12, 17, 0, 5), sig_loc=cat3[90], start=3, stop = 12, Title = 'Visual Cortex KX')

com1=comodulogram('C1', (0, 5, 11, 15), cat1[20, -Fs*60*2:], cat1[20, -Fs*60*2:], '', vmax=0.01)
com2=comodulogram('C2', (6, 11, 11, 15), cat2[3, -Fs*60*2:], cat2[3, -Fs*60*2:], '', vmax=0.01)
com3=comodulogram('C3', (12, 17, 11, 15), catr[20, -Fs*60*2:], catr[20, -Fs*60*2:], '', vmax=0.01)

fig_spec('D', pos = (0, 6, 16, 20), sig_loc=cat2_OB_trans[Fs*2*60:])
fig_gamma_to_hfo('E', (7, 10, 16, 20), cat2_OB_trans, cat3_OB_trans, cat3v2_OB_trans)
fig_spec('F', pos = (11, 17, 16, 20), sig_loc=catr[20, :Fs*60*8], typ='naris')

py.savefig('/Users/Wladek/Dysk Google/Figures for HFO in olfactory bulb/pub2_paper/figs/Figure_2.png')
py.close()