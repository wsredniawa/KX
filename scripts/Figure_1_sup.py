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
from neo.io import Spike2IO
from matplotlib.colors import LogNorm

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
 
py.close('all')

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
    
def fig_power(po, pos, sigs, typ=0, okno=60, Fs= 1394):
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.1, 1.05, letter= po)
    HFO_power = []
    gamma_power = []
    Fs = int(Fs)
    maks_len = int(sigs.shape[1]/(60*Fs))
    for i in range(sigs.shape[0]):
        spec_mtrx = np.zeros((int(Fs/2)+1, maks_len))
        for ii in range(maks_len):
            freq, spec_mtrx[:,ii]= welch(sigs[i, ii*okno*Fs:(ii+1)*okno*Fs],Fs, nperseg=Fs)
        ind1 = np.where(freq < 180)[-1][-1]
        ind2 = np.where(freq > 105)[-1][0]
        norm = np.max(spec_mtrx[ind2:ind1,:5],axis=0)
        HFO_power.append(np.max(spec_mtrx[ind2:ind1,:],axis=0))#/norm.mean())
        ind1 = np.where(freq < 65)[-1][-1]
        ind2 = np.where(freq > 30)[-1][0]
        norm = np.max(spec_mtrx[ind2:ind1,:5],axis=0)
        gamma_power.append(np.max(spec_mtrx[ind2:ind1,:],axis=0))#/norm.mean())
    
    HFO_power = np.asarray(HFO_power)
    gamma_power = np.asarray(gamma_power)
    py.plot(np.linspace(0, maks_len, maks_len), HFO_power.mean(axis=0), color='indianred', label='HFO 105-180 Hz')
    std = HFO_power.std(axis=0)/np.sqrt(sigs.shape[0])
    py.fill_between(np.linspace(0, maks_len, maks_len), HFO_power.mean(axis=0)-std, HFO_power.mean(axis=0)+std, 
                    color='red', alpha=0.3)
    py.plot(np.linspace(0, maks_len, maks_len), gamma_power.mean(axis=0), color='navy', label='Gamma 30-65 Hz')
    std = gamma_power.std(axis=0)/np.sqrt(sigs.shape[0])
    py.fill_between(np.linspace(0, maks_len, maks_len), gamma_power.mean(axis=0)-std, gamma_power.mean(axis=0)+std, 
                    color='blue', alpha=0.3)
    py.yscale('log')
    if typ==0:
        py.axvline(11, color='grey', ls='--', lw=5)
        # py.text(4, 205, 'Xylazine inj.', color='black')
    else:
        py.axvline(20, color='grey', ls='--', lw=5)
        # py.text(10, 205, 'Ketamine 200 mg/kg inj.', color='black')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    py.ylabel('Power of dom. freq.')
    py.xlabel('Time (min)')
    if typ==0:     
        ax1.legend(loc='lower right', bbox_to_anchor=(1.1, -.35), ncol=2, frameon = True, fontsize = 15)
   
    
def fig_spec(po, pos, sig_loc, typ=0):
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.05, 1.05, letter= po)
    freq, time_spec, spec_mtrx = spectrogram(sig_loc, Fs, nperseg=int(Fs))
    py.pcolormesh(time_spec/60, freq, spec_mtrx, cmap = 'Greens', norm = LogNorm(vmin = 1e-6, vmax=1e-3))
    py.ylim(0,200)
    py.colorbar()
    py.xticks(fontsize=15)
    py.yticks(fontsize=15)
    if typ==0:
        py.axvline(11, color='grey', ls='--', lw=5)
        py.text(4, 205, 'Xylazine inj.', color='black')
    else:
        py.axvline(20, color='grey', ls='--', lw=5)
        py.text(10, 205, 'Ketamine 200 mg/kg inj.', color='black')
    # py.axvline(105, color='red', ls='--', lw=2)
    # py.text(80, 210, 'Naris block', color='yellow', fontsize=20)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    py.xlabel('Time (min)')
    py.ylabel('Frequency (Hz)')
    
def load_file(name, channel):
    r =  Spike2IO(filename=name)
    seg = r.read_segment()
    sig = seg.analogsignals[channel]
    print(sig.name)
    return np.asarray(sig)[:,0], sig.sampling_rate

#%%
loaddir2 = './fig1_files/'

rat42, Fs = load_file(loaddir2+'Rat42-xyl.smr', 1)
rat43, Fs = load_file(loaddir2+'Rat43-xyl.smr', 1)
rat44, Fs = load_file(loaddir2+'Rat44-xyl.smr', 1)
rat45, Fs = load_file(loaddir2+'Rat45-xyl.smr', 1)

rat62, Fs = load_file(loaddir2+'Rat62_Ket200.smr', 0)
rat65, Fs = load_file(loaddir2+'Rat65_Ket200.smr', 0)
rat66, Fs = load_file(loaddir2+'Rat66_Ket200.smr', 0)
# [b,a] = butter(3., 2/(Fs/2.0) ,btype = 'lowpass')
# cat3_delta = filtfilt(b,a, catr)
#%%
fig = py.figure(figsize = (16,12), dpi = 250)
gs = gridspec.GridSpec(11, 14, hspace=4, wspace=4)
# figWave('C1', pos = (12, 18, 0, 4), sig_loc=catr[20], start=3, stop = 15, Title = 'Olfactory bulb KX')
# figWave('C2', pos = (12, 18, 4, 8), sig_loc=cat3[40], start=3, stop = 15, Title = 'Thalamus KX')
# figWave('C3', pos = (12, 18, 8, 12), sig_loc=cat3[90], start=3, stop = 15, Title = 'Visual Cortex KX')

fig_spec('A', pos = (0, 7, 0, 5), sig_loc=rat43)
# fig_spec('A', pos = (7, 14, 0, 5), sig_loc=rat44)
# fig_spec('A', pos = (14, 21, 0, 5), sig_loc=rat45)

fig_spec('C', pos = (0, 7, 6, 11), sig_loc=rat65, typ=1)
# fig_spec('B', pos = (7, 14, 5, 10), sig_loc=rat65, typ=1)
# fig_spec('B', pos = (14, 21, 5, 10), sig_loc=rat66, typ=1)
end_time=int(Fs*60*52)
fig_power('B', pos=(8, 14, 0, 5), sigs=np.array([rat42[:end_time], rat43[:end_time], rat45[:end_time]]))#, cat1_prop[5]]))
end_time=int(Fs*60*94)
fig_power('D', pos=(8, 14, 6, 11), sigs=np.array([rat62[:end_time], rat65[:end_time], rat66[:end_time]]), typ=1)

py.savefig('/Users/Wladek/Dysk Google/Figures for HFO in olfactory bulb/pub2_paper/figs/Figure_1_figure supplement 1.png')
py.close()