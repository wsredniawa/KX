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
from scipy.stats import f_oneway
from matplotlib.colors import LogNorm

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False 
py.close('all')
    
def fig_power(po, pos, sigs, typ=0, okno=60, Fs= 1000, low_freq=65):
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.1, 1.05, letter= po)
    rats_num =6
    isokx = np.zeros(rats_num)
    kx_freq = np.zeros(rats_num)
    for n, exp in enumerate(sigs):
        for i, sig in enumerate(exp):
            sig = sig[-et*6:-et*4]
            spec_mtrx = np.zeros((int(Fs/2)+1, int(et/(Fs*okno))))
            for ii in range(int(et/(Fs*okno))):
                freq, spec_mtrx[:,ii]= welch(sig[ii*okno*Fs:(ii+1)*okno*Fs], Fs, nperseg=Fs)
            ind1 = np.where(freq < 150)[-1][-1]
            ind2 = np.where(freq > low_freq)[-1][0]
            dom_freq = np.argmax(spec_mtrx[ind2:ind1], axis=0)+low_freq
            if n==1:
                print(str(rats[i])+' iso: ', dom_freq.mean())
                isokx[i]= np.mean(dom_freq)
            else:
                print(str(rats[i])+' kx: ', dom_freq.mean())
                kx_freq[i] = np.mean(dom_freq)
    py.ylim(50, 150)
    # kx_freq = np.load(loaddir2+'dom_freq_kx.npy')
    mean, std = kx_freq.mean(), kx_freq.std()/np.sqrt(rats_num)
    mean, std = 113.7, 2.4
    py.bar([0], mean, yerr=std, color='grey')
    mean, std = isokx.mean(), isokx.std()/np.sqrt(rats_num)
    mean, std = 94, 2.6
    py.bar([1], mean, yerr=std, color='grey', alpha=0.6)
    print('kx freq: ', kx_freq.mean(), kx_freq.std()/np.sqrt(len(kx_freq)))
    print('iso kx: ', isokx.mean(), isokx.std()/np.sqrt(rats_num))
    pvalue=f_oneway(kx_freq, isokx)[1]
    print(pvalue)
    py.text(0.5, 150, pval(pvalue))
    py.xticks([0,1], ['KX', 'ISO+KX'])
    py.xlim(-1,2)
    py.ylim(50, 200)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    py.ylabel('Frequency (Hz)')
    return freq
   
    
def fig_spec(po, pos, sig_loc, typ=0, vmax=1e-4):
    ax1 = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax1, -0.05, 1.05, letter= po)
    freq, time_spec, spec_mtrx = spectrogram(sig_loc, Fs, nperseg=int(10*Fs))
    py.pcolormesh(time_spec/60, freq, spec_mtrx, cmap = 'Greens', norm = LogNorm(vmin = 1e-6, vmax=vmax))
    py.ylim(0,200)
    py.colorbar()
    py.xticks(fontsize=15)
    py.yticks(fontsize=15)
    py.axvline(11, color='grey', ls='--', lw=5)
    py.text(4, 205, 'ISO + KX inj.', color='black')
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
loaddir2 = './fig3_files/'

rats = [17, 18, 28, 30, 31, 33]
typs = ['_KX' , '_ISOKX']
lista_iso = []
lista_kx = []

for rat in rats:
    for typ in typs:
        if 'ISO' in typ or rat==18: channel=3
        elif '_KX' in typ: channel=0
        if rat==30: channel=0
        sig, Fs = load_file(loaddir2+'Rat_'+str(rat)+typ+'.smr', channel)
        if 'ISO' in typ: lista_iso.append(sig[::int(Fs/1000)])
        else:  lista_kx.append(sig[::int(Fs/1000)])
#%%
Fs = 1000
fig = py.figure(figsize = (16,8), dpi = 250)
gs = gridspec.GridSpec(5, 14, hspace=4, wspace=4)

fig_spec('A', pos = (0, 7, 0, 5), sig_loc=lista_iso[-1][int(Fs)*4400:], vmax=1e-4)
# fig_spec('C', pos = (0, 7, 6, 11), sig_loc=rat65, typ=1)

et=int(Fs*300)
freq = fig_power('B', pos=(8, 14, 0, 5), sigs=[lista_kx, lista_iso], low_freq=80, Fs=Fs)
# end_time=int(Fs*60*94)
# fig_power('D', pos=(8, 14, 6, 11), sigs=np.array([rat62[:end_time], rat65[:end_time], rat66[:end_time]]), typ=1)

py.savefig('/Users/Wladek/Dysk Google/Figures for HFO in olfactory bulb/pub2_paper/figs/Figure_3_figure supplement 1.png')
py.close()