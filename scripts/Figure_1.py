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
import matplotlib.patches as mpatches

py.close('all')
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
lista_rats = ['62','64', '65','66', '67', '86','87', '88','89'] 
bs = 10
early_s = 21
early_f = 26
late_s = 50
late_f = 55
fsize = 17

def plot_mean_podf(po, sz= 200, typ = 'cp', typek ='', pos=(0,4, 4,8)):
    global freq, df
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1.1, letter= letters[po])
    # df = pd.read_excel(saveDir+'g_and_h_mjh.xlsx')
    results = pd.read_excel(saveDir+'results.xlsx')
    df = pd.read_excel(saveDir+'g_and_h.xlsx')
    df2 = pd.read_excel(saveDir+'Injection times.xlsx')
    if 'xyl' in typek: color='purple'
    else: color='red'
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plot_len = 55
    min_minus = 20
    min_plus = 35
    okno = 60
    time_gh =  np.linspace(-20, min_plus, plot_len)
    hfo = np.zeros((len(lista_rats), plot_len))
    gamma = np.zeros((len(lista_rats), plot_len))
    for i in range(len(lista_rats)):
        row = df2.loc[df2['RAT'] == int(lista_rats[i])]
        start = int(row[typek].values[0]/(okno))
        # print(start)
        hfo[i] = df[lista_rats[i]+'HFO_' + typ+typek].values[start-min_minus:start+min_plus]
        gamma[i] = df[lista_rats[i]+ 'gamma_' + typ+typek].values[start-min_minus:start+min_plus]
        # py.plot(hfo[i], color='indianred')
        # py.plot(gamma[i], color = 'blue')
    sem = len(lista_rats)**(1/2)
    m_hfo = hfo.mean(axis=0)
    s_hfo = hfo.std(axis=0)/sem
    m_gamma = gamma.mean(axis=0)
    s_gamma = gamma.std(axis=0)/sem
    py.plot(time_gh, m_gamma, color = 'blue')
    py.fill_between(time_gh, m_gamma - s_gamma, m_gamma + s_gamma, alpha = 0.3, color = 'blue')
    py.plot(time_gh, m_hfo, color = color)
    py.fill_between(time_gh, m_hfo - s_hfo, m_hfo + s_hfo, alpha = 0.3, color = color)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    py.ylabel('Power of dom. freq.($mV^2$)', fontsize = fsize)
    py.yscale('log')
    py.xlabel('Time (min)', fontsize = fsize)
    ax = py.subplot(gs[pos[2]:pos[3], pos[1]+1:pos[1]+5])
    set_axis(ax, -0.05, 1.1, letter= letters[po+1])
    bef_gamma, rly_gamma, lat_gamma = [], [], []
    bef_hfo, rly_hfo, lat_hfo= [], [], []
    for i in range(len(lista_rats)):
        bef_gamma.append(gamma[i,bs-5:bs].mean())
        rly_gamma.append(gamma[i,early_s:early_f].mean())
        lat_gamma.append(gamma[i,late_s:late_f].mean())
        bef_hfo.append(hfo[i,bs-5:bs].mean())
        rly_hfo.append(hfo[i,early_s:early_f].mean())
        lat_hfo.append(hfo[i,late_s:late_f].mean())
        # if typek=='xyl':
        #     py.plot([hfo[i, bs-3:bs].mean(), hfo[i,early_s:early_f].mean(), hfo[i, late_s:late_f].mean()], marker = 'o', color = 'indianred')
        # else:
        py.plot([hfo[i, bs-5:bs].mean(), hfo[i,early_s:early_f].mean(), hfo[i, late_s:late_f].mean()], marker = 'o', color = color)
        py.plot([gamma[i, bs-5:bs].mean(), gamma[i,early_s:early_f].mean(), gamma[i, late_s:late_f].mean()], marker = 'o', color = 'blue')
        # py.text(-.1, hfo[i, bs-3:bs].mean(), lista_rats[i])
    
    results[typek+'bef_gamma'] = bef_gamma
    results[typek+'rly_gamma'] = rly_gamma
    results[typek+'lat_gamma'] = lat_gamma
    
    results[typek+'bef_hfo'] = bef_hfo
    results[typek+'rly_hfo'] = rly_hfo
    results[typek+'lat_hfo'] = lat_hfo
    
    results['rats'] = lista_rats
    results.to_excel(saveDir+ 'results.xlsx', sheet_name='sheet1', index=False)
   
    shift=np.asarray(rly_hfo).mean()/10
    max_ind = np.max(np.array([rly_hfo, lat_hfo])) 
    print('shap', st.shapiro(bef_gamma)[1])
    print('shap', st.shapiro(rly_gamma)[1])
    print('shap', st.shapiro(lat_gamma)[1])
    pvalue = st.ttest_rel(bef_gamma, rly_gamma)[1]
    print('gamma pval', pvalue)
    py.text(.9, max_ind+shift, pval(pvalue), color='blue')  
    pvalue = st.ttest_rel(bef_gamma, lat_gamma)[1]
    py.text(1.9, max_ind+shift, pval(pvalue), color='blue') 
    
    shift=np.asarray(rly_hfo).mean()*2
    print('shap', st.shapiro(bef_hfo)[1])
    print('shap', st.shapiro(rly_hfo)[1])
    print('shap', st.shapiro(lat_hfo)[1])
    pvalue = st.ttest_rel(bef_hfo, rly_hfo)[1]
    print('hfo pval', pvalue)
    py.text(.9, max_ind + shift, pval(pvalue), color=color)  
    pvalue = st.ttest_rel(bef_hfo, lat_hfo)[1]
    py.text(1.9, max_ind + shift, pval(pvalue), color=color) 
    
    py.ylabel('Power of dom. freq.($mV^2$)', fontsize = fsize)
    py.yscale('log')
    py.xticks([0,1,2], ['base', 'early Ket', 'late Ket'], fontsize=fsize) 
    if typek=='xyl': 
        py.xticks([0,1,2], ['base', 'early KX', 'late KX'], fontsize=fsize)
    else:
        ket = mpatches.Patch(color='red', label='HFO after Ket.')
        kx = mpatches.Patch(color='purple', label='HFO after KX')
        gam = mpatches.Patch(color='blue', label='Gamma 30-65 Hz')
        ax.legend(handles=[ket,kx,gam], loc='center', bbox_to_anchor=(1.7, 0.5), ncol=1, 
                  frameon = True, fontsize = 20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    py.xlim(-.2, 2.2)

def fig_freq(po, pos, typek = 'df', okno = 60):
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1.1, letter= letters[po])
    df = pd.read_excel(saveDir+'g_and_h.xlsx')
    df2 = pd.read_excel(saveDir+'Injection times.xlsx')
    results = pd.read_excel(saveDir+'results.xlsx')
    hfo = np.zeros((2,len(lista_rats), 60))
    min_minus = 20
    min_plus = 40
    for i in range(len(lista_rats)):
        row = df2.loc[df2['RAT'] == int(lista_rats[i])]
        start1 = row['_25'].values[0]/okno
        start2 = row['xyl'].values[0]/okno
        hfo[0,i] = df[lista_rats[i]+'HFO_' +typek+'_25'].values[int(start1)-min_minus:int(start1)+min_plus]
        hfo[1,i] = df[lista_rats[i]+'HFO_'+typek+ 'xyl'].values[int(start2)-min_minus:int(start2)+min_plus]
        py.plot([0,2], [hfo[0,i, early_s:early_f].mean(), hfo[1,i, late_s:late_f].mean()], '-o', color='red')
        py.plot([2], [hfo[1,i, late_s:late_f].mean()], 'o', color='purple')
        # py.text(-0.1, hfo[0,i, early_s:early_f].mean(), lista_rats[i])
    m_h_k25, s_h_k25 = hfo[0,:, early_s:early_f].mean(), hfo[0,:, early_s:early_f].std()
    m_h_xyl, s_h_xyl = hfo[1,:, late_s:late_f].mean(), hfo[1,:, late_s:late_f].std()
    results['dom_freq_early_ket_25'] = hfo[0,:, early_s:early_f].mean(axis=-1)
    results['dom_freq_late_kx'] = hfo[1,:, late_s:late_f].mean(axis=-1)
    # results['rats'] = lista_rats
    np.save('dom_freq_kx', hfo[1,:, late_s:late_f].mean(axis=-1))
    results.to_excel(saveDir+ 'results.xlsx', sheet_name='sheet1', index=False)
    print('xyl freq mean and sd: ', m_h_xyl, s_h_xyl)
    print('ket25 freq mean and sd: ',m_h_k25, s_h_k25)
    print('shap', st.shapiro(list(hfo[0,:, early_s:early_f].mean(axis=1)))[1])
    print('shap', st.shapiro(list(hfo[1,:, late_s:late_f].mean(axis=1)))[1])
    pvalue = st.ttest_rel(hfo[0,:, early_s:early_f].mean(axis=1), hfo[1,:, late_s:late_f].mean(axis=1))[1]
    print('freq pvalue: ', str(pvalue))
    py.text(1.8, 150, pval(pvalue), color='purple')
    py.xticks([0, 2], ['Ket.', 'KX'], fontsize = fsize)
    py.xlim(-1, 3)
    py.ylim(100, 200)
    py.ylabel("Frequency (Hz)", fontsize= fsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
def fig_spec(po, nazwa, title = 'HFO', pos=(0,4, 4,8)):
    from matplotlib.colors import LogNorm
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1.1, letter= letters[po])
    spec_mtrx = np.load(saveDir + nazwa)
    print('sec',spec_mtrx.shape)
#    freq2 =spec_mtrx[:, -1]
    im= py.pcolormesh(spec_mtrx[-1,:-1]/60 -20, spec_mtrx[:, -1], spec_mtrx[:-1, :-1],
                      cmap = 'Greens', norm = LogNorm(vmin = 1e-6, vmax=1e-4))
    py.xlim(-20, 50)
    py.ylim(1, 200)
    py.axvline(0, ls ='--', color = 'grey', lw = 3)
    py.colorbar(im)
    py.title(title, fontsize = fsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    py.ylabel('Frequency (Hz)', fontsize = fsize)
    py.xlabel('Time (min)', fontsize = fsize)

def fig_hht(po, pos, rat = '67', part ='k25', name = 'baseline'):
    skok = [0, 0.9, 1.3]
    bt= 24
    dur = 3
    if part[8:10]=='27' or part[8:10]=='10': dur = 2
    ax = py.subplot(gs[pos[2]:pos[3], pos[0]:pos[1]])
    set_axis(ax, -0.05, 1.0, letter= letters[po])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    py.xticks([])
    py.yticks([])
    py.text(1,1, name, fontsize = fsize+4)
    bs = np.load(part)
#    lfp = np.load(rat+part[5:8]+'_LFP.npy')
    time = np.linspace(0, dur*1000, dur*Fss)
    lista_comp = [-1, 0, 2]
    if 'xyl' in part: colors = ['black', 'purple', 'blue']
    else: colors = ['black', 'red', 'blue']
#    czas = int(part[8:10])*60*Fss
    [b_delta,a_delta] =  butter(2.,[0.1/(Fss/2.0), 200/(Fss/2.0)] ,btype = 'bandpass')
    bs[-1] = filtfilt(b_delta,a_delta, bs[-1])/np.max(abs(bs[-1]))
    for n,i in enumerate(lista_comp):
        mean = bs[i, Fss*bt:Fss*(bt+dur)]  - skok[n]
        if n!=0: lw = 0.5
        else: lw = 2
        py.plot(time, mean, color = colors[n], lw = lw)
    py.ylim(-2, 0.7)
    py.plot([0, 1000], [-1.84, -1.84], color = 'black', lw=3)
    py.text(0, -1.8, '1 sec', fontsize = fsize+4)
    py.plot(1000)
    ax = py.subplot(gs[pos[2]+3:pos[3], pos[1]:pos[1]+2])
    nazwa = ['Delta', 'HFO', 'Gamma']
    for n,i in enumerate(lista_comp[1:]):
        freq, sp = welch(bs[i], Fss, nperseg = Fss)
        py.plot(freq, sp, color = colors[n+1], label = nazwa[n+1])
#    py.legend()
    py.xlabel('Frequency')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    py.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    py.yticks(fontsize= 11)
    py.xticks([50,150], [50,150],fontsize=13)
    py.xlim(0, 200)
    py.ylim(0,2*1e-4)
#%%
colors = ['red', 'gray', 'reddrab', 'darkgreen', 'royalblue', 'blue', 'darkorchid', 'red', 'y', 'teal',
          'powderblue', 'b', 'black', 'magenta']
Fss = 1394
df = pd.read_excel("/Users/Wladek/Desktop/dok/schizo/" + 'hist_table.xlsx', index_col = 0)
saveDir = './fig1_files/'
eles = np.linspace(1,32,32)

fig = py.figure(figsize = (20,18), dpi = 300)
gs = gridspec.GridSpec(18, 18, hspace=4, wspace=4)

fig_hht(0, pos = (0, 4, 0, 7), part = saveDir+'hht67k2510.npy', name = 'Baseline')
fig_hht(1, pos = (6, 10, 0, 7), part = saveDir+'hht67k2527.npy', name = 'Ketamine 20 mg/kg')
fig_hht(2, pos = (12, 16, 0, 7), part = saveDir+'hht67xyl40.npy', name = 'Ketamine 100 mg/kg + xylazine 10 mg/kg')

fig_spec(3, 'RAT67_Ket25.npy', title='Ketamine', pos = (0, 5, 9, 13))
fig_spec(6, 'RAT67_Ketyl.npy', title='Ketamine-Xylazine', pos = (0, 5, 14, 18))
plot_mean_podf(4, typek = '_25', typ = 'podf', pos = (6, 10, 9, 13))
plot_mean_podf(7, typek = 'xyl', typ = 'podf', pos = (6, 10, 14, 18))
fig_freq(9, pos = (16, 18, 14, 18))

#fig_hist((1, 5, 10, 15), 'H', name = 'polar_gamma.png', tit = 'baseline gamma')
#fig_hist((6, 10, 10, 15), 'I', name = 'polar_ket25.png', tit = 'ketamine 20 mg/kg')
# fig_polar((3, 8, 21, 25), 'K', 'gamma.npy', 'blue')
# fig_polar((8, 13, 21, 25), 'L', 'ket25.npy', 'indianred')
# fig_polar((13, 18, 21, 25), 'M', 'ketxyl.npy', 'purple')

py.savefig('/Users/Wladek/Dysk Google/Figures for HFO in olfactory bulb/pub2_paper/figs/Figure_1.png')
py.close()
#figD(9)