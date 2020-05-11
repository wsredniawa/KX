import matplotlib.pyplot as plt

def set_axis(ax, x, y, letter=None):
    ax.text(
        x,
        y,
        letter,
        fontsize=20,
        weight='bold',
        transform=ax.transAxes)
    return ax

def pval(pvalue):
    if pvalue<0.05: nap = '*'
    else: nap='n.s.'
    if pvalue<0.01: nap='**' 
    if pvalue<0.001: nap='***'
    return nap

plt.rcParams.update({
    
    'xtick.labelsize': 17,
    'xtick.major.size': 17,
    'ytick.labelsize': 17,
    'ytick.major.size': 17,
    'font.size': 17,
    'axes.labelsize': 17,
    'axes.titlesize': 12,
    'axes.titlepad' : 5,
    'legend.fontsize': 10,
    'figure.subplot.wspace': 0,
    'figure.subplot.hspace': 0.2,
    'figure.subplot.left': 0.1,
})

