



import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
import mne
import pandas as pd
import joblib
import xarray as xr

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False


chan_list
data = load_data_sujet('12BD', 'CO2', 'o')
chan_i = chan_list.index('Fp2')

x = data[chan_i,:]
respi = data[-3,:]

fcutoff = 40
transw  = .2
order   = np.round( 7*srate/fcutoff )
shape   = [ 0,0,1,1 ]
frex    = [ 0, fcutoff, fcutoff+fcutoff*transw, srate/2 ]
filtkern = scipy.signal.firls(order+1,frex,shape,fs=srate)
x = scipy.signal.filtfilt(filtkern,1,x)


fcutoff = 100
transw  = .2
order   = np.round( 7*srate/fcutoff )
shape   = [ 1,1,0,0 ]
frex    = [ 0, fcutoff, fcutoff+fcutoff*transw, srate/2 ]
filtkern = scipy.signal.firls(order,frex,shape,fs=srate)
x = scipy.signal.filtfilt(filtkern,1,x)

plt.plot(zscore(respi))
plt.plot(zscore(x))
plt.show()








########################
######## PPI ######## 
########################


reref A1 A2

filtre : 0.05 40
notch 50


artifact rejection : enlever les epochs qui posent problème
zscore sur l'ensemble des epochs
vire ceux qui sont > à 3SD
potentiellement faire une deuxième vague de zscore après une première rejection

attention à pas enlever endessous de 100

#### PPI
-1.5 -> 0.5
pente > -0.5 = PPI
fit_lm de matlab

plot avec l'axe des y inveré pas * -1


import matplotlib.pyplot as plt
import numpy as np

def onpick(event):
    artist = event.artist
    print(artist.get_offsets())

fig, ax = plt.subplots()
fig.canvas.mpl_connect('pick_event', onpick)

srate = 100
times = np.arange(srate*60)/srate
for i in range(times.shape[0]):
    ax.scatter(times, 10*np.sin(2*np.pi*10*times), picker=True, marker='o', linestyle='None', pickradius=1)
plt.show()
