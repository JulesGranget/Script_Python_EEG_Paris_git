



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




def viewer(sujet, cond, odor, chan_selection, filter=False):

    data = load_data_sujet(sujet, cond, odor)

    if len(chan_selection) == 1:

        chan_i = chan_list.index(chan_selection[0])

        x = data[chan_i,:]
        respi = data[-3,:]

        if filter:

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
        plt.plot(zscore(x)+3)
        plt.show()

    else:

        chan_i_list = [chan_list.index(_chan) for _chan in chan_selection]

        respi = data[-3,:]

        if filter:

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

        for i, chan_i in enumerate(chan_i_list):
        
            x = data[chan_i,:]
            plt.plot(zscore(x)+3*(i+1), label=chan_selection[i])
        
        plt.title(f"{sujet} {cond} {odor}")
        plt.legend()
        plt.show()


if __name__ == '__main__':

    
    sujet_list = ['01PD','02MJ','03VN','04GB','05LV','06EF','07PB','08DM','09TA','10BH','11FA','12BD','13FP',
    '14MD','15LG','16GM','17JR','18SE','19TM','20TY','21ZV','22DI','23LF','24TJ','25DF','26MN','27BD','28NT','29SC',
    '30AR','31HJ','32CM','33MA']

    sujet = '12BD'
    
    cond = 'FR_CV_1'
    cond = 'MECA'
    cond = 'CO2'
    cond = 'FR_CV_2'

    odor = 'o'
    odor = '+'
    odor = '-'

    chan_selection = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1']
    chan_selection = ['Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2']

    chan_list = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 
                'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 
                'PRESS', 'ECG', 'ECG_cR']

    viewer(sujet, cond, odor, chan_selection)




