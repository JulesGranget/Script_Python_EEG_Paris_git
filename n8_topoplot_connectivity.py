

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
import pickle
import xarray as xr

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False


################################
######## GENERATE XR ########
################################


def get_Cxy_Surrogates_allcond(sujet, session_eeg):

    source_path = os.getcwd()
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    with open(f'{sujet}_s{session_eeg+1}_Cxy_allcond.pkl', 'rb') as f:
        Cxy_allcond = pickle.load(f)

    with open(f'{sujet}_s{session_eeg+1}_surrogates_allcond.pkl', 'rb') as f:
        surrogates_allcond = pickle.load(f)

    os.chdir(source_path)

    return Cxy_allcond, surrogates_allcond


def get_Pxx(sujet, session_eeg):

    source_path = os.getcwd()
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    with open(f'{sujet}_s{session_eeg+1}_Pxx_allcond.pkl', 'rb') as f:
        Pxx_allcond = pickle.load(f)

    os.chdir(source_path)

    return Pxx_allcond



def compute_Cxy_xarray(Cxy_allsession, Surrogates_allsession, respi_mean_allcond, around_respi_Cxy, prms):

    print('#### COMPUTE XR CXY ####')
    
    hzCxy = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #### generate data
    data_Cxy = np.zeros((3, len(conditions_allsubjects), len(prms['chan_list_ieeg']), 2, 1))
    for session_eeg in range(3):
        for cond_i, cond in enumerate(conditions_allsubjects):
            respi_mean_i = respi_mean_allcond[f's{session_eeg+1}'][cond]
            respi_mean_up = respi_mean_i + around_respi_Cxy
            respi_mean_dw = respi_mean_i - around_respi_Cxy
            mask_respi_hzCxy = (hzCxy>=respi_mean_dw) & (hzCxy<respi_mean_up)
            for nchan in range(len(prms['chan_list_ieeg'])):
                for data_type in ['data', 'surrogates']:
                    if data_type == 'data':
                        data_Cxy[session_eeg, cond_i, nchan, 0, :] = np.mean(Cxy_allsession[f's{session_eeg+1}'][cond][nchan][mask_respi_hzCxy])
                    else :
                        data_Cxy[session_eeg, cond_i, nchan, 1, :] = np.mean(Surrogates_allsession[f's{session_eeg+1}']['Cxy'][cond][nchan][mask_respi_hzCxy])



    #### generate xarray
    dims_Cxy = ['odor', 'cond', 'chan', 'data', 'Cxy_at_respi']
    coords_Cxy = {'odor': [odor_i for odor_i in odor_order[sujet].values()], 'cond':conditions_allsubjects, 'chan':prms['chan_list_ieeg'] , 'data':['Cxy', 'surrogates'], 'Cxy_at_respi':['Cxy_at_respi']}
    xr_Cxy = xr.DataArray(data=data_Cxy, dims=dims_Cxy, coords=coords_Cxy)

    return xr_Cxy


def compute_Pxx_xarray(Pxx_allsession, prms):

    print('#### COMPUTE XR PXX ####')
    
    hzPxx = np.linspace(0,srate/2,int(prms['nfft']/2+1))

    #### identify number of band
    n_band = 0
    band_list = []
    for band_prep in band_prep_list:
        for band in list(freq_band_dict[band_prep]):
            n_band += 1
            band_list.append(band)

    #### generate data
    data_Pxx = np.zeros((len(band_prep_list), 3, len(conditions_allsubjects), n_band, len(prms['chan_list_ieeg']), 1))
    for band_prep_i, band_prep in enumerate(band_prep_list):
        for band_i, (band, freq) in enumerate(freq_band_dict[band_prep].items()):     
            for session_eeg in range(3):
                for cond_i, cond in enumerate(conditions_allsubjects):
                    for nchan in range(len(prms['chan_list_ieeg'])):
                        mask = (hzPxx >= freq[0]) & (hzPxx <= freq[1])
                        Pxx_mean = np.mean(Pxx_allsession[f's{session_eeg+1}'][band_prep][cond][nchan][mask])
                        data_Pxx[band_prep_i, session_eeg, cond_i, band_i, nchan, :] = Pxx_mean



    #### generate xarray
    dims_Pxx = ['band_prep', 'odor', 'cond', 'band', 'chan', 'Pxx_mean']
    coords_Pxx = {'band_prep' : band_prep_list, 'odor': [odor_i for odor_i in odor_order[sujet].values()], 'cond':conditions_allsubjects, 'band' : band_list, 'chan':prms['chan_list_ieeg']}
    xr_Pxx = xr.DataArray(data=data_Pxx, dims=dims_Pxx, coords=coords_Pxx)

    return xr_Pxx










########################
######## PLOT ########
########################




def plot_save_Cxy(xr_Cxy, info, prms):

    odor_list = [odor_i for odor_i in odor_order[sujet].values()]

    vmax = np.max(xr_Cxy.loc[:, :, :, 'Cxy', :].values)

    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions_allsubjects))
    plt.suptitle(f'{sujet}_Cxy_at_respi')

    #odor_i = '+'
    for r, odor_i in enumerate(odor_list):

        for c, cond in enumerate(conditions_allsubjects):

            ax = axs[r,c]
            if r == 0:
                ax.set_xlabel(cond)
            if c == 0:
                ax.set_ylabel(f'odor_{odor_i}')
            mne.viz.plot_topomap(xr_Cxy.loc[odor_i, cond, :, 'Cxy', :].values.reshape(len(prms['chan_list_ieeg'])), info, axes=ax, vmin=0, vmax=vmax, show=False)

    #plt.show()
    #### save
    os.chdir(os.path.join(path_results, sujet, 'TOPOPLOT'))
    fig.savefig(f'{sujet}_Cxy_at_respi.jpeg', dpi=600)
    plt.close()







    
def plot_save_Pxx(xr_Pxx, info, prms):

    odor_list = [odor_i for odor_i in odor_order[sujet].values()]

    for band_prep in band_prep_list:

        for band in list(freq_band_dict[band_prep]):

            vmax = np.max(xr_Pxx.loc[band_prep, :, :, band, :].values)
            vmin = np.min(xr_Pxx.loc[band_prep, :, :, band, :].values)

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions_allsubjects))
            plt.suptitle(f'{sujet}_Pxx_{band_prep}_{band}')

            #odor_i = '+'
            for r, odor_i in enumerate(odor_list):

                for c, cond in enumerate(conditions_allsubjects):

                    ax = axs[r,c]
                    if r == 0:
                        ax.set_xlabel(cond)
                    if c == 0:
                        ax.set_ylabel(f'odor_{odor_i}')
                    mne.viz.plot_topomap(xr_Pxx.loc[band_prep, odor_i, cond, band, :, :].values.reshape(len(prms['chan_list_ieeg'])), info, axes=ax, vmin=vmin, vmax=vmax, show=False)

            #plt.show()
            #### save
            os.chdir(os.path.join(path_results, sujet, 'TOPOPLOT'))
            fig.savefig(f'{sujet}_Pxx_{band_prep}_{band}.jpeg', dpi=600)
            plt.close()

    









################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #### params
    around_respi_Cxy = 0.05

    prms = get_params(sujet)
    respfeatures_allcond, respi_mean_allcond = load_respfeatures(sujet)

    #### get data
    Pxx_allsession = {}
    Cxy_allsession = {}
    Surrogates_allsession = {}

    for session_eeg in range(3):

        Cxy_allcond, surrogates_allcond = get_Cxy_Surrogates_allcond(sujet, session_eeg)
        Cxy_allsession[f's{session_eeg+1}'] = Cxy_allcond
        Surrogates_allsession[f's{session_eeg+1}'] = surrogates_allcond

        Pxx_allcond = get_Pxx(sujet, session_eeg)
        Pxx_allsession[f's{session_eeg+1}'] = Pxx_allcond

    #### get xr and info
    xr_Cxy = compute_Cxy_xarray(Cxy_allsession, Surrogates_allsession, respi_mean_allcond, around_respi_Cxy, prms)
    xr_Pxx = compute_Pxx_xarray(Pxx_allsession, prms)
    info = get_pos_file(sujet)

    #### plot
    plot_save_Cxy(xr_Cxy, info, prms)
    plot_save_Pxx(xr_Pxx, info, prms)




