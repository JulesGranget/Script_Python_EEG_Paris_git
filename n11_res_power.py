
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import cv2

import pickle
import gc

from n00_config_params import *
from n00bis_config_analysis_functions import *


debug = False







########################################
######## PSD & COH PRECOMPUTE ########
########################################



#dict2reduce = cyclefreq_binned_allcond
def reduce_data(dict2reduce, prms):

    #### identify count
    dict_count = {}
        #### for cyclefreq & Pxx
    if list(dict2reduce.keys())[0] in band_prep_list:

        for cond in prms['conditions']:
            dict_count[cond] = len(dict2reduce[band_prep_list[0]][cond])
        #### for surrogates
    elif len(list(dict2reduce.keys())) == 4 and list(dict2reduce.keys())[0] not in prms['conditions']:

        for cond in prms['conditions']:
            dict_count[cond] = len(dict2reduce[list(dict2reduce.keys())[0]][cond])
        #### for Cxy & MVL
    else:

        for cond in prms['conditions']:
            dict_count[cond] = len(dict2reduce[cond])    

    #### for Pxx & Cyclefreq reduce
    if np.sum([True for i in list(dict2reduce.keys()) if i in band_prep_list]) > 0:
    
        #### generate dict
        dict_reduced = {}

        for band_prep in band_prep_list:
            dict_reduced[band_prep] = {}

            for cond in prms['conditions']:
                dict_reduced[band_prep][cond] = np.zeros(( dict2reduce[band_prep][cond][0].shape ))

        #### fill
        for band_prep in band_prep_list:

            for cond in prms['conditions']:

                for session_i in range(dict_count[cond]):

                    dict_reduced[band_prep][cond] += dict2reduce[band_prep][cond][session_i]

                dict_reduced[band_prep][cond] /= dict_count[cond]

    #### for Cxy & MVL reduce
    elif np.sum([True for i in list(dict2reduce.keys()) if i in prms['conditions']]) > 0:

        #### generate dict
        dict_reduced = {}

        for cond in prms['conditions']:

            dict_reduced[cond] = np.zeros(( dict2reduce[cond][0].shape ))

        #### fill
        for cond in prms['conditions']:

            for session_i in range(dict_count[cond]):

                dict_reduced[cond] += dict2reduce[cond][session_i]

            dict_reduced[cond] /= dict_count[cond]

    #### for surrogates
    else:
        
        #### generate dict
        dict_reduced = {}
        for key in list(dict2reduce.keys()):
            dict_reduced[key] = {}
            for cond in prms['conditions']:
                dict_reduced[key][cond] = np.zeros(( dict2reduce[key][cond][0].shape ))

        #### fill
        #key = 'Cxy'
        for key in list(dict2reduce.keys()):

            for cond in prms['conditions']:

                for session_i in range(dict_count[cond]):

                    dict_reduced[key][cond] += dict2reduce[key][cond][session_i]

                dict_reduced[key][cond] /= dict_count[cond]

    #### verify
        #### for cyclefreq & Pxx
    if list(dict2reduce.keys())[0] in band_prep_list:

        for band_prep in band_prep_list:
            for cond in prms['conditions']:
                try: 
                    _ = dict_reduced[band_prep][cond].shape
                except:
                    raise ValueError('reducing wrong')
        
        #### for surrogates
    elif len(list(dict2reduce.keys())) == 4 and list(dict2reduce.keys())[0] not in prms['conditions']:

        list_surr = list(dict2reduce.keys())

        for surr_i in list_surr:
        
            for cond in prms['conditions']:
                try: 
                    _ = dict_reduced[surr_i][cond].shape
                except:
                    raise ValueError('reducing wrong')
    
        #### for Cxy & MVL
    else:

        for cond in prms['conditions']:
            try: 
                _ = dict_reduced[cond].shape
            except:
                raise ValueError('reducing wrong')

    return dict_reduced






def load_surrogates(sujet):

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    surrogates_allcond = {}

    for data_type in ['Cxy', 'cyclefreq_wb', 'MVL']:

        surrogates_allcond[data_type] = {}

        for cond in conditions:

            surrogates_allcond[data_type][cond] = {}

            for odor_i in odor_list:

                if data_type == 'Cxy':
                    surrogates_allcond['Cxy'][cond][odor_i] = np.load(f'{sujet}_{cond}_{odor_i}_Coh.npy')
                if data_type == 'cyclefreq_wb':
                    surrogates_allcond['cyclefreq_wb'][cond][odor_i] = np.load(f'{sujet}_{cond}_{odor_i}_cyclefreq_wb.npy')
                if data_type == 'MVL':
                    surrogates_allcond['MVL'][cond][odor_i] = np.load(f'{sujet}_{cond}_{odor_i}_MVL_wb.npy')

    return surrogates_allcond







#### compute Pxx & Cxy & Cyclefreq
def compute_PxxCxyCyclefreq_for_cond_session(sujet, cond, odor_i, band_prep):
    
    print(cond, odor_i, flush=True)

    #### extract data
    respfeatures_allcond = load_respfeatures(sujet)
    prms = get_params()
    chan_i = prms['chan_list'].index('PRESS')
    respi = load_data_sujet(sujet, cond, odor_i)[chan_i,:]
    data_tmp = load_data_sujet(sujet, cond, odor_i)[:len(chan_list_eeg),:]

    #### prepare analysis
    hzPxx = np.linspace(0,prms['srate']/2,int(prms['nfft']/2+1))
    hzCxy = np.linspace(0,prms['srate']/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #### compute
    Cxy_for_cond = np.zeros(( data_tmp.shape[0], len(hzCxy)))
    Pxx_for_cond = np.zeros(( data_tmp.shape[0], len(hzPxx)))
    cyclefreq_for_cond = np.zeros(( data_tmp.shape[0], stretch_point_surrogates))
    # MI_for_cond = np.zeros(( data_tmp.shape[0] ))
    MVL_for_cond = np.zeros(( data_tmp.shape[0] ))
    # cyclefreq_binned_for_cond = np.zeros(( data_tmp.shape[0], MI_n_bin))

    # MI_bin_i = int(stretch_point_surrogates / MI_n_bin)

    #n_chan = 0
    for n_chan in range(data_tmp.shape[0]):

        #### Pxx, Cxy, CycleFreq
        x = data_tmp[n_chan,:]
        hzPxx, Pxx = scipy.signal.welch(x, fs=prms['srate'], window=prms['hannw'], nperseg=prms['nwind'], noverlap=prms['noverlap'], nfft=prms['nfft'])

        y = respi
        hzPxx, Cxy = scipy.signal.coherence(x, y, fs=prms['srate'], window=prms['hannw'], nperseg=prms['nwind'], noverlap=prms['noverlap'], nfft=prms['nfft'])

        x_stretch, trash = stretch_data(respfeatures_allcond[cond][odor_i], stretch_point_surrogates, x, prms['srate'])
        x_stretch_mean = np.mean(x_stretch, 0)
        x_stretch_mean = x_stretch_mean - x_stretch_mean.mean() 

        Cxy_for_cond[n_chan,:] = Cxy[mask_hzCxy]
        Pxx_for_cond[n_chan,:] = Pxx
        cyclefreq_for_cond[n_chan,:] = x_stretch_mean

        #### MVL
        x_zscore = zscore(x)
        x_stretch, trash = stretch_data(respfeatures_allcond[cond][odor_i], stretch_point_surrogates, x_zscore, prms['srate'])

        MVL_for_cond[n_chan] = get_MVL(np.mean(x_stretch,axis=0)-np.mean(x_stretch,axis=0).min())

        if debug:

            plt.plot(zscore(x))
            plt.plot(zscore(y))
            plt.show()

            plt.plot(hzPxx, Pxx)
            plt.show()

            plt.plot(hzPxx, Cxy)
            plt.show()

            plt.plot(x_stretch_mean)
            plt.show()

            plt.plot(np.mean(x_stretch,axis=0)-np.mean(x_stretch,axis=0).min())
            plt.show()

        # #### MI
        # x = x_stretch_mean

        # x_bin = np.zeros(( MI_n_bin ))

        # for bin_i in range(MI_n_bin):
        #     x_bin[bin_i] = np.mean(x[MI_bin_i*bin_i:MI_bin_i*(bin_i+1)])

        # cyclefreq_binned_for_cond[n_chan,:] = x_bin

        # x_bin += np.abs(x_bin.min())*2 #supress zero values
        # x_bin = x_bin/np.sum(x_bin) #transform into probabilities
            
        # MI_for_cond[n_chan] = Shannon_MI(x_bin)

    if debug:

        for nchan in range(data_tmp.shape[0]):

            # plt.plot(hzPxx, Pxx_for_cond[nchan,:])
            # plt.plot(hzPxx[mask_hzCxy], Cxy_for_cond[nchan,:])
            plt.plot(zscore(cyclefreq_for_cond[nchan,:]))

        plt.show()


    return Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond, MVL_for_cond

        





def compute_all_PxxCxyCyclefreq(sujet, band_prep):

    data_allcond = {}

    for data_type in ['Pxx', 'Cxy', 'cyclefreq', 'MVL']:

        data_allcond[data_type] = {}

        for cond in conditions:

            data_allcond[data_type][cond] = {}

            for odor_i in odor_list:

                data_allcond[data_type][cond][odor_i] = []

    for cond in conditions:

        for odor_i in odor_list:

            data_allcond[data_type][cond][odor_i] = []        

            Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond, MVL_for_cond = compute_PxxCxyCyclefreq_for_cond_session(sujet, cond, odor_i, band_prep)

            data_allcond['Pxx'][cond][odor_i] = Pxx_for_cond
            data_allcond['Cxy'][cond][odor_i] = Cxy_for_cond
            data_allcond['cyclefreq'][cond][odor_i] = cyclefreq_for_cond
            data_allcond['MVL'][cond][odor_i] = MVL_for_cond

    return data_allcond




def compute_PxxCxyCyclefreqSurrogates(sujet, band_prep):

    #### load params
    surrogates_allcond = load_surrogates(sujet)

    compute_token = False
        
    if os.path.exists(os.path.join(path_precompute, sujet, 'PSD_Coh', f'allcond_{sujet}_Pxx.pkl')) == False:

        compute_token = True

    if compute_token:
    
        #### compute metrics
        data_allcond = compute_all_PxxCxyCyclefreq(sujet, band_prep)

        #### save 
        os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

        with open(f'allcond_{sujet}_Pxx.pkl', 'wb') as f:
            pickle.dump(data_allcond['Pxx'], f)

        with open(f'allcond_{sujet}_Cxy.pkl', 'wb') as f:
            pickle.dump(data_allcond['Cxy'], f)

        with open(f'allcond_{sujet}_surrogates.pkl', 'wb') as f:
            pickle.dump(surrogates_allcond, f)

        with open(f'allcond_{sujet}_cyclefreq.pkl', 'wb') as f:
            pickle.dump(data_allcond['cyclefreq'], f)

        with open(f'allcond_{sujet}_MVL.pkl', 'wb') as f:
            pickle.dump(data_allcond['MVL'], f)

    else:

        print('ALREADY COMPUTED', flush=True)

    print('done', flush=True) 











################################################
######## PLOT & SAVE PSD AND COH ########
################################################




def get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet):

    source_path = os.getcwd()
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
                
    with open(f'allcond_{sujet}_Pxx.pkl', 'rb') as f:
        Pxx_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_Cxy.pkl', 'rb') as f:
        Cxy_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_surrogates.pkl', 'rb') as f:
        surrogates_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_cyclefreq.pkl', 'rb') as f:
        cyclefreq_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_MVL.pkl', 'rb') as f:
        MVL_allcond = pickle.load(f)

    os.chdir(source_path)

    return Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond



#n_chan, chan_name = 0, chan_list_eeg[0]
def plot_save_PSD_Cxy_CF_MVL(sujet, n_chan, chan_name, band_prep):

    #### load data
    Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond = get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet)
    prms = get_params()
    respfeatures_allcond = load_respfeatures(sujet)
    
    #### plot
    print_advancement(n_chan, len(chan_list_eeg), steps=[25, 50, 75])

    hzPxx = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    hzCxy = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #odor_i = odor_list[0]
    for odor_i in odor_list:

        fig, axs = plt.subplots(nrows=4, ncols=len(conditions))
        plt.suptitle(f'{sujet}_{chan_name}_{odor_i}')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(conditions):

            #### identify respi mean
            respi_mean = np.round(respfeatures_allcond[cond][odor_i]['cycle_freq'].median(), 3)
                    
            #### plot
            ax = axs[0, c]
            ax.set_title(cond, fontweight='bold', rotation=0)
            ax.semilogy(hzPxx, Pxx_allcond[cond][odor_i][n_chan,:], color='k')
            ax.vlines(respi_mean, ymin=0, ymax=Pxx_allcond[cond][odor_i][n_chan,:].max(), color='r')
            ax.set_xlim(0,60)
 
            ax = axs[1, c]
            Pxx_sel_min = Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:][np.where(hzPxx[remove_zero_pad:] < 2)[0]].min()
            Pxx_sel_max = Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:][np.where(hzPxx[remove_zero_pad:] < 2)[0]].max()
            ax.semilogy(hzPxx[remove_zero_pad:], Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:], color='k')
            ax.set_xlim(0, 2)
            ax.set_ylim(Pxx_sel_min, Pxx_sel_max)
            ax.vlines(respi_mean, ymin=0, ymax=Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:].max(), color='r')

            ax = axs[2, c]
            ax.plot(hzCxy,Cxy_allcond[cond][odor_i][n_chan,:], color='k')
            ax.plot(hzCxy,surrogates_allcond['Cxy'][cond][odor_i][n_chan,:], color='c')
            ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

            ax = axs[3, c]
            MVL_i = np.round(MVL_allcond[cond][odor_i][n_chan], 5)
            MVL_surr = np.percentile(surrogates_allcond['MVL'][cond][odor_i][n_chan,:], 99)
            if MVL_i > MVL_surr:
                MVL_p = f'MVL : {MVL_i}, *** {int(MVL_i * 100 / MVL_surr)}%'
            else:
                MVL_p = f'MVL : {MVL_i}, NS {int(MVL_i * 100 / MVL_surr)}%'
            # ax.set_title(MVL_p, rotation=0)
            ax.set_xlabel(MVL_p)

            ax.plot(cyclefreq_allcond[cond][odor_i][n_chan,:], color='k')
            ax.plot(surrogates_allcond['cyclefreq_wb'][cond][odor_i][0, n_chan,:], color='b')
            ax.plot(surrogates_allcond['cyclefreq_wb'][cond][odor_i][1, n_chan,:], color='c', linestyle='dotted')
            ax.plot(surrogates_allcond['cyclefreq_wb'][cond][odor_i][2, n_chan,:], color='c', linestyle='dotted')
            ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=surrogates_allcond['cyclefreq_wb'][cond][odor_i][2, n_chan,:].min(), ymax=surrogates_allcond['cyclefreq_wb'][cond][odor_i][1, n_chan,:].max(), colors='r')
            #plt.show() 

        #### save
        os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'summary'))
        fig.savefig(f'{sujet}_{chan_name}_odor_{odor_i}_{band_prep}.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    #cond = 'FR_CV_1'
    for cond in conditions:

        fig, axs = plt.subplots(nrows=4, ncols=len(odor_list))
        plt.suptitle(f'{sujet}_{chan_name}_{cond}')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, odor_i = 0, odor_list[0]
        for c, odor_i in enumerate(odor_list):

            #### identify respi mean
            respi_mean = np.round(respfeatures_allcond[cond][odor_i]['cycle_freq'].median(), 3)
                    
            #### plot
            ax = axs[0, c]
            ax.set_title(odor_i, fontweight='bold', rotation=0)
            ax.semilogy(hzPxx, Pxx_allcond[cond][odor_i][n_chan,:], color='k')
            ax.vlines(respi_mean, ymin=0, ymax=Pxx_allcond[cond][odor_i][n_chan,:].max(), color='r')
            ax.set_xlim(0,60)
 
            ax = axs[1, c]
            Pxx_sel_min = Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:][np.where(hzPxx[remove_zero_pad:] < 2)[0]].min()
            Pxx_sel_max = Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:][np.where(hzPxx[remove_zero_pad:] < 2)[0]].max()
            ax.semilogy(hzPxx[remove_zero_pad:], Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:], color='k')
            ax.set_xlim(0, 2)
            ax.set_ylim(Pxx_sel_min, Pxx_sel_max)
            ax.vlines(respi_mean, ymin=0, ymax=Pxx_allcond[cond][odor_i][n_chan,remove_zero_pad:].max(), color='r')

            ax = axs[2, c]
            ax.plot(hzCxy,Cxy_allcond[cond][odor_i][n_chan,:], color='k')
            ax.plot(hzCxy,surrogates_allcond['Cxy'][cond][odor_i][n_chan,:], color='c')
            ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

            ax = axs[3, c]
            MVL_i = np.round(MVL_allcond[cond][odor_i][n_chan], 5)
            MVL_surr = np.percentile(surrogates_allcond['MVL'][cond][odor_i][n_chan,:], 99)
            if MVL_i > MVL_surr:
                MVL_p = f'MVL : {MVL_i}, *** {int(MVL_i * 100 / MVL_surr)}%'
            else:
                MVL_p = f'MVL : {MVL_i}, NS {int(MVL_i * 100 / MVL_surr)}%'
            # ax.set_title(MVL_p, rotation=0)
            ax.set_xlabel(MVL_p)

            ax.plot(cyclefreq_allcond[cond][odor_i][n_chan,:], color='k')
            ax.plot(surrogates_allcond['cyclefreq_wb'][cond][odor_i][0, n_chan,:], color='b')
            ax.plot(surrogates_allcond['cyclefreq_wb'][cond][odor_i][1, n_chan,:], color='c', linestyle='dotted')
            ax.plot(surrogates_allcond['cyclefreq_wb'][cond][odor_i][2, n_chan,:], color='c', linestyle='dotted')
            ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=surrogates_allcond['cyclefreq_wb'][cond][odor_i][2, n_chan,:].min(), ymax=surrogates_allcond['cyclefreq_wb'][cond][odor_i][1, n_chan,:].max(), colors='r')
            #plt.show() 

        #### save
        os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'summary'))
        fig.savefig(f'{sujet}_{chan_name}_cond_{cond}_{band_prep}.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

        








    

################################
######## TOPOPLOT ########
################################


#n_chan, chan_name = 0, chan_list_eeg[0]
def plot_save_Cxy_TOPOPLOT(sujet, band_prep):

    #### create montage
    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    mask_params = dict(markersize=15, markerfacecolor='y')

    #### load data
    Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond = get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet)
    prms = get_params()
    respfeatures_allcond = load_respfeatures(sujet)

    #### params
    hzPxx = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    hzCxy = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    cond_sel = ['FR_CV_1', 'CO2']

    #### reduce data
    topoplot_data = {}

    for cond in cond_sel:

        topoplot_data[cond] = {}

        for odor_i in odor_list:

            topoplot_data[cond][odor_i] = {}

            mean_resp = respfeatures_allcond[cond][odor_i]['cycle_freq'].mean()
            hzCxy_mask = (hzCxy > (mean_resp - around_respi_Cxy)) & (hzCxy < (mean_resp + around_respi_Cxy))

            Cxy_allchan_i = Cxy_allcond[cond][odor_i][:,hzCxy_mask].mean(axis=1)
            topoplot_data[cond][odor_i]['Cxy'] = Cxy_allchan_i

            Cxy_allchan_surr_i = surrogates_allcond['Cxy'][cond][odor_i][:len(chan_list_eeg),hzCxy_mask].mean(axis=1)
            topoplot_data[cond][odor_i]['Cxy_surr_val'] = Cxy_allchan_surr_i
            topoplot_data[cond][odor_i]['Cxy_surr'] = np.array(Cxy_allchan_i > Cxy_allchan_surr_i)*1

    #### check val
    if debug:

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_sel))
        plt.suptitle(f'{sujet}_Cxy')
        fig.set_figheight(10)
        fig.set_figwidth(10) 

        for c, cond in enumerate(cond_sel):

            #r, odor_i = 0, odor_list[0]
            for r, odor_i in enumerate(odor_list):

                ax = axs[r, c]

                ax.plot(topoplot_data[cond][odor_i]['Cxy'], label='Cxy')
                ax.plot(topoplot_data[cond][odor_i]['Cxy_surr_val'], label='Cxy_surr')
                ax.set_ylim(0,1)
        
        plt.legend()
        plt.show()


    #### plot Cxy
    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_sel))
    plt.suptitle(f'{sujet}_Cxy')
    fig.set_figheight(10)
    fig.set_figwidth(10)

    #c, cond = 0, 'FR_CV_1'
    for c, cond in enumerate(cond_sel):

        #r, odor_i = 0, odor_list[0]
        for r, odor_i in enumerate(odor_list):

            #### plot
            ax = axs[r, c]

            if r == 0:
                ax.set_title(cond, fontweight='bold', rotation=0)
            if c == 0:
                ax.set_ylabel(f'{odor_i}')

            mask_signi = topoplot_data[cond][odor_i][f'Cxy_surr'].astype('bool')

            im, _ = mne.viz.plot_topomap(data=topoplot_data[cond][odor_i][f'Cxy'], axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    mask=mask_signi, mask_params=mask_params, vlim=(0, 1), cmap='OrRd')
            
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Amplitude')

    # plt.show() 

    #### save
    os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'topoplot'))
    fig.savefig(f'{sujet}_Cxy_topo.jpeg', dpi=150)
    os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot_allsujet'))
    fig.savefig(f'Cxy_{sujet}_topo.jpeg', dpi=150)
    fig.clf()
    plt.close('all')
    gc.collect()



#n_chan, chan_name = 0, chan_list_eeg[0]
def plot_save_PSD_Cxy_CF_MVL_TOPOPLOT(sujet, band_prep):

    #### create montage
    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    #### load data
    Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond = get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet)
    prms = get_params()
    respfeatures_allcond = load_respfeatures(sujet)

    #### params
    hzPxx = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    hzCxy = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #### reduce data
    topoplot_data = {}

    for cond in conditions:

        topoplot_data[cond] = {}

        for odor_i in odor_list:

            topoplot_data[cond][odor_i] = {}

            mean_resp = respfeatures_allcond[cond][odor_i]['cycle_freq'].mean()
            hzCxy_mask = (hzCxy > (mean_resp - around_respi_Cxy)) & (hzCxy < (mean_resp + around_respi_Cxy))

            Cxy_allchan_i = Cxy_allcond[cond][odor_i][:,hzCxy_mask].mean(axis=1)
            topoplot_data[cond][odor_i]['Cxy'] = Cxy_allchan_i
            MVL_allchan_i = MVL_allcond[cond][odor_i]
            topoplot_data[cond][odor_i]['MVL'] = MVL_allchan_i

            Cxy_allchan_surr_i = surrogates_allcond['Cxy'][cond][odor_i][:len(chan_list_eeg),hzCxy_mask].mean(axis=1)
            topoplot_data[cond][odor_i]['Cxy_surr'] = np.array(Cxy_allchan_i > Cxy_allchan_surr_i)*1
            MVL_allchan_surr_i = np.array([np.percentile(surrogates_allcond['MVL'][cond][odor_i][nchan,:], 99) for nchan, _ in enumerate(chan_list_eeg)])
            topoplot_data[cond][odor_i]['MVL_surr'] = np.array(MVL_allchan_i > MVL_allchan_surr_i)*1

    for cond in conditions:

        for odor_i in odor_list:

            #band, freq = 'theta', [4, 8]
            for band, freq in freq_band_fc_analysis.items():

                hzPxx_mask = (hzPxx >= freq[0]) & (hzPxx <= freq[-1])
                Pxx_mean_i = Pxx_allcond[cond][odor_i][:,hzPxx_mask].mean(axis=1)
                topoplot_data[cond][odor_i][f'Pxx_{band}'] = Pxx_mean_i

    #### scales
    scales_cond = {}
    scales_odor = {}
    scales_allband = {}

    data_type_list = [f'Pxx_{band}' for band in freq_band_fc_analysis.keys()] + ['MVL']

    #data_type = data_type_list[0]
    for data_type in data_type_list:

        if data_type == 'MVL':

            scales_cond[data_type] = {}
            scales_odor[data_type] = {}

            for odor_i in odor_list:

                scales_cond[data_type][odor_i] = {}

                val = np.array([])

                for cond in conditions:

                    val = np.append(val, topoplot_data[cond][odor_i][data_type])

                scales_cond[data_type][odor_i]['min'] = val.min()
                scales_cond[data_type][odor_i]['max'] = val.max()

            for cond in conditions:

                scales_odor[data_type][cond] = {}

                val = np.array([])

                for odor_i in odor_list:

                    val = np.append(val, topoplot_data[cond][odor_i][data_type])

                scales_odor[data_type][cond]['min'] = val.min()
                scales_odor[data_type][cond]['max'] = val.max()

        scales_allband[data_type] = {}

        val = np.array([])

        for odor_i in odor_list:

            for cond in conditions:

                val = np.append(val, topoplot_data[cond][odor_i][data_type])

        scales_allband[data_type]['min'] = val.min()
        scales_allband[data_type]['max'] = val.max()

    #### plot topo summary
    #band, freq = 'theta', [4, 8]
    for band, freq in freq_band_fc_analysis.items():

        #### plot Pxx
        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
        plt.suptitle(f'{sujet}_{band}_Pxx')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(conditions):

            #r, odor_i = 0, odor_list[0]
            for r, odor_i in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(cond, fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor_i}')
                
                mne.viz.plot_topomap(topoplot_data[cond][odor_i][f'Pxx_{band}'], info, axes=ax, vlim=(scales_allband[f'Pxx_{band}']['min'], 
                                     scales_allband[f'Pxx_{band}']['max']), show=False)

        # plt.show() 

        #### save
        os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'topoplot'))
        fig.savefig(f'{sujet}_Pxx_{band}_{band_prep}_topo.jpeg', dpi=150)
        os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot_allsujet'))
        fig.savefig(f'Pxx_{band}_{sujet}_{band_prep}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    #### plot Cxy, MVL
    for data_type in ['raw', 'stats']:

        for metric_type in ['Cxy', 'MVL']:

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
            plt.suptitle(f'{sujet}_{metric_type}_{data_type}')
            fig.set_figheight(10)
            fig.set_figwidth(10)

            #c, cond = 0, 'FR_CV_1'
            for c, cond in enumerate(conditions):

                #r, odor_i = 0, odor_list[0]
                for r, odor_i in enumerate(odor_list):

                    #### plot
                    ax = axs[r, c]

                    if r == 0:
                        ax.set_title(cond, fontweight='bold', rotation=0)
                    if c == 0:
                        ax.set_ylabel(f'{odor_i}')
                    
                    if metric_type == 'Cxy' and data_type == 'stats':
                        mne.viz.plot_topomap(topoplot_data[cond][odor_i][f'Cxy_surr'], info, axes=ax, vlim=(0,1), show=False)
                        
                    if metric_type == 'Cxy' and data_type == 'raw':
                        mne.viz.plot_topomap(topoplot_data[cond][odor_i][f'Cxy'], info, axes=ax, vlim=(0,1), show=False)

                    if metric_type == 'MVL' and data_type == 'stats':
                        mne.viz.plot_topomap(topoplot_data[cond][odor_i][f'MVL_surr'], info, axes=ax, vlim=(0,1), show=False)
                        
                    if metric_type == 'MVL' and data_type == 'raw':
                        vmin = np.round(scales_allband['MVL']['min'], 2)
                        vmax = np.round(scales_allband['MVL']['max'], 2)
                        mne.viz.plot_topomap(topoplot_data[cond][odor_i]['MVL'], info, axes=ax, vlim=(vmin,vmax), show=False)

            # plt.show() 

            #### save
            os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'topoplot'))
            fig.savefig(f'{sujet}_{metric_type}_{data_type}_{band_prep}_topo.jpeg', dpi=150)
            os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot_allsujet'))
            fig.savefig(f'{data_type}_{metric_type}_{sujet}_{band_prep}_topo.jpeg', dpi=150)
            fig.clf()
            plt.close('all')
            gc.collect()








################################
######## LOAD TF & ITPC ########
################################


def compute_TF_ITPC(sujet):

    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:
    
        if tf_mode == 'TF':
            print('######## LOAD TF ########', flush=True)
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            if os.path.exists(os.path.join(path_precompute, sujet, 'TF', f'allcond_{sujet}_tf_stretch.pkl')):
                print('ALREADY COMPUTED')
                continue
            
        elif tf_mode == 'ITPC':
            print('######## LOAD ITPC ########', flush=True)
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

            if os.path.exists(os.path.join(path_precompute, sujet, 'ITPC', f'allcond_{sujet}_itpc_stretch.pkl')):
                print('ALREADY COMPUTED', flush=True)
                continue

        #### load file with reducing to one TF
        tf_stretch_allcond = {}

        #band_prep = 'wb'
        for band_prep in band_prep_list:

            tf_stretch_allcond[band_prep] = {}

            #### chose nfrex
            _, nfrex = get_wavelets(band_prep, list(freq_band_dict[band_prep].values())[0])  

            #cond = 'FR_CV'
            for cond in conditions:

                tf_stretch_allcond[band_prep][cond] = {}

                for odor_i in odor_list:

                    tf_stretch_allcond[band_prep][cond][odor_i] = {}

                    #### impose good order in dict
                    for band, freq in freq_band_dict[band_prep].items():
                        tf_stretch_allcond[band_prep][cond][odor_i][band] = np.zeros(( len(chan_list_eeg), nfrex, stretch_point_TF ))

                    #### load file
                    for band, freq in freq_band_dict[band_prep].items():
                        
                        for file_i in os.listdir(): 

                            if file_i.find(f'{freq[0]}_{freq[1]}_{cond}_{odor_i}') != -1 and file_i.find('STATS') == -1:
                                file_to_load = file_i
                            else:
                                continue
                        
                        tf_stretch_allcond[band_prep][cond][odor_i][band] += np.load(file_to_load)
               
        #### save
        if tf_mode == 'TF':
            with open(f'allcond_{sujet}_tf_stretch.pkl', 'wb') as f:
                pickle.dump(tf_stretch_allcond, f)
        elif tf_mode == 'ITPC':
            with open(f'allcond_{sujet}_itpc_stretch.pkl', 'wb') as f:
                pickle.dump(tf_stretch_allcond, f)

    print('done', flush=True)








########################################
######## PLOT & SAVE TF & ITPC ########
########################################

#tf, nchan = tf_plot, n_chan
def get_tf_stats(tf, pixel_based_distrib):

    #### thresh data
    tf_thresh = tf.copy()
    #wavelet_i = 0
    for wavelet_i in range(tf.shape[0]):
        mask = np.logical_or(tf_thresh[wavelet_i, :] < pixel_based_distrib[wavelet_i, 0], tf_thresh[wavelet_i, :] > pixel_based_distrib[wavelet_i, 1])
        tf_thresh[wavelet_i, mask] = 1
        tf_thresh[wavelet_i, np.logical_not(mask)] = 0

    if debug:

        plt.pcolormesh(tf_thresh)
        plt.show()

    #### if empty return
    if tf_thresh.sum() == 0:

        return tf_thresh

    #### thresh cluster
    tf_thresh = tf_thresh.astype('uint8')
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(tf_thresh)
    #### nb_blobs, im_with_separated_blobs, stats = nb clusters, clusters image with labeled clusters, info on clusters
    sizes = stats[1:, -1]
    nb_blobs -= 1
    min_size = np.percentile(sizes,tf_stats_percentile_cluster)  

    if debug:

        plt.hist(sizes, bins=100)
        plt.vlines(np.percentile(sizes,95), ymin=0, ymax=20, colors='r')
        plt.show()

    tf_thresh = np.zeros_like(im_with_separated_blobs)
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            tf_thresh[im_with_separated_blobs == blob + 1] = 1

    if debug:
    
        time = np.arange(tf.shape[-1])

        plt.pcolormesh(time, frex, tf, shading='gouraud', cmap='seismic')
        plt.contour(time, frex, tf_thresh, levels=0, colors='g')
        plt.yscale('log')
        plt.show()

    return tf_thresh





    


def compilation_compute_TF_ITPC(sujet, band_prep, stats_plot):

    # compute_TF_ITPC(sujet)

    # if os.path.exists(os.path.join(path_results, sujet, 'TF', 'summary', f'{sujet}_Fp1_inter_{band_prep}.jpeg')):
    #     print('TF PLOT ALREADY COMPUTED', flush=True)
    #     return
    
    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:
        
        if tf_mode == 'TF':
            print('######## PLOT & SAVE TF ########', flush=True)
        if tf_mode == 'ITPC':
            continue
            print('######## PLOT & SAVE ITPC ########', flush=True)

        #### load data
        os.chdir(os.path.join(path_precompute, sujet, tf_mode))

        data_allcond = {}

        for cond in conditions:

            data_allcond[cond] = {}

            for odor_i in odor_list:

                data_allcond[cond][odor_i] = np.median(np.load(f'{sujet}_{tf_mode.lower()}_conv_{cond}_{odor_i}.npy'), axis=1)

        #### scale
        vals = np.array([])

        for cond in conditions:

            for odor_i in odor_list:

                vals = np.append(vals, data_allcond[cond][odor_i].reshape(-1))

        if debug:

            vals_diff = np.abs(vals - np.median(vals))

            count, _, _ = plt.hist(vals_diff, bins=500)
            thresh = np.percentile(vals_diff, tf_plot_percentile_scale)
            val_max = vals_diff.max()
            plt.vlines([thresh, val_max], ymin=0, ymax=count.max(), color='r')
            plt.show()

        # median_diff = np.max([np.abs(np.median(vals) - vals.min()), np.abs(np.median(vals) + vals.max())])
        median_diff = np.percentile(np.abs(vals - np.median(vals)), tf_plot_percentile_scale)

        vmin = np.median(vals) - median_diff
        vmax = np.median(vals) + median_diff

        del vals

        #### inspect stats
        if debug:

            tf_stats_type = 'inter'
            n_chan, chan_name = 11, chan_list_eeg[11]
            r, odor_i = 2, odor_list[2]
            c, cond = 0, conditions[0]

            tf_plot = data_allcond[cond][odor_i][n_chan,:,:]

            plt.pcolormesh(time, frex, tf_plot, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
            plt.yscale('log')

            if stats_plot:
                os.chdir(os.path.join(path_precompute, sujet, 'TF'))
                pixel_based_distrib = np.load(f'{sujet}_{tf_mode.lower()}_STATS_{cond}_{odor_i}_{tf_stats_type}.npy')[n_chan]
                plt.contour(time, frex, get_tf_stats(tf_plot, pixel_based_distrib), levels=0, colors='g')

            plt.show()

            plt.pcolormesh(get_tf_stats(tf_plot, pixel_based_distrib))
            plt.show()

            #wavelet_i = 0
            for wavelet_i in range(tf_plot.shape[0]):

                plt.plot(tf_plot[wavelet_i,:], color='b')
                plt.hlines([pixel_based_distrib[wavelet_i,0], pixel_based_distrib[wavelet_i,1]], xmin=0, xmax=tf_plot.shape[-1] ,color='r')

                plt.title(f'{np.round(wavelet_i/tf_plot.shape[0],2)}')

                plt.show()


        #### plot
        if tf_mode == 'TF':
            os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))
        elif tf_mode == 'ITPC':
            os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary'))

        #tf_stats_type = 'inter'
        for tf_stats_type in ['inter', 'intra']:

            #n_chan, chan_name = 0, chan_list_eeg[0]
            for n_chan, chan_name in enumerate(chan_list_eeg):

                print_advancement(n_chan, len(chan_list_eeg), steps=[25, 50, 75])

                #### plot
                fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

                plt.suptitle(f'{sujet}_{chan_name}')

                fig.set_figheight(10)
                fig.set_figwidth(15)

                time = range(stretch_point_TF)

                #r, odor_i = 0, odor_list[0]
                for r, odor_i in enumerate(odor_list):

                    #c, cond = 1, conditions[1]
                    for c, cond in enumerate(conditions):

                        tf_plot = data_allcond[cond][odor_i][n_chan,:,:]
                    
                        ax = axs[r,c]

                        if r == 0 :
                            ax.set_title(cond, fontweight='bold', rotation=0)

                        if c == 0:
                            ax.set_ylabel(odor_i)

                        ax.pcolormesh(time, frex, tf_plot, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
                        ax.set_yscale('log')

                        if stats_plot:

                            if tf_mode == 'TF' and cond != 'FR_CV_1' and tf_stats_type == 'intra':
                                os.chdir(os.path.join(path_precompute, sujet, 'TF'))

                                pixel_based_distrib = np.load(f'{sujet}_{tf_mode.lower()}_STATS_{cond}_{odor_i}_intra.npy')[n_chan]
                                
                                if get_tf_stats(tf_plot, pixel_based_distrib).sum() != 0:
                                    ax.contour(time, frex, get_tf_stats(tf_plot, pixel_based_distrib), levels=0, colors='g')

                            if tf_mode == 'TF' and odor_i != 'o' and tf_stats_type == 'inter':
                                os.chdir(os.path.join(path_precompute, sujet, 'TF'))

                                pixel_based_distrib = np.load(f'{sujet}_{tf_mode.lower()}_STATS_{cond}_{odor_i}_inter.npy')[n_chan]
                                
                                if get_tf_stats(tf_plot, pixel_based_distrib).sum() != 0:
                                    ax.contour(time, frex, get_tf_stats(tf_plot, pixel_based_distrib), levels=0, colors='g')

                        ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=frex[0], ymax=frex[-1], colors='g')
                        ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])

                #plt.show()

                #### save
                if tf_mode == 'TF':
                    os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))
                elif tf_mode == 'ITPC':
                    os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary'))

                if tf_stats_type == 'inter':
                    fig.savefig(f'{sujet}_{chan_name}_inter_{band_prep}.jpeg', dpi=150)
                if tf_stats_type == 'intra':
                    fig.savefig(f'{sujet}_{chan_name}_intra_{band_prep}.jpeg', dpi=150)

                #### save for allsujet
                if tf_mode == 'TF':
                    os.chdir(os.path.join(path_results, 'allplot', 'TF', 'allsujet', chan_name))
                elif tf_mode == 'ITPC':
                    os.chdir(os.path.join(path_results, 'allplot', 'ITPC', 'allsujet', chan_name))

                if tf_stats_type == 'inter':
                    fig.savefig(f'{sujet}_{chan_name}_inter_{band_prep}.jpeg', dpi=150)
                if tf_stats_type == 'intra':
                    fig.savefig(f'{sujet}_{chan_name}_intra_{band_prep}.jpeg', dpi=150)
                    
                fig.clf()
                plt.close('all')
                gc.collect()






########################################
######## COMPILATION FUNCTION ########
########################################

def compilation_compute_Pxx_Cxy_Cyclefreq_MVL(sujet, band_prep):

    #### compute & reduce surrogates
    print('######## COMPUTE PSD AND COH ########', flush=True)
    compute_PxxCxyCyclefreqSurrogates(sujet, band_prep)
    
    #### compute joblib
    print('######## PLOT & SAVE PSD AND COH ########', flush=True)
    # if os.path.exists(os.path.join(path_results, sujet, 'PSD_Coh', 'summary', f'{sujet}_Fp1_odor_o_{band_prep}.jpeg')):
    #     print('NCHAN PLOT ALREADY COMPUTED', flush=True)
    # else:    
    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(plot_save_PSD_Cxy_CF_MVL)(sujet, n_chan, chan_name, band_prep) for n_chan, chan_name in enumerate(chan_list_eeg))

    print('######## PLOT & SAVE TOPOPLOT ########', flush=True)
    # if os.path.exists(os.path.join(path_results, sujet, 'PSD_Coh', 'topoplot', f'{sujet}_Cxy_raw_{band_prep}_topo.jpeg')) and os.path.exists(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot_allsujet', f'raw_Cxy_{sujet}_{band_prep}_topo.jpeg')):
    #     print('TOPOPLOT ALREADY COMPUTED', flush=True)
    # else:    
    
    # plot_save_Cxy_TOPOPLOT(sujet, band_prep)
    # plot_save_PSD_Cxy_CF_MVL_TOPOPLOT(sujet, band_prep)

    print('done', flush=True)






################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    band_prep = 'wb'
    # stats_plot = False

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        print(sujet, flush=True)

        #### Cxy
        plot_save_Cxy_TOPOPLOT(sujet, band_prep)

        #### Pxx CycleFreq
        # compilation_compute_Pxx_Cxy_Cyclefreq_MVL(sujet, band_prep)
        # execute_function_in_slurm_bash_mem_choice('n11_res_power', 'compilation_compute_Pxx_Cxy_Cyclefreq_MVL', [sujet, band_prep], '15G')

        #### TF & ITPC
        # compilation_compute_TF_ITPC(sujet, band_prep, stats_plot)
        # execute_function_in_slurm_bash_mem_choice('n11_res_power', 'compilation_compute_TF_ITPC', [sujet, band_prep, stats_plot], '15G')


