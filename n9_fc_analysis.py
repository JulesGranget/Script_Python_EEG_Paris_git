

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib

from n2_prep_respi_analysis import analyse_resp

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False




#######################################
############# ISPC & PLI #############
#######################################

#session_eeg, band_prep, freq, band, cond, session_i, prms = 0, 'wb', [2, 10], 'theta', 'FR_CV', 0, prms
def compute_fc_metrics_mat(session_eeg, band_prep, freq, band, cond, session_i, prms):
    
    #### check if already computed
    pli_mat = np.array([0])
    ispc_mat = np.array([0])

    os.chdir(os.path.join(path_precompute, sujet, 'FC'))
    if os.path.exists(f'{sujet}_s{session_eeg+1}_ISPC_{band}_{cond}_{str(session_i+1)}.npy'):
        print(f'ALREADY COMPUTED : {sujet}_s{session_eeg+1}_ISPC_{band}_{cond}_{str(session_i+1)}')
        ispc_mat = np.load(f'{sujet}_s{session_eeg+1}_ISPC_{band}_{cond}_{str(session_i+1)}.npy')

    if os.path.exists(f'{sujet}_s{session_eeg+1}_PLI_{band}_{cond}_{str(session_i+1)}.npy'):
        print(f'ALREADY COMPUTED : {sujet}_s{session_eeg+1}_PLI_{band}_{cond}_{str(session_i+1)}')
        pli_mat = np.load(f'{sujet}_s{session_eeg+1}_PLI_{band}_{cond}_{str(session_i+1)}.npy')

    if len(ispc_mat) != 1 and len(pli_mat) != 1:
        return pli_mat, ispc_mat 
    
    #### load_data
    data = load_data_sujet(sujet, band_prep, cond, session_i)

    #### select wavelet parameters
    if band_prep == 'wb':
        wavetime = np.arange(-2,2,1/srate)
        nfrex = nfrex_lf
        ncycle_list = np.linspace(ncycle_list_wb[0], ncycle_list_wb[1], nfrex) 

    if band_prep == 'lf':
        wavetime = np.arange(-2,2,1/srate)
        nfrex = nfrex_lf
        ncycle_list = np.linspace(ncycle_list_lf[0], ncycle_list_lf[1], nfrex) 

    if band_prep == 'hf':
        wavetime = np.arange(-.5,.5,1/srate)
        nfrex = nfrex_hf
        ncycle_list = np.linspace(ncycle_list_hf[0], ncycle_list_hf[1], nfrex)

    #### compute wavelets
    frex  = np.linspace(freq[0],freq[1],nfrex)
    wavelets = np.zeros((nfrex,len(wavetime)) ,dtype=complex)

    # create Morlet wavelet family
    for fi in range(0,nfrex):
        
        s = ncycle_list[fi] / (2*np.pi*frex[fi])
        gw = np.exp(-wavetime**2/ (2*s**2)) 
        sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
        mw =  gw * sw

        wavelets[fi,:] = mw

    #### get only EEG chan
    data = data[:len(prms['chan_list_ieeg']),:]

    #### compute all convolution
    os.chdir(path_memmap)
    convolutions = np.memmap(f'{sujet}_s{session_eeg+1}_{band_prep}_{band}_{cond}_{session_i}_fc_convolutions.dat', dtype=np.complex128, mode='w+', shape=(len(prms['chan_list_ieeg']), nfrex, data.shape[1]))

    print('CONV')

    def convolution_x_wavelets_nchan(nchan):

        if nchan/np.size(data,0) % .25 <= .01:
            print("{:.2f}".format(nchan/len(prms['chan_list_ieeg'])))
        
        nchan_conv = np.zeros((nfrex, np.size(data,1)), dtype='complex')

        x = data[nchan,:]

        for fi in range(nfrex):

            nchan_conv[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

        convolutions[nchan,:,:] = nchan_conv

        return

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(convolution_x_wavelets_nchan)(nchan) for nchan in range(np.size(data,0)))

    #### compute metrics
    pli_mat = np.zeros((np.size(data,0),np.size(data,0)))
    ispc_mat = np.zeros((np.size(data,0),np.size(data,0)))

    print('COMPUTE')

    for seed in range(np.size(data,0)) :

        if seed/len(prms['chan_list_ieeg']) % .25 <= .01:
            print("{:.2f}".format(seed/len(prms['chan_list_ieeg'])))

        def compute_ispc_pli(nchan):

            if nchan == seed : 
                return
                
            else :

                # initialize output time-frequency data
                ispc = np.zeros((nfrex))
                pli  = np.zeros((nfrex))

                # compute metrics
                for fi in range(nfrex):
                    
                    as1 = convolutions[seed][fi,:]
                    as2 = convolutions[nchan][fi,:]

                    # collect "eulerized" phase angle differences
                    cdd = np.exp(1j*(np.angle(as1)-np.angle(as2)))
                    
                    # compute ISPC and PLI (and average over trials!)
                    ispc[fi] = np.abs(np.mean(cdd))
                    pli[fi] = np.abs(np.mean(np.sign(np.imag(cdd))))

            # compute mean
            mean_ispc = np.mean(ispc,0)
            mean_pli = np.mean(pli,0)

            return mean_ispc, mean_pli

        compute_ispc_pli_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_pli)(nchan) for nchan in range(np.size(data,0)))
        
        for nchan in range(np.size(data,0)) :
                
            if nchan == seed:

                continue

            else:
                    
                ispc_mat[seed,nchan] = compute_ispc_pli_res[nchan][0]
                pli_mat[seed,nchan] = compute_ispc_pli_res[nchan][1]

    #### save matrix
    os.chdir(os.path.join(path_precompute, sujet, 'FC'))
    np.save(f'{sujet}_s{session_eeg+1}_ISPC_{band}_{cond}_{str(session_i+1)}.npy', ispc_mat)

    np.save(f'{sujet}_s{session_eeg+1}_PLI_{band}_{cond}_{str(session_i+1)}.npy', pli_mat)

    #### supress mmap
    os.chdir(path_memmap)
    os.remove(f'{sujet}_s{session_eeg+1}_{band_prep}_{band}_{cond}_{session_i}_fc_convolutions.dat')
    
    return pli_mat, ispc_mat




#session_eeg=0
def compute_pli_ispc_allband(sujet, session_eeg):

    #### get params
    prms = get_params(sujet)

    #### compute
    pli_allband = {}
    ispc_allband = {}

    #band_prep_i, band_prep = 0, 'lf'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #band = 'theta'
        for band, freq in freq_band_dict[band_prep].items():

            if band == 'whole' :

                continue

            else: 

                pli_allcond = {}
                ispc_allcond = {}

                #cond_i, cond = 0, conditions[0]
                #session_i = 0
                for cond_i, cond in enumerate(conditions_allsubjects) :

                    print(band, cond)

                    if prms['count_session'][f's{session_eeg+1}'][cond] == 1:

                        pli_mat, ispc_mat = compute_fc_metrics_mat(session_eeg, band_prep, freq, band, cond, session_i, prms)
                        pli_allcond[cond] = [pli_mat]
                        ispc_allcond[cond] = [ispc_mat]

                    elif prms['count_session'][f's{session_eeg+1}'][cond] > 1:

                        load_ispc = []
                        load_pli = []

                        for session_i in range(prms['count_session'][f's{session_eeg+1}'][cond]):
                            
                            pli_mat, ispc_mat = compute_fc_metrics_mat(session_eeg, band_prep, freq, band, cond, session_i, prms)
                            load_ispc.append(ispc_mat)
                            load_pli.append(pli_mat)

                        pli_allcond[cond] = load_pli
                        ispc_allcond[cond] = load_ispc

                pli_allband[band] = pli_allcond
                ispc_allband[band] = ispc_allcond

    #### verif

    if debug == True:
                
        for band, freq in freq_band_fc_analysis.items():

            for cond_i, cond in enumerate(conditions_allsubjects) :

                print(band, cond, len(pli_allband[band][cond]))
                print(band, cond, len(ispc_allband[band][cond]))







def get_pli_ispc_allsession(sujet):

    #### get params
    prms = get_params(sujet)

    #### compute
    pli_allband = {}
    ispc_allband = {}

    for session_eeg in range(3):

        pli_allband[f's{session_eeg+1}'] = {}
        ispc_allband[f's{session_eeg+1}'] = {}

        #band_prep_i, band_prep = 0, 'lf'
        for band_prep_i, band_prep in enumerate(band_prep_list):

            #band = 'theta'
            for band, freq in freq_band_dict[band_prep].items():

                if band == 'whole' :

                    continue

                else: 

                    pli_allcond = {}
                    ispc_allcond = {}

                    #cond_i, cond = 0, conditions[0]
                    #session_i = 0
                    for cond_i, cond in enumerate(conditions_allsubjects) :

                        print(band, cond)

                        if prms['count_session'][f's{session_eeg+1}'][cond] == 1:

                            pli_mat, ispc_mat = compute_fc_metrics_mat(session_eeg, band_prep, freq, band, cond, session_i, prms)
                            pli_allcond[cond] = [pli_mat]
                            ispc_allcond[cond] = [ispc_mat]

                        elif prms['count_session'][f's{session_eeg+1}'][cond] > 1:

                            load_ispc = []
                            load_pli = []

                            for session_i in range(prms['count_session'][f's{session_eeg+1}'][cond]):
                                
                                pli_mat, ispc_mat = compute_fc_metrics_mat(session_eeg, band_prep, freq, band, cond, session_i, prms)
                                load_ispc.append(ispc_mat)
                                load_pli.append(pli_mat)

                            pli_allcond[cond] = load_pli
                            ispc_allcond[cond] = load_ispc

                    pli_allband[f's{session_eeg+1}'][band] = pli_allcond
                    ispc_allband[f's{session_eeg+1}'][band] = ispc_allcond

    #### verif

    if debug == True:
                
        for band_prep in band_prep_list:
            
            for band, freq in freq_band_dict[band_prep].items():

                for cond_i, cond in enumerate(conditions_allsubjects) :

                    print(band, cond, len(pli_allband[band][cond]))
                    print(band, cond, len(ispc_allband[band][cond]))


        #### reduce to one cond
    #### generate dict to fill
    ispc_allband_reduced = {}
    pli_allband_reduced = {}

    for session_eeg in range(3):

        ispc_allband_reduced[f's{session_eeg+1}'] = {}
        pli_allband_reduced[f's{session_eeg+1}'] = {}

        for band_prep in band_prep_list:

            for band, freq in freq_band_dict[band_prep].items():

                ispc_allband_reduced[f's{session_eeg+1}'][band] = {}
                pli_allband_reduced[f's{session_eeg+1}'][band] = {}

                for cond_i, cond in enumerate(conditions_allsubjects) :

                    ispc_allband_reduced[f's{session_eeg+1}'][band][cond] = []
                    pli_allband_reduced[f's{session_eeg+1}'][band][cond] = []

    #### fill
    for session_eeg in range(3):
    
        for band_prep_i, band_prep in enumerate(band_prep_list):

            for band, freq in freq_band_dict[band_prep].items():

                if band == 'whole' :

                    continue

                else:

                    for cond_i, cond in enumerate(conditions_allsubjects) :

                        if prms['count_session'][f's{session_eeg+1}'][cond] == 1:

                            ispc_allband_reduced[f's{session_eeg+1}'][band][cond] = ispc_allband[f's{session_eeg+1}'][band][cond][0]
                            pli_allband_reduced[f's{session_eeg+1}'][band][cond] = pli_allband[f's{session_eeg+1}'][band][cond][0]

                        elif prms['count_session'][f's{session_eeg+1}'][cond] > 1:

                            load_ispc = []
                            load_pli = []

                            for session_i in range(prms['count_session'][f's{session_eeg+1}'][cond]):

                                if session_i == 0 :

                                    load_ispc.append(ispc_allband[f's{session_eeg+1}'][band][cond][session_i])
                                    load_pli.append(pli_allband[f's{session_eeg+1}'][band][cond][session_i])

                                else :
                                
                                    load_ispc = (load_ispc[0] + ispc_allband[f's{session_eeg+1}'][band][cond][session_i]) / 2
                                    load_pli = (load_pli[0] + pli_allband[f's{session_eeg+1}'][band][cond][session_i]) / 2

                            pli_allband_reduced[f's{session_eeg+1}'][band][cond] = load_pli
                            ispc_allband_reduced[f's{session_eeg+1}'][band][cond] = load_ispc

    return pli_allband_reduced, ispc_allband_reduced



################################
######## SAVE FIG ########
################################

def save_fig_FC(pli_allband_reduced, ispc_allband_reduced, prms):

    print('######## SAVEFIG FC ########')

    #### sort matrix

    #def sort_mat(mat):

    #    mat_sorted = np.zeros((np.size(mat,0), np.size(mat,1)))
    #    for i_before_sort_r, i_sort_r in enumerate(df_sorted.index.values):
    #        for i_before_sort_c, i_sort_c in enumerate(df_sorted.index.values):
    #            mat_sorted[i_sort_r,i_sort_c] = mat[i_before_sort_r,i_before_sort_c]

    #    return mat_sorted

    #### verify sorting
    #mat = pli_allband_reduced.get(band).get(cond)
    #mat_sorted = sort_mat(mat)
    #plt.matshow(mat_sorted)
    #plt.show()

    #### prepare sort
    #df_sorted = df_loca.sort_values(['lobes', 'ROI'])
    #chan_name_sorted = df_sorted['ROI'].values.tolist()


    #chan_name_sorted_mat = []
    #rep_count = 0
    #for i, name_i in enumerate(chan_name_sorted):
    #    if i == 0:
    #        chan_name_sorted_mat.append(name_i)
    #        continue
    #    else:
    #        if name_i == chan_name_sorted[i-(rep_count+1)]:
    #            chan_name_sorted_mat.append('')
    #            rep_count += 1
    #            continue
    #        if name_i != chan_name_sorted[i-(rep_count+1)]:
    #            chan_name_sorted_mat.append(name_i)
    #            rep_count = 0
    #            continue
    #            

    #### identify scale
    scale = {}
    for session_eeg in range(3):

        scale[f's{session_eeg+1}'] = {'ispc' : {'min' : {}, 'max' : {}}, 'pli' : {'min' : {}, 'max' : {}}}

        scale[f's{session_eeg+1}']['ispc']['max'] = {}
        scale[f's{session_eeg+1}']['ispc']['min'] = {}
        scale[f's{session_eeg+1}']['pli']['max'] = {}
        scale[f's{session_eeg+1}']['pli']['min'] = {}
    
        for band_prep in band_prep_list:

            for band, freq in freq_band_dict[band_prep].items():

                band_ispc = {'min' : [], 'max' : []}
                band_pli = {'min' : [], 'max' : []}

                for cond_i, cond in enumerate(conditions_allsubjects):
                    band_ispc['max'].append(np.max(ispc_allband_reduced[f's{session_eeg+1}'][band][cond]))
                    band_ispc['min'].append(np.min(ispc_allband_reduced[f's{session_eeg+1}'][band][cond]))
                    
                    band_pli['max'].append(np.max(pli_allband_reduced[f's{session_eeg+1}'][band][cond]))
                    band_pli['min'].append(np.min(pli_allband_reduced[f's{session_eeg+1}'][band][cond]))

                scale[f's{session_eeg+1}']['ispc']['max'][band] = np.max(band_ispc['max'])
                scale[f's{session_eeg+1}']['ispc']['min'][band] = np.min(band_ispc['min'])
                scale[f's{session_eeg+1}']['pli']['max'][band] = np.max(band_pli['max'])
                scale[f's{session_eeg+1}']['pli']['min'][band] = np.min(band_pli['min'])


    #### ISPC

    os.chdir(os.path.join(path_results, sujet, 'FC', 'ISPC'))

    odor_list = odor_order[sujet]
    nrows, ncols = len(odor_list), len(conditions_allsubjects)

    #band_prep, band, freq = 'wb', 'theta', [2, 10]
    for band_prep in band_prep_list:

        for band, freq in freq_band_dict[band_prep].items():

            #### graph
            fig = plt.figure(facecolor='black')
            for session_eeg in range(3):
                count_odor = (session_eeg)*(ncols)
                for cond_i, cond in enumerate(conditions_allsubjects):
                    mne.viz.plot_connectivity_circle(ispc_allband_reduced[f's{session_eeg+1}'][band][cond], node_names=prms['chan_list_ieeg'], n_lines=None, title=cond, show=False, padding=7, fig=fig, subplot=(nrows, ncols, count_odor+cond_i+1))
            plt.suptitle('ISPC_' + band, color='w')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            #fig.show()

            fig.savefig(sujet + '_ISPC_' + band + '_graph.jpeg', dpi = 100)

        
            #### matrix
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,10))

            for session_eeg in range(3):
                r = session_eeg
                for c, cond_i in enumerate(conditions_allsubjects):
                    ax = axs[r, c]
                    ax.matshow(ispc_allband_reduced[f's{session_eeg+1}'][band][cond], vmin=scale[f's{session_eeg+1}']['ispc']['min'][band], vmax=scale[f's{session_eeg+1}']['ispc']['max'][band])
                    if r == 0:
                        ax.set_title(cond)
                    if c == 0:
                        ax.set_ylabel(f"odor_{odor_list[f's{session_eeg+1}']}")
                    ax.set_yticks(range(len(prms['chan_list_ieeg'])))
                    ax.set_yticklabels(prms['chan_list_ieeg'])
                        
            plt.suptitle('ISPC_' + band)
            #plt.show()
                        
            fig.savefig(sujet + '_ISPC_' + band + '_mat.jpeg', dpi = 100)


    #### PLI

    os.chdir(os.path.join(path_results, sujet, 'FC', 'PLI'))

    odor_list = odor_order[sujet]
    nrows, ncols = len(odor_list), len(conditions_allsubjects)

    #band_prep, band, freq = 'wb', 'theta', [2, 10]
    for band_prep in band_prep_list:

        for band, freq in freq_band_dict[band_prep].items():

            #### graph
            fig = plt.figure(facecolor='black')
            for session_eeg in range(3):
                count_odor = (session_eeg)*(ncols)
                for cond_i, cond in enumerate(conditions_allsubjects):
                    mne.viz.plot_connectivity_circle(pli_allband_reduced[f's{session_eeg+1}'][band][cond], node_names=prms['chan_list_ieeg'], n_lines=None, title=cond, show=False, padding=7, fig=fig, subplot=(nrows, ncols, count_odor+cond_i+1))
            plt.suptitle('PLI_' + band, color='w')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            #fig.show()

            fig.savefig(sujet + '_PLI_' + band + '_graph.jpeg', dpi = 100)

        
            #### matrix
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,10))

            for session_eeg in range(3):
                r = session_eeg
                for c, cond_i in enumerate(conditions_allsubjects):
                    ax = axs[r, c]
                    ax.matshow(pli_allband_reduced[f's{session_eeg+1}'][band][cond], vmin=scale[f's{session_eeg+1}']['pli']['min'][band], vmax=scale[f's{session_eeg+1}']['pli']['max'][band])
                    if r == 0:
                        ax.set_title(cond)
                    if c == 0:
                        ax.set_ylabel(f"odor_{odor_list[f's{session_eeg+1}']}")
                    ax.set_yticks(range(len(prms['chan_list_ieeg'])))
                    ax.set_yticklabels(prms['chan_list_ieeg'])
                        
            plt.suptitle('PLI_' + band)
            #plt.show()
                        
            fig.savefig(sujet + '_PLI_' + band + '_mat.jpeg', dpi = 100)






def save_fig_for_allsession(sujet):

    prms = get_params(sujet)

    pli_allband_reduced, ispc_allband_reduced = get_pli_ispc_allsession(sujet)

    save_fig_FC(pli_allband_reduced, ispc_allband_reduced, prms)




################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    #### params
    compute_metrics = True
    plot_fig = False

    #### compute fc metrics
    if compute_metrics:
        #session_eeg = 2
        for session_eeg in range(3):
            #compute_pli_ispc_allband(sujet, session_eeg)
            execute_function_in_slurm_bash('n9_fc_analysis', 'compute_pli_ispc_allband', [sujet, session_eeg])

    #### save fig
    if plot_fig:

        save_fig_for_allsession(sujet)

