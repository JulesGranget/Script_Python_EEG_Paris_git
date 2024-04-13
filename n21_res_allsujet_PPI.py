
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
from sklearn import linear_model
import gc
import xarray as xr
import seaborn as sns
import pickle
import cv2

from n0_config_params import *
from n0bis_config_analysis_functions import *
from n0ter_stats import *
from n21bis_res_allsujet_PPI import *

from mne.stats import permutation_cluster_test
from mne.stats import permutation_cluster_1samp_test

debug = False




################################
######## PERMUTATION ########
################################




def compute_topoplot_stats_allsujet_perm(xr_data, perm_type):

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    times = xr_data['time'].values

    ######## INTRA ########
    #### scale
    min = np.array([])
    max = np.array([])

    for odor_i, odor in enumerate(['o', '+', '-']):

        data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :, :].values

        for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_perm = data_cond_red - data_baseline_red 
            data_perm_topo = data_perm.mean(axis=0)

            min = np.append(min, data_perm_topo.min())
            max = np.append(max, data_perm_topo.max())

    min = min.min()
    max = max.max()
    vlim = np.array([max, min]).max()

    #### plot
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

    for odor_i, odor in enumerate(['o', '+', '-']):

        data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :, :].values

        for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

            print(perm_type, 'intra', odor, cond)

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_perm = data_cond_red - data_baseline_red 
            data_perm_topo = data_perm.mean(axis=0)

            #chan_i, chan = 0, chan_list_eeg[0]
            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                if perm_type == 'mne':

                    data_perm = data_cond_chan - data_baseline_chan 

                    T_obs, clusters, clusters_p_values, H0 = permutation_cluster_1samp_test(
                        data_perm,
                        n_permutations=1000,
                        threshold=None,
                        tail=0,
                        n_jobs=4,
                        out_type="mask",
                        verbose=False
                    )

                    if (clusters_p_values < 0.05).any():
                        
                        mask_signi[chan_i] = True

                else:

                    perm_vec = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

                    if perm_vec.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 
                        mask_signi[chan_i] = True 

            ax = axs[odor_i, cond_i]

            ax.set_title(f"{cond} {odor}")

            mne.viz.plot_topomap(data=data_perm_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim), cmap='seismic')

    plt.tight_layout()

    plt.suptitle(f'ALLSUJET INTRA (cond-baseline) {np.round(-vlim,2)}:{np.round(vlim,2)}')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
    fig.savefig(f"perm_{perm_type}_intra_allsujet.jpeg")

    plt.close('all')
    
    # plt.show()

    ######## INTER ########
    #### scale
    min = np.array([])
    max = np.array([])

    for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

        data_baseline = xr_data.loc[:, cond, 'o', :, :].values

        for odor_i, odor in enumerate(['+', '-']):

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_perm = data_cond_red - data_baseline_red 
            data_perm_topo = data_perm.mean(axis=0)

            min = np.append(min, data_perm_topo.min())
            max = np.append(max, data_perm_topo.max())

    min = min.min()
    max = max.max()
    vlim = np.array([max, min]).max()

    #### plot
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15,15))

    for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

        data_baseline = xr_data.loc[:, cond, 'o', :, :].values

        for odor_i, odor in enumerate(['+', '-']):

            print(perm_type, 'inter', odor, cond)

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_perm = data_cond_red - data_baseline_red 
            data_perm_topo = data_perm.mean(axis=0)

            #chan_i, chan = 0, chan_list_eeg[0]
            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                if perm_type == 'mne':

                    data_perm = data_cond_chan - data_baseline_chan 

                    T_obs, clusters, clusters_p_values, H0 = permutation_cluster_1samp_test(
                        data_perm,
                        n_permutations=1000,
                        threshold=None,
                        tail=0,
                        n_jobs=4,
                        out_type="mask",
                        verbose=False
                    )

                    if (clusters_p_values < 0.05).any():
                            
                        mask_signi[chan_i] = True

                else:

                    perm_vec = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

                    if perm_vec.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 
                        mask_signi[chan_i] = True 
            
            ax = axs[odor_i, cond_i]

            ax.set_title(f"{cond} {odor}")

            mne.viz.plot_topomap(data=data_perm_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim), cmap='seismic')

    plt.tight_layout()

    plt.suptitle(f'ALLSUJET INTER (cond-baseline) {np.round(-vlim,2)}:{np.round(vlim,2)}')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
    fig.savefig(f"perm_{perm_type}_inter_allsujet.jpeg")

    plt.close('all')

    # plt.show()







def compute_topoplot_stats_repnorep_perm(xr_data, perm_type):

    mask_params = dict(markersize=15, markerfacecolor='y')

    sujet_no_respond_rev = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_no_respond])

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    mat_adjacency = mne.channels.find_ch_adjacency(info, 'eeg')[0]

    times = xr_data['time'].values

    #sujet_sel = 'rep'
    for sujet_sel in ['rep', 'no_rep']:

        #### select data
        if sujet_sel == 'rep':
            xr_data_sel = xr_data.loc[sujet_best_list_rev, :, :, :, :]
        elif sujet_sel == 'no_rep':
            xr_data_sel = xr_data.loc[sujet_no_respond_rev, :, :, :, :]

        ######## INTRA ########
        #### scale
        min = np.array([])
        max = np.array([])

        # for sujet_sel_vlim in ['rep', 'no_rep']:

        #     if sujet_sel_vlim == 'rep':
        #         xr_data_vlim = xr_data.loc[sujet_best_list_rev, :, :, :, :]
        #     elif sujet_sel_vlim == 'no_rep':
        #         xr_data_vlim = xr_data.loc[sujet_no_respond_rev, :, :, :, :]

        #     for odor_i, odor in enumerate(['o', '+', '-']):

        #         data_baseline = xr_data_vlim.loc[:, 'FR_CV_1', odor, :, :].values

        #         for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

        #             data_cond = xr_data_vlim.loc[:, cond, odor, :, :].values

        #             data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
        #             data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

        #             data_perm = data_cond_red - data_baseline_red 
        #             data_perm_topo = data_perm.mean(axis=0)

        #             min = np.append(min, data_perm_topo.min())
        #             max = np.append(max, data_perm_topo.max())

        for odor_i, odor in enumerate(['o', '+', '-']):

            data_baseline = xr_data_sel.loc[:, 'FR_CV_1', odor, :, :].values

            for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

                data_cond = xr_data_sel.loc[:, cond, odor, :, :].values

                data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
                data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

                data_perm = data_cond_red - data_baseline_red 
                data_perm_topo = data_perm.mean(axis=0)

                min = np.append(min, data_perm_topo.min())
                max = np.append(max, data_perm_topo.max())

        min = min.min()
        max = max.max()
        vlim = np.array([max, min]).max()

        #### plot
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

        for odor_i, odor in enumerate(['o', '+', '-']):

            data_baseline = xr_data_sel.loc[:, 'FR_CV_1', odor, :, :].values

            for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

                print(perm_type, 'intra', odor, cond)

                data_cond = xr_data_sel.loc[:, cond, odor, :, :].values

                mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

                data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
                data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

                data_perm = data_cond_red - data_baseline_red 
                data_perm_topo = data_perm.mean(axis=0)

                #chan_i, chan = 0, chan_list_eeg[0]
                for chan_i, chan in enumerate(chan_list_eeg):

                    data_baseline_chan = data_baseline[:, chan_i, :]
                    data_cond_chan = data_cond[:, chan_i, :] 

                    if perm_type == 'mne':

                        data_perm = data_cond_chan - data_baseline_chan 

                        T_obs, clusters, clusters_p_values, H0 = permutation_cluster_1samp_test(
                            data_perm,
                            n_permutations=1000,
                            threshold=None,
                            tail=0,
                            n_jobs=4,
                            out_type="mask",
                            verbose=False
                        )

                        if (clusters_p_values < 0.05).any():
                            
                            mask_signi[chan_i] = True

                    else:

                        perm_vec = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

                        if perm_vec.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 
                            mask_signi[chan_i] = True 

                ax = axs[odor_i, cond_i]

                ax.set_title(f"{cond} {odor}")

                mne.viz.plot_topomap(data=data_perm_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim), cmap='seismic')

        plt.tight_layout()

        plt.suptitle(f"{sujet_sel} INTRA (cond-baseline) {np.round(-vlim,2)}:{np.round(vlim,2)}")

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
        fig.savefig(f"perm_{perm_type}_intra_{sujet_sel}.jpeg")
        
        # plt.show()

        plt.close('all')

        ######## INTER ########
        #### scale
        min = np.array([])
        max = np.array([])

        # for sujet_sel_vlim in ['rep', 'no_rep']:

        #     if sujet_sel_vlim == 'rep':
        #         xr_data_vlim = xr_data.loc[sujet_best_list_rev, :, :, :, :]
        #     elif sujet_sel_vlim == 'no_rep':
        #         xr_data_vlim = xr_data.loc[sujet_no_respond_rev, :, :, :, :]

        #     for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

        #         data_baseline = xr_data_vlim.loc[:, cond, 'o', :, :].values

        #         for odor_i, odor in enumerate(['+', '-']):

        #             data_cond = xr_data_vlim.loc[:, cond, odor, :, :].values

        #             data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
        #             data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

        #             data_perm = data_cond_red - data_baseline_red 
        #             data_perm_topo = data_perm.mean(axis=0)

        #             min = np.append(min, data_perm_topo.min())
        #             max = np.append(max, data_perm_topo.max())

        for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

            data_baseline = xr_data_sel.loc[:, cond, 'o', :, :].values

            for odor_i, odor in enumerate(['+', '-']):

                data_cond = xr_data_sel.loc[:, cond, odor, :, :].values

                data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
                data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

                data_perm = data_cond_red - data_baseline_red 
                data_perm_topo = data_perm.mean(axis=0)

                min = np.append(min, data_perm_topo.min())
                max = np.append(max, data_perm_topo.max())

        min = min.min()
        max = max.max()
        vlim = np.array([max, min]).max()

        #### plot
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15,15))

        for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

            data_baseline = xr_data_sel.loc[:, cond, 'o', :, :].values

            for odor_i, odor in enumerate(['+', '-']):

                print(perm_type, 'inter', odor, cond)

                data_cond = xr_data_sel.loc[:, cond, odor, :, :].values

                mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

                data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
                data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

                data_perm = data_cond_red - data_baseline_red 
                data_perm_topo = data_perm.mean(axis=0)

                #chan_i, chan = 0, chan_list_eeg[0]
                for chan_i, chan in enumerate(chan_list_eeg):

                    data_baseline_chan = data_baseline[:, chan_i, :]
                    data_cond_chan = data_cond[:, chan_i, :]

                    if perm_type == 'mne':

                        data_perm = data_cond_chan - data_baseline_chan 

                        T_obs, clusters, clusters_p_values, H0 = permutation_cluster_1samp_test(
                            data_perm,
                            n_permutations=1000,
                            threshold=None,
                            tail=0,
                            n_jobs=4,
                            out_type="mask",
                            verbose=False
                        )

                        if (clusters_p_values < 0.05).any():
                                
                            mask_signi[chan_i] = True

                    else:

                        perm_vec = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

                        if perm_vec.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 
                            mask_signi[chan_i] = True 
                
                ax = axs[odor_i, cond_i]

                ax.set_title(f"{cond} {odor}")

                mne.viz.plot_topomap(data=data_perm_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim), cmap='seismic')

        plt.tight_layout()

        plt.suptitle(f"{sujet_sel} INTER (cond-baseline) {np.round(-vlim,2)}:{np.round(vlim,2)}")

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
        fig.savefig(f"perm_{perm_type}_inter_{sujet_sel}.jpeg")

        # plt.show()

        plt.close('all')



def compute_topoplot_stats_repnorep_diff_perm(xr_data, perm_type):

    mask_params = dict(markersize=15, markerfacecolor='y')

    sujet_no_respond_rev = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_no_respond])

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    mat_adjacency = mne.channels.find_ch_adjacency(info, 'eeg')[0]

    times = xr_data['time'].values

    #### sel data
    xr_data_rep = xr_data.loc[sujet_best_list_rev, :, :, :, :]
    xr_data_norep = xr_data.loc[sujet_no_respond_rev, :, :, :, :]

    #### scale
    min = np.array([])
    max = np.array([])

    for odor_i, odor in enumerate(['o', '+', '-']):

        for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

            data_baseline = xr_data_rep.loc[:, cond, odor, :, :].values
            data_cond = xr_data_norep.loc[:, cond, odor, :, :].values

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

            min = np.append(min, data_topo.min())
            max = np.append(max, data_topo.max())

    min = min.min()
    max = max.max()
    vlim = np.array([max, min]).max()

    #### plot
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(15,15))

    for odor_i, odor in enumerate(['o', '+', '-']):

        for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

            print(perm_type, 'intra', odor, cond)

            data_baseline = xr_data_rep.loc[:, cond, odor, :, :].values
            data_cond = xr_data_norep.loc[:, cond, odor, :, :].values

            mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

            #chan_i, chan = 0, chan_list_eeg[0]
            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                if perm_type == 'mne':

                    T_obs, clusters, clusters_p_values, H0 = permutation_cluster_test(
                        [data_baseline_chan, data_cond_chan],
                        n_permutations=1000,
                        threshold=None,
                        tail=0,
                        n_jobs=4,
                        out_type="mask",
                        verbose=False
                    )

                    if (clusters_p_values < 0.05).any():
                        
                        mask_signi[chan_i] = True

                else:

                    perm_vec = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

                    if perm_vec.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 
                        mask_signi[chan_i] = True 

            ax = axs[odor_i, cond_i]

            ax.set_title(f"{cond} {odor}")

            mne.viz.plot_topomap(data=data_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim))

    # plt.tight_layout()

    plt.suptitle(f"(norep - rep) {np.round(-vlim,2)}:{np.round(vlim,2)}")

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
    fig.savefig(f"perm_{perm_type}_repnorep.jpeg")
    
    # plt.show()

    plt.close('all')






########################
######## MINMAX ########
########################



def compute_topoplot_stats_allsujet_minmax(xr_data):

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    ######## INTRA ########
    #### scale
    min = np.array([])
    max = np.array([])

    for odor_i, odor in enumerate(['o', '+', '-']):

        data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :, :].values

        for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

            min = np.append(min, data_topo.min())
            max = np.append(max, data_topo.max())

    min = min.min()
    max = max.max()
    vlim = np.array([max, min]).max()

    #### plot
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

    for odor_i, odor in enumerate(['o', '+', '-']):

        data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :, :].values

        for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

            print('intra', odor, cond)

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

            #chan_i, chan = 0, chan_list_eeg[0]
            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                data_baseline_red = data_baseline_chan.max(axis=1) - data_baseline_chan.min(axis=1) 
                data_cond_red = data_cond_chan.max(axis=1) - data_cond_chan.min(axis=1)

                pval = pg.ttest(data_baseline_red, data_cond_red, paired=True, alternative='two-sided', correction=None, confidence=0.95)['p-val'].values[0]

                if pval < 0.05:
                    
                    mask_signi[chan_i] = True

            ax = axs[odor_i, cond_i]

            ax.set_title(f"{cond} {odor}")

            mne.viz.plot_topomap(data=data_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim))

    plt.tight_layout()

    plt.suptitle(f'minmax ALLSUJET INTRA (cond-baseline) {np.round(-vlim,2)}:{np.round(vlim,2)}')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
    fig.savefig(f"minmax_intra_allsujet.jpeg")
    
    # plt.show()

    ######## INTER ########
    #### scale
    min = np.array([])
    max = np.array([])

    for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

        data_baseline = xr_data.loc[:, cond, 'o', :, :].values

        for odor_i, odor in enumerate(['+', '-']):

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

            min = np.append(min, data_topo.min())
            max = np.append(max, data_topo.max())

    min = min.min()
    max = max.max()
    vlim = np.array([max, min]).max()

    #### plot
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15,15))

    for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

        data_baseline = xr_data.loc[:, cond, 'o', :, :].values

        for odor_i, odor in enumerate(['+', '-']):

            print('inter', odor, cond)

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

            #chan_i, chan = 0, chan_list_eeg[0]
            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                data_baseline_red = data_baseline_chan.max(axis=1) - data_baseline_chan.min(axis=1) 
                data_cond_red = data_cond_chan.max(axis=1) - data_cond_chan.min(axis=1) 

                pval = pg.ttest(data_baseline_red, data_cond_red, paired=True, alternative='two-sided', correction=None, confidence=0.95)['p-val'].values[0]

                if pval < 0.05:
                        
                    mask_signi[chan_i] = True
            
            ax = axs[odor_i, cond_i]

            ax.set_title(f"{cond} {odor}")

            mne.viz.plot_topomap(data=data_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim))

    plt.tight_layout()

    plt.suptitle(f'minmax ALLSUJET INTER (cond-baseline) {np.round(-vlim,2)}:{np.round(vlim,2)}')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
    fig.savefig(f"minmax_inter_allsujet.jpeg")

    # plt.show()







def compute_topoplot_stats_repnorep_minmax(xr_data):

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    #sujet_sel = 'rep'
    for sujet_sel in ['rep', 'no_rep']:

        #### select data
        if sujet_sel == 'rep':
            xr_data_sel = xr_data.loc[sujet_best_list_rev, :, :, :, :]
        elif sujet_sel == 'no_rep':
            xr_data_sel = xr_data.loc[sujet_no_respond_rev, :, :, :, :]

        ######## INTRA ########
        #### scale
        min = np.array([])
        max = np.array([])

        # for sujet_sel_vlim in ['rep', 'no_rep']:

        #     if sujet_sel_vlim == 'rep':
        #         xr_data_vlim = xr_data.loc[sujet_best_list_rev, :, :, :, :]
        #     elif sujet_sel_vlim == 'no_rep':
        #         xr_data_vlim = xr_data.loc[sujet_no_respond_rev, :, :, :, :]

        #     for odor_i, odor in enumerate(['o', '+', '-']):

        #         data_baseline = xr_data_vlim.loc[:, 'FR_CV_1', odor, :, :].values

        #         for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

        #             data_cond = xr_data_vlim.loc[:, cond, odor, :, :].values

        #             data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
        #             data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

        #             data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

        #             min = np.append(min, data_topo.min())
        #             max = np.append(max, data_topo.max())

        for odor_i, odor in enumerate(['o', '+', '-']):

            data_baseline = xr_data_sel.loc[:, 'FR_CV_1', odor, :, :].values

            for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

                data_cond = xr_data_sel.loc[:, cond, odor, :, :].values

                data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
                data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

                data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

                min = np.append(min, data_topo.min())
                max = np.append(max, data_topo.max())

        min = min.min()
        max = max.max()
        vlim = np.array([max, min]).max()

        #### plot
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

        for odor_i, odor in enumerate(['o', '+', '-']):

            data_baseline = xr_data_sel.loc[:, 'FR_CV_1', odor, :, :].values

            for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

                print('intra', odor, cond)

                data_cond = xr_data_sel.loc[:, cond, odor, :, :].values

                mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

                data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
                data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

                data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

                #chan_i, chan = 0, chan_list_eeg[0]
                for chan_i, chan in enumerate(chan_list_eeg):

                    data_baseline_chan = data_baseline[:, chan_i, :]
                    data_cond_chan = data_cond[:, chan_i, :] 

                    data_baseline_red = data_baseline_chan.max(axis=1) - data_baseline_chan.min(axis=1) 
                    data_cond_red = data_cond_chan.max(axis=1) - data_cond_chan.min(axis=1) 

                    pval = pg.ttest(data_baseline_red, data_cond_red, paired=True, alternative='two-sided', correction=None, confidence=0.95)['p-val'].values[0]

                    if pval < 0.05:
                        
                        mask_signi[chan_i] = True

                ax = axs[odor_i, cond_i]

                ax.set_title(f"{cond} {odor}")

                mne.viz.plot_topomap(data=data_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim))

        plt.tight_layout()

        plt.suptitle(f"minmax {sujet_sel} INTRA (cond-baseline) {np.round(-vlim,2)}:{np.round(vlim,2)}")

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
        fig.savefig(f"minmax_intra_{sujet_sel}.jpeg")
        
        # plt.show()

        ######## INTER ########
        #### scale
        min = np.array([])
        max = np.array([])

        # for sujet_sel_vlim in ['rep', 'no_rep']:

        #     if sujet_sel_vlim == 'rep':
        #         xr_data_vlim = xr_data.loc[sujet_best_list_rev, :, :, :, :]
        #     elif sujet_sel_vlim == 'no_rep':
        #         xr_data_vlim = xr_data.loc[sujet_no_respond_rev, :, :, :, :]

        #     for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

        #         data_baseline = xr_data_vlim.loc[:, cond, 'o', :, :].values

        #         for odor_i, odor in enumerate(['+', '-']):

        #             data_cond = xr_data_vlim.loc[:, cond, odor, :, :].values

        #             data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
        #             data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

        #             data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0) 

        #             min = np.append(min, data_topo.min())
        #             max = np.append(max, data_topo.max())
            
        for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

            data_baseline = xr_data_sel.loc[:, cond, 'o', :, :].values

            for odor_i, odor in enumerate(['+', '-']):

                data_cond = xr_data_sel.loc[:, cond, odor, :, :].values

                data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
                data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

                data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0) 

                min = np.append(min, data_topo.min())
                max = np.append(max, data_topo.max())

        min = min.min()
        max = max.max()
        vlim = np.array([max, min]).max()

        #### plot
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15,15))

        for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

            data_baseline = xr_data_sel.loc[:, cond, 'o', :, :].values

            for odor_i, odor in enumerate(['+', '-']):

                print('inter', odor, cond)

                data_cond = xr_data_sel.loc[:, cond, odor, :, :].values

                mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

                data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
                data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

                data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0) 

                #chan_i, chan = 0, chan_list_eeg[0]
                for chan_i, chan in enumerate(chan_list_eeg):

                    data_baseline_chan = data_baseline[:, chan_i, :]
                    data_cond_chan = data_cond[:, chan_i, :] 

                    data_baseline_red = data_baseline_chan.max(axis=1) - data_baseline_chan.min(axis=1) 
                    data_cond_red = data_cond_chan.max(axis=1) - data_cond_chan.min(axis=1) 

                    pval = pg.ttest(data_baseline_red, data_cond_red, paired=True, alternative='two-sided', correction=None, confidence=0.95)['p-val'].values[0]

                    if pval < 0.05:
                            
                        mask_signi[chan_i] = True
                
                ax = axs[odor_i, cond_i]

                ax.set_title(f"{cond} {odor}")

                mne.viz.plot_topomap(data=data_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim))

        plt.tight_layout()

        plt.suptitle(f"minmax {sujet_sel} INTER (cond-baseline) {np.round(-vlim,2)}:{np.round(vlim,2)}")

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
        fig.savefig(f"minmax_inter_{sujet_sel}.jpeg")

        # plt.show()



def compute_topoplot_stats_repnorep_diff_minmax(xr_data):

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    #### sel data
    xr_data_rep = xr_data.loc[sujet_best_list_rev, :, :, :, :]
    xr_data_norep = xr_data.loc[sujet_no_respond_rev, :, :, :, :]

    #### scale
    min = np.array([])
    max = np.array([])

    for odor_i, odor in enumerate(['o', '+', '-']):

        for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

            data_baseline = xr_data_rep.loc[:, cond, odor, :, :].values
            data_cond = xr_data_norep.loc[:, cond, odor, :, :].values

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

            min = np.append(min, data_topo.min())
            max = np.append(max, data_topo.max())

    min = min.min()
    max = max.max()
    vlim = np.array([max, min]).max()

    #### plot
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(15,15))

    for odor_i, odor in enumerate(['o', '+', '-']):

        for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

            print('intra', odor, cond)

            data_baseline = xr_data_rep.loc[:, cond, odor, :, :].values
            data_cond = xr_data_norep.loc[:, cond, odor, :, :].values

            mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

            #chan_i, chan = 0, chan_list_eeg[0]
            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                data_baseline_red = data_baseline_chan.max(axis=1) - data_baseline_chan.min(axis=1) 
                data_cond_red = data_cond_chan.max(axis=1) - data_cond_chan.min(axis=1) 

                pval = pg.ttest(data_baseline_red, data_cond_red, paired=False, alternative='two-sided', correction=None, confidence=0.95)['p-val'].values[0]


                if pval < 0.05:
                    
                    mask_signi[chan_i] = True

            ax = axs[odor_i, cond_i]

            ax.set_title(f"{cond} {odor}")

            mne.viz.plot_topomap(data=data_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim))

    # plt.tight_layout()

    plt.suptitle(f"minmax norep - rep {np.round(-vlim,2)}:{np.round(vlim,2)}")

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
    fig.savefig(f"minmax_repnorep.jpeg")
    
    # plt.show()





################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #### ERP
    # cond_erp = ['VS', 'MECA', 'CO2']

    print(f'#### compute allsujet ####', flush=True)

    xr_data, xr_data_sem = compute_ERP()
    df_stats_interintra = get_df_stats(xr_data)
    xr_lm_data, xr_lm_pred, xr_lm_pred_coeff = compute_lm_on_ERP(xr_data)

    cluster_stats_type = 'manual_perm'
    # cluster_stats, cluster_stats_rep_norep = get_cluster_stats(xr_data)
    cluster_stats, cluster_stats_rep_norep = get_cluster_stats_manual_prem(xr_data)

    xr_PPI_count = get_PPI_count(xr_data)
    
    # shuffle_way = 'inter_cond'
    # shuffle_way = 'intra_cond'
    shuffle_way = 'linear_based'
    xr_surr = compute_surr_ERP(xr_data, shuffle_way)
    
    print(f'#### plot allsujet ####', flush=True)

    plot_ERP(xr_data, xr_lm_data, xr_lm_pred, xr_lm_pred_coeff, xr_surr, xr_PPI_count)

    plot_ERP_response_profile(xr_data, xr_data_sem)

    plot_ERP_diff(xr_data, df_stats_interintra, cluster_stats, cluster_stats_type) # used for analysis

    plot_ERP_rep_norep(xr_data, cluster_stats_rep_norep, cluster_stats_type)

    plot_erp_response_stats(xr_data)

    #### plot PPI across subject
    plot_PPI_proportion(xr_PPI_count)

    #### respi
    plot_mean_respi()

    #### manual evaluation
    generate_ppi_evaluation(xr_data)

    #### reg 
    print(f'#### plot discomfort / slope ####', flush=True)
    plot_slope_versus_discomfort(xr_data, xr_lm_data)

    #### stats and save topoplot
    compute_topoplot_stats_allsujet_minmax(xr_data)
    compute_topoplot_stats_repnorep_minmax(xr_data)
    compute_topoplot_stats_repnorep_diff_minmax(xr_data)

    for perm_type in ['mne', 'inhouse']:
        compute_topoplot_stats_allsujet_perm(xr_data, perm_type)
        compute_topoplot_stats_repnorep_perm(xr_data, perm_type)
        compute_topoplot_stats_repnorep_diff_perm(xr_data, perm_type)

