
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
from n21bis_res_allsujet_ERP import *

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






################################
######## TIMING ERP ########
################################


def timing_ERP_IE_SUM_export_df(xr_data):

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    time_vec = xr_data['time'].data

    RRP_time_ampl = {'group' : [], 'odor' : [], 'phase' : [], 'time' : [], 'amplitude' : []}

    #subgroup_type = 'allsujet'
    for subgroup_type in ['allsujet', 'rep', 'no_rep']:

        ######## INTRA ########
        #### load stats
        timing_data = np.zeros((len(odor_list), len(['expi', 'inspi']), len(chan_list_eeg), len(['value', 'time'])))

        for odor_i, odor in enumerate(['o', '+', '-']):

            if subgroup_type == 'allsujet':
                data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :, :].values

            elif subgroup_type == 'rep':
                data_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, :, :].values

            elif subgroup_type == 'no_rep':
                data_baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, :, :].values

            cond = 'CO2'

            print('intra', subgroup_type, odor, cond)

            if subgroup_type == 'allsujet':
                data_cond = xr_data.loc[:, cond, odor, :, :].values
                
            elif subgroup_type == 'rep':
                data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, :, :].values

            elif subgroup_type == 'no_rep':
                data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, :, :].values

            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                mask_signi = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

                for respi_phase_i, respi_phase in enumerate(['expi', 'inspi']):

                    if respi_phase == 'inspi':
                        mask_signi_phase = np.concatenate(( np.zeros(int(time_vec.size/2)).astype('bool'), mask_signi[int(time_vec.size/2):] ))

                    elif respi_phase == 'expi':
                        mask_signi_phase = np.concatenate(( mask_signi[:int(time_vec.size/2)], np.zeros(int(time_vec.size/2)).astype('bool') ))

                    if mask_signi_phase.sum() == 0:
                        continue     

                    else:
                        mask_signi_phase[0] = False
                        mask_signi_phase[-1] = False   

                    if np.diff(mask_signi_phase).sum() > 2: 

                        start_stop_chunk = np.where(np.diff(mask_signi_phase))[0]

                        max_chunk_signi = []
                        max_chunk_time = []
                        for start_i in np.arange(0, start_stop_chunk.size, 2):

                            _argmax = np.argmax(np.abs(data_cond_chan.mean(axis=0))[start_stop_chunk[start_i]:start_stop_chunk[start_i+1]])
                            max_chunk_time.append(start_stop_chunk[start_i] +_argmax)
                            max_chunk_signi.append(data_cond_chan.mean(axis=0)[_argmax])

                        max_rep = max_chunk_signi[np.argmax(np.abs(max_chunk_signi))]
                        time_max_rep = time_vec[max_chunk_time[np.where(max_chunk_signi == max_rep)[0][0]]]

                    else:

                        start_stop_chunk = np.where(np.diff(mask_signi_phase))[0]

                        _argmax = np.argmax(np.abs(data_cond_chan.mean(axis=0))[start_stop_chunk[0]:start_stop_chunk[1]])
                        max_rep = data_cond_chan.mean(axis=0)[start_stop_chunk[0] + _argmax]
                        time_max_rep = time_vec[start_stop_chunk[0] +_argmax]

                    timing_data[odor_i, respi_phase_i, chan_i, 0] = max_rep
                    timing_data[odor_i, respi_phase_i, chan_i, 1] = time_max_rep

        xr_coords = {'odor' : odor_list, 'phase' : ['expi', 'inspi'], 'chan' : chan_list_eeg, 'type' : ['value', 'time']}
        xr_timing = xr.DataArray(data=timing_data, coords=xr_coords)       

        #### export df
        for odor_i, odor in enumerate(odor_list):

            for phase in ['expi', 'inspi']:

                angles_vec = np.linspace(0, 2*np.pi, num=time_vec.size)
                mask_sel = (xr_timing.loc[odor, phase, :, 'time'] != 0).data
                time_responses_filter = xr_timing.loc[odor, phase, :, 'time'].data[mask_sel]
                ampl_responses_filter = xr_timing.loc[odor, phase, :, 'value'].data[mask_sel]

                time_responses_i = [np.where(time_vec == time_val)[0][0] for time_val in time_responses_filter]
                angle_responses = angles_vec[time_responses_i]

                _phase_mean = np.angle(np.mean(np.exp(1j*angle_responses)))
                
                mean_time = np.round(time_vec[[angle_i for angle_i, angle_val in enumerate(angles_vec) if np.mod(_phase_mean, 2 * np.pi) < angle_val][0]], 3)

                RRP_time_ampl['group'].append(subgroup_type)
                RRP_time_ampl['odor'].append(odor)
                RRP_time_ampl['phase'].append(phase)
                RRP_time_ampl['time'].append(mean_time)
                RRP_time_ampl['amplitude'].append(np.mean(ampl_responses_filter))

    df_export = pd.DataFrame(RRP_time_ampl)
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'time'))
    df_export.to_excel('df_intra_allgroup.xlsx')


def timing_ERP_IE_SUM_plot(xr_data):   

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    time_vec = xr_data['time'].data

    ######## PLOT RESPONSE FOR ALL GROUPS ########

    #subgroup_type = 'allsujet'
    for subgroup_type in ['allsujet', 'rep', 'no_rep']:

        ######## INTRA ########
        #### load stats
        timing_data = np.zeros((len(odor_list), len(['expi', 'inspi']), len(chan_list_eeg), len(['value', 'time'])))

        for odor_i, odor in enumerate(['o', '+', '-']):

            if subgroup_type == 'allsujet':
                data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :, :].values

            elif subgroup_type == 'rep':
                data_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, :, :].values

            elif subgroup_type == 'no_rep':
                data_baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, :, :].values

            cond = 'CO2'

            print('intra', subgroup_type, odor, cond)

            if subgroup_type == 'allsujet':
                data_cond = xr_data.loc[:, cond, odor, :, :].values
                
            elif subgroup_type == 'rep':
                data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, :, :].values

            elif subgroup_type == 'no_rep':
                data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, :, :].values

            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                mask_signi = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

                for respi_phase_i, respi_phase in enumerate(['expi', 'inspi']):

                    if respi_phase == 'inspi':
                        mask_signi_phase = np.concatenate(( np.zeros(int(time_vec.size/2)).astype('bool'), mask_signi[int(time_vec.size/2):] ))

                    elif respi_phase == 'expi':
                        mask_signi_phase = np.concatenate(( mask_signi[:int(time_vec.size/2)], np.zeros(int(time_vec.size/2)).astype('bool') ))

                    if mask_signi_phase.sum() == 0:
                        continue     

                    else:
                        mask_signi_phase[0] = False
                        mask_signi_phase[-1] = False   

                    if np.diff(mask_signi_phase).sum() > 2: 

                        start_stop_chunk = np.where(np.diff(mask_signi_phase))[0]

                        max_chunk_signi = []
                        max_chunk_time = []
                        for start_i in np.arange(0, start_stop_chunk.size, 2):

                            _argmax = np.argmax(np.abs(data_cond_chan.mean(axis=0))[start_stop_chunk[start_i]:start_stop_chunk[start_i+1]])
                            max_chunk_time.append(start_stop_chunk[start_i] +_argmax)
                            max_chunk_signi.append(data_cond_chan.mean(axis=0)[_argmax])

                        max_rep = max_chunk_signi[np.argmax(np.abs(max_chunk_signi))]
                        time_max_rep = time_vec[max_chunk_time[np.where(max_chunk_signi == max_rep)[0][0]]]

                    else:

                        start_stop_chunk = np.where(np.diff(mask_signi_phase))[0]

                        _argmax = np.argmax(np.abs(data_cond_chan.mean(axis=0))[start_stop_chunk[0]:start_stop_chunk[1]])
                        max_rep = data_cond_chan.mean(axis=0)[start_stop_chunk[0] + _argmax]
                        time_max_rep = time_vec[start_stop_chunk[0] +_argmax]

                    timing_data[odor_i, respi_phase_i, chan_i, 0] = max_rep
                    timing_data[odor_i, respi_phase_i, chan_i, 1] = time_max_rep

        xr_coords = {'odor' : odor_list, 'phase' : ['expi', 'inspi'], 'chan' : chan_list_eeg, 'type' : ['value', 'time']}
        xr_timing = xr.DataArray(data=timing_data, coords=xr_coords)     

        ### plot
        min, max = xr_timing.loc[:,:,:,'value'].data.min(), xr_timing.loc[:,:,:,'value'].data.max()
        vlim = np.max([np.abs(min), np.abs(max)])
        if subgroup_type == 'rep':
            vlim = 0.22
        color_phase = {'expi' : 'tab:green', 'inspi' : 'tab:orange'}

        fig, axs = plt.subplots(ncols=3, subplot_kw={'projection': 'polar'}, figsize=(10,8))

        for odor_i, odor in enumerate(odor_list):

            ax = axs[odor_i]
            mean_vec = {}

            for phase in ['expi', 'inspi']:

                angles_vec = np.linspace(0, 2*np.pi, num=time_vec.size)
                mask_sel = (xr_timing.loc[odor, phase, :, 'time'] != 0).data
                time_responses_filter = xr_timing.loc[odor, phase, :, 'time'].data[mask_sel]
                ampl_responses_filter = xr_timing.loc[odor, phase, :, 'value'].data[mask_sel]

                time_responses_i = [np.where(time_vec == time_val)[0][0] for time_val in time_responses_filter]
                angle_responses = angles_vec[time_responses_i]

                _phase_mean = np.angle(np.mean(np.exp(1j*angle_responses)))
                
                mean_vec[phase] = np.round(time_vec[[angle_i for angle_i, angle_val in enumerate(angles_vec) if np.mod(_phase_mean, 2 * np.pi) < angle_val][0]], 2)

                ax.scatter(angle_responses, ampl_responses_filter, color=color_phase[phase], s=10)

                ax.plot(angles_vec, np.zeros((time_vec.size)), color='k')

                ax.plot([_phase_mean, _phase_mean], [0, np.mean(ampl_responses_filter)], color=color_phase[phase])

            ax.set_xticks(np.linspace(0, 2 * np.pi, num=4, endpoint=False))
            ax.set_xticklabels(np.round(np.linspace(-2.5,2.5, num=4, endpoint=False), 2))
            ax.set_yticks(np.round(np.linspace(-vlim,vlim, num=3, endpoint=True), 2))
            ax.set_rlim([-vlim,vlim])
            ax.set_title(f"{odor} \n inspi : {mean_vec['inspi']}, expi : {mean_vec['expi']}")

        plt.suptitle(subgroup_type)
        plt.tight_layout()

        # plt.show()

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'time'))
        plt.savefig(f"{subgroup_type}_CO2.png")

    




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

    ######## LOAD DATA ########

    # cond_erp = ['VS', 'MECA', 'CO2']

    print(f'#### compute allsujet ####', flush=True)

    xr_data, xr_data_sem = compute_ERP()
    df_stats_interintra = get_df_stats(xr_data)
    xr_lm_data, xr_lm_pred, xr_lm_pred_coeff = compute_lm_on_ERP(xr_data)


    ######## LINEAR REGRESSION PLOT ########

    xr_PPI_count = get_PPI_count(xr_data)
    
    # shuffle_way = 'inter_cond'
    # shuffle_way = 'intra_cond'
    shuffle_way = 'linear_based'
    xr_surr = compute_surr_ERP(xr_data, shuffle_way)
    
    print(f'#### plot allsujet ####', flush=True)

    plot_ERP(xr_data, xr_lm_data, xr_lm_pred, xr_lm_pred_coeff, xr_surr, xr_PPI_count)

    ######## IDENTIFY WHEN AND WHERE ERP OCCURE ########

    plot_ERP_response_profile(xr_data, xr_data_sem)

    ######## ERP ANALYSIS ########

    cluster_stats_type = 'manual_perm'
    # cluster_stats, cluster_stats_rep_norep = get_cluster_stats(xr_data)
    cluster_stats, cluster_stats_rep_norep = get_cluster_stats_manual_prem(xr_data)
    xr_cluster_based_perm = get_cluster_stats_manual_prem_subject_wise()

    df_ERP_metrics_allsujet, df_ERP_metric_A2_ratings = get_df_ERP_metric_allsujet(xr_data, xr_cluster_based_perm)

    plot_ERP_metrics_response(df_ERP_metrics_allsujet)
    plot_ERP_metrics_A2_lm(df_ERP_metric_A2_ratings)

    plot_ERP_diff(xr_data, cluster_stats, cluster_stats_type) # used for analysis

    plot_ERP_rep_norep(xr_data, cluster_stats_rep_norep, cluster_stats_type)

    plot_erp_response_stats(xr_data)

    ######## ERP TIME ########

    timing_ERP_IE_SUM_plot(xr_data)
    timing_ERP_IE_SUM_export_df(xr_data)

    ######## PPI ########

    #### plot PPI across subject
    plot_PPI_proportion(xr_PPI_count)

    #### manual evaluation
    generate_ppi_evaluation(xr_data)

    ######## MEAN RESPI ########

    plot_mean_respi()

    ######## REG ########

    print(f'#### plot discomfort / slope ####', flush=True)
    plot_slope_versus_discomfort(xr_data, xr_lm_data)

    ######## TOPOPLOTS AND STATS ########

    compute_topoplot_stats_allsujet_minmax(xr_data)
    compute_topoplot_stats_repnorep_minmax(xr_data)
    compute_topoplot_stats_repnorep_diff_minmax(xr_data)

    for perm_type in ['mne', 'inhouse']:
        compute_topoplot_stats_allsujet_perm(xr_data, perm_type)
        compute_topoplot_stats_repnorep_perm(xr_data, perm_type)
        compute_topoplot_stats_repnorep_diff_perm(xr_data, perm_type)




