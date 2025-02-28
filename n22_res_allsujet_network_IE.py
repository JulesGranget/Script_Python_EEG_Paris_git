
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

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *
from n21bis_res_allsujet_ERP import *

debug = False



########################################
######## STATS FUNCTIONS ########
########################################

#tf = tf_plot
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

################################
######## PSEUDO NETWORK ########
################################


def compute_topoplot_IE_network(xr_data):

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    time_vec = xr_data.shape[-1]

    #subgroup_type = 'allsujet'
    for subgroup_type in ['allsujet', 'rep', 'no_rep']:

        ######## INTRA ########
        #### load stats
        data_stats_cluster_intra = {}

        for odor_i, odor in enumerate(['o', '+', '-']):

            data_stats_cluster_intra[odor] = {}

            if subgroup_type == 'allsujet':
                data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :, :].values

            elif subgroup_type == 'rep':
                data_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, :, :].values

            elif subgroup_type == 'no_rep':
                data_baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, :, :].values

            for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

                print(subgroup_type, odor, cond)

                data_stats_cluster_intra[odor][cond] = np.zeros((len(chan_list_eeg), time_vec))

                if subgroup_type == 'allsujet':
                    data_cond = xr_data.loc[:, cond, odor, :, :].values
                    
                elif subgroup_type == 'rep':
                    data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, :, :].values

                elif subgroup_type == 'no_rep':
                    data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, :, :].values

                for chan_i, chan in enumerate(chan_list_eeg):

                    data_baseline_chan = data_baseline[:, chan_i, :]
                    data_cond_chan = data_cond[:, chan_i, :] 

                    data_stats_cluster_intra[odor][cond][chan_i,:] = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

        #### scale
        cluster_size = np.array([])

        for phase_i, phase in enumerate(['inspi', 'expi']):

            for odor_i, odor in enumerate(['o', '+', '-']):

                for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

                    for chan_i, chan in enumerate(chan_list_eeg):

                        if phase == 'inspi':
                            cluster_size = np.append( cluster_size, np.round((data_stats_cluster_intra[odor][cond][chan_i,int(time_vec/2):]*1).sum() / (time_vec/2)*100, 3) )
                        if phase == 'expi':
                            cluster_size = np.append( cluster_size, np.round((data_stats_cluster_intra[odor][cond][chan_i,:int(time_vec/2)]*1).sum() / (time_vec/2)*100, 3) )

        vlim = np.percentile(cluster_size, 99)

        #### plot
        for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):    

            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15,15))

            for phase_i, phase in enumerate(['inspi', 'expi']):

                for odor_i, odor in enumerate(['o', '+', '-']):

                    print('intra', phase, odor, cond)

                    mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')
                    data_topoplot = np.zeros((len(chan_list_eeg)))

                    #chan_i, chan = 0, chan_list_eeg[0]
                    for chan_i, chan in enumerate(chan_list_eeg):

                        if phase == 'inspi':
                            perm_vec_phase = data_stats_cluster_intra[odor][cond][chan_i,int(time_vec/2):]
                        if phase == 'expi':
                            perm_vec_phase = data_stats_cluster_intra[odor][cond][chan_i,:int(time_vec/2)]

                        if perm_vec_phase.sum() > 0: 

                            if phase == 'inspi':
                                data_topoplot[chan_i] = np.round(perm_vec_phase.sum()/(time_vec/2)*100, 3)
                            if phase == 'expi':
                                data_topoplot[chan_i] = np.round(perm_vec_phase.sum()/(time_vec/2)*100, 3)

                            mask_signi[chan_i] = True

                    ax = axs[odor_i, phase_i]

                    ax.set_title(f"{odor} {phase}")

                    mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    mask=mask_signi, mask_params=mask_params, vlim=(0, vlim), cmap='seismic')

            plt.tight_layout()

            plt.suptitle(f'{cond} {subgroup_type} INTRA {np.round(vlim,2)}')

            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network'))
            fig.savefig(f"{subgroup_type}_intra_{cond}.jpeg")

            plt.close('all')
            
            # plt.show()

        ######## INTER ########
        #### load stats
        data_stats_cluster_inter = {}

        for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

            data_stats_cluster_inter[cond] = {}

            data_baseline = xr_data.loc[:, cond, 'o', :, :].values

            if subgroup_type == 'allsujet':
                data_baseline = xr_data.loc[:, cond, 'o', :, :].values
                
            elif subgroup_type == 'rep':
                data_baseline = xr_data.loc[sujet_best_list_rev, cond, 'o', :, :].values

            elif subgroup_type == 'no_rep':
                data_baseline = xr_data.loc[sujet_no_respond_rev, cond, 'o', :, :].values

            for odor_i, odor in enumerate(['+', '-']):

                print(odor, cond)

                data_stats_cluster_inter[cond][odor] = np.zeros((len(chan_list_eeg), time_vec))

                if subgroup_type == 'allsujet':
                    data_cond = xr_data.loc[:, cond, odor, :, :].values
                    
                elif subgroup_type == 'rep':
                    data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, :, :].values

                elif subgroup_type == 'no_rep':
                    data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, :, :].values

                for chan_i, chan in enumerate(chan_list_eeg):

                    data_baseline_chan = data_baseline[:, chan_i, :]
                    data_cond_chan = data_cond[:, chan_i, :] 

                    data_stats_cluster_inter[cond][odor][chan_i,:] = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

        #### scale
        cluster_size = np.array([])

        for phase_i, phase in enumerate(['inspi', 'expi']):

            for odor_i, odor in enumerate(['+', '-']):

                for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

                    for chan_i, chan in enumerate(chan_list_eeg):

                        if phase == 'inspi':
                            cluster_size = np.append( cluster_size, np.round((data_stats_cluster_inter[cond][odor][chan_i,int(time_vec/2):]*1).sum()/(time_vec/2)*100, 3) )
                        if phase == 'expi':
                            cluster_size = np.append( cluster_size, np.round((data_stats_cluster_inter[cond][odor][chan_i,:int(time_vec/2)]*1).sum()/(time_vec/2)*100, 3) )

        vlim = np.percentile(cluster_size, 99)

        #### plot
        for odor_i, odor in enumerate(['+', '-']):    

            fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15,15))

            for phase_i, phase in enumerate(['inspi', 'expi']):

                for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

                    print('intra', phase, odor, cond)

                    mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')
                    data_topoplot = np.zeros((len(chan_list_eeg)))

                    #chan_i, chan = 0, chan_list_eeg[0]
                    for chan_i, chan in enumerate(chan_list_eeg):

                        if phase == 'inspi':
                            perm_vec_phase = data_stats_cluster_inter[cond][odor][chan_i,int(time_vec/2):]
                        if phase == 'expi':
                            perm_vec_phase = data_stats_cluster_inter[cond][odor][chan_i,:int(time_vec/2)]

                        if perm_vec_phase.sum() > 0: 

                            if phase == 'inspi':
                                data_topoplot[chan_i] = np.round(perm_vec_phase.sum()/(time_vec/2)*100, 3)
                            if phase == 'expi':
                                data_topoplot[chan_i] = np.round(perm_vec_phase.sum()/(time_vec/2)*100, 3)

                            mask_signi[chan_i] = True

                    ax = axs[cond_i, phase_i]

                    ax.set_title(f"{cond} {phase}")

                    mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    mask=mask_signi, mask_params=mask_params, vlim=(0, vlim), cmap='seismic')

            plt.tight_layout()

            plt.suptitle(f'{odor} {subgroup_type} INTER {np.round(vlim,2)}')

            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network'))
            fig.savefig(f"{subgroup_type}_inter_{odor}.jpeg")

            plt.close('all')
            
            # plt.show()







def compute_topoplot_IE_network_SUM(stretch=True):

    if stretch:
        xr_data, xr_data_sem = compute_ERP_stretch()
        cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep = get_cluster_stats_manual_prem(stretch=True)
    else:
        xr_data, xr_data_sem = compute_ERP()
        cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep = get_cluster_stats_manual_prem()

    mask_params = dict(markersize=15, markerfacecolor='y')

    cond_sel = ['FR_CV_1', 'MECA', 'CO2']
    phase_vec_list = {'whole' : np.arange(xr_data.shape[-1]), 'inspi' : np.arange(0, int(xr_data.shape[-1]/2)), 'expi' : np.arange(int(xr_data.shape[-1]/2), xr_data.shape[-1])}

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    #subgroup_type = 'non_rep'
    for subgroup_type in ['allsujet', 'rep', 'non_rep']:

        ######## INTRA ########

        print(f'compute intra {subgroup_type}')

        #### compute topoplot values
        topoplot_val_allcond = np.zeros([len(phase_list), len(cond_sel)-1, len(odor_list), len(chan_list_eeg)])

        for phase_i, phase in enumerate(['whole', 'inspi', 'expi']):

            for odor_i, odor in enumerate(['o', '+', '-']):

                for cond_i, cond in enumerate(['MECA', 'CO2']):

                    #chan_i, chan = 6, chan_list_eeg[6]
                    for chan_i, chan in enumerate(chan_list_eeg):

                        mask_signi = cluster_stats_intra.loc[subgroup_type, chan, odor, cond, phase_vec_list[phase]].values.astype('bool')

                        if mask_signi.sum() > 0: 

                            if subgroup_type == 'allsujet':
                                data_baseline = np.median(xr_data.loc[:, 'FR_CV_1', odor, chan, phase_vec_list[phase]].values, axis=0)

                            elif subgroup_type == 'rep':
                                data_baseline = np.median(xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, chan, phase_vec_list[phase]].values, axis=0)

                            elif subgroup_type == 'non_rep':
                                data_baseline = np.median(xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, chan, phase_vec_list[phase]].values, axis=0)

                            if subgroup_type == 'allsujet':
                                data_cond = np.median(xr_data.loc[:, cond, odor, chan, phase_vec_list[phase]].values, axis=0)
                                
                            elif subgroup_type == 'rep':
                                data_cond = np.median(xr_data.loc[sujet_best_list_rev, cond, odor, chan, phase_vec_list[phase]].values, axis=0)

                            elif subgroup_type == 'non_rep':
                                data_cond = np.median(xr_data.loc[sujet_no_respond_rev, cond, odor, chan, phase_vec_list[phase]].values, axis=0)

                            data_diff_sel = (data_cond - data_baseline)[mask_signi]
                            topoplot_val_allcond[phase_i, cond_i, odor_i, chan_i] = data_diff_sel.sum()
        
        #### scale
        vlim = np.percentile(topoplot_val_allcond[1:,:,:,:], 99)
        vlim_allphase = np.percentile(topoplot_val_allcond[0,:,:,:], 99)

        #### get allchan response
        allchan_response_intra = {'region' : [], 'cond' : [], 'phase' : [], 'odor' : [], 'sum' : []}

        for region in chan_list_lobes:

            for cond_i, cond in enumerate(['MECA', 'CO2']):    

                for phase_i, phase in enumerate(phase_list):

                    for odor_i, odor in enumerate(['o', '+', '-']):

                        data_topoplot = topoplot_val_allcond[phase_i,cond_i,odor_i,:]

                        allchan_response_intra['cond'].append(cond)
                        allchan_response_intra['phase'].append(phase)
                        allchan_response_intra['odor'].append(odor)
                        allchan_response_intra['sum'].append(data_topoplot.sum()/len(chan_list_lobes[region]))
                        allchan_response_intra['region'].append(region)
        
        df_allchan_response_intra = pd.DataFrame(allchan_response_intra)

        #### plot inspi expi
        
        #cond_i, cond = 1, 'CO2'
        for cond_i, cond in enumerate(['MECA', 'CO2']):    

            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15,15))

            for phase_i, phase in enumerate(['inspi', 'expi']):

                for odor_i, odor in enumerate(['o', '+', '-']):

                    ax = axs[odor_i, phase_i]

                    mask_signi_topoplot = topoplot_val_allcond[phase_i+1,cond_i,odor_i,:] != 0
                    data_topoplot = topoplot_val_allcond[phase_i+1,cond_i,odor_i,:]

                    ax.set_title(f"{odor} {phase}")

                    mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    mask=mask_signi_topoplot, mask_params=mask_params, vlim=(-vlim, vlim), cmap='seismic')

            plt.tight_layout()

            plt.suptitle(f'{cond} {subgroup_type} INTRA {np.round(vlim,2)}')

            # plt.show()

            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network'))

            if stretch:
                fig.savefig(f"stretch_SUM_{subgroup_type}_intra_{cond}.jpeg")
            else:
                fig.savefig(f"nostretch_SUM_{subgroup_type}_intra_{cond}.jpeg")

            plt.close('all')
            
            

        #### plot allphase    
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15,15))

        for cond_i, cond in enumerate(['MECA', 'CO2']):

            for odor_i, odor in enumerate(['o', '+', '-']):

                mask_signi_topoplot = topoplot_val_allcond[0,cond_i,odor_i,:] != 0
                data_topoplot = topoplot_val_allcond[0,cond_i,odor_i,:]

                ax = axs[odor_i, cond_i]

                ax.set_title(f"{odor} {cond}")

                mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                mask=mask_signi_topoplot, mask_params=mask_params, vlim=(-vlim_allphase, vlim_allphase), cmap='seismic')

        plt.tight_layout()

        plt.suptitle(f'{subgroup_type} INTRA {np.round(vlim_allphase,2)}')

        # plt.show()

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network'))

        if stretch:
            fig.savefig(f"stretch_SUM_ALLPHASE_{subgroup_type}_intra.jpeg")
        else:
            fig.savefig(f"nostretch_SUM_ALLPHASE_{subgroup_type}_intra.jpeg")

        plt.close('all')

        #### plot allchan response
        for region in chan_list_lobes:

            for cond in ['MECA', 'CO2']:

                sns.barplot(data=df_allchan_response_intra.query(f"cond == '{cond}' and region == '{region}'"), x='phase', y='sum', hue='odor', hue_order=["o", "+", "-"], order=['expi', 'inspi'])

                os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network', 'allchan_sum'))
                plt.title(f'{subgroup_type} intra {cond} {region}')
                
                if stretch:
                    plt.savefig(f"stretch_ALLCHAN_{subgroup_type}_{cond}_{region}_intra.jpeg")
                else:
                    plt.savefig(f"nostretch_ALLCHAN_{subgroup_type}_{cond}_{region}_intra.jpeg")

                plt.close('all')

        sns.barplot(data=df_allchan_response_intra.query(f"cond == 'CO2' and odor == 'o'"), x='region', y='sum', hue='phase', hue_order=['expi', 'inspi'])

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network', 'allchan_sum'))
        plt.title(f'{subgroup_type} intra CO2')
        
        if stretch:
            plt.savefig(f"stretch_ALLCHAN_ALLREGION_{subgroup_type}_{cond}_intra.jpeg")
        else:
            plt.savefig(f"nostretch_ALLCHAN_ALLREGION_{subgroup_type}_{cond}_intra.jpeg")

        plt.close('all')

        #### values for CO2
        df_allchan_response_intra.query(f"cond == 'CO2' and region == 'all'")





    

        ######## INTER ########

        print(f'compute inter {subgroup_type}')

        #### compute topoplot values
        topoplot_val_allcond = np.zeros([len(phase_list), len(cond_sel), len(odor_list)-1, len(chan_list_eeg)])

        for phase_i, phase in enumerate(['whole', 'inspi', 'expi']):

            for odor_i, odor in enumerate(['+', '-']):

                for cond_i, cond in enumerate([cond_sel]):

                    #chan_i, chan = 6, chan_list_eeg[6]
                    for chan_i, chan in enumerate(chan_list_eeg):

                        mask_signi = cluster_stats_inter.loc[subgroup_type, chan, odor, cond, phase_vec_list[phase]].values.astype('bool')

                        if mask_signi.sum() > 0: 

                            if subgroup_type == 'allsujet':
                                data_baseline = np.median(xr_data.loc[:, cond, 'o', chan, phase_vec_list[phase]].values, axis=0)

                            elif subgroup_type == 'rep':
                                data_baseline = np.median(xr_data.loc[sujet_best_list_rev, cond, 'o', chan, phase_vec_list[phase]].values, axis=0)

                            elif subgroup_type == 'non_rep':
                                data_baseline = np.median(xr_data.loc[sujet_no_respond_rev, cond, 'o', chan, phase_vec_list[phase]].values, axis=0)

                            if subgroup_type == 'allsujet':
                                data_cond = np.median(xr_data.loc[:, cond, odor, chan, phase_vec_list[phase]].values, axis=0)
                                
                            elif subgroup_type == 'rep':
                                data_cond = np.median(xr_data.loc[sujet_best_list_rev, cond, odor, chan, phase_vec_list[phase]].values, axis=0)

                            elif subgroup_type == 'non_rep':
                                data_cond = np.median(xr_data.loc[sujet_no_respond_rev, cond, odor, chan, phase_vec_list[phase]].values, axis=0)

                            data_diff_sel = (data_cond - data_baseline)[mask_signi]
                            topoplot_val_allcond[phase_i, cond_i, odor_i, chan_i] = data_diff_sel.sum()

        #### scale
        vlim = np.percentile(topoplot_val_allcond[1:,:,:,:], 99)
        vlim_allphase = np.percentile(topoplot_val_allcond[0,:,:,:], 99)

        #### get allchan response
        allchan_response_inter = {'region' : [], 'cond' : [], 'phase' : [], 'odor' : [], 'sum' : []}

        for region in chan_list_lobes:

            for cond_i, cond in enumerate([cond_sel]):    

                for phase_i, phase in enumerate([phase_list]):

                    for odor_i, odor in enumerate(['+', '-']):

                        data_topoplot = topoplot_val_allcond[phase_i, cond_i, odor_i, :]

                        allchan_response_inter['cond'].append(cond)
                        allchan_response_inter['phase'].append(phase)
                        allchan_response_inter['odor'].append(odor)
                        allchan_response_inter['sum'].append(data_topoplot.sum()/len(chan_list_lobes[region]))
                        allchan_response_inter['region'].append(region)
        
        df_allchan_response_inter = pd.DataFrame(allchan_response_inter)

        #### plot inspi expi

        for odor_i, odor in enumerate(['+', '-']):    

            fig, axs = plt.subplots(nrows=len(cond_sel), ncols=2, figsize=(15,15))

            for phase_i, phase in enumerate(['inspi', 'expi']):

                for cond_i, cond in enumerate(cond_sel):

                    ax = axs[cond_i, phase_i]

                    mask_signi = topoplot_val_allcond[phase_i+1,cond_i,odor_i,:] != 0
                    data_topoplot = topoplot_val_allcond[phase_i+1,cond_i,odor_i,:]

                    ax.set_title(f"{cond} {phase}")

                    mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim), cmap='seismic')

            plt.tight_layout()

            plt.suptitle(f'{odor} {subgroup_type} INTER {np.round(vlim,2)}')

            # plt.show()

            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network'))

            if stretch:
                fig.savefig(f"stretch_SUM_{subgroup_type}_inter_{odor}.jpeg")
            else:
                fig.savefig(f"nostretch_SUM_{subgroup_type}_inter_{odor}.jpeg")

            plt.close('all')

        #### plot allphase

        fig, axs = plt.subplots(nrows=2, ncols=len(cond_sel), figsize=(15,15))
    
        for odor_i, odor in enumerate(['+', '-']):  

            for cond_i, cond in enumerate(cond_sel):

                mask_signi = topoplot_val_allcond[0,cond_i,odor_i,:] != 0
                data_topoplot = topoplot_val_allcond[0,cond_i,odor_i,:]

                ax = axs[odor_i, cond_i]

                ax.set_title(f"{cond} {odor}")

                mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                mask=mask_signi, mask_params=mask_params, vlim=(-vlim_allphase, vlim_allphase), cmap='seismic')

        # plt.tight_layout()

        plt.suptitle(f'{subgroup_type} INTER {np.round(vlim_allphase,2)}')

        # plt.show()

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network'))

        if stretch:
            fig.savefig(f"stretch_SUM_ALLPHASE_{subgroup_type}_inter.jpeg")
        else:
            fig.savefig(f"nostretch_SUM_ALLPHASE_{subgroup_type}_inter.jpeg")

        plt.close('all')
        
        

        #### plot allchan response
        for region in chan_list_lobes:

            for cond in ['FR_CV_1', 'MECA', 'CO2']:

                sns.barplot(data=df_allchan_response_inter.query(f"cond == '{cond}' and region == '{region}'"), x='phase', y='sum', hue='odor', hue_order=["+", "-"], order=['expi', 'inspi'])

                os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network', 'allchan_sum'))
                plt.title(f'{subgroup_type} inter {cond} {region}')
                
                if stretch:
                    plt.savefig(f"stretch_ALLCHAN_{subgroup_type}_{cond}_{region}_inter.jpeg")
                else:
                    plt.savefig(f"nostretch_ALLCHAN_{subgroup_type}_{cond}_{region}_inter.jpeg")

                plt.close('all')












def compute_topoplot_IE_network_repnorep(xr_data):

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    time_vec = xr_data.shape[-1]

    #### load stats
    data_stats_cluster = {}

    for odor_i, odor in enumerate(odor_list):

        data_stats_cluster[odor] = {}

        for cond_i, cond in enumerate(conditions):

            print(odor, cond)

            data_baseline = xr_data.loc[sujet_best_list_rev, cond, odor, :, :].values

            data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, :, :].values

            data_stats_cluster[odor][cond] = np.zeros((len(chan_list_eeg), time_vec))

            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                data_stats_cluster[odor][cond][chan_i,:] = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

    #### scale
    cluster_size = np.array([])

    for phase_i, phase in enumerate(['inspi', 'expi']):

        for odor_i, odor in enumerate(odor_list):

            for cond_i, cond in enumerate(conditions):

                for chan_i, chan in enumerate(chan_list_eeg):

                    if phase == 'inspi':
                        cluster_size = np.append( cluster_size, np.round((data_stats_cluster[odor][cond][chan_i,int(time_vec/2):]*1).sum() / (time_vec/2)*100, 3) )
                    if phase == 'expi':
                        cluster_size = np.append( cluster_size, np.round((data_stats_cluster[odor][cond][chan_i,:int(time_vec/2)]*1).sum() / (time_vec/2)*100, 3) )

    vlim = np.percentile(cluster_size, 99)

    #### plot   
    for phase_i, phase in enumerate(['inspi', 'expi']):

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions), figsize=(15,15))

        for cond_i, cond in enumerate(conditions):  

            for odor_i, odor in enumerate(odor_list):

                print(phase, odor, cond)

                mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')
                data_topoplot = np.zeros((len(chan_list_eeg)))

                #chan_i, chan = 0, chan_list_eeg[0]
                for chan_i, chan in enumerate(chan_list_eeg):

                    if phase == 'inspi':
                        perm_vec_phase = data_stats_cluster[odor][cond][chan_i,int(time_vec/2):]
                    if phase == 'expi':
                        perm_vec_phase = data_stats_cluster[odor][cond][chan_i,:int(time_vec/2)]

                    if perm_vec_phase.sum() > 0: 

                        if phase == 'inspi':
                            data_topoplot[chan_i] = np.round(perm_vec_phase.sum()/(time_vec/2)*100, 3)
                        if phase == 'expi':
                            data_topoplot[chan_i] = np.round(perm_vec_phase.sum()/(time_vec/2)*100, 3)

                        mask_signi[chan_i] = True

                ax = axs[odor_i, cond_i]

                ax.set_title(f"{odor} {cond}")

                mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                mask=mask_signi, mask_params=mask_params, vlim=(0, vlim), cmap='seismic')

        plt.tight_layout()

        plt.suptitle(f'{phase} REP_NOREP {np.round(vlim,2)}')

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network'))
        fig.savefig(f"repnorep_{phase}.jpeg")

        plt.close('all')
        
        # plt.show()



def compute_topoplot_IE_network_repnorep_SUM(stretch=True):

    if stretch:
        xr_data, xr_data_sem = compute_ERP_stretch()
        cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep = get_cluster_stats_manual_prem(stretch=True)
    else:
        xr_data, xr_data_sem = compute_ERP()
        cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep = get_cluster_stats_manual_prem()

    phase_vec_list = {'whole' : np.arange(xr_data.shape[-1]), 'inspi' : np.arange(0, int(xr_data.shape[-1]/2)), 'expi' : np.arange(int(xr_data.shape[-1]/2), xr_data.shape[-1])}

    mask_params = dict(markersize=15, markerfacecolor='y')

    cond_sel = ['FR_CV_1', 'MECA', 'CO2']

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    #### compute topoplot values
    topoplot_val_allcond = np.zeros([len(phase_list), len(cond_sel), len(odor_list), len(chan_list_eeg)])

    for phase_i, phase in enumerate(['whole', 'inspi', 'expi']):

        for odor_i, odor in enumerate(odor_list):

            for cond_i, cond in enumerate(cond_sel):

                #chan_i, chan = 6, chan_list_eeg[6]
                for chan_i, chan in enumerate(chan_list_eeg):

                    mask_signi = cluster_stats_rep_norep.loc[chan, odor, cond, phase_vec_list[phase]].values.astype('bool')

                    if mask_signi.sum() > 0: 

                        data_baseline = np.median(xr_data.loc[sujet_no_respond_rev, cond, odor, chan, phase_vec_list[phase]].values, axis=0)
                        data_cond = np.median(xr_data.loc[sujet_best_list_rev, cond, odor, chan, phase_vec_list[phase]].values, axis=0)

                        data_diff_sel = (data_cond - data_baseline)[mask_signi]
                        topoplot_val_allcond[phase_i, cond_i, odor_i, chan_i] = data_diff_sel.sum()
    
    #### scale
    vlim = np.percentile(topoplot_val_allcond, 99)
    vlim_allphase = np.percentile(topoplot_val_allcond, 99)

    #### get allchan response
    allchan_response_repnorep = {'region' : [], 'cond' : [], 'phase' : [], 'odor' : [], 'sum' : []}

    for region in chan_list_lobes:

        for cond_i, cond in enumerate([cond_sel]):

            for phase_i, phase in enumerate([phase_list]):

                for odor_i, odor in enumerate([odor_list]):

                    data_topoplot = topoplot_val_allcond[phase_i,cond_i,odor_i,:]

                    allchan_response_repnorep['cond'].append(cond)
                    allchan_response_repnorep['phase'].append(phase)
                    allchan_response_repnorep['odor'].append(odor)
                    allchan_response_repnorep['sum'].append(data_topoplot.sum()/len(chan_list_lobes[region]))
                    allchan_response_repnorep['region'].append(region)
    
    df_allchan_response_repnorep = pd.DataFrame(allchan_response_repnorep)

    #### plot inspi expi  
    for phase_i, phase in enumerate(['inspi', 'expi']):

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_sel), figsize=(15,15))

        for cond_i, cond in enumerate(cond_sel):  

            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, cond_i]

                mask_signi = topoplot_val_allcond[phase_i+1,cond_i,odor_i,:] != 0
                data_topoplot = topoplot_val_allcond[phase_i+1,cond_i,odor_i,:]

                ax.set_title(f"{odor} {cond}")

                mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim), cmap='seismic')

        plt.tight_layout()

        plt.suptitle(f'{phase} REP_NOREP {np.round(vlim,2)}')

        # plt.show()

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network'))

        if stretch:
            fig.savefig(f"stretch_SUM_repnorep_{phase}.jpeg")
        else:
            fig.savefig(f"nostretch_SUM_repnorep_{phase}.jpeg")

        plt.close('all')

    #### plot allphase 
    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_sel), figsize=(15,15))

    for cond_i, cond in enumerate(cond_sel):  

        for odor_i, odor in enumerate(odor_list):

            mask_signi = topoplot_val_allcond[0,cond_i,odor_i,:] != 0
            data_topoplot = topoplot_val_allcond[0,cond_i,odor_i,:]

            ax = axs[odor_i, cond_i]

            ax.set_title(f"{odor} {cond}")

            mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            mask=mask_signi, mask_params=mask_params, vlim=(-vlim_allphase, vlim_allphase), cmap='seismic')

    plt.tight_layout()

    plt.suptitle(f'ALLPHASE REP_NOREP {np.round(vlim_allphase,2)}')

    # plt.show()

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network'))

    if stretch:
        fig.savefig(f"stretch_SUM_ALLPHASE_repnorep.jpeg")
    else:
        fig.savefig(f"nostretch_SUM_ALLPHASE_repnorep.jpeg")

    plt.close('all')
    
    #### plot allchan response
    for region in chan_list_lobes:

        for cond in ['FR_CV_1', 'MECA', 'CO2']:

            sns.barplot(data=df_allchan_response_repnorep.query(f"cond == '{cond}' and region == '{region}'"), x='phase', y='sum', hue='odor', hue_order=["o", "+", "-"], order=['expi', 'inspi'])

            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network', 'allchan_sum'))
            plt.title(f'repnorep {cond} {region}')
            
            if stretch:
                plt.savefig(f"stretch_ALLCHAN_repnorep_{cond}_{region}.jpeg")
            else:
                plt.savefig(f"nostretch_ALLCHAN_repnorep_{cond}_{region}.jpeg")

            plt.close('all')

    #### value for CO2
    df_allchan_response_repnorep.query(f"cond == 'CO2' and region == 'all'")
    df_allchan_response_repnorep.query(f"cond == 'FR_CV_1' and region == 'all'")
















def network_TF_IE():

    tf_mode = 'TF'

    print('COMPUTE', flush=True)

    os.chdir(os.path.join(path_precompute, 'allsujet', tf_mode))
    xr_allsujet = xr.open_dataarray(f'allsujet_{tf_mode}.nc')

    group_list = ['allsujet', 'rep', 'no_rep']
    band_list = ['theta', 'gamma']
    respi_phase_list = ['all', 'inspi', 'expi']
    Pxx_modification_list = ['all', 'incre', 'decre']
    time = np.arange(stretch_point_TF)

    #### get data
    os.chdir(os.path.join(path_results, 'allplot', tf_mode))

    cluster_size_dict = {'stat_type' : ['inter', 'intra'], 'group' : group_list, 'band' : band_list, 'cond' : conditions, 'odor' : odor_list, 'phase' : respi_phase_list, 'Pxx' : Pxx_modification_list, 'chan' : chan_list_eeg}
    cluster_size_data = np.zeros((2, len(group_list), len(band_list), len(conditions), len(odor_list), len(respi_phase_list), len(Pxx_modification_list), len(chan_list_eeg)))

    #tf_stats_type = 'inter'
    for tf_stats_type_i, tf_stats_type in enumerate(['inter', 'intra']):

        #group = group_list[1]
        for group_i, group in enumerate(group_list):

            for band_i, band in enumerate(band_list):

                print(tf_stats_type, group, band)

                #r, odor_i = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    #c, cond = 1, conditions[1]
                    for cond_i, cond in enumerate(conditions):

                        for Pxx_type_i, Pxx_type in enumerate(Pxx_modification_list):

                            #n_chan, chan_name = 0, chan_list_eeg[0]
                            for n_chan, chan_name in enumerate(chan_list_eeg):

                                tf_plot = xr_allsujet.loc[chan_name, group, cond, odor, :, :].values

                                if cond != 'FR_CV_1' and tf_stats_type == 'intra':
                                    os.chdir(os.path.join(path_precompute, 'allsujet', tf_mode))

                                    pixel_based_distrib = np.median(np.load(f'allsujet_tf_STATS_nchan{n_chan}_{cond}_{odor}_intra.npy'), axis=1)
                                    
                                elif odor != 'o' and tf_stats_type == 'inter':
                                    os.chdir(os.path.join(path_precompute, 'allsujet', tf_mode))

                                    pixel_based_distrib = np.median(np.load(f'allsujet_tf_STATS_nchan{n_chan}_{cond}_{odor}_inter.npy'), axis=1)

                                else:

                                    continue
                                    
                                tf_stats_pre = get_tf_stats(tf_plot, pixel_based_distrib).astype('bool')

                                if Pxx_type == 'all':
                                    tf_mask_Pxx = np.ones(tf_plot.shape).astype('bool')

                                if Pxx_type == 'incre':
                                    tf_mask_Pxx = tf_plot > 0

                                if Pxx_type == 'decre':
                                    tf_mask_Pxx = tf_plot < 0

                                tf_stats = (tf_stats_pre & tf_mask_Pxx) * 1

                                if band == 'theta':
                                    band_inf, band_sup = 4, 8
                                if band == 'gamma':
                                    band_inf, band_sup = 50, 151
                                    
                                frex_sel = (frex >= band_inf) & (frex <= band_sup)

                                for phase_i, phase in enumerate(respi_phase_list):

                                    if phase == 'all':
                                        time_sel = time > -1
                                    if phase == 'inspi':
                                        time_sel = time < stretch_point_TF/2
                                    if phase == 'expi':
                                        time_sel = time > stretch_point_TF/2

                                    tf_stats_phase = tf_stats[frex_sel,:]
                                    cluster_size_data[tf_stats_type_i, group_i, band_i, cond_i, odor_i, phase_i, Pxx_type_i, n_chan] = tf_stats_phase[:,time_sel].sum()*100 / tf_stats_phase[:,time_sel].size

    xr_cluster = xr.DataArray(data=cluster_size_data, dims=cluster_size_dict.keys(), coords=cluster_size_dict.values())
    df_cluster = xr_cluster.to_dataframe(name='cluster_size').reset_index(drop=False)

    #### scale allsujet
    ylim = {}

    for tf_stats_type_i, tf_stats_type in enumerate(['inter', 'intra']):

        ylim[tf_stats_type] = {}

        for group in ['allsujet', 'repnorep', 'rep', 'no_rep']:

            ylim[tf_stats_type][group] = {}

            for band_i, band in enumerate(band_list):

                print(tf_stats_type, group, band)

                scale = np.array([])
                std = np.array([])

                #r, odor_i = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    #c, cond = 1, conditions[1]
                    for cond_i, cond in enumerate(conditions):

                        for Pxx_type_i, Pxx_type in enumerate(Pxx_modification_list):

                            if Pxx_type == 'all':
                                continue

                            for phase_i, phase in enumerate(respi_phase_list):

                                if phase == 'all':
                                    continue

                                if tf_stats_type == 'intra':
                                
                                    df_plot = df_cluster.query(f"stat_type == '{tf_stats_type}' and cond != 'FR_CV_1'")

                                if tf_stats_type == 'inter':
                                
                                    df_plot = df_cluster.query(f"stat_type == '{tf_stats_type}' and odor != 'o'")    

                                if group == 'allsujet':
                                    scale = np.append(scale, df_plot.query(f"group == 'allsujet' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}' and Pxx == '{Pxx_type}'")['cluster_size'].values.mean())
                                    std = np.append(scale, df_plot.query(f"group == 'allsujet' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}' and Pxx == '{Pxx_type}'")['cluster_size'].values.std())

                                if group == 'rep':
                                    scale = np.append(scale, df_plot.query(f"group == 'rep' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}' and Pxx == '{Pxx_type}'")['cluster_size'].values.mean())
                                    std = np.append(scale, df_plot.query(f"group == 'rep' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}' and Pxx == '{Pxx_type}'")['cluster_size'].values.std())

                                if group == 'no_rep':
                                    scale = np.append(scale, df_plot.query(f"group == 'no_rep' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}' and Pxx == '{Pxx_type}'")['cluster_size'].values.mean())
                                    std = np.append(scale, df_plot.query(f"group == 'no_rep' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}' and Pxx == '{Pxx_type}'")['cluster_size'].values.std())

                                if group == 'repnorep':
                                    scale = np.append(scale, df_plot.query(f"group != 'allsujet' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}' and Pxx == '{Pxx_type}'")['cluster_size'].values.mean())
                                    std = np.append(scale, df_plot.query(f"group != 'allsujet' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}' and Pxx == '{Pxx_type}'")['cluster_size'].values.std())

                ylim[tf_stats_type][group][band] = scale[~np.isnan(scale)].max() + std[~np.isnan(std)].mean()

    #### export value
    df_cluster_export = pd.DataFrame(columns=['stat_type', 'group', 'band', 'cond', 'odor', 'phase', 'Pxx', 'cluster_size', 'std'])
    
    for tf_stats_type_i, tf_stats_type in enumerate(['inter', 'intra']):

        for group in group_list:

            for band_i, band in enumerate(band_list):

                print(tf_stats_type, group, band)

                #r, odor_i = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    #c, cond = 1, conditions[1]
                    for cond_i, cond in enumerate(conditions):

                        for Pxx_type_i, Pxx_type in enumerate(Pxx_modification_list):

                            for phase_i, phase in enumerate(respi_phase_list):

                                if tf_stats_type == 'intra' and cond == 'FR_CV_1':
                                
                                    continue

                                if tf_stats_type == 'inter' and odor == 'o':
                                
                                    continue

                                _cluster_size = df_cluster.query(f"stat_type == '{tf_stats_type}' and group == '{group}' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}' and Pxx == '{Pxx_type}'")['cluster_size'].values.mean()
                                _cluster_size_std = df_cluster.query(f"stat_type == '{tf_stats_type}' and group == '{group}' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}' and Pxx == '{Pxx_type}'")['cluster_size'].values.std()


                                _df = pd.DataFrame({'stat_type' : [tf_stats_type], 'group' : [group], 'band' : [band], 'cond' : [cond], 
                                                    'odor' : [odor], 'phase' : [phase], 'Pxx' : [Pxx_type], 'cluster_size' : [_cluster_size],
                                                    'std' : [_cluster_size_std]})

                                df_cluster_export = pd.concat([df_cluster_export, _df], axis=0)
                            
    os.chdir(os.path.join(path_results, 'allplot', tf_mode, 'Pxx'))

    df_cluster_export.to_excel(f'cluster_values_allcond.xlsx')

    #### plot allsujet
    pd.DataFrame.iteritems = pd.DataFrame.items

    for group in group_list:

        for tf_stats_type_i, tf_stats_type in enumerate(['inter', 'intra']):

            for band_i, band in enumerate(band_list):

                for Pxx_type_i, Pxx_type in enumerate(Pxx_modification_list):

                    if Pxx_type == 'all':
                        continue

                    print(tf_stats_type, band)

                    if tf_stats_type == 'intra':
                    
                        df_plot = df_cluster.query(f"phase != 'all' and group == '{group}' and stat_type == '{tf_stats_type}' and cond != 'FR_CV_1' and band == '{band}' and Pxx == '{Pxx_type}'")

                    if tf_stats_type == 'inter':
                    
                        df_plot = df_cluster.query(f"phase != 'all' and group == '{group}' and stat_type == '{tf_stats_type}' and odor != 'o' and band == '{band}' and Pxx == '{Pxx_type}'")    

                    # g = sns.FacetGrid(df_plot, col="phase", height=5, aspect=1)
                    # g.map(sns.barplot, "cond", "cluster_size", 'odor', palette='flare')
                    # plt.legend()
                    # plt.suptitle(f"allsujet {tf_stats_type} {band} Pxx:{Pxx_type}")
                    # plt.tight_layout()
                    # plt.ylim(0,ylim[tf_stats_type]['allsujet'][band])
                    # plt.show()

                    g = sns.FacetGrid(df_plot, col="cond", height=5, aspect=1)
                    g.map(sns.barplot, "phase", "cluster_size", 'odor', palette='flare')
                    plt.legend()
                    plt.suptitle(f"{group} {tf_stats_type} {band} Pxx:{Pxx_type}")
                    plt.tight_layout()
                    plt.ylim(0,ylim[tf_stats_type][group][band])
                    # plt.show()

                    os.chdir(os.path.join(path_results, 'allplot', tf_mode, 'Pxx'))

                    plt.savefig(f'{group}_{tf_stats_type}_{band}_Pxx{Pxx_type}.jpeg', dpi=150)
                        
                    plt.close('all')
                    gc.collect()


    #### plot repnorep
    for tf_stats_type_i, tf_stats_type in enumerate(['inter', 'intra']):

        for band_i, band in enumerate(band_list):

            for Pxx_type_i, Pxx_type in enumerate(Pxx_modification_list):

                if Pxx_type == 'all':
                    continue

                for cond in conditions:

                    print(tf_stats_type, band)

                    if tf_stats_type == 'intra':

                        if cond == 'FR_CV_1':
                            continue
                    
                        df_plot = df_cluster.query(f"phase != 'all' and group != 'allsujet' and stat_type == '{tf_stats_type}' and cond == '{cond}' and band == '{band}' and Pxx == '{Pxx_type}'")

                    if tf_stats_type == 'inter':
                    
                        df_plot = df_cluster.query(f"phase != 'all' and group != 'allsujet' and stat_type == '{tf_stats_type}' and odor != 'o' and cond == '{cond}' and band == '{band}' and Pxx == '{Pxx_type}'")    

                    # g = sns.FacetGrid(df_plot, col="phase", height=5, aspect=1)
                    # g.map(sns.barplot, "cond", "cluster_size", 'odor', palette='flare')
                    # plt.legend()
                    # plt.suptitle(f"allsujet {tf_stats_type} {band} Pxx:{Pxx_type}")
                    # plt.tight_layout()
                    # plt.ylim(0,ylim[tf_stats_type]['allsujet'][band])
                    # plt.show()

                    g = sns.FacetGrid(df_plot, col="odor", height=5, aspect=1)
                    g.map(sns.barplot, "phase", "cluster_size", 'group', palette='flare')
                    plt.legend()
                    plt.suptitle(f"repnorep {cond} {tf_stats_type} {band} Pxx:{Pxx_type}")
                    plt.tight_layout()
                    # plt.show()

                    os.chdir(os.path.join(path_results, 'allplot', tf_mode, 'Pxx'))

                    plt.savefig(f'repnorep_{tf_stats_type}_{cond}_{band}_Pxx{Pxx_type}.jpeg', dpi=150)
                        
                    plt.close('all')
                    gc.collect()





         
            

def network_TF_IE_power_extraction():

    tf_mode = 'TF'

    print('COMPUTE', flush=True)

    os.chdir(os.path.join(path_precompute, 'allsujet', tf_mode))
    xr_allsujet = xr.open_dataarray(f'allsujet_{tf_mode}.nc')

    group_list = ['allsujet', 'rep', 'no_rep']
    Pxx_modification_list = ['all', 'incre', 'decre']
    time = np.arange(stretch_point_TF)

    band_list = {'theta' : [4, 8], 'alpha' : [8, 12], 'beta' : [12, 30], 'gamma' : [50, 150]}
    respi_phase_list = {'all' : time, 'inspi' : np.arange(time.shape[0]/2).astype('int'), 'expi' : np.arange(time.shape[0]/2, time.shape[0]).astype('int')}


    #### get data
    os.chdir(os.path.join(path_results, 'allplot', tf_mode))

    stats_pxx_dict = {'stat_type' : ['inter', 'intra'], 'group' : group_list, 'band' : list(band_list), 'cond' : conditions, 'odor' : odor_list, 'phase' : list(respi_phase_list), 'chan' : chan_list_eeg}
    stats_pxx_data = np.zeros((2, len(group_list), len(band_list), len(conditions), len(odor_list), len(respi_phase_list), len(chan_list_eeg)))
    stats_pxx_data_p = np.zeros((2, len(group_list), len(band_list), len(conditions), len(odor_list), len(respi_phase_list), len(chan_list_eeg)))

    #tf_stats_type = 'inter'
    for tf_stats_type_i, tf_stats_type in enumerate(['inter', 'intra']):

        #group = group_list[0]
        for group_i, group in enumerate(group_list):

            #odor_i, odor = 1, odor_list[1]
            for odor_i, odor in enumerate(odor_list):

                print(tf_stats_type, group)

                #c, cond = 1, conditions[1]
                for cond_i, cond in enumerate(conditions):

                    #n_chan, chan_name = 0, chan_list_eeg[0]
                    for n_chan, chan_name in enumerate(chan_list_eeg):

                        tf_plot = xr_allsujet.loc[chan_name, group, cond, odor, :, :].values

                        if cond != 'FR_CV_1' and tf_stats_type == 'intra':
                            os.chdir(os.path.join(path_precompute, 'allsujet', tf_mode))

                            pixel_based_distrib = np.median(np.load(f'allsujet_tf_STATS_nchan{n_chan}_{cond}_{odor}_intra.npy'), axis=1)
                            
                        elif odor != 'o' and tf_stats_type == 'inter':
                            os.chdir(os.path.join(path_precompute, 'allsujet', tf_mode))

                            pixel_based_distrib = np.median(np.load(f'allsujet_tf_STATS_nchan{n_chan}_{cond}_{odor}_inter.npy'), axis=1)

                        else:

                            continue
                            
                        tf_stats = get_tf_stats(tf_plot, pixel_based_distrib).astype('bool')

                        #band_i, band = 0, band_list[0]
                        for band_i, band in enumerate(band_list.keys()):

                            #phase_i, phase = 0, 'all'
                            for phase_i, phase in enumerate(respi_phase_list.keys()):

                                tf_stats_chunk = tf_stats[band_list[band][0]:band_list[band][1],respi_phase_list[phase]]
                                tf_plot_chunk = tf_plot[band_list[band][0]:band_list[band][1],respi_phase_list[phase]]

                                _p = tf_stats_chunk.sum() / tf_stats_chunk.size
                                
                                if _p >= 0.05:

                                    stats_pxx_data_p[tf_stats_type_i, group_i, band_i, cond_i, odor_i, phase_i, n_chan] = 1

                                else :

                                    stats_pxx_data_p[tf_stats_type_i, group_i, band_i, cond_i, odor_i, phase_i, n_chan] = 0

                                if tf_stats_chunk.sum() == 0:

                                    stats_pxx_data[tf_stats_type_i, group_i, band_i, cond_i, odor_i, phase_i, n_chan] = 0

                                else:

                                    stats_pxx_data[tf_stats_type_i, group_i, band_i, cond_i, odor_i, phase_i, n_chan] = np.median(tf_plot_chunk[tf_stats_chunk])
                                
    xr_stats_pxx = xr.DataArray(data=stats_pxx_data, dims=stats_pxx_dict.keys(), coords=stats_pxx_dict.values())
    xr_stats_pxx_p = xr.DataArray(data=stats_pxx_data_p, dims=stats_pxx_dict.keys(), coords=stats_pxx_dict.values())
    df_stats_pxx = xr_stats_pxx.to_dataframe(name='Pxx').reset_index(drop=False)
    df_stats_pxx_p = xr_stats_pxx_p.to_dataframe(name='Pxx').reset_index(drop=False)

    #### scale allsujet
    ylim_max = {}
    ylim_min = {}

    for tf_stats_type_i, tf_stats_type in enumerate(['inter', 'intra']):

        ylim_max[tf_stats_type] = {}
        ylim_min[tf_stats_type] = {}

        for group in ['allsujet', 'repnorep', 'rep', 'no_rep']:

            ylim_max[tf_stats_type][group] = {}
            ylim_min[tf_stats_type][group] = {}

            for band_i, band in enumerate(band_list):

                print(tf_stats_type, group, band)

                scale = np.array([])
                std = np.array([])

                #r, odor_i = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    #c, cond = 1, conditions[1]
                    for cond_i, cond in enumerate(conditions):

                        for phase_i, phase in enumerate(respi_phase_list):

                            if phase == 'all':
                                continue

                            if tf_stats_type == 'intra':
                            
                                df_plot = df_stats_pxx.query(f"stat_type == '{tf_stats_type}' and cond != 'FR_CV_1'")

                            if tf_stats_type == 'inter':
                            
                                df_plot = df_stats_pxx.query(f"stat_type == '{tf_stats_type}' and odor != 'o'")    

                            if group == 'allsujet':
                                scale = np.append(scale, df_plot.query(f"group == 'allsujet' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}'")['Pxx'].values.mean())
                                std = np.append(scale, df_plot.query(f"group == 'allsujet' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}'")['Pxx'].values.std())

                            if group == 'rep':
                                scale = np.append(scale, df_plot.query(f"group == 'rep' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}'")['Pxx'].values.mean())
                                std = np.append(scale, df_plot.query(f"group == 'rep' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}'")['Pxx'].values.std())

                            if group == 'no_rep':
                                scale = np.append(scale, df_plot.query(f"group == 'no_rep' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}'")['Pxx'].values.mean())
                                std = np.append(scale, df_plot.query(f"group == 'no_rep' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}'")['Pxx'].values.std())

                            if group == 'repnorep':
                                scale = np.append(scale, df_plot.query(f"group != 'allsujet' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}'")['Pxx'].values.mean())
                                std = np.append(scale, df_plot.query(f"group != 'allsujet' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}'")['Pxx'].values.std())

                if np.isnan(scale).sum() == scale.shape[0]:
                    continue  
                else:
                    ylim_max[tf_stats_type][group][band] = scale[~np.isnan(scale)].max() + std[~np.isnan(std)].mean()
                    ylim_min[tf_stats_type][group][band] = scale[~np.isnan(scale)].min() - std[~np.isnan(std)].mean()

    #### plot allsujet
    pd.DataFrame.iteritems = pd.DataFrame.items

    for group in group_list:

        for tf_stats_type_i, tf_stats_type in enumerate(['inter', 'intra']):

            for band_i, band in enumerate(band_list):

                #brain_region = 'frontal'
                for brain_region in chan_list_lobes.keys():

                    print(brain_region, tf_stats_type, band)

                    if tf_stats_type == 'intra':
                    
                        df_plot = df_stats_pxx.query(f"phase != 'all' and group == '{group}' and stat_type == '{tf_stats_type}' and cond != 'FR_CV_1' and band == '{band}' and chan in {chan_list_lobes[brain_region]}")
                        df_plot = df_plot.query(f"Pxx != 0")

                    if tf_stats_type == 'inter':
                    
                        df_plot = df_stats_pxx.query(f"phase != 'all' and group == '{group}' and stat_type == '{tf_stats_type}' and odor != 'o' and band == '{band}' and chan in {chan_list_lobes[brain_region]}")    
                        df_plot = df_plot.query(f"Pxx != 0")

                    # g = sns.FacetGrid(df_plot, col="phase", height=5, aspect=1)
                    # g.map(sns.barplot, "cond", "cluster_size", 'odor', palette='flare')
                    # plt.legend()
                    # plt.suptitle(f"allsujet {tf_stats_type} {band} Pxx:{Pxx_type}")
                    # plt.tight_layout()
                    # plt.ylim(0,ylim[tf_stats_type]['allsujet'][band])
                    # plt.show()

                    g = sns.FacetGrid(df_plot, col="cond", height=5, aspect=1)
                    g.map(sns.barplot, "phase", "Pxx", 'odor', palette='flare', hue_order=['o', '+', '-'])
                    plt.legend()
                    plt.suptitle(f"{brain_region} {group} {tf_stats_type} {band}")
                    plt.tight_layout()
                    # plt.ylim(ylim_min[tf_stats_type][group][band],ylim_max[tf_stats_type][group][band])
                    # plt.show()

                    os.chdir(os.path.join(path_results, 'allplot', tf_mode, 'Pxx', 'cluster_value'))

                    plt.savefig(f'{brain_region}_{group}_{tf_stats_type}_{band}.jpeg', dpi=150)
                        
                    plt.close('all')
                    gc.collect()

    #### just for CO2
                    
    for tf_stats_type_i, tf_stats_type in enumerate(['inter', 'intra']):

        for band_i, band in enumerate(band_list):

            #brain_region = 'frontal'
            for brain_region in chan_list_lobes.keys():

                print(brain_region, tf_stats_type, band)

                if tf_stats_type == 'intra':
                
                    df_plot = df_stats_pxx.query(f"phase != 'all' and cond == 'CO2' and stat_type == '{tf_stats_type}' and cond != 'FR_CV_1' and band == '{band}' and chan in {chan_list_lobes[brain_region]}")
                    df_plot = df_plot.query(f"Pxx != 0")

                if tf_stats_type == 'inter':
                
                    df_plot = df_stats_pxx.query(f"phase != 'all' and cond == 'CO2' and stat_type == '{tf_stats_type}' and odor != 'o' and band == '{band}' and chan in {chan_list_lobes[brain_region]}")    
                    df_plot = df_plot.query(f"Pxx != 0")

                # g = sns.FacetGrid(df_plot, col="phase", height=5, aspect=1)
                # g.map(sns.barplot, "cond", "cluster_size", 'odor', palette='flare')
                # plt.legend()
                # plt.suptitle(f"allsujet {tf_stats_type} {band} Pxx:{Pxx_type}")
                # plt.tight_layout()
                # plt.ylim(0,ylim[tf_stats_type]['allsujet'][band])
                # plt.show()

                g = sns.FacetGrid(df_plot, col="group", height=5, aspect=1)
                g.map(sns.barplot, "phase", "Pxx", 'odor', palette='flare', hue_order=['o', '+', '-'])
                plt.legend()
                plt.suptitle(f"{brain_region} CO2 {tf_stats_type} {band}")
                plt.tight_layout()
                # plt.ylim(ylim_min[tf_stats_type][group][band],ylim_max[tf_stats_type][group][band])
                # plt.show()

                os.chdir(os.path.join(path_results, 'allplot', tf_mode, 'Pxx', 'cluster_value', 'CO2'))

                plt.savefig(f'{brain_region}_CO2_{tf_stats_type}_{band}.jpeg', dpi=150)
                    
                plt.close('all')
                gc.collect()

    #### plot topo
    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    tf_stats_type = 'intra'

    min, max = df_stats_pxx.query(f"stat_type == '{tf_stats_type}'")['Pxx'].min(), df_stats_pxx.query(f"stat_type == '{tf_stats_type}'")['Pxx'].max()

    vlim = np.abs([min, max]).max()

    #### plot
    for band in band_list.keys():

        for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):    

            fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(15,15))

            for odor_i, odor in enumerate(['o', '+', '-']):

                x_i = -1

                for phase_i, phase in enumerate(['inspi', 'expi']):

                    for group_i, group in enumerate(group_list):

                        x_i += 1

                        print('intra', phase, odor, cond)

                        mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')
                        data_topoplot = np.zeros((len(chan_list_eeg)))

                        #chan_i, chan = 0, chan_list_eeg[0]
                        for chan_i, chan in enumerate(chan_list_eeg):

                            data_topoplot[chan_i] = df_stats_pxx.query(f"odor == '{odor}' and phase == '{phase}' and group == '{group}' and stat_type == '{tf_stats_type}' and cond == '{cond}' and band == '{band}' and chan == '{chan}'")['Pxx'].values[0]

                            # _p = df_stats_pxx_p.query(f"odor == '{odor}' and phase == '{phase}' and group == '{group}' and stat_type == '{tf_stats_type}' and cond == '{cond}' and band == '{band}' and chan == '{chan}'")['Pxx'].values[0]
                            # mask_signi[chan_i] = _p.astype('bool')

                            if data_topoplot[chan_i] != 0:
                                mask_signi[chan_i] = True
                            else:
                                mask_signi[chan_i] = False
                                

                        ax = axs[odor_i, x_i]

                        ax.set_title(f"{odor} {phase} {group}")

                        # mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                        #                 mask=mask_signi, mask_params=mask_params, vlim=(0, vlim), cmap='seismic')
                        
                        mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                        mask_params=mask_params, vlim=(0, vlim), cmap='seismic')

                plt.tight_layout()

                plt.suptitle(f'{cond} {band} INTRA {np.round(vlim,2)}')

                os.chdir(os.path.join(path_results, 'allplot', 'TF', 'Pxx', 'topoplot'))
                fig.savefig(f"{band}_intra_{cond}.jpeg")

                plt.close('all')
                
                # plt.show()









################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #### MAIN WORKFLOW

    # compute_topoplot_IE_network_SUM(stretch=False)
    compute_topoplot_IE_network_SUM(stretch=True)

    # compute_topoplot_IE_network_repnorep_SUM(stretch=False)
    compute_topoplot_IE_network_repnorep_SUM(stretch=True)






    #### ACCESORY FUNCTION

    compute_topoplot_IE_network()

    compute_topoplot_IE_network_repnorep()

    network_TF_IE()

    network_TF_IE_power_extraction()

