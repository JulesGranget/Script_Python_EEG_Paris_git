
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False










################################
######## COMPUTE STATS ########
################################




#chan_i, chan = 0, chan_list_eeg[0]
def precompute_tf_STATS_allsujet(chan_i, chan):
    
    group_list = ['allsujet', 'rep', 'norep']

    ######## LOAD DATA ########
    print(f'LOAD DATA {chan}')

    tf_conv_allsujet = np.zeros((len(sujet_list), len(conditions), len(odor_list), nfrex, stretch_point_TF))

    os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'conv'))
        
    for sujet_i, sujet in enumerate(sujet_list):

        tf_conv_allsujet[sujet_i] = np.load(f'{sujet}_tf_conv_allcond.npy')[:,:,chan_i]

    xr_dict = {'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'nfrex' : np.arange(nfrex), 'phase' : np.arange(stretch_point_TF)}
    xr_tf_conv = xr.DataArray(data=tf_conv_allsujet, dims=xr_dict.keys(), coords=xr_dict.values())



    ######## INTRA ########
    os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'stats'))

    if os.path.exists(f'tf_STATS_chan{chan}_intra.nc'):
        print(f'INTRA {chan} ALREADY COMPUTED', flush=True)

    else:
    
        print(f'#### COMPUTE TF STATS INTRA chan:{chan} ####', flush=True)

        cond_sel = ['MECA', 'CO2', 'FR_CV_2']
        odor_sel = odor_list

        data_intra = np.zeros((len(group_list), len(cond_sel), len(odor_sel), nfrex, stretch_point_TF))

        #group_i, group = 2, group_list[2]
        for group_i, group in enumerate(group_list):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_sel):
                
                #cond_i, cond = 0, cond_sel[0]
                for cond_i, cond in enumerate(cond_sel):

                    print(f'COMPUTE {group} {cond} {odor}', flush=True)

                    if group == 'allsujet':
                        data_baseline = xr_tf_conv.loc[:, 'FR_CV_1', odor, :, :].values
                        data_cond = xr_tf_conv.loc[:, cond, odor, :, :].values
                    elif group == 'rep':
                        data_baseline = xr_tf_conv.loc[sujet_best_list_rev, 'FR_CV_1', odor, :, :].values
                        data_cond = xr_tf_conv.loc[sujet_best_list_rev, cond, odor, :, :].values
                    elif group == 'norep':
                        data_baseline = xr_tf_conv.loc[sujet_no_respond_rev, 'FR_CV_1', odor, :, :].values
                        data_cond = xr_tf_conv.loc[sujet_no_respond_rev, cond, odor, :, :].values

                    mask = get_permutation_cluster_2d(data_baseline, data_cond, ERP_n_surrogate,
                                                        stat_design='within', mode_grouped=mode_grouped_ERP_STATS, 
                                                        mode_generate_surr=mode_generate_surr_ERP_STATS, percentile_thresh=percentile_thresh_ERP_STATS,
                                                        mode_select_thresh=mode_select_thresh_ERP_STATS, size_thresh_alpha=size_thresh_alpha_ERP_STATS)

                    #### verif tf
                    if debug:

                        fig, ax = plt.subplots()
                        ax.pcolormesh(np.median(data_cond - data_baseline, axis=0), shading='gouraud', cmap=plt.get_cmap('seismic'))
                        ax.contour(mask, levels=0, colors='g')
                        plt.show()

                    data_intra[group_i, cond_i, odor_i, :, :] = mask

        #### save

        print(f'SAVE', flush=True)

        xr_dict = {'group' : group_list, 'cond' : cond_sel, 'odor' : odor_sel, 'nfrex' : np.arange(nfrex), 'phase' : np.arange(stretch_point_TF)}
        xr_intra = xr.DataArray(data=data_intra, dims=xr_dict.keys(), coords=xr_dict.values())

        os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'stats'))

        xr_intra.to_netcdf(f'tf_STATS_chan{chan}_intra.nc')


        
    ######## INTER ########
    os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'stats'))

    if os.path.exists(f'tf_STATS_chan{chan}_inter.nc'):
        print(f'INTER {chan} ALREADY COMPUTED', flush=True)

    else:

        print(f'#### COMPUTE TF STATS INTER chan:{chan} ####', flush=True)

        cond_sel = conditions
        odor_sel = ['+', '-']

        data_inter = np.zeros((len(group_list), len(cond_sel), len(odor_sel), nfrex, stretch_point_TF))

        for group_i, group in enumerate(group_list):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_sel):
                
                #cond = 'MECA'
                for cond_i, cond in enumerate(cond_sel):

                    print(f'COMPUTE {group} {cond} {odor}', flush=True)

                    if group == 'allsujet':
                        data_baseline = xr_tf_conv.loc[:, cond, 'o', :, :].values
                        data_cond = xr_tf_conv.loc[:, cond, odor, :, :].values
                    elif group == 'rep':
                        data_baseline = xr_tf_conv.loc[sujet_best_list_rev, cond, 'o', :, :].values
                        data_cond = xr_tf_conv.loc[sujet_best_list_rev, cond, odor, :, :].values
                    elif group == 'norep':
                        data_baseline = xr_tf_conv.loc[sujet_no_respond_rev, cond, 'o', :, :].values
                        data_cond = xr_tf_conv.loc[sujet_no_respond_rev, cond, odor, :, :].values

                    mask = get_permutation_cluster_2d(data_baseline, data_cond, ERP_n_surrogate,
                                                        stat_design='within', mode_grouped=mode_grouped_ERP_STATS, 
                                                        mode_generate_surr=mode_generate_surr_ERP_STATS, percentile_thresh=percentile_thresh_ERP_STATS,
                                                        mode_select_thresh=mode_select_thresh_ERP_STATS, size_thresh_alpha=size_thresh_alpha_ERP_STATS)

                    #### verif tf
                    if debug:

                        fig, ax = plt.subplots()
                        ax.pcolormesh(np.median(data_cond - data_baseline, axis=0) , shading='gouraud', cmap=plt.get_cmap('seismic'))
                        ax.contour(mask, levels=0, colors='g')

                        plt.show()

                    data_inter[group_i, cond_i, odor_i, :, :] = mask

        #### save

        print(f'SAVE', flush=True)

        xr_dict = {'group' : group_list, 'cond' : cond_sel, 'odor' : odor_sel, 'nfrex' : np.arange(nfrex), 'phase' : np.arange(stretch_point_TF)}
        xr_inter = xr.DataArray(data=data_inter, dims=xr_dict.keys(), coords=xr_dict.values())

        os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'stats'))

        xr_inter.to_netcdf(f'tf_STATS_chan{chan}_inter.nc')


    ######## REPNOREP ########
    os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'stats'))

    if os.path.exists(f'tf_STATS_chan{chan}_repnorep.nc'):
        print(f'REPNOREP {chan} ALREADY COMPUTED', flush=True)

    else:

        print(f'#### COMPUTE TF STATS REPNOREP chan:{chan} ####', flush=True)

        cond_sel = conditions
        odor_sel = odor_list

        data_repnorep = np.zeros((len(cond_sel), len(odor_sel), nfrex, stretch_point_TF))

        #odor_i, odor = 0, odor_list[0]
        for odor_i, odor in enumerate(odor_sel):
            
            #cond = 'MECA'
            for cond_i, cond in enumerate(cond_sel):

                print(f'COMPUTE {cond} {odor}', flush=True)

                data_baseline = xr_tf_conv.loc[sujet_no_respond_rev, cond, odor, :, :].values
                data_cond = xr_tf_conv.loc[sujet_best_list_rev, cond, odor, :, :].values

                mask = get_permutation_cluster_2d(data_baseline, data_cond, ERP_n_surrogate,
                                                    stat_design='between', mode_grouped=mode_grouped_ERP_STATS, 
                                                    mode_generate_surr=mode_generate_surr_ERP_STATS, percentile_thresh=percentile_thresh_ERP_STATS,
                                                    mode_select_thresh=mode_select_thresh_ERP_STATS, size_thresh_alpha=size_thresh_alpha_ERP_STATS)

                #### verif tf
                if debug:

                    fig, ax = plt.subplots()
                    ax.pcolormesh(np.median(data_cond - data_baseline, axis=0) , shading='gouraud', cmap=plt.get_cmap('seismic'))
                    ax.contour(mask, levels=0, colors='g')

                    plt.show()

                data_repnorep[cond_i, odor_i, :, :] = mask

        #### save
        print(f'SAVE', flush=True)

        xr_dict = {'cond' : cond_sel, 'odor' : odor_sel, 'nfrex' : np.arange(nfrex), 'phase' : np.arange(stretch_point_TF)}
        xr_repnorep = xr.DataArray(data=data_repnorep, dims=xr_dict.keys(), coords=xr_dict.values())

        os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'stats'))

        xr_repnorep.to_netcdf(f'tf_STATS_chan{chan}_repnorep.nc')






########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    execute_function_in_slurm_bash('n17bis_precompute_allsujet_TF_STATS', 'precompute_tf_STATS_allsujet', [[chan_i, chan] for chan_i, chan in enumerate(chan_list_eeg)], n_core=n_core, mem='15G')
    # sync_folders__push_to_crnldata()
    



        







