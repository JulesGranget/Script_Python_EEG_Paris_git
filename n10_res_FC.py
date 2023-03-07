
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib import cm
import xarray as xr
import joblib
import mne_connectivity
import copy

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False





########################################
######## ANALYSIS FUNCTIONS ########
########################################



#dfc_data = allband_data[band][cond][odor_i].loc[cf_metric,:,phase,:].values
def dfc_pairs_to_mat(dfc_data, allpairs):

    mat_dfc = np.zeros(( len(chan_list_eeg_fc), len(chan_list_eeg_fc) ))

    #x_i, x_name = 0, chan_list_eeg_fc[0]
    for x_i, x_name in enumerate(chan_list_eeg_fc):
        #y_i, y_name = 2, chan_list_eeg_fc[2]
        for y_i, y_name in enumerate(chan_list_eeg_fc):
            if x_name == y_name:
                continue

            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'
            
            x = dfc_data[allpairs == pair_to_find, :]
            x_rev = dfc_data[allpairs == pair_to_find_rev, :]

            x_mean_pair = np.median(np.concatenate((x, x_rev), axis=0), axis=0)

            x_mean_pair_nfrex = np.median(x_mean_pair, axis=0)

            #### extract value dfc
            mat_dfc[x_i, y_i] = x_mean_pair_nfrex

    if debug:

        plt.matshow(mat_dfc)
        plt.show()

    return mat_dfc




def plot_all_verif(allband_data, allpairs, cond_to_load, band_prep):

    os.chdir(os.path.join(path_results, sujet, 'DFC', 'verif'))

    #cf_metric_i, cf_metric = 0, 'ispc'
    for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):
        #pair_i = 200
        for pair_i in range(allpairs.shape[0]):

            if pair_i % 200 == 0:

                fig, axs = plt.subplots(ncols=len(cond_to_load), nrows=3, figsize=(15,15))

                plt.suptitle(f'{cf_metric}_pair{pair_i}', color='k')

                #band = 'theta'
                for r, band in enumerate(freq_band_dict_FC[band_prep]):
                    #cond = 'AC'
                    for c, cond in enumerate(cond_to_load):
                    
                        ax = axs[r,c]
                        
                        if cond == 'AL':

                            _med = allband_data[band][cond][cf_metric_i,pair_i,:,:].median(axis=0).median(axis=0)
                            _mad = get_mad(allband_data[band][cond][cf_metric_i,pair_i,:,:].median(axis=0), axis=0)
                            ax.plot(_med)

                        else:

                            ax.plot(allband_data[band][cond][cf_metric_i,pair_i,:,:].mean(axis=0))

                        if r == 0:
                            ax.set_title(f'{cond}')
                        if c == 0:
                            ax.set_ylabel(f'{band}')
                        # plt.show()

                plt.savefig(f'cf_spectre_pair{pair_i}_{cf_metric}_{band_prep}.png')
                plt.close('all')

    #### select pairs to plot
    pair_to_plot = []

    for pair_i, pair in enumerate(pair_unique):

        if pair_i % 30 == 0:
            pair_to_plot.append(pair)

    #x_i, x_name = 0, roi_in_data[0]
    for x_i, x_name in enumerate(roi_in_data):
        #y_i, y_name = 2, roi_in_data[2]
        for y_i, y_name in enumerate(roi_in_data):
            if x_name == y_name:
                continue

            pair_to_find = f'{x_name}-{y_name}'
            pair_to_find_rev = f'{y_name}-{x_name}'

            if pair_to_find in pair_to_plot or pair_to_find_rev in pair_to_plot:

                try:
                    pair_i = np.where(pair_unique == pair_to_find)[0][0]
                except:
                    pair_i = np.where(pair_unique == pair_to_find_rev)[0][0]

                #cf_metric_i, cf_metric = 0, 'ispc'
                for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

                    fig, axs = plt.subplots(ncols=len(cond_to_load), nrows=len(freq_band_dict_FC_function[band_prep]), figsize=(15,15))

                    #band = 'theta'
                    for r, band in enumerate(freq_band_dict_FC_function[band_prep]):
                        #cond = 'RD_SV'
                        for c, cond in enumerate(cond_to_load):

                            fc_to_plot = allband_data[band][cond][cf_metric_i,pair_i,:,:]
                            
                            ax = axs[r,c]
                            
                            ax.plot(fc_to_plot.mean(axis=0), label='mean')
                            ax.plot(fc_to_plot.mean(axis=0) + fc_to_plot.std(axis=0), color='r', label='1SD')
                            ax.plot(fc_to_plot.mean(axis=0) - fc_to_plot.std(axis=0), color='r', label='1SD')
                            ax.plot([np.percentile(fc_to_plot, 10)]*fc_to_plot.shape[-1], linestyle='--', color='g', label='10p')
                            ax.plot([np.percentile(fc_to_plot, 25)]*fc_to_plot.shape[-1], linestyle='-.', color='g', label='25p')
                            ax.plot([np.percentile(fc_to_plot, 40)]*fc_to_plot.shape[-1], linestyle=':', color='g', label='40p')
                            ax.plot([np.percentile(fc_to_plot, 60)]*fc_to_plot.shape[-1], linestyle=':', color='g', label='60p')
                            ax.plot([np.percentile(fc_to_plot, 75)]*fc_to_plot.shape[-1], linestyle='-.', color='g', label='75p')
                            ax.plot([np.percentile(fc_to_plot, 90)]*fc_to_plot.shape[-1], linestyle='--', color='g', label='90p')

                            if r == 0:
                                ax.set_title(f'{cond}')
                            if c == 0:
                                ax.set_ylabel(f'{band}')

                    plt.suptitle(f'{cf_metric}_{pair_to_find}_count : {fc_to_plot.shape[0]}', color='k')

                    ax.legend()

                    # plt.show()

                    plt.savefig(f'cf_mean_allpair{pair_i}_{cf_metric}_{band_prep}.png')
                    plt.close('all')
                    
    #### export mat count pairs
    fig, ax = plt.subplots(figsize=(15,15))

    cax = ax.matshow(mat_count_pairs)

    fig.colorbar(cax, ax=ax)

    ax.set_yticks(np.arange(roi_in_data.shape[0]))
    ax.set_yticklabels(roi_in_data)

    # plt.show()
    fig.savefig(f'{sujet}_MAT_COUNT.png')
    plt.close('all')

     



















################################
######## SAVE FIG ########
################################


#FR_CV_normalized = False
def process_fc_res(sujet, FR_CV_normalized, plot_circle_dfc=False, plot_verif=False):

    print(f'######## DFC ########')

    phase_list = ['whole', 'inspi', 'expi']
    phase_plot = 'whole'

    if FR_CV_normalized:
        cond_to_plot = conditions[1:]
    else:
        cond_to_plot = conditions    

    cond_to_load = conditions

    #### LOAD DATA ####

    #### get params
    os.chdir(os.path.join(path_precompute, sujet, 'FC'))

    file_to_load = [i for i in os.listdir() if ( i.find('allpairs') != -1 and i.find(band_name_fc_dfc[0]) != -1 and i.find('bi') == -1)]
    
    cf_metrics_list = xr.open_dataarray(file_to_load[0])['mat_type'].values

    #band_prep = 'wb'
    for band_prep in band_prep_list:

        #### load data 
        allband_data = {}

        #band = 'theta'
        for band in freq_band_dict_FC[band_prep]:

            allband_data[band] = {}

            #cond = 'AL'
            for cond in cond_to_load:

                allband_data[band][cond] = {}

                for odor_i in odor_list:

                    file_to_load = f'{sujet}_FC_wpli_ispc_{cond}_{odor_i}_{band}_allpairs.nc'
                    
                    allband_data[band][cond][odor_i] = xr.open_dataarray(file_to_load)
                    allpairs = xr.open_dataarray(file_to_load)['pairs'].data

        #### plot verif
        if plot_verif:

            plot_all_verif(allband_data, allpairs, cond_to_load, band_prep)
                
        #### mean
        allband_dfc_phase = {}

        #band = 'theta'
        for band in freq_band_dict_FC[band_prep]:

            print(band)

            allband_dfc_phase[band] = {}

            #cond = 'FR_CV_1'
            for cond in cond_to_load:

                allband_dfc_phase[band][cond] = {}

                #phase = 'whole'
                for phase_i, phase in enumerate(phase_list):

                    allband_dfc_phase[band][cond][phase] = {}

                    mat_fc_i = np.zeros((2, len(chan_list_eeg), len(chan_list_eeg)))

                    #cf_metric_i, cf_metric = 0, 'ispc'
                    for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

                        for odor_i in odor_list:
                            
                            mat_fc_i[cf_metric_i, :, :] = dfc_pairs_to_mat(allband_data[band][cond][odor_i].loc[cf_metric,:,phase,:].values, allpairs)

                            allband_dfc_phase[band][cond][phase][odor_i] = mat_fc_i

        if debug:

            for band in freq_band_dict_FC[band_prep]:

                for cond in cond_to_load:

                    for odor_i in odor_list:

                        for cf_metric_i, cf_metric in enumerate(['ispc', 'wpli']):

                            plt.matshow(allband_dfc_phase[band][cond][phase_plot][odor_i][cf_metric_i,:,:])
                            plt.show()

        del allband_data

        #### normalization
        if FR_CV_normalized:
                
            #band = 'theta'
            for band in freq_band_dict_FC[band_prep]:
                #cond = 'RD_SV'
                for cond in cond_to_plot:

                    #phase = 'whole'
                    for phase_i, phase in enumerate(phase_list):
                        #cf_metric_i, cf_metric = 0, 'ispc'

                        for odor_i in odor_list:

                            allband_dfc_phase[band][cond][phase][odor_i] = allband_dfc_phase[band][cond][phase][odor_i] - allband_dfc_phase[band]['FR_CV_1'][phase][odor_i]

        #### identify scales
        scales_abs = {}

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            scales_abs[mat_type] = {}

            #band = 'theta'
            for band in freq_band_dict_FC[band_prep]:

                max_list = np.array(())

                scales_abs[mat_type][band] = {}

                #cond = 'RD_SV'
                for cond in cond_to_plot:

                    for odor_i in odor_list:

                        for phase_i, phase in enumerate(phase_list):

                            max_list = np.append(max_list, np.abs(allband_dfc_phase[band][cond][phase][odor_i][mat_type_i,:,:].min()))
                            max_list = np.append(max_list, allband_dfc_phase[band][cond][phase][odor_i][mat_type_i,:,:].max())

                scales_abs[mat_type][band]['max'] = max_list.max()

                if FR_CV_normalized:
                    scales_abs[mat_type][band]['min'] = -max_list.max()
                else:
                    scales_abs[mat_type][band]['min'] = 0    

        #### thresh on previous plot
        percentile_thresh_up = 99
        percentile_thresh_down = 1

        mat_dfc_clean = copy.deepcopy(allband_dfc_phase)

        for mat_type_i, mat_type in enumerate(cf_metrics_list):

            #band = 'theta'
            for band in freq_band_dict_FC[band_prep]:

                for cond in cond_to_plot:

                    for phase_i, phase in enumerate(phase_list):

                        for odor_i in odor_list:

                            thresh_up = np.percentile(allband_dfc_phase[band][cond][phase][odor_i][mat_type_i,:,:].reshape(-1), percentile_thresh_up)
                            thresh_down = np.percentile(allband_dfc_phase[band][cond][phase][odor_i][mat_type_i,:,:].reshape(-1), percentile_thresh_down)

                            for x in range(mat_dfc_clean[band][cond][phase][odor_i][mat_type_i,:,:].shape[1]):
                                for y in range(mat_dfc_clean[band][cond][phase][odor_i][mat_type_i,:,:].shape[1]):
                                    if mat_type_i == 0:
                                        if mat_dfc_clean[band][cond][phase][odor_i][mat_type_i,x,y] < thresh_up:
                                            mat_dfc_clean[band][cond][phase][odor_i][mat_type_i,x,y] = 0
                                    else:
                                        if (mat_dfc_clean[band][cond][phase][odor_i][mat_type_i,x,y] < thresh_up) & (mat_dfc_clean[band][cond][phase][odor_i][mat_type_i,x,y] > thresh_down):
                                            mat_dfc_clean[band][cond][phase][odor_i][mat_type_i,x,y] = 0


        ######## PLOT ########


        #### go to results
        os.chdir(os.path.join(path_results, sujet, 'FC'))

        if FR_CV_normalized:
            plot_color = cm.seismic
        else:
            plot_color = cm.YlGn

        #### RAW

        #band = 'theta'
        for band in freq_band_dict_FC[band_prep]:

            #mat_type_i, mat_type = 0, 'ispc'
            for mat_type_i, mat_type in enumerate(cf_metrics_list):
            
                ######## NO THRESH ########

                #### mat plot raw

                fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_to_plot), figsize=(15,15))

                plt.suptitle(f'{mat_type} {band}')

                #cond = 'MECA'
                for c, cond in enumerate(cond_to_plot):

                    for r, odor_i in enumerate(odor_list):
                        
                        ax = axs[r, c]

                        if c == 0:
                            ax.set_ylabel(odor_i)
                        if r == 0:
                            ax.set_title(f'{cond}')
                        
                        cax = ax.matshow(allband_dfc_phase[band][cond][phase_plot][odor_i][mat_type_i,:,:], vmin=scales_abs[mat_type][band]['min'], vmax=scales_abs[mat_type][band]['max'], cmap=plot_color)

                        if c == len(cond_to_plot)-1:
                            fig.colorbar(cax, ax=ax)

                        ax.set_yticks(np.arange(len(chan_list_eeg_fc)))
                        ax.set_yticklabels(chan_list_eeg_fc)

                # plt.show()

                if FR_CV_normalized:
                    fig.savefig(f'MAT_{mat_type}_{band}_norm_{band_prep}.png')
                else:
                    fig.savefig(f'MAT_{mat_type}_{band}_{band_prep}.png')
                    
                plt.close('all')

                #### circle plot RAW
                    
                if plot_circle_dfc:
                    
                    nrows, ncols = len(freq_band_dict_FC[band_prep]), len(phase_list)
                    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))

                    for r, band in enumerate(freq_band_dict_FC[band_prep]):

                        for c, phase in enumerate(phase_list):

                            mne_connectivity.viz.plot_connectivity_circle(allband_dfc_phase[band][cond][phase][odor_i][mat_type_i,:,:], node_names=chan_list_eeg_fc, n_lines=None, 
                                                        title=f'{band} {phase}', show=False, padding=7, ax=axs[r, c],
                                                        vmin=scales_abs[mat_type][band]['min'], vmax=scales_abs[mat_type][band]['max'], colormap=plot_color, facecolor='w', 
                                                        textcolor='k')

                    plt.suptitle(f'{cond}_{mat_type}', color='k')
                    
                    fig.set_figheight(10)
                    fig.set_figwidth(12)
                    # fig.show()

                    if FR_CV_normalized:
                        fig.savefig(f'CIRCLE_{mat_type}_{cond}_norm_{band_prep}.png')
                    else:
                        fig.savefig(f'CIRCLE_{mat_type}_{cond}_{band_prep}.png')
                    
                    plt.close('all')


                ######## THRESH ########

                #### mat plot raw

                fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_to_plot), figsize=(15,15))

                plt.suptitle(f'{mat_type} {band}')

                #cond = 'MECA'
                for c, cond in enumerate(cond_to_plot):

                    for r, odor_i in enumerate(odor_list):
                        
                        ax = axs[r, c]

                        if c == 0:
                            ax.set_ylabel(odor_i)
                        if r == 0:
                            ax.set_title(f'{cond}')
                        
                        cax = ax.matshow(mat_dfc_clean[band][cond][phase_plot][odor_i][mat_type_i,:,:], vmin=scales_abs[mat_type][band]['min'], vmax=scales_abs[mat_type][band]['max'], cmap=plot_color)

                        if c == len(cond_to_plot)-1:
                            fig.colorbar(cax, ax=ax)

                        ax.set_yticks(np.arange(len(chan_list_eeg_fc)))
                        ax.set_yticklabels(chan_list_eeg_fc)

                # plt.show()

                if FR_CV_normalized:
                    fig.savefig(f'MAT_THRESH_{mat_type}_{band}_norm_{band_prep}.png')
                else:
                    fig.savefig(f'MAT_THRESH_{mat_type}_{band}_{band_prep}.png')
                    
                plt.close('all')

                #### circle plot RAW
                    
                if plot_circle_dfc:
                    
                    nrows, ncols = len(freq_band_dict_FC[band_prep]), len(phase_list)
                    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True))

                    for r, band in enumerate(freq_band_dict_FC[band_prep]):

                        for c, phase in enumerate(phase_list):

                            mne_connectivity.viz.plot_connectivity_circle(mat_dfc_clean[band][cond][phase][odor_i][mat_type_i,:,:], node_names=chan_list_eeg_fc, n_lines=None, 
                                                        title=f'{band} {phase}', show=False, padding=7, ax=axs[r, c],
                                                        vmin=scales_abs[mat_type][band]['min'], vmax=scales_abs[mat_type][band]['max'], colormap=plot_color, facecolor='w', 
                                                        textcolor='k')

                    plt.suptitle(f'{cond}_{mat_type}', color='k')
                    
                    fig.set_figheight(10)
                    fig.set_figwidth(12)
                    # fig.show()

                    if FR_CV_normalized:
                        fig.savefig(f'CIRCLE_{mat_type}_{cond}_norm_{band_prep}.png')
                    else:
                        fig.savefig(f'CIRCLE_{mat_type}_{cond}_{band_prep}.png')
                    
                    plt.close('all')










################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        for FR_CV_normalized in [True, False]:

            process_fc_res(sujet, FR_CV_normalized)
            # execute_function_in_slurm_bash('n10_res_FC', 'process_fc_res', [sujet, FR_CV_normalized])

    