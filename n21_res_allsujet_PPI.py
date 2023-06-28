
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
from sklearn import linear_model
import gc
import xarray as xr

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False





################################
######## ERP ANALYSIS ########
################################


def compute_ERP(cond_dys):

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

    xr_dict = {'sujet' : sujet_list_erp, 'cond' : cond_dys, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'time' : time_vec}
    xr_data = xr.DataArray(data=np.zeros((len(sujet_list_erp), len(cond_dys), len(odor_list), len(chan_list_eeg), time_vec.shape[0])), dims=xr_dict.keys(), coords=xr_dict.values())

    for sujet_i, sujet in enumerate(sujet_list_erp):

        print_advancement(sujet_i, len(sujet_list_erp), [25, 50, 75])

        respfeatures = load_respfeatures(sujet)

        #cond = 'MECA'
        for cond in cond_dys:

            #odor = odor_list[0]
            for odor in odor_list:

                #### load
                data = load_data_sujet(sujet, cond, odor)
                data = data[:len(chan_list_eeg),:]

                respfeatures_i = respfeatures[cond][odor]
                inspi_starts = respfeatures_i.query(f"select == 1")['inspi_index'].values

                #nchan_i, nchan = 0, chan_list_eeg[0]
                for nchan_i, nchan in enumerate(chan_list_eeg):

                    #### chunk
                    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                    data_stretch = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

                    x = data[nchan_i,:]

                    x_mean, x_std = x.mean(), x.std()

                    for start_i, start_time in enumerate(inspi_starts):

                        t_start = int(start_time + t_start_PPI*srate)
                        t_stop = int(start_time + t_stop_PPI*srate)

                        data_stretch[start_i, :] = (x[t_start: t_stop] - x_mean) / x_std

                    if debug:

                        time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                        for inspi_i, _ in enumerate(inspi_starts):

                            plt.plot(time_vec, data_stretch[inspi_i, :], alpha=0.3)

                        plt.vlines(0, ymax=data_stretch.max(), ymin=data_stretch.min(), color='k')
                        plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                        plt.plot(time_vec, data_stretch.mean(axis=0), color='r')
                        plt.title(f'{cond} {odor} : {data_stretch.shape[0]}')
                        plt.show()

                    #### clean
                    data_stretch_clean = data_stretch[~((data_stretch >= 3) | (data_stretch <= -3)).any(axis=1),:]

                    if debug:

                        time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                        for inspi_i, _ in enumerate(inspi_starts):

                            plt.plot(time_vec, data_stretch_clean[inspi_i, :], alpha=0.3)

                        plt.vlines(0, ymax=data_stretch_clean.max(), ymin=data_stretch_clean.min(), color='k')
                        plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                        plt.plot(time_vec, data_stretch_clean.mean(axis=0), color='r')
                        plt.title(f'{cond} {odor} : {data_stretch_clean.shape[0]}')
                        plt.show()

                    xr_data.loc[sujet, cond, odor, nchan, :] = data_stretch_clean.mean(axis=0)

    return xr_data
            








def compute_lm_on_ERP(xr_data, cond_dys):

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    sujet_no_respond = np.array([sujet for sujet in sujet_list_erp if sujet not in sujet_best_list_erp])

    sujet_group = ['allsujet', 'rep', 'non_rep']
    data_type = ['coeff', 'pval', 'slope']

    xr_dict = {'sujet' : sujet_list_erp, 'cond' : cond_dys, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'type' : data_type}
    xr_lm_data = xr.DataArray(data=np.zeros((len(sujet_list_erp), len(cond_dys), len(odor_list), len(chan_list_eeg), len(data_type))), dims=xr_dict.keys(), coords=xr_dict.values())

    time_vec_lm = np.arange(PPI_lm_start, PPI_lm_stop, 1/srate)
    xr_dict = {'group' : sujet_group, 'cond' : cond_dys, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'time' : time_vec_lm}
    xr_lm_pred = xr.DataArray(data=np.zeros((len(sujet_group), len(cond_dys), len(odor_list), len(chan_list_eeg), len(time_vec_lm))), dims=xr_dict.keys(), coords=xr_dict.values())

    xr_dict = {'group' : sujet_group, 'cond' : cond_dys, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'type' : data_type}
    xr_lm_pred_coeff = xr.DataArray(data=np.zeros((len(sujet_group), len(cond_dys), len(odor_list), len(chan_list_eeg), len(data_type))), dims=xr_dict.keys(), coords=xr_dict.values())

    #sujet_i, sujet = 0, sujet_list_erp[0]
    for sujet_i, sujet in enumerate(sujet_list_erp):

        print_advancement(sujet_i, len(sujet_list_erp), [25, 50, 75])

        #cond = 'MECA'
        for cond in cond_dys:

            #odor = odor_list[0]
            for odor in odor_list:

                for nchan_i, nchan in enumerate(chan_list_eeg):
                
                    data = xr_data.loc[sujet, cond, odor, nchan, :].values
                    time_vec = np.linspace(t_start_PPI, t_stop_PPI, data.shape[0])
                    time_vec_mask = (time_vec > PPI_lm_start) & (time_vec < PPI_lm_stop)
                    Y = data[time_vec_mask].reshape(-1,1)
                    X = time_vec[time_vec_mask].reshape(-1,1)

                    lm = linear_model.LinearRegression()

                    lm.fit(X, Y)

                    Y_pred = lm.predict(X)

                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X.reshape(-1), Y.reshape(-1))

                    xr_lm_data.loc[sujet, cond, odor, nchan, 'coeff'] = np.round(r_value**2, 5)
                    xr_lm_data.loc[sujet, cond, odor, nchan, 'pval'] = np.round(p_value, 5)
                    xr_lm_data.loc[sujet, cond, odor, nchan, 'slope'] = np.round(slope, 5)

                    #### verif
                    if debug:
            
                        plt.plot(X, Y)
                        plt.plot(X, Y_pred, color="b", linewidth=3)

                        plt.show()

    #### lm pred for mean
    #group = sujet_group[0]
    for group in sujet_group:

        #cond = 'MECA'
        for cond in cond_dys:

            #odor = odor_list[0]
            for odor in odor_list:

                for nchan_i, nchan in enumerate(chan_list_eeg):

                    if group == 'allsujet':
                        data = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                    elif group == 'rep':
                        data = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :].mean('sujet').values
                    elif group == 'non_rep':
                        data = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                    
                    time_vec = np.linspace(t_start_PPI, t_stop_PPI, data.shape[0])
                    time_vec_mask = (time_vec > PPI_lm_start) & (time_vec < PPI_lm_stop)
                    Y = data[time_vec_mask].reshape(-1,1)
                    X = time_vec[time_vec_mask].reshape(-1,1)

                    lm = linear_model.LinearRegression()

                    lm.fit(X, Y)

                    Y_pred = lm.predict(X)

                    xr_lm_pred.loc[group, cond, odor, nchan, :] = Y_pred.reshape(-1)

                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X.reshape(-1), Y.reshape(-1))

                    xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'coeff'] = np.round(r_value**2, 5)
                    xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'pval'] = np.round(p_value, 5)
                    xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'slope'] = np.round(slope, 5)

                    #### verif
                    if debug:
            
                        plt.plot(X, Y)
                        plt.plot(X, Y_pred, color="b", linewidth=3)

                        plt.show()

    return xr_lm_data, xr_lm_pred, xr_lm_pred_coeff





################################
######## ERP PLOT ########
################################



def plot_ERP(xr_data, xr_lm_data, xr_lm_pred, xr_lm_pred_coeff, cond_dys):

    print('ERP PLOT')

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    sujet_group = ['allsujet', 'rep', 'non_rep']
    sujet_no_respond = np.array([sujet for sujet in sujet_list_erp if sujet not in sujet_best_list_erp])

    ######## SUMMARY NCHAN ########

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary'))

    for group in sujet_group:

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_dys))

            plt.suptitle(f'{nchan} {group}')

            fig.set_figheight(10)
            fig.set_figwidth(10)

            scales_val = {'min' : [], 'max' : []}

            for cond_i, cond in enumerate(cond_dys):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :].mean('sujet').values
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                    scales_val['min'].append(data_stretch.min())
                    scales_val['max'].append(data_stretch.max())

            scales_val['min'] = np.array(scales_val['min']).min()
            scales_val['max'] = np.array(scales_val['max']).max()

            #cond_i, cond = 1, 'MECA'
            for cond_i, cond in enumerate(cond_dys):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :].mean('sujet').values
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values

                    ax = axs[odor_i, cond_i]

                    if cond_i ==0:
                        ax.set_ylabel(odor)

                    ax.set_title(f"{cond} \n slope : {np.round(xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'slope'].values, 3)} / r2 : {np.round(xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'coeff'].values, 3)} / pval : {xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'pval'].values}", fontweight='bold')

                    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                    time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

                    ax.plot(time_vec, data_stretch)
                    # ax.plot(time_vec, data_stretch.std(axis=0), color='k', linestyle='--')
                    # ax.plot(time_vec, -data_stretch.std(axis=0), color='k', linestyle='--')

                    time_vec_lm = np.arange(PPI_lm_start, PPI_lm_stop, 1/srate)
                    ax.plot(time_vec_lm, xr_lm_pred.loc[group, cond, odor, nchan, :], color='r', linewidth=3)

                    ax.invert_yaxis()

                    ax.vlines(0, ymin=scales_val['min'], ymax=scales_val['max'], colors='g')  

            fig.tight_layout()

            # plt.show()

            #### save
            fig.savefig(f'{nchan}_{group}.jpeg', dpi=150)

            fig.clf()
            plt.close('all')
            gc.collect()

    ######## TOPOPLOT SLOPE ########

    #### scales
    val = np.array([])

    for odor_i in odor_list:

        for cond in cond_dys:

            val = np.append(val, xr_lm_pred_coeff.loc[cond, odor, :, 'slope'])

    scale_min = val.min()
    scale_max = val.max()

    #### plot
    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_dys))
    plt.suptitle(f'allsujet slope (s{xr_data.shape[0]})')
    fig.set_figheight(10)
    fig.set_figwidth(10)

    #c, cond = 0, 'FR_CV_1'
    for c, cond in enumerate(cond_dys):

        #r, odor_i = 0, odor_list[0]
        for r, odor in enumerate(odor_list):

            #### plot
            ax = axs[r, c]

            if r == 0:
                ax.set_title(cond, fontweight='bold', rotation=0)
            if c == 0:
                ax.set_ylabel(f'{odor}')
            
            data_plot = xr_lm_pred_coeff.loc['allsujet', cond, odor, :, 'slope'].values*-1

            mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

    # plt.show()

    #### save
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
    fig.savefig(f'allsujet_slope_topo.jpeg', dpi=150)
    fig.clf()
    plt.close('all')
    gc.collect()

    #### for cond
    for c, cond in enumerate(cond_dys):

        #### scales
        val = np.array([])
            
        for sujet_type_i, sujet_type in enumerate(['respond', 'no_respond']):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):
                
                if sujet_type == 'respond':
                    data_plot = xr_lm_data.loc[sujet_best_list_erp, cond, odor, :, 'slope'].mean('sujet').values*-1
                if sujet_type == 'no_respond':
                    data_plot = xr_lm_data.loc[sujet_no_respond, cond, odor, :, 'slope'].mean('sujet').values*-1

                val = np.append(val, data_plot)

        scale_min = val.min()
        scale_max = val.max()

        #### plot
        fig, axs = plt.subplots(nrows=len(odor_list), ncols=2)
        plt.suptitle(f'slope (s{xr_data.shape[0]})')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        for c, sujet_type in enumerate(['respond', 'no_respond']):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(f'{cond} {sujet_type}', fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                if sujet_type == 'respond':
                    data_plot = xr_lm_data.loc[sujet_best_list_erp, cond, odor, :, 'slope'].mean('sujet').values*-1
                if sujet_type == 'no_respond':
                    data_plot = xr_lm_data.loc[sujet_no_respond, cond, odor, :, 'slope'].mean('sujet').values*-1

                mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
        fig.savefig(f'{cond}_slope_topo_respond.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    ######## TOPOPLOT PPI ########

    PPI_topoplot_data_mean = {}

    for sujet_type in ['allsujet', 'respond', 'no_respond']:

        PPI_topoplot_data_mean[sujet_type] = {}

        for c, cond in enumerate(cond_dys):

            PPI_topoplot_data_mean[sujet_type][cond] = {}

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):
        
                PPI_topoplot_data_mean[sujet_type][cond][odor] = np.zeros((len(chan_list_eeg)))

    for sujet_type in ['allsujet', 'respond', 'no_respond']:

        for sujet in sujet_list_erp:

            for c, cond in enumerate(cond_dys):

                #r, odor_i = 0, odor_list[0]
                for r, odor in enumerate(odor_list):

                    if sujet_type == 'allsujet':
                        PPI_topoplot_data_mean[sujet_type][cond][odor] += (xr_lm_data.loc[sujet, cond, odor, :, 'slope'].values < -0.5) * 1 
                    if sujet_type == 'respond':
                        if sujet in sujet_best_list_erp:
                            PPI_topoplot_data_mean[sujet_type][cond][odor] += (xr_lm_data.loc[sujet, cond, odor, :, 'slope'].values < -0.5) * 1 
                    if sujet_type == 'no_respond':
                        if sujet in sujet_no_respond:
                            PPI_topoplot_data_mean[sujet_type][cond][odor] += (xr_lm_data.loc[sujet, cond, odor, :, 'slope'].values < -0.5) * 1 

    #### plot allsujet

    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_dys))
    plt.suptitle(f'PPI allsujet (s{xr_data.shape[0]})')
    fig.set_figheight(10)
    fig.set_figwidth(10)

    #c, cond = 0, 'FR_CV_1'
    for c, cond in enumerate(cond_dys):

        #r, odor_i = 0, odor_list[0]
        for r, odor in enumerate(odor_list):

            #### plot
            ax = axs[r, c]

            if r == 0:
                ax.set_title(f'{cond}', fontweight='bold', rotation=0)
            if c == 0:
                ax.set_ylabel(f'{odor}')
            
            mne.viz.plot_topomap(PPI_topoplot_data_mean['allsujet'][cond][odor], info, axes=ax, show=False, cmap='summer')

    # plt.show()

    #### save
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
    fig.savefig(f'allsujet_PPI_topo.jpeg', dpi=150)
    fig.clf()
    plt.close('all')
    gc.collect()

    #### PPI for cond

    for c, cond in enumerate(cond_dys):

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=2)
        plt.suptitle(f'PPI {cond} (s{xr_data.shape[0]})')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, sujet_type in enumerate(['respond', 'no_respond']):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(sujet_type, fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                mne.viz.plot_topomap(PPI_topoplot_data_mean[sujet_type][cond][odor], info, axes=ax, show=False, cmap='summer')

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
        fig.savefig(f'{cond}_PPI_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()







################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    cond_dys = ['MECA', 'CO2']

    print(f'#### compute allsujet ####', flush=True)

    xr_data = compute_ERP(cond_dys)
    xr_lm_data, xr_lm_pred, xr_lm_pred_coeff = compute_lm_on_ERP(xr_data, cond_dys)
    
    print(f'#### plot allsujet ####', flush=True)

    plot_ERP(xr_data, xr_lm_data, xr_lm_pred, xr_lm_pred_coeff, cond_dys)







