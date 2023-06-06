
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
from sklearn import linear_model
import gc

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False





################################
######## ERP ANALYSIS ########
################################


def compute_ERP(sujet, cond_dys):

    respfeatures = load_respfeatures(sujet)

    data_stretch_allcond = {}

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    #cond = 'MECA'
    for cond in cond_dys:

        data_stretch_allcond[cond] = {}

        #odor = odor_list[0]
        for odor in odor_list:

            data_stretch_allcond[cond][odor] = {}

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

                data_stretch_allcond[cond][odor][nchan] = data_stretch_clean

    return data_stretch_allcond
            








def compute_lm_on_ERP(data_stretch_allcond, cond_dys):

    lm_data = {}

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    #cond = 'MECA'
    for cond in cond_dys:

        lm_data[cond] = {}

        #odor = odor_list[0]
        for odor in odor_list:

            lm_data[cond][odor] = {}
            lm_data[cond][odor]['coeff'] = np.zeros((len(chan_list_eeg)))
            lm_data[cond][odor]['pval'] = np.zeros((len(chan_list_eeg)))
            lm_data[cond][odor]['slope'] = np.zeros((len(chan_list_eeg)))

            time_vec_lm = np.arange(PPI_lm_start, PPI_lm_stop, 1/srate)
            lm_data[cond][odor]['Y_pred'] = np.zeros((len(chan_list_eeg),time_vec_lm.shape[0]))

            for nchan_i, nchan in enumerate(chan_list_eeg):
            
                data = data_stretch_allcond[cond][odor][nchan].mean(axis=0)
                time_vec = np.linspace(t_start_PPI, t_stop_PPI, data.shape[0])
                time_vec_mask = (time_vec > PPI_lm_start) & (time_vec < PPI_lm_stop)
                Y = data[time_vec_mask].reshape(-1,1)
                X = time_vec[time_vec_mask].reshape(-1,1)

                lm = linear_model.LinearRegression()

                lm.fit(X, Y)

                Y_pred = lm.predict(X)

                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X.reshape(-1), Y.reshape(-1))

                lm_data[cond][odor]['coeff'][nchan_i] = np.round(r_value**2, 5)
                lm_data[cond][odor]['pval'][nchan_i] = np.round(p_value, 5)
                lm_data[cond][odor]['slope'][nchan_i] = np.round(slope, 5)

                lm_data[cond][odor]['Y_pred'][nchan_i, :] = Y_pred.reshape(-1)

                #### verif
                if debug:
        
                    plt.plot(X, Y)
                    plt.plot(X, Y_pred, color="b", linewidth=3)

                    plt.show()

    return lm_data





################################
######## ERP PLOT ########
################################



def plot_ERP(sujet, data_stretch_allcond, lm_data, cond_dys):

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    os.chdir(os.path.join(path_results, sujet, 'ERP', 'summary'))

    #nchan_i, nchan = 0, prms['chan_list'][:-3][0]
    for nchan_i, nchan in enumerate(chan_list_eeg):

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_dys))

        plt.suptitle(f'{nchan}')

        fig.set_figheight(10)
        fig.set_figwidth(10)

        #cond_i, cond = 1, 'MECA'
        for cond_i, cond in enumerate(cond_dys):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                data_stretch = data_stretch_allcond[cond][odor][nchan]

                ax = axs[odor_i, cond_i]

                if cond_i ==0:
                    ax.set_ylabel(odor)

                ax.set_title(f"{cond} : {data_stretch.shape[0]} \n slope : {np.round(lm_data[cond][odor]['slope'][nchan_i], 3)} / r2 : {np.round(lm_data[cond][odor]['coeff'][nchan_i], 3)} / pval : {lm_data[cond][odor]['pval'][nchan_i]}", fontweight='bold')

                stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

                ax.plot(time_vec, data_stretch.mean(axis=0), color='b')
                # ax.plot(time_vec, data_stretch.std(axis=0), color='k', linestyle='--')
                # ax.plot(time_vec, -data_stretch.std(axis=0), color='k', linestyle='--')

                ax.invert_yaxis()

                max_plot = np.stack((data_stretch.mean(axis=0), data_stretch.std(axis=0))).max()

                ax.vlines(0, ymin=max_plot*-1, ymax=max_plot, colors='g')

                time_vec_lm = np.arange(PPI_lm_start, PPI_lm_stop, 1/srate)

                ax.plot(time_vec_lm, lm_data[cond][odor]['Y_pred'][nchan_i,:], color='r', linewidth=3)

        fig.tight_layout()

        # plt.show()

        #### save
        fig.savefig(f'{sujet}_{nchan}.jpeg', dpi=150)

        fig.clf()
        plt.close('all')
        gc.collect()

    #### topoplot
    
    #### scales
    val = np.array([])

    for odor_i in odor_list:

        for cond in cond_dys:

            val = np.append(val, lm_data[cond][odor]['slope'])

    scale_min = val.min()
    scale_max = val.max()

    #### plot
    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_dys))
    plt.suptitle(f'{sujet}_slope')
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
            
            mne.viz.plot_topomap(lm_data[cond][odor]['slope']*-1, info, axes=ax, vlim=(scale_min, scale_max), show=False)

    # plt.show()

    #### save
    os.chdir(os.path.join(path_results, sujet, 'ERP', 'topoplot'))
    fig.savefig(f'{sujet}_slope_topo.jpeg', dpi=150)
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot'))
    fig.savefig(f'slope_{sujet}_topo.jpeg', dpi=150)
    fig.clf()
    plt.close('all')
    gc.collect()



    #### topoplot PPI
    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_dys))
    plt.suptitle(f'{sujet}_PPI')
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

            topo_data_PPI = (lm_data[cond][odor]['slope'] < -0.5) *1 
            
            mne.viz.plot_topomap(topo_data_PPI, info, axes=ax, vlim=(0, 1), show=False)

    # plt.show()

    #### save
    os.chdir(os.path.join(path_results, sujet, 'ERP', 'topoplot'))
    fig.savefig(f'{sujet}_PPI_topo.jpeg', dpi=150)
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot'))
    fig.savefig(f'PPI_{sujet}_topo.jpeg', dpi=150)
    fig.clf()
    plt.close('all')
    gc.collect()







################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    cond_dys = ['MECA', 'CO2']
    
    #sujet = sujet_list_erp[0]
    for sujet in sujet_list_erp:

        print(f'#### {sujet} ####', flush=True)
        print(f'COMPUTE', flush=True)

        data_stretch_allcond = compute_ERP(sujet, cond_dys)
        lm_data = compute_lm_on_ERP(data_stretch_allcond, cond_dys)
        
        print('PLOT', flush=True)

        plot_ERP(sujet, data_stretch_allcond, lm_data, cond_dys)







