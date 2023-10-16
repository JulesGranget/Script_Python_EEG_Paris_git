
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



########################
######## FILTER ########
########################

#sig = data
def iirfilt(sig, srate, lowcut=None, highcut=None, order=4, ftype='butter', verbose=False, show=False, axis=0):

    if len(sig.shape) == 1:

        axis = 0

    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = scipy.signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)

    filtered_sig = scipy.signal.sosfiltfilt(sos, sig, axis=axis)

    return filtered_sig







################################
######## ERP ANALYSIS ########
################################


def compute_ERP(sujet):

    respfeatures = load_respfeatures(sujet)

    data_chunk_allcond = {}
    data_value_microV = {}

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    #cond = 'FR_CV_1'
    for cond in conditions:

        data_chunk_allcond[cond] = {}
        data_value_microV[cond] = {}

        #odor = odor_list[0]
        for odor in odor_list:

            print(sujet, cond, odor)

            data_chunk_allcond[cond][odor] = {}
            data_value_microV[cond][odor] = {}

            #### load
            data = load_data_sujet(sujet, cond, odor)
            data = data[:len(chan_list_eeg),:]

            respfeatures_i = respfeatures[cond][odor]
            inspi_starts = respfeatures_i.query(f"select == 1")['inspi_index'].values

            #### low pass 45Hz
            for chan_i, chan in enumerate(chan_list_eeg):
                data[chan_i,:] = iirfilt(data[chan_i,:], srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)

            #nchan_i, nchan = 23, chan_list_eeg[23]
            for nchan_i, nchan in enumerate(chan_list_eeg):

                #### chunk
                stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                data_chunk = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

                x = data[nchan_i,:]

                x_mean, x_std = x.mean(), x.std()
                microV_SD = int(x_std*1e6)

                for start_i, start_time in enumerate(inspi_starts):

                    t_start = int(start_time + t_start_PPI*srate)
                    t_stop = int(start_time + t_stop_PPI*srate)

                    data_chunk[start_i, :] = (x[t_start: t_stop] - x_mean) / x_std

                if debug:

                    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                    for inspi_i, _ in enumerate(inspi_starts):

                        plt.plot(time_vec, data_chunk[inspi_i, :], alpha=0.3)

                    plt.vlines(0, ymax=data_chunk.max(), ymin=data_chunk.min(), color='k')
                    # plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                    plt.plot(time_vec, data_chunk.mean(axis=0), color='r')
                    plt.title(f'{cond} {odor} : {data_chunk.shape[0]}, 3SD : {microV_SD}')
                    plt.gca().invert_yaxis()
                    plt.show()

                #### clean
                data_stretch_clean = data_chunk[~((data_chunk >= 3) | (data_chunk <= -3)).any(axis=1),:]

                if debug:

                    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                    for inspi_i, _ in enumerate(inspi_starts):

                        plt.plot(time_vec, data_stretch_clean[inspi_i, :], alpha=0.3)

                    plt.vlines(0, ymax=data_stretch_clean.max(), ymin=data_stretch_clean.min(), color='k')
                    plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                    plt.plot(time_vec, data_stretch_clean.mean(axis=0), color='r')
                    plt.title(f'{cond} {odor} : {data_stretch_clean.shape[0]}')
                    plt.show()

                data_chunk_allcond[cond][odor][nchan] = data_stretch_clean
                data_value_microV[cond][odor][nchan] = microV_SD

    #### regroup FR_CV
    data_chunk_allcond['VS'] = {}
    data_value_microV['VS'] = {}

    for odor in odor_list:

        data_chunk_allcond['VS'][odor] = {}
        data_value_microV['VS'][odor] = {}

        #### low pass 45Hz
        for nchan_i, nchan in enumerate(chan_list_eeg):

            data_chunk_allcond['VS'][odor][nchan] = np.concatenate([data_chunk_allcond['FR_CV_1'][odor][nchan], data_chunk_allcond['FR_CV_2'][odor][nchan]], axis=0)
            data_value_microV['VS'][odor][nchan] = data_value_microV['FR_CV_1'][odor][nchan] + data_value_microV['FR_CV_2'][odor][nchan] / 2

    data_chunk_allcond['FR_CV_1'] = {}
    data_chunk_allcond['FR_CV_2'] = {}

    data_value_microV['FR_CV_1'] = {}
    data_value_microV['FR_CV_2'] = {}

    return data_chunk_allcond, data_value_microV
            








def compute_lm_on_ERP(data_chunk_allcond, cond_erp):

    lm_data = {}

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    #cond = 'MECA'
    for cond in cond_erp:

        lm_data[cond] = {}

        #odor = odor_list[0]
        for odor in odor_list:

            lm_data[cond][odor] = {}
            lm_data[cond][odor]['coeff'] = np.zeros((len(chan_list_eeg)))
            lm_data[cond][odor]['pval'] = np.zeros((len(chan_list_eeg)))
            lm_data[cond][odor]['slope'] = np.zeros((len(chan_list_eeg)))
            lm_data[cond][odor]['d_int'] = np.zeros((len(chan_list_eeg)))

            time_vec_lm = np.arange(PPI_lm_start, PPI_lm_stop, 1/srate)
            lm_data[cond][odor]['Y_pred'] = np.zeros((len(chan_list_eeg),time_vec_lm.shape[0]))

            #nchan_i, nchan = 0, chan_list_eeg[0]
            for nchan_i, nchan in enumerate(chan_list_eeg):
            
                data = data_chunk_allcond[cond][odor][nchan].mean(axis=0)

                time_vec = np.linspace(t_start_PPI, t_stop_PPI, data.shape[0])
                time_vec_mask = (time_vec > PPI_lm_start) & (time_vec < PPI_lm_stop)
                Y = data[time_vec_mask].reshape(-1,1)
                X = time_vec[time_vec_mask].reshape(-1,1)

                #### get reg linear model
                lm = linear_model.LinearRegression()

                lm.fit(X, Y)

                Y_pred = lm.predict(X)

                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X.reshape(-1), Y.reshape(-1))

                lm_data[cond][odor]['coeff'][nchan_i] = np.round(r_value**2, 5)
                lm_data[cond][odor]['pval'][nchan_i] = np.round(p_value, 5)
                lm_data[cond][odor]['slope'][nchan_i] = np.round(slope, 5)

                lm_data[cond][odor]['Y_pred'][nchan_i, :] = Y_pred.reshape(-1)

                #### get derivee integral
                diff_data = np.insert(np.diff(data), 0, np.median(np.diff(data)))
                diff_integral = np.trapz(diff_data[time_vec_mask])
                lm_data[cond][odor]['d_int'][nchan_i] = np.round(diff_integral, 5)

                #### verif
                if debug:
        
                    plt.plot(X, Y)
                    plt.plot(X, Y_pred, color="b", linewidth=3)

                    plt.show()

    return lm_data



def shuffle_data_ERP(data):

    ERP_shuffle = np.zeros(data.shape)

    for erp_i in range(data.shape[0]):

        cut = np.random.randint(0, data.shape[1], 1)[0]

        ERP_shuffle[erp_i,:data[:,cut:].shape[1]] = data[erp_i,cut:]
        ERP_shuffle[erp_i,data[:,cut:].shape[1]:] = data[erp_i,:cut]
        
    return ERP_shuffle






def compute_surr_ERP(data_chunk_allcond, shuffle_way):
    
    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    time_vec = np.linspace(t_start_PPI, t_stop_PPI, data_chunk_allcond[cond_erp[0]][odor_list[0]][chan_list[0]].shape[-1])
    time_vec_mask = (time_vec > PPI_lm_start) & (time_vec < PPI_lm_stop)

    ERP_surr = {}

    if shuffle_way == 'intra_cond':

        cond = 'VS'

        ERP_surr[cond] = {}

        #odor = odor_list[0]
        for odor in odor_list:

            ERP_surr[cond][odor] = {}

            #nchan_i, nchan = 0, chan_list_eeg[0]
            for nchan_i, nchan in enumerate(chan_list_eeg):
            
                data = data_chunk_allcond[cond][odor][nchan]

                _ERP_surr = np.zeros((ERP_n_surrogate, data.shape[1]))

                for surr_i in range(ERP_n_surrogate):

                    _ERP_surr[surr_i,:] = shuffle_data_ERP(data).mean(axis=0)

                # up_percentile = np.std(_ERP_surr[:,:], axis=0)*3
                # down_percentile = np.percentile(_ERP_surr, 99, axis=0)

                up_percentile = np.std(_ERP_surr, axis=0)*3
                down_percentile = -np.std(_ERP_surr, axis=0)*3

                ERP_surr[cond][odor][nchan] = np.vstack((up_percentile, down_percentile))

        ERP_surr['MECA'] = ERP_surr['VS']
        ERP_surr['CO2'] = ERP_surr['VS']
        ERP_surr['VS'] = {}

    if shuffle_way == 'inter_cond':

        #cond = 'MECA'
        for cond in cond_erp:

            if cond == 'VS':
                continue

            ERP_surr[cond] = {}

            #odor = odor_list[0]
            for odor in odor_list:

                ERP_surr[cond][odor] = {}

                #nchan_i, nchan = 0, chan_list_eeg[0]
                for nchan_i, nchan in enumerate(chan_list_eeg):
                
                    data_cond = data_chunk_allcond[cond][odor][nchan]
                    data_baseline = data_chunk_allcond['VS'][odor][nchan]

                    min_shape = np.array([data_cond.shape[0], data_baseline.shape[0]]).min()

                    _ERP_surr = np.zeros((ERP_n_surrogate, data_cond.shape[1]))

                    for surr_i in range(ERP_n_surrogate):

                        balance_selection_baseline = np.random.choice(np.arange(data_baseline.shape[0]), min_shape, replace=False)
                        balance_selection_cond = np.random.choice(np.arange(data_cond.shape[0]), min_shape, replace=False)

                        rand_selection_shuffle = np.random.randint(low=0, high=2, size=min_shape)
                        rand_selection_shuffle_i = np.where(rand_selection_shuffle == 1)[0]

                        data_cond_shuffle = data_cond[balance_selection_cond,:]
                        data_baseline_shuffle = data_baseline[balance_selection_baseline,:]
                        
                        data_cond_shuffle[rand_selection_shuffle_i,:] = data_baseline[balance_selection_baseline,:][rand_selection_shuffle_i,:]
                        data_baseline_shuffle[rand_selection_shuffle_i,:] = data_cond[balance_selection_cond,:][rand_selection_shuffle_i,:]

                        _ERP_surr[surr_i,:] = data_cond_shuffle.mean(axis=0)

                    up_percentile = np.percentile(_ERP_surr, 98, axis=0)
                    down_percentile = np.percentile(_ERP_surr, 2, axis=0)

                    ERP_surr[cond][odor][nchan] = np.vstack((up_percentile, down_percentile))

    if shuffle_way == 'linear_based':

        #cond = 'CO2'
        for cond in cond_erp:

            ERP_surr[cond] = {}

            #odor = odor_list[0]
            for odor in odor_list:

                print(cond, odor)

                ERP_surr[cond][odor] = {}

                #nchan_i, nchan = 0, chan_list_eeg[0]
                for nchan_i, nchan in enumerate(chan_list_eeg):
                
                    data_cond = data_chunk_allcond[cond][odor][nchan]

                    _ERP_surr = np.zeros((ERP_n_surrogate))

                    for surr_i in range(ERP_n_surrogate):

                        surr_i_mean = shuffle_data_ERP(data_cond).mean(axis=0)

                        Y = surr_i_mean[time_vec_mask]
                        X = time_vec[time_vec_mask]

                        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
                        
                        _ERP_surr[surr_i] = slope

                    if debug:

                        Y = data_cond.mean(0)[time_vec_mask]
                        X = time_vec[time_vec_mask]
                        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)

                        plt.plot(time_vec, data_cond.mean(axis=0))
                        plt.gca().invert_yaxis()
                        plt.title(nchan)
                        plt.show()

                        count, hist, _ = plt.hist(_ERP_surr, bins=50)
                        plt.gca().invert_xaxis()
                        plt.vlines(slope, ymin=0, ymax=count.max(), color='r', label='raw')
                        plt.vlines(np.percentile(_ERP_surr, 5), ymin=0, ymax=count.max(), color='b', label='99')
                        plt.legend()
                        plt.title(nchan)
                        plt.show()

                    ERP_surr[cond][odor][nchan] = np.percentile(_ERP_surr, 5)

    return ERP_surr




################################
######## ERP PLOT ########
################################



def plot_ERP(sujet, data_chunk_allcond, data_value_microV, ERP_surr, lm_data, cond_erp):

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
    time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)
    mask_time_PPI = (time_vec > -2.5) & (time_vec < 0)

    os.chdir(os.path.join(path_results, sujet, 'ERP', 'summary'))

    #nchan_i, nchan = 23, chan_list_eeg[23]
    for nchan_i, nchan in enumerate(chan_list_eeg):

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_erp))

        plt.suptitle(f'{nchan}')

        fig.set_figheight(10)
        fig.set_figwidth(10)

        #cond_i, cond = 1, 'MECA'
        for cond_i, cond in enumerate(cond_erp):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                data_chunk = data_chunk_allcond[cond][odor][nchan]
                x_mean = data_chunk.mean(axis=0)

                ax = axs[odor_i, cond_i]

                if cond_i ==0:
                    ax.set_ylabel(odor)

                # if cond != 'VS':

                #     if ((x_mean < ERP_surr[cond][odor][nchan][1,:]) | (x_mean > ERP_surr[cond][odor][nchan][0,:])).any():

                #         up_int = np.trapz(x_mean[(x_mean - ERP_surr[cond][odor][nchan][0,:]) > 0])
                #         down_int = np.trapz(x_mean[(x_mean - ERP_surr[cond][odor][nchan][1,:]) < 0])
                        
                #         p_surr = np.round(up_int + np.abs(down_int),2)

                #     if (x_mean[mask_time_PPI] < ERP_surr[cond][odor][nchan][1,:][mask_time_PPI]).any() and (x_mean[mask_time_PPI] > ERP_surr[cond][odor][nchan][0,:][mask_time_PPI]).any():
                        
                #         PPI_presence = 1

                #         if debug:

                #             plt.plot(x_mean[mask_time_PPI])
                #             plt.plot(ERP_surr[cond][odor][nchan][1,:][mask_time_PPI])
                #             plt.plot(ERP_surr[cond][odor][nchan][0,:][mask_time_PPI])
                #             plt.show()

                #     else:

                #         PPI_presence = 0

                if ERP_surr[cond][odor][nchan] >= lm_data[cond][odor]['slope'][nchan_i]:

                    PPI_presence = 1

                else:

                    # p_surr = 0
                    PPI_presence = 0

                # ax.set_title(f"{cond}:{data_chunk.shape[0]}, SD:{data_value_microV[cond][odor][nchan]}microV \n slope:{np.round(lm_data[cond][odor]['slope'][nchan_i], 3)} / r2 : {np.round(lm_data[cond][odor]['coeff'][nchan_i], 3)} / pval : {p_surr}", fontweight='bold')
                ax.set_title(f"{cond}:{data_chunk.shape[0]}, SD:{data_value_microV[cond][odor][nchan]}microV \n slope:{np.round(lm_data[cond][odor]['slope'][nchan_i], 3)} / PPI:{PPI_presence} ", fontweight='bold')

                ax.plot(time_vec, x_mean, color='b')
                # ax.plot(time_vec, data_chunk.std(axis=0), color='k', linestyle='--')
                # ax.plot(time_vec, -data_chunk.std(axis=0), color='k', linestyle='--')

                # if cond != 'VS':
                #     ax.plot(time_vec, ERP_surr[cond][odor][nchan][0,:], color='k', linestyle='--')
                #     ax.plot(time_vec, ERP_surr[cond][odor][nchan][1,:], color='k', linestyle='--')

                ax.set_ylim(-1,1)

                ax.invert_yaxis()

                max_plot = np.stack((x_mean, data_chunk.std(axis=0))).max()

                ax.vlines(0, ymin=-1, ymax=1, colors='g')

                time_vec_lm = np.arange(PPI_lm_start, PPI_lm_stop, 1/srate)

                ax.plot(time_vec_lm, lm_data[cond][odor]['Y_pred'][nchan_i,:], color='r', linewidth=3)

        fig.tight_layout()

        # plt.show()

        #### save
        fig.savefig(f'{sujet}_{nchan}.jpeg', dpi=150)

        fig.clf()
        plt.close('all')
        gc.collect()

    #### topoplot SLOPE
    
    #### scales
    val = np.array([])

    for odor_i in odor_list:

        for cond in cond_erp:

            val = np.append(val, lm_data[cond][odor]['slope'])

    scale_min = val.min()
    scale_max = val.max()

    #### plot
    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_erp))

    if sujet in sujet_best_list:
        plt.suptitle(f'{sujet} : slope, rep')
    else:
        plt.suptitle(f'{sujet} : slope, no_rep')

    fig.set_figheight(10)
    fig.set_figwidth(10)

    #c, cond = 0, 'FR_CV_1'
    for c, cond in enumerate(cond_erp):

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


    #### topoplot diff int
    
    #### scales
    val = np.array([])

    for odor_i in odor_list:

        for cond in cond_erp:

            val = np.append(val, lm_data[cond][odor]['d_int'])

    scale_min = val.min()
    scale_max = val.max()

    #### plot
    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_erp))

    if sujet in sujet_best_list:
        plt.suptitle(f'{sujet} : slope, rep')
    else:
        plt.suptitle(f'{sujet} : slope, no_rep')

    fig.set_figheight(10)
    fig.set_figwidth(10)

    #c, cond = 0, 'FR_CV_1'
    for c, cond in enumerate(cond_erp):

        #r, odor_i = 0, odor_list[0]
        for r, odor in enumerate(odor_list):

            #### plot
            ax = axs[r, c]

            if r == 0:
                ax.set_title(cond, fontweight='bold', rotation=0)
            if c == 0:
                ax.set_ylabel(f'{odor}')
            
            mne.viz.plot_topomap(lm_data[cond][odor]['d_int']*-1, info, axes=ax, vlim=(scale_min, scale_max), show=False)

    # plt.show()

    #### save
    os.chdir(os.path.join(path_results, sujet, 'ERP', 'topoplot'))
    fig.savefig(f'{sujet}_d_int_topo.jpeg', dpi=150)
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot'))
    fig.savefig(f'd_int_{sujet}_topo.jpeg', dpi=150)
    fig.clf()
    plt.close('all')
    gc.collect()



    #### topoplot PPI
    PPI_count = {}

    for c, cond in enumerate(cond_erp):

        if cond == 'VS':
            continue

        PPI_count[cond] = {}

        #r, odor_i = 0, odor_list[0]
        for r, odor in enumerate(odor_list):

            _PPI_count = np.zeros(len(chan_list_eeg))

            for nchan_i, nchan in enumerate(chan_list_eeg):
                
                x_mean = data_chunk_allcond[cond][odor][nchan].mean(axis=0)

                # if (x_mean[mask_time_PPI] < ERP_surr[cond][odor][nchan][1,:][mask_time_PPI]).any()  and (x_mean[mask_time_PPI] > ERP_surr[cond][odor][nchan][0,:][mask_time_PPI]).any():
                #     _PPI_count[nchan_i] = 1

                if ERP_surr[cond][odor][nchan] >= lm_data[cond][odor]['slope'][nchan_i]:
                    _PPI_count[nchan_i] = 1

                if debug :

                    plt.plot(time_vec[mask_time_PPI], x_mean[mask_time_PPI])
                    plt.plot(time_vec[mask_time_PPI], ERP_surr[cond][odor][nchan][1,:][mask_time_PPI])
                    plt.plot(time_vec[mask_time_PPI], ERP_surr[cond][odor][nchan][0,:][mask_time_PPI])
                    plt.show()

            PPI_count[cond][odor] = _PPI_count

    cond_topoplot = ['MECA', 'CO2']

    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_topoplot))

    if sujet in sujet_best_list:
        plt.suptitle(f'{sujet} : slope, rep')
    else:
        plt.suptitle(f'{sujet} : slope, no_rep')
        
    fig.set_figheight(10)
    fig.set_figwidth(10)

    #c, cond = 0, 'FR_CV_1'
    for c, cond in enumerate(cond_topoplot):

        #r, odor_i = 0, odor_list[0]
        for r, odor in enumerate(odor_list):

            #### plot
            ax = axs[r, c]

            if r == 0:
                ax.set_title(cond, fontweight='bold', rotation=0)
            if c == 0:
                ax.set_ylabel(f'{odor}')

            # topo_data_PPI = (lm_data[cond][odor]['slope'] < -0.5) *1
            
            mne.viz.plot_topomap(PPI_count[cond][odor], info, axes=ax, vlim=(0, 1), show=False)

    # plt.show()

    #### save
    os.chdir(os.path.join(path_results, sujet, 'ERP', 'topoplot'))
    fig.savefig(f'{sujet}_PPI_topo.jpeg', dpi=150)
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot'))
    fig.savefig(f'PPI_{sujet}_topo.jpeg', dpi=150)
    fig.clf()
    plt.close('all')
    gc.collect()





########################################
######## RESPI ERP ANALYSIS ########
########################################


def get_data_itl():

    t_start_PPI = mean_respi_ERP_time_vec[0]
    t_stop_PPI = mean_respi_ERP_time_vec[1]

    #### load data ITL
    data_respi_ITL = {}

    os.chdir(os.path.join(path_data, 'data_itl_leo'))

    sujet_list_itl = np.unique(np.array([file[:4] for file in os.listdir()]))

    #sujet_itl = sujet_list_itl[0]
    for sujet_itl in sujet_list_itl:

        data_respi_ITL[sujet_itl] = {}

        for cond in ['VS', 'CO2', 'ITL']:

            respi_file_name = [file for file in os.listdir() if file.find(sujet_itl) != -1 and file.find(f'{cond}.edf') != -1][0]
            inspi_file_name = [file for file in os.listdir() if file.find(sujet_itl) != -1 and file.find(f'{cond}.Markers') != -1][0]
            
            respi = mne.io.read_raw_edf(respi_file_name).get_data()[-1,:]
            f = open(inspi_file_name, "r")
            inspi_starts = [int(line.split(',')[2][1:]) for line_i, line in enumerate(f.read().split('\n')) if len(line.split(',')) == 5 and line.split(',')[0] == 'Response']
            inspi_starts = np.array(inspi_starts)

            if debug:
                fig, ax = plt.subplots()
                plt.plot(respi[:10000])
                plt.vlines(inspi_starts[inspi_starts<10000], ymin=respi.min(), ymax=respi.max(), color='r')
                export_fig(f'respi.png', fig)

            #### chunk
            stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)

            clean_cycles_i = [start_i for start_i, start_time in enumerate(inspi_starts) if int(start_time + t_start_PPI*srate) > 0]
            inspi_starts = inspi_starts[clean_cycles_i]
            clean_cycles_i = [start_i for start_i, start_time in enumerate(inspi_starts) if int(start_time + t_stop_PPI*srate) < respi.shape[0]]
            inspi_starts = inspi_starts[clean_cycles_i]

            respi_stretch = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

            x_mean, x_std = respi.mean(), respi.std()

            for start_i, start_time in enumerate(inspi_starts):

                t_start = int(start_time + t_start_PPI*srate)
                t_stop = int(start_time + t_stop_PPI*srate)

                # respi_stretch[start_i, :] = (respi[t_start: t_stop] - x_mean) / x_std
                respi_stretch[start_i, :] = respi[t_start: t_stop] - respi[t_start: t_stop].mean()

            data_respi_ITL[sujet_itl][cond] = respi_stretch

            if debug:

                time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                fig, ax = plt.subplots()
                plt.plot(time_vec, respi_stretch.mean(axis=0)*-1)
                plt.vlines(0, ymin=(respi_stretch.mean(axis=0)*-1).min(), ymax=(respi_stretch.mean(axis=0)*-1).max(), color='r')
                export_fig(f'respi.png', fig)

    #### normalize by VS
    val = np.array([])

    for sujet_itl in sujet_list_itl:

        val = np.concatenate((val, data_respi_ITL[sujet_itl]['VS'].reshape(-1)), axis=0)

    VS_mean, VS_std = val.mean(), val.std()

    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)
    dict = {'sujet' : sujet_list_itl, 'cond' : ['VS', 'CO2', 'ITL'], 'times' : time_vec}
    data = np.zeros((sujet_list_itl.shape[0], 3, time_vec.shape[0]))
    xr_itl = xr.DataArray(data=data, dims=dict.keys(), coords=dict.values())

    for sujet_itl in sujet_list_itl:

        for cond in ['VS', 'CO2', 'ITL']:

            _zscore = (data_respi_ITL[sujet_itl][cond] - VS_mean) / VS_std
            xr_itl.loc[sujet_itl, cond, :] = np.median(_zscore, axis=0)

    return xr_itl



def plot_mean_respi(sujet, xr_itl):

    #### load data
    respfeatures = load_respfeatures(sujet)

    data_respi = {}

    t_start_PPI = mean_respi_ERP_time_vec[0]
    t_stop_PPI = mean_respi_ERP_time_vec[1]

    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

    #cond = 'FR_CV_1'
    for cond in conditions:

        data_respi[cond] = {}

        #odor = odor_list[0]
        for odor in odor_list:

            data_respi[cond][odor] = {}

            #### load
            respi = load_data_sujet(sujet, cond, odor)[chan_list.index('PRESS'),:]

            resp_clean = physio.preprocess(respi, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
            respi = physio.smooth_signal(resp_clean, srate, win_shape='gaussian', sigma_ms=40.0)

            if debug:

                plt.plot(respi)
                plt.show()

            respfeatures_i = respfeatures[cond][odor]
            inspi_starts = respfeatures_i.query(f"select == 1")['inspi_index'].values

            #### chunk
            stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
            respi_stretch = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

            x_mean, x_std = respi.mean(), respi.std()

            for start_i, start_time in enumerate(inspi_starts):

                t_start = int(start_time + t_start_PPI*srate)
                t_stop = int(start_time + t_stop_PPI*srate)

                # respi_stretch[start_i, :] = (respi[t_start: t_stop] - x_mean) / x_std
                respi_stretch[start_i, :] = respi[t_start: t_stop] - x_mean

            if debug:

                time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                for inspi_i, _ in enumerate(inspi_starts):

                    plt.plot(time_vec, respi_stretch[inspi_i, :], alpha=0.3)

                plt.vlines(0, ymax=respi_stretch.max(), ymin=respi_stretch.min(), color='k')
                plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                plt.plot(time_vec, respi_stretch.mean(axis=0), color='r')
                plt.title(f'{cond} {odor} : {respi_stretch.shape[0]}')
                plt.show()

            data_respi[cond][odor] = respi_stretch

    #### normalize by VS
    val = np.array([])

    for odor_i, odor in enumerate(odor_list):

        val = np.concatenate((val, data_respi['FR_CV_1'][odor].reshape(-1)), axis=0)
        val = np.concatenate((val, data_respi['FR_CV_2'][odor].reshape(-1)), axis=0)

    VS_mean, VS_std = val.mean(), val.std()

    for cond in conditions:

        for odor_i, odor in enumerate(odor_list):

            data_respi[cond][odor] = (data_respi[cond][odor] - VS_mean) / VS_std

    #### plot
    os.chdir(os.path.join(path_results, sujet, 'RESPI'))

    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

    fig, axs = plt.subplots(ncols=len(odor_list), figsize=(15,10))

    for odor_i, odor in enumerate(odor_list):

        ax = axs[odor_i]

        ax.set_title(f"{odor}")

        ax.plot(time_vec, xr_itl.median('sujet').loc['VS',:].values*-1, label='VS_itl', color='g', linestyle=':', dashes=(5, 10))
        ax.plot(time_vec, xr_itl.median('sujet').loc['CO2',:].values*-1, label='CO2_itl', color='r', linestyle=':', dashes=(5, 10))
        ax.plot(time_vec, xr_itl.median('sujet').loc['ITL',:].values*-1, label='ITL_itl', color='b', linestyle=':', dashes=(5, 10))

        ax.plot(time_vec, np.median(data_respi['FR_CV_1'][odor], axis=0), label=f'VS_1', color='g')
        ax.plot(time_vec, np.median(data_respi['FR_CV_2'][odor], axis=0), label=f'VS_2', color='g')
        ax.plot(time_vec, np.median(data_respi['CO2'][odor], axis=0), label=f'CO2', color='r')
        ax.plot(time_vec, np.median(data_respi['MECA'][odor], axis=0), label=f'MECA', color='b')

        ax.vlines(0, ymin=xr_itl.median('sujet').min(), ymax=xr_itl.median('sujet').max(), color='k')

    plt.legend()
    
    plt.suptitle(f"{sujet} comparison ITL")

    # plt.show()

    plt.savefig(f"{sujet}_ERP_comparison_ITL.png")
    plt.close('all')


    #### plot sujet respi cond
    min = []
    max = []

    for cond_i, cond in enumerate(conditions):

        for odor_i, odor in enumerate(odor_list):

            min.append(np.median(data_respi[cond][odor], axis=0).min())
            max.append(np.median(data_respi[cond][odor], axis=0).max())

    min = np.array(min).min()
    max = np.array(max).max()

    fig, axs = plt.subplots(ncols=len(conditions), figsize=(15,10))

    for cond_i, cond in enumerate(conditions):

        ax = axs[cond_i]

        ax.set_title(f"{cond} \n o:{data_respi[cond]['o'].shape[0]}, -:{data_respi[cond]['-'].shape[0]}, +:{data_respi[cond]['+'].shape[0]}")

        ax.plot(time_vec, np.median(data_respi[cond]['o'], axis=0), label=f'o', color='b')
        ax.plot(time_vec, np.median(data_respi[cond]['-'], axis=0), label=f'-', color='r')
        ax.plot(time_vec, np.median(data_respi[cond]['+'], axis=0), label=f'+', color='g')

        ax.set_ylim(min, max)

        ax.vlines(0, ymin=min, ymax=max, color='k')

    plt.legend()
    
    plt.suptitle(f"{sujet} ERP")

    # plt.show()

    plt.savefig(f"{sujet}_ERP_mean.png")
    
    plt.close('all')


    #### plot sujet respi odor
    min = []
    max = []

    for odor_i, odor in enumerate(odor_list):

        for cond_i, cond in enumerate(conditions):

            min.append(np.median(data_respi[cond][odor], axis=0).min())
            max.append(np.median(data_respi[cond][odor], axis=0).max())

    min = np.array(min).min()
    max = np.array(max).max()

    fig, axs = plt.subplots(ncols=len(odor_list), figsize=(15,10))

    for odor_i, odor in enumerate(odor_list):

        ax = axs[odor_i]

        ax.set_title(f"{odor} \n FRCV1:{data_respi['FR_CV_1'][odor].shape[0]}, MECA:{data_respi['MECA'][odor].shape[0]}, CO2:{data_respi['CO2'][odor].shape[0]}, FRCV2:{data_respi['FR_CV_2'][odor].shape[0]}")

        ax.plot(time_vec, np.median(data_respi['FR_CV_1'][odor], axis=0), label=f'FR_CV_1', color='b')
        ax.plot(time_vec, np.median(data_respi['FR_CV_2'][odor], axis=0), label=f'FR_CV_2', color='c')
        ax.plot(time_vec, np.median(data_respi['MECA'][odor], axis=0), label=f'MECA', color='g')
        ax.plot(time_vec, np.median(data_respi['CO2'][odor], axis=0), label=f'CO2', color='r')

        ax.set_ylim(min, max)

        ax.vlines(0, ymin=min, ymax=max, color='k')

    plt.legend()
    
    plt.suptitle(f"{sujet} ERP")

    # plt.show()

    plt.savefig(f"{sujet}_ERP_mean.png")
    
    plt.close('all')











################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    cond_erp = ['VS', 'MECA', 'CO2']

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        print(f'#### {sujet} ####', flush=True)
        print(f'COMPUTE', flush=True)

        data_chunk_allcond, data_value_microV = compute_ERP(sujet)
        lm_data = compute_lm_on_ERP(data_chunk_allcond, cond_erp)

        # shuffle_way = 'inter_cond'
        # shuffle_way = 'intra_cond'
        shuffle_way = 'linear_based'

        ERP_surr = compute_surr_ERP(data_chunk_allcond, shuffle_way)
        
        print('PLOT', flush=True)

        plot_ERP(sujet, data_chunk_allcond, data_value_microV, ERP_surr, lm_data, cond_erp)


    xr_itl = get_data_itl()

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        print(sujet)

        #### plot mean
        plot_mean_respi(sujet, xr_itl)






