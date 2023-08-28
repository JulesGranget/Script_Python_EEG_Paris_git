
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


def compute_ERP():

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'allsujet_ERP_data.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        xr_data = xr.open_dataarray('allsujet_ERP_data.nc')

    else:

        t_start_PPI = PPI_time_vec[0]
        t_stop_PPI = PPI_time_vec[1]

        time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

        xr_dict = {'sujet' : sujet_list_erp, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'time' : time_vec}
        xr_data = xr.DataArray(data=np.zeros((len(sujet_list_erp), len(conditions), len(odor_list), len(chan_list_eeg), time_vec.shape[0])), dims=xr_dict.keys(), coords=xr_dict.values())

        #sujet_i, sujet = 0, sujet_list_erp[0]
        for sujet_i, sujet in enumerate(sujet_list_erp):

            print(sujet)

            respfeatures = load_respfeatures(sujet)

            #nchan_i, nchan = 0, chan_list_eeg[0]
            for nchan_i, nchan in enumerate(chan_list_eeg):

                erp_data = {}

                #cond = 'MECA'
                for cond in conditions:

                    erp_data[cond] = {}

                    #odor = odor_list[0]
                    for odor in odor_list:

                        #### load
                        data = load_data_sujet(sujet, cond, odor)
                        data = data[:len(chan_list_eeg),:]

                        respfeatures_i = respfeatures[cond][odor]
                        inspi_starts = respfeatures_i.query(f"select == 1")['inspi_index'].values

                        #### chunk
                        stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                        data_stretch = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

                        #### low pass 45Hz
                        x = data[nchan_i,:]
                        x = iirfilt(x, srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)

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

                        erp_data[cond][odor] = data_stretch_clean

                #### regroup FR_CV
                # erp_data['VS'] = {}

                # for odor in odor_list:

                #     erp_data['VS'][odor] = np.concatenate([erp_data['FR_CV_1'][odor], erp_data['FR_CV_2'][odor]], axis=0)

                #### load
                for cond in conditions:

                    #odor = odor_list[0]
                    for odor in odor_list:
                        
                        xr_data.loc[sujet, cond, odor, nchan, :] = erp_data[cond][odor].mean(axis=0)

        #### save data
        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_data.to_netcdf('allsujet_ERP_data.nc')

    return xr_data
            








def compute_lm_on_ERP(xr_data):

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'allsujet_ERP_lm_data.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        xr_lm_data = xr.open_dataarray('allsujet_ERP_lm_data.nc')
        xr_lm_pred = xr.open_dataarray('allsujet_ERP_lm_pred.nc')
        xr_lm_pred_coeff = xr.open_dataarray('allsujet_ERP_lm_pred_coeff.nc')

        return xr_lm_data, xr_lm_pred, xr_lm_pred_coeff

    else:

        t_start_PPI = PPI_time_vec[0]
        t_stop_PPI = PPI_time_vec[1]

        PPI_lm_start = PPI_lm_time[0]
        PPI_lm_stop = PPI_lm_time[1] 

        sujet_best_list_erp = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
        sujet_no_respond = np.array([sujet for sujet in sujet_list_erp if sujet not in sujet_best_list_erp])

        sujet_group = ['allsujet', 'rep', 'non_rep']
        data_type = ['coeff', 'pval', 'slope', 'd_int']

        xr_dict = {'sujet' : sujet_list_erp, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'type' : data_type}
        xr_lm_data = xr.DataArray(data=np.zeros((len(sujet_list_erp), len(conditions), len(odor_list), len(chan_list_eeg), len(data_type))), dims=xr_dict.keys(), coords=xr_dict.values())

        time_vec_lm = np.arange(PPI_lm_start, PPI_lm_stop, 1/srate)
        xr_dict = {'group' : sujet_group, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'time' : time_vec_lm}
        xr_lm_pred = xr.DataArray(data=np.zeros((len(sujet_group), len(conditions), len(odor_list), len(chan_list_eeg), len(time_vec_lm))), dims=xr_dict.keys(), coords=xr_dict.values())

        xr_dict = {'group' : sujet_group, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'type' : data_type}
        xr_lm_pred_coeff = xr.DataArray(data=np.zeros((len(sujet_group), len(conditions), len(odor_list), len(chan_list_eeg), len(data_type))), dims=xr_dict.keys(), coords=xr_dict.values())

        #sujet_i, sujet = 22, sujet_list_erp[22]
        for sujet_i, sujet in enumerate(sujet_list_erp):

            print(sujet)

            #cond = 'MECA'
            for cond in conditions:

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

                        #### get derivee integral
                        diff_data = np.insert(np.diff(data), 0, np.median(np.diff(data)))
                        diff_integral = np.trapz(diff_data[time_vec_mask])
                        
                        xr_lm_data.loc[sujet, cond, odor, nchan, 'd_int'] = np.round(diff_integral, 5)

                        #### verif
                        if debug:
                
                            plt.plot(X, Y)
                            plt.plot(X, Y_pred, color="b", linewidth=3)

                            plt.show()

        #### lm pred for mean
        #group = sujet_group[0]
        for group in sujet_group:

            #cond = 'MECA'
            for cond in conditions:

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

        #### save data
        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_lm_data.to_netcdf('allsujet_ERP_lm_data.nc')
        xr_lm_pred.to_netcdf('allsujet_ERP_lm_pred.nc')
        xr_lm_pred_coeff.to_netcdf('allsujet_ERP_lm_pred_coeff.nc')

        return xr_lm_data, xr_lm_pred, xr_lm_pred_coeff






def shuffle_data_ERP(data):

    ERP_shuffle = np.zeros(data.shape)

    for ERP_i in range(data.shape[0]):

        cut = np.random.randint(0, data.shape[1], 1)[0]
        ERP_shuffle[ERP_i,:data[ERP_i,cut:].shape[0]] = data[ERP_i,cut:]
        ERP_shuffle[ERP_i,data[ERP_i,cut:].shape[0]:] = data[ERP_i,:cut]

    return ERP_shuffle.mean(0)




def shuffle_data_ERP_linear_based(data):

    ERP_shuffle = np.zeros(data.shape)

    for ERP_i in range(data.shape[0]):

        cut = np.random.randint(0, data.shape[1], 1)[0]
        ERP_shuffle[ERP_i,:data[ERP_i,cut:].shape[0]] = data[ERP_i,cut:]
        ERP_shuffle[ERP_i,data[ERP_i,cut:].shape[0]:] = data[ERP_i,:cut]

    return ERP_shuffle





def compute_surr_ERP(xr_data, shuffle_way):

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', f'allsujet_ERP_surrogates_{shuffle_way}.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print()

        print('ALREADY COMPUTED', flush=True)

        xr_surr = xr.open_dataarray(f'allsujet_ERP_surrogates_{shuffle_way}.nc')

    else:

        #### get data
        t_start_PPI = PPI_time_vec[0]
        t_stop_PPI = PPI_time_vec[1]

        PPI_lm_start = PPI_lm_time[0]
        PPI_lm_stop = PPI_lm_time[1] 

        time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)
        time_vec_mask = (time_vec > PPI_lm_start) & (time_vec < PPI_lm_stop)

        xr_dict = {'sujet' : sujet_list_erp, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'percentile_type' : ['up', 'down'], 'time' : time_vec}
        xr_surr = xr.DataArray(data=np.zeros((len(sujet_list_erp), len(conditions), len(odor_list), len(chan_list_eeg), 2, time_vec.shape[0])), dims=xr_dict.keys(), coords=xr_dict.values())

        if shuffle_way == 'linear_based':

            sujet_group = ['allsujet', 'rep', 'non_rep']

            xr_dict = {'group' : sujet_group, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'percentile_type' : ['up', 'down']}
            xr_surr = xr.DataArray(data=np.zeros((len(sujet_group), len(conditions), len(odor_list), len(chan_list_eeg), 2)), dims=xr_dict.keys(), coords=xr_dict.values())

            sujet_best_list_erp = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
            sujet_no_respond = np.array([sujet for sujet in sujet_list_erp if sujet not in sujet_best_list_erp])

            for group in sujet_group:

                #cond = 'CO2'
                for cond in conditions:

                    #odor = odor_list[0]
                    for odor in odor_list:

                        print(group, cond, odor)

                        #nchan_i, nchan = 0, chan_list_eeg[0]
                        for nchan_i, nchan in enumerate(chan_list_eeg):
                        
                            if group == 'allsujet':
                                data_cond = xr_data.loc[:,cond,odor,nchan,:].values
                            if group == 'rep':
                                data_cond = xr_data.loc[sujet_best_list_erp,cond,odor,nchan,:].values
                            if group == 'no_rep':
                                data_cond = xr_data.loc[sujet_no_respond,cond,odor,nchan,:].values

                            _ERP_surr = np.zeros((ERP_n_surrogate))

                            for surr_i in range(ERP_n_surrogate):

                                surr_i_mean = shuffle_data_ERP_linear_based(data_cond).mean(axis=0)

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

                            xr_surr.loc[group, cond, odor, nchan, 'up'] = np.percentile(_ERP_surr, 95, axis=0)
                            xr_surr.loc[group, cond, odor, nchan, 'down'] = np.percentile(_ERP_surr, 5, axis=0)

        else:
            
            for sujet in sujet_list_erp:

                print(sujet)

                respfeatures = load_respfeatures(sujet)

                data_chunk_allcond = {}

                t_start_PPI = PPI_time_vec[0]
                t_stop_PPI = PPI_time_vec[1]

                #cond = 'MECA'
                for cond in conditions:

                    data_chunk_allcond[cond] = {}

                    #odor = odor_list[0]
                    for odor in odor_list:

                        data_chunk_allcond[cond][odor] = {}

                        #### load
                        data = load_data_sujet(sujet, cond, odor)
                        data = data[:len(chan_list_eeg),:]

                        respfeatures_i = respfeatures[cond][odor]
                        inspi_starts = respfeatures_i.query(f"select == 1")['inspi_index'].values

                        #### low pass 45Hz
                        for chan_i, chan in enumerate(chan_list_eeg):
                            data[chan_i,:] = iirfilt(data[chan_i,:], srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)

                        #nchan_i, nchan = 0, chan_list_eeg[0]
                        for nchan_i, nchan in enumerate(chan_list_eeg):

                            #### chunk
                            stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                            data_stretch = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

                            #### low pass 45Hz
                            x = data[nchan_i,:]
                            x = iirfilt(x, srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)

                            x_mean, x_std = x.mean(), x.std()
                            microV_SD = int(x_std*1e6)

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
                                plt.title(f'{cond} {odor} : {data_stretch.shape[0]}, 3SD : {microV_SD}')
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

                            data_chunk_allcond[cond][odor][nchan] = data_stretch_clean

                #### regroup FR_CV
                data_chunk_allcond['VS'] = {}

                for odor in odor_list:

                    data_chunk_allcond['VS'][odor] = {}

                    #### low pass 45Hz
                    for nchan_i, nchan in enumerate(chan_list_eeg):

                        data_chunk_allcond['VS'][odor][nchan] = np.concatenate([data_chunk_allcond['FR_CV_1'][odor][nchan], data_chunk_allcond['FR_CV_2'][odor][nchan]], axis=0)

                data_chunk_allcond['FR_CV_1'] = {}
                data_chunk_allcond['FR_CV_2'] = {}

                #### compute ERP surr
                if shuffle_way == 'intra_cond':

                    cond = 'VS'

                    #odor = odor_list[0]
                    for odor in odor_list:

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

                            xr_surr.loc[sujet, cond, odor, nchan, 'up', :] = up_percentile
                            xr_surr.loc[sujet, cond, odor, nchan, 'down', :] = down_percentile

                if shuffle_way == 'inter_cond':

                    ERP_surr = {}

                    #cond = 'MECA'
                    for cond in conditions:

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

                    #cond = 'MECA'
                    for cond in conditions:

                        #odor = odor_list[0]
                        for odor in odor_list:

                            #nchan_i, nchan = 0, chan_list_eeg[0]
                            for nchan_i, nchan in enumerate(chan_list_eeg):
                            
                                data = data_chunk_allcond[cond][odor][nchan]

                                _ERP_surr = np.zeros((ERP_n_surrogate, data.shape[1]))

                                for surr_i in range(ERP_n_surrogate):

                                    _ERP_surr[surr_i,:] = shuffle_data_ERP(data)

                                xr_surr.loc[sujet, cond, odor, nchan, 'up', :] = np.percentile(_ERP_surr, 98, axis=0)
                                xr_surr.loc[sujet, cond, odor, nchan, 'down', :] = np.percentile(_ERP_surr, 2, axis=0)

        #### save data
        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_surr.to_netcdf(f'allsujet_ERP_surrogates_{shuffle_way}.nc')

    return xr_surr



def get_PPI_count(xr_data):

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', f'PPI_count_linear_based.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        xr_PPI_count = xr.open_dataarray(f'PPI_count_linear_based.nc')

    else:

        t_start_PPI = PPI_time_vec[0]
        t_stop_PPI = PPI_time_vec[1]

        PPI_lm_start = PPI_lm_time[0]
        PPI_lm_stop = PPI_lm_time[1] 

        time_vec = np.linspace(t_start_PPI, t_stop_PPI, xr_data['time'].shape[-1])
        time_vec_mask = (time_vec > PPI_lm_start) & (time_vec < PPI_lm_stop)

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'blind_evaluation'))
        df_blind_eva = pd.read_excel('df_PPI_blind_evaluation.xlsx')

        examinateur_list = ['JG', 'MCN', 'TS']

        xr_dict = {'examinateur' : examinateur_list, 'sujet' : sujet_list_erp, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg}
        xr_PPI_count = xr.DataArray(data=np.zeros((len(examinateur_list), len(sujet_list_erp), len(conditions), len(odor_list), len(chan_list_eeg))), dims=xr_dict.keys(), coords=xr_dict.values())

        #examinateur = examinateur_list[2]
        for examinateur in examinateur_list:

            if examinateur == 'JG':

                for sujet in sujet_list_erp:

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

                            print('compute erp')
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

                            #nchan_i, nchan = 0, chan_list_eeg[0]
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
                                    plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                                    plt.plot(time_vec, data_chunk.mean(axis=0), color='r')
                                    plt.title(f'{cond} {odor} : {data_chunk.shape[0]}, 3SD : {microV_SD}')
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
                    # data_chunk_allcond['VS'] = {}
                    # data_value_microV['VS'] = {}

                    # for odor in odor_list:

                    #     data_chunk_allcond['VS'][odor] = {}
                    #     data_value_microV['VS'][odor] = {}

                    #     #### low pass 45Hz
                    #     for nchan_i, nchan in enumerate(chan_list_eeg):

                    #         data_chunk_allcond['VS'][odor][nchan] = np.concatenate([data_chunk_allcond['FR_CV_1'][odor][nchan], data_chunk_allcond['FR_CV_2'][odor][nchan]], axis=0)
                    #         data_value_microV['VS'][odor][nchan] = data_value_microV['FR_CV_1'][odor][nchan] + data_value_microV['FR_CV_2'][odor][nchan] / 2

                    # data_chunk_allcond['FR_CV_1'] = {}
                    # data_chunk_allcond['FR_CV_2'] = {}

                    # data_value_microV['FR_CV_1'] = {}
                    # data_value_microV['FR_CV_2'] = {}

                    #cond = 'CO2'
                    for cond in conditions:

                        #odor = odor_list[0]
                        for odor in odor_list:

                            print('compute surr')
                            print(sujet, cond, odor)

                            #nchan_i, nchan = 0, chan_list_eeg[0]
                            for nchan_i, nchan in enumerate(chan_list_eeg):
                            
                                data_cond = data_chunk_allcond[cond][odor][nchan]

                                Y = data_cond.mean(axis=0)[time_vec_mask]
                                X = time_vec[time_vec_mask]

                                slope_observed, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)

                                _ERP_surr = np.zeros((ERP_n_surrogate))

                                for surr_i in range(ERP_n_surrogate):

                                    ERP_shuffle = np.zeros(data_cond.shape)

                                    for erp_i in range(data_cond.shape[0]):

                                        cut = np.random.randint(0, data_cond.shape[1], 1)[0]

                                        ERP_shuffle[erp_i,:data_cond[:,cut:].shape[1]] = data_cond[erp_i,cut:]
                                        ERP_shuffle[erp_i,data_cond[:,cut:].shape[1]:] = data_cond[erp_i,:cut]
                                            
                                    surr_i_mean = ERP_shuffle.mean(axis=0)

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

                                if slope_observed < np.percentile(_ERP_surr, 5):

                                    xr_PPI_count.loc[examinateur, sujet, cond, odor, nchan] = 1

            else:

                #sujet = sujet_list_erp[0]
                for sujet in sujet_list_erp:

                    if sujet in ['28NT']:
                        continue

                    #cond = 'CO2'
                    for cond in conditions:

                        if cond in ['FR_CV_1', 'FR_CV_2']:
                            continue

                        #odor = odor_list[0]
                        for odor in odor_list:

                            #nchan_i, nchan = 0, chan_list_eeg[0]
                            for nchan_i, nchan in enumerate(chan_list_eeg):

                                if nchan in ['Cz', 'Fz']:

                                    _eva = df_blind_eva.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}' and nchan == '{nchan}'")[examinateur].values[0]
                                    xr_PPI_count.loc[examinateur, sujet, cond, odor, nchan] = _eva

        #### save data
        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_PPI_count.to_netcdf(f'PPI_count_linear_based.nc')

    return xr_PPI_count






################################
######## ERP PLOT ########
################################



def plot_ERP(xr_data, xr_lm_data, xr_lm_pred, xr_lm_pred_coeff, xr_surr):

    print('ERP PLOT', flush=True)

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    sujet_group = ['allsujet', 'rep', 'non_rep']

    sujet_best_list_erp = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list_erp if sujet not in sujet_best_list_erp])

    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
    time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)
    mask_time_PPI = (time_vec > -2.5) & (time_vec < 0)

    ######## SUMMARY NCHAN ########

    print('PLOT SUMMARY ERP')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary'))

    #group = sujet_group[0]
    for group in sujet_group:

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

            fig.set_figheight(10)
            fig.set_figwidth(10)

            scales_val = {'min' : [], 'max' : []}

            #cond_i, cond = 2, conditions[2]
            for cond_i, cond in enumerate(conditions):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                        n_sujet = xr_data.loc[:, cond, odor, nchan, :]['sujet'].shape[0]
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :].mean('sujet').values
                        n_sujet = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :]['sujet'].shape[0]
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                        n_sujet = xr_data.loc[sujet_no_respond, cond, odor, nchan, :]['sujet'].shape[0]
                    scales_val['min'].append(data_stretch.min())
                    scales_val['max'].append(data_stretch.max())

            scales_val['min'] = np.array(scales_val['min']).min()
            scales_val['max'] = np.array(scales_val['max']).max()

            plt.suptitle(f'{nchan} {group} {n_sujet}')

            #cond_i, cond = 1, 'MECA'
            for cond_i, cond in enumerate(conditions):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[:, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[:, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :].mean('sujet').values
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_best_list_erp, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_best_list_erp, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'down', :].mean('sujet').values

                    data_surr_down = xr_surr.loc[group, cond, odor, nchan, 'down'].values

                    ax = axs[odor_i, cond_i]

                    # if cond != 'VS':

                    #     if (data_stretch[mask_time_PPI] < data_surr_down[mask_time_PPI]).any() and (data_stretch[mask_time_PPI] > data_surr_up[mask_time_PPI]).any():
                            
                    #         PPI_presence = 1

                    #         if debug:

                    #             plt.plot(data_stretch[mask_time_PPI])
                    #             plt.plot(data_surr_down[mask_time_PPI])
                    #             plt.plot(data_surr_up[mask_time_PPI])
                    #             plt.show()

                    #     else:

                    #         PPI_presence = 0

                    if xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'slope'].values < data_surr_down:
                        
                        PPI_presence = 1

                        if debug:

                            plt.plot(data_stretch[mask_time_PPI])
                            plt.plot(data_surr_down[mask_time_PPI])
                            plt.show()

                            for nchan in chan_list_eeg:

                                plt.plot(time_vec, xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values)
                                plt.title(nchan)
                                plt.gca().invert_yaxis()
                                plt.show()

                                count, bins, _ = plt.hist(xr_surr.loc[sujet_best_list_erp, cond, odor, nchan, 'down', time_vec[0]], bins=50)
                                plt.vlines(xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'slope'].values, ymin=count.min(), ymax=count.max(), color='r')
                                plt.vlines(np.median(xr_surr.loc[sujet_best_list_erp, cond, odor, nchan, 'down', time_vec[0]]), ymin=count.min(), ymax=count.max(), color='g')
                                plt.show()

                            for nchan in chan_list_eeg:

                                for sujet in sujet_no_respond:

                                    plt.plot(time_vec, xr_data.loc[sujet, cond, odor, nchan, :].values, alpha=0.5)

                                plt.plot(time_vec, xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values, alpha=1)
                                plt.title(nchan)
                                plt.gca().invert_yaxis()
                                plt.show()     

                    else:

                        PPI_presence = 0

                    if cond_i ==0:
                        ax.set_ylabel(odor)

                    ax.set_title(f"{cond} \n slope:{np.round(xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'slope'].values, 3)} / r2:{np.round(xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'coeff'].values, 3)} \n PPI:{PPI_presence}", fontweight='bold')

                    ax.set_ylim(allplot_erp_ylim[0], allplot_erp_ylim[1])

                    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                    time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

                    ax.plot(time_vec, data_stretch)
                    # ax.plot(time_vec, data_stretch.std(axis=0), color='k', linestyle='--')
                    # ax.plot(time_vec, -data_stretch.std(axis=0), color='k', linestyle='--')

                    # if cond != 'VS':
                    #     ax.plot(time_vec, data_surr_up, color='k', linestyle='--')
                    #     ax.plot(time_vec, data_surr_down, color='k', linestyle='--')

                    time_vec_lm = np.arange(PPI_lm_start, PPI_lm_stop, 1/srate)
                    ax.plot(time_vec_lm, xr_lm_pred.loc[group, cond, odor, nchan, :], color='r', linewidth=3)

                    ax.invert_yaxis()

                    ax.vlines(0, ymin=allplot_erp_ylim[0], ymax=allplot_erp_ylim[1], colors='g')  

            fig.tight_layout()

            # plt.show()

            #### save
            fig.savefig(f'{nchan}_{group}.jpeg', dpi=150)

            fig.clf()
            plt.close('all')
            gc.collect()

    ######## TOPOPLOT SLOPE ########

    print('TOPOPLOT SLOPE')

    #### scales
    val = np.array([])

    for odor_i in odor_list:

        for cond in conditions:

            val = np.append(val, xr_lm_pred_coeff.loc['allsujet', cond, odor, :, 'slope'])

    scale_min = val.min()
    scale_max = val.max()

    #### plot
    for group in sujet_group:

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

        if group == 'allsujet':
            plt.suptitle(f'{group} slope (s{xr_data.shape[0]})')
        if group == 'rep':
            plt.suptitle(f'{group} slope (s{len(sujet_best_list_erp)})')
        if group == 'non_rep':
            plt.suptitle(f'{group} slope (s{len(sujet_no_respond)})')

        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(conditions):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(cond, fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                data_plot = xr_lm_pred_coeff.loc[group, cond, odor, :, 'slope'].values*-1

                mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
        fig.savefig(f'slope_{group}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    #### for cond
    for c, cond in enumerate(conditions):

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
        plt.suptitle(f'slope')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        for c, sujet_type in enumerate(['respond', 'no_respond']):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    if sujet_type == 'respond':
                        ax.set_title(f'{cond} {sujet_type} (s{len(sujet_best_list_erp)})', fontweight='bold', rotation=0)
                    if sujet_type == 'no_respond':
                        ax.set_title(f'{cond} {sujet_type} (s{len(sujet_no_respond)})', fontweight='bold', rotation=0)

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
        fig.savefig(f'slope_{cond}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    ######## TOPOPLOT D_INT ########

    print('TOPOPLOT D INT')

    #### scales
    val = np.array([])

    for odor_i in odor_list:

        for cond in conditions:

            val = np.append(val, xr_lm_data.loc[:, cond, odor, :, 'd_int'].mean('sujet'))

    scale_min = val.min()
    scale_max = val.max()

    #### plot
    for group in sujet_group:

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

        if group == 'allsujet':
            plt.suptitle(f'{group} d_int (s{xr_data.shape[0]})')
        if group == 'rep':
            plt.suptitle(f'{group} d_int (s{len(sujet_best_list_erp)})')
        if group == 'non_rep':
            plt.suptitle(f'{group} d_int (s{len(sujet_no_respond)})')

        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(conditions):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(cond, fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                if group == 'allsujet':
                    data_plot = xr_lm_data.loc[:, cond, odor, :, 'd_int'].mean('sujet').values*-1
                if group == 'rep':
                    data_plot = xr_lm_data.loc[sujet_best_list_erp, cond, odor, :, 'd_int'].mean('sujet').values*-1
                if group == 'non_rep':
                    data_plot = xr_lm_data.loc[sujet_no_respond, cond, odor, :, 'd_int'].mean('sujet').values*-1

                mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
        fig.savefig(f'd_int_{group}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    #### for cond
    for c, cond in enumerate(conditions):

        #### scales
        val = np.array([])
            
        for sujet_type_i, sujet_type in enumerate(['respond', 'no_respond']):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):
                
                if sujet_type == 'respond':
                    data_plot = xr_lm_data.loc[sujet_best_list_erp, cond, odor, :, 'd_int'].mean('sujet').values*-1
                if sujet_type == 'no_respond':
                    data_plot = xr_lm_data.loc[sujet_no_respond, cond, odor, :, 'd_int'].mean('sujet').values*-1

                val = np.append(val, data_plot)

        scale_min = val.min()
        scale_max = val.max()

        #### plot
        fig, axs = plt.subplots(nrows=len(odor_list), ncols=2)

        plt.suptitle(f'd_int')

        fig.set_figheight(10)
        fig.set_figwidth(10)

        for c, sujet_type in enumerate(['respond', 'no_respond']):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    if sujet_type == 'respond':
                        ax.set_title(f'{cond} {sujet_type} (s{len(sujet_best_list_erp)})', fontweight='bold', rotation=0)
                    if sujet_type == 'no_respond':
                        ax.set_title(f'{cond} {sujet_type} (s{len(sujet_no_respond)})', fontweight='bold', rotation=0)

                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                if sujet_type == 'respond':
                    data_plot = xr_lm_data.loc[sujet_best_list_erp, cond, odor, :, 'd_int'].mean('sujet').values*-1
                if sujet_type == 'no_respond':
                    data_plot = xr_lm_data.loc[sujet_no_respond, cond, odor, :, 'd_int'].mean('sujet').values*-1

                mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
        fig.savefig(f'd_int_{cond}_topo_respond.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    ######## TOPOPLOT PPI ########

    print('TOPOPLOT PPI')

    #### plot allsujet

    scale = {}

    for group in sujet_group:

        _max = []

        for c, cond in enumerate(conditions):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):
                
                if group == 'allsujet':
                    PPI_count = xr_PPI_count.loc['JG', :, cond, odor, :].sum('sujet')
                if group == 'rep':
                    PPI_count = xr_PPI_count.loc['JG', sujet_best_list_erp, cond, odor, :].sum('sujet')
                if group == 'non_rep':
                    PPI_count = xr_PPI_count.loc['JG', sujet_no_respond, cond, odor, :].sum('sujet')

                _max.append(PPI_count.max())

        scale[group] = np.array(_max).max()

    for group in sujet_group:

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

        if group == 'allsujet':
            plt.suptitle(f'PPI {group} (s{xr_data.shape[0]})')
        if group == 'rep':
            plt.suptitle(f'PPI {group} (s{len(sujet_best_list_erp)})')
        if group == 'non_rep':
            plt.suptitle(f'PPI {group} (s{len(sujet_no_respond)})')

        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(conditions):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(f'{cond}', fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                if group == 'allsujet':
                    PPI_count = xr_PPI_count.loc['JG', :, cond, odor, :].sum('sujet')
                    vmax = xr_data.shape[0]
                if group == 'rep':
                    PPI_count = xr_PPI_count.loc['JG', sujet_best_list_erp, cond, odor, :].sum('sujet')
                    vmax = 15
                if group == 'non_rep':
                    PPI_count = xr_PPI_count.loc['JG', sujet_no_respond, cond, odor, :].sum('sujet')
                    vmax = 15
                
                mne.viz.plot_topomap(PPI_count, info, vlim=(0, vmax), axes=ax, show=False)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
        fig.savefig(f'PPI_{group}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    #### PPI for cond

    for c, cond in enumerate(conditions):

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=2)

        plt.suptitle(f'PPI {cond}')

        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, sujet_type in enumerate(['respond', 'no_respond']):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    if sujet_type == 'respond':
                        ax.set_title(f"{sujet_type} (s{len(sujet_best_list_erp)})", fontweight='bold', rotation=0)
                    if sujet_type == 'no_respond':
                        ax.set_title(f"{sujet_type} (s{len(sujet_no_respond)})", fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')

                if sujet_type == 'respond':
                    PPI_count = xr_PPI_count.loc['JG', sujet_best_list_erp, cond, odor, :].sum('sujet')
                if sujet_type == 'no_respond':
                    PPI_count = xr_PPI_count.loc['JG', sujet_no_respond, cond, odor, :].sum('sujet')
                
                mne.viz.plot_topomap(PPI_count, info, vlim=(0, 15), axes=ax, show=False)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
        fig.savefig(f'PPI_{cond}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()




def plot_ERP_CO2(xr_data):

    print('ERP PLOT', flush=True)

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    sujet_group = ['allsujet', 'rep', 'non_rep']

    sujet_best_list_erp = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list_erp if sujet not in sujet_best_list_erp])

    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
    time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)
    mask_time_PPI = (time_vec > -2.5) & (time_vec < 0)

    conditions_diff = ['MECA', 'CO2', 'FR_CV_2']

    ######## SUMMARY NCHAN ########

    print('PLOT SUMMARY ERP')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff'))

    conditions = [cond for cond in conditions if cond != 'FR_CV_1']

    #group = sujet_group[0]
    for group in sujet_group:

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

            fig.set_figheight(10)
            fig.set_figwidth(10)

            scales_val = {'min' : [], 'max' : []}

            #cond_i, cond = 2, conditions[2]
            for cond_i, cond in enumerate(conditions):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                        n_sujet = xr_data.loc[:, cond, odor, nchan, :]['sujet'].shape[0]
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :].mean('sujet').values
                        n_sujet = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :]['sujet'].shape[0]
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                        n_sujet = xr_data.loc[sujet_no_respond, cond, odor, nchan, :]['sujet'].shape[0]
                    scales_val['min'].append(data_stretch.min())
                    scales_val['max'].append(data_stretch.max())

            scales_val['min'] = np.array(scales_val['min']).min()
            scales_val['max'] = np.array(scales_val['max']).max()

            plt.suptitle(f'{nchan} {group} {n_sujet}')

            #cond_i, cond = 1, 'MECA'
            for cond_i, cond in enumerate(conditions_diff):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                        sem = xr_data.loc[:, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        sem_baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, 'FR_CV_1', odor, nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[:, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[:, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :].mean('sujet').values
                        sem = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[sujet_best_list_erp, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        sem_baseline = xr_data.loc[sujet_best_list_erp, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_erp, 'FR_CV_1', odor, nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_best_list_erp, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_best_list_erp, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                        sem = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[sujet_no_respond, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        sem_baseline = xr_data.loc[sujet_no_respond, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond, 'FR_CV_1', odor, nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'down', :].mean('sujet').values

                    ax = axs[odor_i, cond_i]

                    ax.set_title(f"{cond}")

                    ax.set_ylim(allplot_erp_ylim[0], allplot_erp_ylim[1])

                    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                    time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

                    ax.plot(time_vec, data_stretch, label=cond, color='r')
                    ax.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

                    ax.plot(time_vec, baseline, label='VS', color='b')
                    ax.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')
                    
                    ax.invert_yaxis()

                    ax.vlines(0, ymin=allplot_erp_ylim[0], ymax=allplot_erp_ylim[1], colors='g')  

            fig.tight_layout()
            plt.legend()

            # plt.show()

            #### save
            fig.savefig(f'{nchan}_{group}.jpeg', dpi=150)

            fig.clf()
            plt.close('all')
            gc.collect()

    ######## TOPOPLOT DIFF ########

    print('TOPOPLOT DIFF')

    time_diff = [-2, -1.5, -1, -0.5, 0, 0.5, 1]
    time_diff_step = 0.5
    dict_diff = {'group' : sujet_group, 'cond' : conditions_diff, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'time' : time_diff}
    data_diff = np.zeros((len(sujet_group), len(conditions_diff), len(odor_list), len(chan_list_eeg), len(time_diff)))

    xr_data_diff_mean = xr.DataArray(data=data_diff, dims=dict_diff.keys(), coords=dict_diff.values())
    xr_data_diff_point = xr.DataArray(data=data_diff, dims=dict_diff.keys(), coords=dict_diff.values())

    #group = sujet_group[0]
    for group in sujet_group:

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            for cond_i, cond in enumerate(conditions_diff):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                        baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :].mean('sujet').values
                        baseline = xr_data.loc[sujet_best_list_erp, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                        baseline = xr_data.loc[sujet_no_respond, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        
                    diff = data_stretch - baseline

                    for time_start in time_diff:

                        time_stop = time_start + time_diff_step

                        mask_sel_time_diff_mean = (time_vec >= time_start) & (time_vec <= time_stop)
                        mask_sel_time_diff_point = (time_vec >= time_start-0.01) & (time_vec <= time_start+0.01)

                        xr_data_diff_mean.loc[group, cond, odor, nchan, time_start] = np.median(diff[mask_sel_time_diff_mean])

                        xr_data_diff_point.loc[group, cond, odor, nchan, time_start] = np.median(diff[mask_sel_time_diff_point])


    #### select params
    # xr_data_diff = xr_data_diff_mean
    xr_data_diff = xr_data_diff_point

    classing_letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

    #### scales
    scale_min = xr_data_diff.loc['allsujet', :, :, :, :].min().values
    scale_max = xr_data_diff.loc['allsujet', :, :, :, :].max().values

    #### plot
    for group in sujet_group:

        for time_start_i, time_start in enumerate(time_diff):

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions_diff))

            if group == 'allsujet':
                plt.suptitle(f'{group} {time_start} diff (s{xr_data.shape[0]})')
            if group == 'rep':
                plt.suptitle(f'{group} {time_start} diff (s{len(sujet_best_list_erp)})')
            if group == 'non_rep':
                plt.suptitle(f'{group} {time_start} diff (s{len(sujet_no_respond)})')

            fig.set_figheight(10)
            fig.set_figwidth(10)

            #c, cond = 0, 'FR_CV_1'
            for c, cond in enumerate(conditions_diff):

                #r, odor_i = 0, odor_list[0]
                for r, odor in enumerate(odor_list):

                    #### plot
                    ax = axs[r, c]

                    if r == 0:
                        ax.set_title(cond, fontweight='bold', rotation=0)
                    if c == 0:
                        ax.set_ylabel(f'{odor}')
                    
                    data_plot = xr_data_diff.loc[group, cond, odor, :, time_start].values*-1

                    mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

            # plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_diff'))
            fig.savefig(f'diff_baseline_{group}_{classing_letter[time_start_i]}{time_start}_topo.jpeg', dpi=150)
            fig.clf()
            plt.close('all')
            gc.collect()

    #### for cond
    for c, cond in enumerate(conditions_diff):

        for time_start_i, time_start in enumerate(time_diff):

            #### scales
            scale_min = xr_data_diff.loc[group, cond, :, :, :].values.min()
            scale_max = xr_data_diff.loc[group, cond, :, :, :].values.max()

            #### plot
            fig, axs = plt.subplots(nrows=len(odor_list), ncols=2)
            plt.suptitle(f'{time_start} diff')
            fig.set_figheight(10)
            fig.set_figwidth(10)

            for c, sujet_type in enumerate(['respond', 'no_respond']):

                #r, odor_i = 0, odor_list[0]
                for r, odor in enumerate(odor_list):

                    #### plot
                    ax = axs[r, c]

                    if r == 0:
                        if sujet_type == 'respond':
                            ax.set_title(f'{cond} {sujet_type} (s{len(sujet_best_list_erp)})', fontweight='bold', rotation=0)
                        if sujet_type == 'no_respond':
                            ax.set_title(f'{cond} {sujet_type} (s{len(sujet_no_respond)})', fontweight='bold', rotation=0)

                    if c == 0:
                        ax.set_ylabel(f'{odor}')
                    
                    if sujet_type == 'respond':
                        data_plot = xr_data_diff.loc['rep', cond, odor, :, time_start].values
                    if sujet_type == 'no_respond':
                        data_plot = xr_data_diff.loc['non_rep', cond, odor, :, time_start].values

                    mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

            # plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_diff'))
            fig.savefig(f'{cond}_diff_baseline_{classing_letter[time_start_i]}{time_start}_topo.jpeg', dpi=150)
            fig.clf()
            plt.close('all')
            gc.collect()

    #### across time
    for group in sujet_group:

        for c, cond in enumerate(conditions_diff):

            #### scales
            scale_min = xr_data_diff.loc[group, cond, :, :, :].values.min()
            scale_max = xr_data_diff.loc[group, cond, :, :, :].values.max()

            #### plot
            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(time_diff))
            plt.suptitle(f'{group} {cond} diff')
            fig.set_figheight(10)
            fig.set_figwidth(10)

            for c, time_start in enumerate(time_diff):

                #r, odor_i = 0, odor_list[0]
                for r, odor in enumerate(odor_list):

                    #### plot
                    ax = axs[r, c]

                    if r == 0:
                        ax.set_title(f'{time_start}', fontweight='bold', rotation=0)

                    if c == 0:
                        ax.set_ylabel(f'{odor}')
                    
                    data_plot = xr_data_diff.loc[group, cond, odor, :, time_start].values

                    mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

            # plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_diff'))
            fig.savefig(f'time_{cond}_{group}_diff_baseline_topo.jpeg', dpi=150)
            fig.clf()
            plt.close('all')
            gc.collect()










def plot_ERP_rep_norep(xr_data):

    print('ERP PLOT', flush=True)

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    sujet_group = ['allsujet', 'rep', 'non_rep']

    sujet_best_list_erp = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list_erp if sujet not in sujet_best_list_erp])

    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
    time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)
    mask_time_PPI = (time_vec > -2.5) & (time_vec < 0)

    ######## SUMMARY NCHAN ########

    print('PLOT SUMMARY ERP')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary'))

    #nchan_i, nchan = 0, chan_list_eeg[0]
    for nchan_i, nchan in enumerate(chan_list_eeg):

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

        fig.set_figheight(10)
        fig.set_figwidth(10)

        scales_val = {'min' : [], 'max' : []}

        #cond_i, cond = 2, conditions[2]
        for cond_i, cond in enumerate(conditions):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                data_stretch = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :].mean('sujet').values
                n_sujet_rep = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :]['sujet'].shape[0]

                scales_val['min'].append(data_stretch.min())
                scales_val['max'].append(data_stretch.max())

                data_stretch = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                n_sujet_norep = xr_data.loc[sujet_no_respond, cond, odor, nchan, :]['sujet'].shape[0]

                scales_val['min'].append(data_stretch.min())
                scales_val['max'].append(data_stretch.max())

        scales_val['min'] = np.array(scales_val['min']).min()
        scales_val['max'] = np.array(scales_val['max']).max()

        plt.suptitle(f'{nchan} rep:{n_sujet_rep} no_rep:{n_sujet_norep}')

        #cond_i, cond = 1, 'MECA'
        for cond_i, cond in enumerate(conditions):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, cond_i]

                if cond_i == 0:
                    ax.set_ylabel(odor)

                data_stretch_rep = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :].mean('sujet').values
                sem_rep = xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_erp, cond, odor, nchan, :].shape[0])

                data_stretch_norep = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                sem_norep = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond, cond, odor, nchan, :].shape[0])

                ax.set_title(f"{cond}")

                ax.set_ylim(allplot_erp_ylim[0], allplot_erp_ylim[1])

                stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

                ax.plot(time_vec, data_stretch_rep, label='rep', color='r')
                ax.fill_between(time_vec, data_stretch_rep+sem_rep, data_stretch_rep-sem_rep, alpha=0.25, color='m')

                ax.plot(time_vec, data_stretch_norep, label='no_rep', color='b')
                ax.fill_between(time_vec, data_stretch_norep+sem_norep, data_stretch_norep-sem_norep, alpha=0.25, color='c')
                
                ax.invert_yaxis()

                ax.vlines(0, ymin=allplot_erp_ylim[0], ymax=allplot_erp_ylim[1], colors='g')  

        fig.tight_layout()
        plt.legend()

        # plt.show()

        #### save
        fig.savefig(f'{nchan}_rep_norep.jpeg', dpi=150)

        fig.clf()
        plt.close('all')
        gc.collect()


    ######## TOPOPLOT DIFF ########

    print('TOPOPLOT DIFF')

    time_diff = [-2, -1.5, -1, -0.5, 0, 0.5, 1]
    time_diff_step = 0.5
    dict_diff = {'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'time' : time_diff}
    data_diff = np.zeros((len(conditions), len(odor_list), len(chan_list_eeg), len(time_diff)))

    dict_diff_alltime = {'metric' : ['minmax', 'sum'], 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg}
    data_diff_alltime = np.zeros((len(['minmax', 'sum']), len(conditions), len(odor_list), len(chan_list_eeg)))

    xr_data_diff_mean = xr.DataArray(data=data_diff, dims=dict_diff.keys(), coords=dict_diff.values())
    xr_data_diff_point = xr.DataArray(data=data_diff, dims=dict_diff.keys(), coords=dict_diff.values())
    xr_data_diff_alltime = xr.DataArray(data=data_diff_alltime, dims=dict_diff_alltime.keys(), coords=dict_diff_alltime.values())

    sujet_best_list_erp_balanced = [sujet for sujet in sujet_best_list_erp if sujet not in ['01PD', '03VN']]

    #nchan_i, nchan = 0, chan_list_eeg[0]
    for nchan_i, nchan in enumerate(chan_list_eeg):

        for cond_i, cond in enumerate(conditions):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):
                    
                data_stretch_rep = xr_data.loc[sujet_best_list_erp_balanced, cond, odor, nchan, :].values
                
                data_stretch_norep = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].values

                diff = (data_stretch_norep - data_stretch_rep).mean(axis=0)
                
                xr_data_diff_alltime.loc['sum', cond, odor, nchan] = np.sum(diff)
                xr_data_diff_alltime.loc['minmax', cond, odor, nchan] = diff.max() - diff.min()

                for time_start in time_diff:

                    time_stop = time_start + time_diff_step

                    mask_sel_time_diff_mean = (time_vec >= time_start) & (time_vec <= time_stop)
                    mask_sel_time_diff_point = (time_vec >= time_start-0.01) & (time_vec <= time_start+0.01)

                    xr_data_diff_mean.loc[cond, odor, nchan, time_start] = np.median(diff[mask_sel_time_diff_mean])

                    xr_data_diff_point.loc[cond, odor, nchan, time_start] = np.median(diff[mask_sel_time_diff_point])


    #### select params
    # xr_data_diff = xr_data_diff_mean
    xr_data_diff = xr_data_diff_point

    classing_letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

    #### scales
    scale_min = xr_data_diff.min().values
    scale_max = xr_data_diff.max().values

    #### plot
    for time_start_i, time_start in enumerate(time_diff):

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

        plt.suptitle(f'{time_start} diff rep_norep')

        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(conditions):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(cond, fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                data_plot = xr_data_diff.loc[cond, odor, :, time_start].values*-1

                mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_diff'))
        fig.savefig(f'diff_repnorep_{classing_letter[time_start_i]}{time_start}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    #### across time
    for c, cond in enumerate(conditions):

        #### scales
        scale_min = xr_data_diff.loc[cond, :, :, :].values.min()
        scale_max = xr_data_diff.loc[cond, :, :, :].values.max()

        #### plot
        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(time_diff))
        plt.suptitle(f'{cond} diff')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        for c, time_start in enumerate(time_diff):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(f'{time_start}', fontweight='bold', rotation=0)

                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                data_plot = xr_data_diff.loc[cond, odor, :, time_start].values

                mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

        plt.suptitle(cond)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_diff'))
        fig.savefig(f'time_repnorep_{cond}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    #### all time

    #### scales
    for metric_alltime in ['minmax', 'sum']:

        scale_min = xr_data_diff_alltime.loc[metric_alltime].values.min()
        scale_max = xr_data_diff_alltime.loc[metric_alltime].values.max()

        #### plot
        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
        plt.suptitle(f'alltime diff {metric_alltime}')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        for c, cond in enumerate(conditions):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(f'{cond}', fontweight='bold', rotation=0)

                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                data_plot = xr_data_diff_alltime.loc[metric_alltime, cond, odor, :].values

                mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_diff'))
        fig.savefig(f'alltime_repnorep_{metric_alltime}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()







################################################
######## GENERATE PPI EVALUATION ########
################################################



def generate_ppi_evaluation(xr_data):

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    #### generate parameters
    dict = {'sujet' : [], 'cond' : [], 'odor' : [], 'nchan' : [], 'indice' : [], 'PPI' : []}

    chan_list_blind_evaluation = ['Cz', 'Fz']

    for sujet in sujet_list_erp:

        for cond_i, cond in enumerate(conditions):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                for nchan_i, nchan in enumerate(chan_list_blind_evaluation):

                    dict['sujet'].append(sujet)
                    dict['cond'].append(cond)
                    dict['odor'].append(odor)
                    dict['nchan'].append(nchan)
                    dict['indice'].append(0)
                    dict['PPI'].append(0)

    df_PPI_blind_evaluation = pd.DataFrame(dict)
    df_PPI_blind_evaluation = df_PPI_blind_evaluation.sample(frac=1).reset_index(drop=True)
    df_PPI_blind_evaluation['indice'] = np.arange(df_PPI_blind_evaluation.shape[0])+1

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'blind_evaluation'))
    df_PPI_blind_evaluation.to_excel('df_PPI_blind_evaluation.xlsx')

    ######## SUMMARY NCHAN ########

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'blind_evaluation'))

    for indice in range(df_PPI_blind_evaluation.shape[0]):

        print_advancement(indice, df_PPI_blind_evaluation.shape[0], [25, 50, 75])

        fig, ax = plt.subplots()

        plt.suptitle(indice+1)

        fig.set_figheight(10)
        fig.set_figwidth(10)

        sujet, cond, odor, nchan = df_PPI_blind_evaluation.iloc[indice].values[0], df_PPI_blind_evaluation.iloc[indice].values[1], df_PPI_blind_evaluation.iloc[indice].values[2], df_PPI_blind_evaluation.iloc[indice].values[3]

        data_stretch = xr_data.loc[sujet, cond, odor, nchan, :].values

        ax.set_ylim(-3, 3)

        stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
        time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

        ax.plot(time_vec, data_stretch)
        # ax.plot(time_vec, data_stretch.std(axis=0), color='k', linestyle='--')
        # ax.plot(time_vec, -data_stretch.std(axis=0), color='k', linestyle='--')

        ax.invert_yaxis()

        ax.vlines(0, ymin=-3, ymax=3, colors='g')  
        ax.hlines(0, xmin=PPI_time_vec[0], xmax=PPI_time_vec[-1], colors='g') 

        # plt.show()

        #### save
        fig.savefig(f'{indice+1}.jpeg', dpi=150)

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





def get_data_erp_respi():

    t_start_PPI = -4
    t_stop_PPI = 4

    #### extract zscore params
    zscore_prms = {'mean' : {}, 'std' : {}}

     #odor = odor_list[1]
    for odor in odor_list:

        zscore_prms['mean'][odor] = {}
        zscore_prms['std'][odor] = {}

    #sujet_i, sujet = 0, sujet_list_erp[0]
    for sujet_i, sujet in enumerate(sujet_list_erp):

        print_advancement(sujet_i, len(sujet_list_erp), [25, 50, 75])

        #odor = odor_list[0]
        for odor in odor_list:

            respi_concat = []

            #cond = 'FR_CV_1'
            for cond in conditions:

                #### load
                respi_concat = np.concatenate((respi_concat, load_data_sujet(sujet, cond, odor)[chan_list.index('PRESS'),:]), axis=0)

            if debug:

                plt.plot(respi_concat)
                plt.show()

            zscore_prms['mean'][odor][sujet] = np.array(respi_concat).mean()
            zscore_prms['std'][odor][sujet] = np.array(respi_concat).std()

    #### extract data
    data_respi = {}

    #sujet_i, sujet = 26, sujet_list_erp[26]
    for sujet_i, sujet in enumerate(sujet_list_erp):

        print_advancement(sujet_i, len(sujet_list_erp), [25, 50, 75])

        data_respi[sujet] = {}

        respfeatures = load_respfeatures(sujet)

        #cond = 'MECA'
        for cond in conditions:

            data_respi[sujet][cond] = {}

            #odor = odor_list[1]
            for odor in odor_list:

                data_respi[sujet][cond][odor] = {}

                #### load
                respi = load_data_sujet(sujet, cond, odor)[chan_list.index('PRESS'),:]
                respi = (respi - zscore_prms['mean'][odor][sujet]) / zscore_prms['std'][odor][sujet]

                if debug:

                    plt.plot(respi)
                    plt.show()

                respfeatures_i = respfeatures[cond][odor]
                inspi_starts = respfeatures_i.query(f"select == 1")['inspi_index'].values

                #### chunk
                stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                respi_stretch = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

                # x_mean, x_std = respi.mean(), respi.std()

                for start_i, start_time in enumerate(inspi_starts):

                    t_start = int(start_time + t_start_PPI*srate)
                    t_stop = int(start_time + t_stop_PPI*srate)

                    # respi_stretch[start_i, :] = (respi[t_start: t_stop] - x_mean) / x_std
                    # respi_stretch[start_i, :] = (respi[t_start: t_stop] - zscore_prms['mean'][sujet]) / zscore_prms['std'][sujet]
                    # respi_stretch[start_i, :] = respi[t_start: t_stop] - x_mean
                    respi_stretch[start_i, :] = respi[t_start: t_stop]

                if debug:

                    plt.plot(respi)
                    plt.vlines(inspi_starts, ymax=respi.max(), ymin=respi.min(), color='k')
                    plt.show()

                    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                    for inspi_i, _ in enumerate(inspi_starts):

                        plt.plot(time_vec, respi_stretch[inspi_i, :], alpha=0.3)

                    plt.vlines(0, ymax=respi_stretch.max(), ymin=respi_stretch.min(), color='k')
                    # plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                    plt.plot(time_vec, respi_stretch.mean(axis=0), color='r')
                    plt.title(f'{cond} {odor} : {respi_stretch.shape[0]}')
                    plt.show()

                data_respi[sujet][cond][odor] = respi_stretch

    # for sujet_i, sujet in enumerate(sujet_list_erp):

    #     #### normalize by VS
    #     val = np.array([])

    #     for odor_i, odor in enumerate(odor_list):

    #         val = np.concatenate((val, data_respi[sujet]['FR_CV_1'][odor].reshape(-1)), axis=0)
    #         val = np.concatenate((val, data_respi[sujet]['FR_CV_2'][odor].reshape(-1)), axis=0)

    #     VS_mean, VS_std = val.mean(), val.std()

    #     for cond in conditions:

    #         for odor_i, odor in enumerate(odor_list):

    #             data_respi[sujet][cond][odor] = (data_respi[sujet][cond][odor] - VS_mean) / VS_std

    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)
    dict = {'sujet' : sujet_list_erp, 'odor' : odor_list, 'cond' : conditions, 'times' : time_vec}
    data = np.zeros((len(sujet_list_erp), len(odor_list), len(conditions), time_vec.shape[0]))
    xr_respi = xr.DataArray(data=data, dims=dict.keys(), coords=dict.values())

    for sujet in sujet_list_erp:

        for odor in odor_list:

            for cond in conditions:

                xr_respi.loc[sujet, odor, cond, :] = np.mean(data_respi[sujet][cond][odor], axis=0)

    return xr_respi







def plot_mean_respi():

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', f'allsujet_ERP_respi.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('xr_respi ALREADY COMPUTED', flush=True)

        xr_respi = xr.open_dataarray(f'allsujet_ERP_respi.nc')
        xr_itl = xr.open_dataarray(f'allsujet_ERP_respi_itl.nc')

    else:

        xr_respi = get_data_erp_respi()
        xr_itl = get_data_itl()

        #### save data
        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_respi.to_netcdf('allsujet_ERP_respi.nc')
        xr_itl.to_netcdf('allsujet_ERP_respi_itl.nc')

    t_start_PPI = -4
    t_stop_PPI = 4

    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

    sujet_best_list_erp = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list_erp if sujet not in sujet_best_list_erp])

    max = np.array([(xr_itl.mean('sujet').values*-1).max(), (xr_respi.mean('sujet').values).max()]).max()
    min = np.array([(xr_itl.mean('sujet').values*-1).min(), (xr_respi.mean('sujet').min().values).min()]).min()

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'respi'))

    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

    fig, axs = plt.subplots(ncols=len(odor_list), figsize=(15,10))

    for odor_i, odor in enumerate(odor_list):

        ax = axs[odor_i]

        ax.set_title(f"{odor}")

        ax.plot(time_vec, xr_itl.mean('sujet').loc['VS',:].values*-1, label='VS_itl', color='g', linestyle=':', dashes=(5, 10))
        ax.plot(time_vec, xr_itl.mean('sujet').loc['CO2',:].values*-1, label='CO2_itl', color='r', linestyle=':', dashes=(5, 10))
        ax.plot(time_vec, xr_itl.mean('sujet').loc['ITL',:].values*-1, label='ITL_itl', color='b', linestyle=':', dashes=(5, 10))

        ax.plot(time_vec, xr_respi.loc[:, odor, 'FR_CV_1', :].mean('sujet'), label=f'VS_1', color='g')
        ax.plot(time_vec, xr_respi.loc[:, odor, 'FR_CV_2', :].mean('sujet'), label=f'VS_2', color='g')
        ax.plot(time_vec, xr_respi.loc[:, odor, 'CO2', :].mean('sujet'), label=f'CO2', color='r')
        ax.plot(time_vec, xr_respi.loc[:, odor, 'MECA', :].mean('sujet'), label=f'MECA', color='b')

        ax.vlines(0, ymin=min, ymax=max, color='k')

    plt.legend()
    
    plt.suptitle(f"comparison ITL")

    # plt.show()

    plt.savefig(f"allsujet_ERP_comparison_ITL.png")
    plt.close('all')


    #### plot sujet respi
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'respi'))
    sujet_group = ['allsujet', 'rep', 'no_rep']

    for group in sujet_group:

        if group == 'allsujet':
            xr_data = xr_respi
        if group == 'rep':
            xr_data = xr_respi.loc[sujet_best_list_erp]
        if group == 'no_rep':
            xr_data = xr_respi.loc[sujet_no_respond]

        max = (xr_data.mean('sujet').values).max()
        min = (xr_data.mean('sujet').values).min()

        fig, axs = plt.subplots(ncols=len(conditions), figsize=(15,10))

        for cond_i, cond in enumerate(conditions):

            ax = axs[cond_i]

            ax.set_title(f"{cond}")

            ax.plot(time_vec, xr_data.loc[:, 'o', cond, :].mean('sujet'), label=f'o', color='b')
            ax.plot(time_vec, xr_data.loc[:, '-', cond, :].mean('sujet'), label=f'-', color='r')
            ax.plot(time_vec, xr_data.loc[:, '+', cond, :].mean('sujet'), label=f'+', color='g')

            ax.set_ylim(min, max)

            ax.vlines(0, ymin=min, ymax=max, color='k')

        plt.legend()
        
        plt.suptitle(f"{group} ERP, n:{xr_data['sujet'].shape[0]}")

        # plt.show()

        plt.savefig(f"ERP_COND_mean_{group}.png")
        
        plt.close('all')

    for group in sujet_group:

        if group == 'allsujet':
            xr_data = xr_respi
        if group == 'rep':
            xr_data = xr_respi.loc[sujet_best_list_erp]
        if group == 'no_rep':
            xr_data = xr_respi.loc[sujet_no_respond]

        max = (xr_data.mean('sujet').values).max()
        min = (xr_data.mean('sujet').values).min()

        fig, axs = plt.subplots(ncols=len(odor_list), figsize=(15,10))

        for odor_i, odor in enumerate(odor_list):

            ax = axs[odor_i]

            ax.set_title(f"{odor}")

            ax.plot(time_vec, xr_data.loc[:, odor, 'FR_CV_1', :].mean('sujet'), label=f'FR_CV_1', color='c')
            ax.plot(time_vec, xr_data.loc[:,  odor, 'FR_CV_2', :].mean('sujet'), label=f'FR_CV_2', color='b')
            ax.plot(time_vec, xr_data.loc[:,  odor, 'CO2', :].mean('sujet'), label=f'CO2', color='r')
            ax.plot(time_vec, xr_data.loc[:,  odor, 'MECA', :].mean('sujet'), label=f'MECA', color='g')

            ax.set_ylim(min, max)

            ax.vlines(0, ymin=min, ymax=max, color='k')

        plt.legend()
        
        plt.suptitle(f"{group} ERP, n:{xr_data['sujet'].shape[0]}")

        # plt.show()

        plt.savefig(f"ERP_ODOR_mean_{group}.png")
        
        plt.close('all')

    if debug:

        cond = 'FR_CV_1'
        odor = 'o'

        for sujet in sujet_list_erp:

            plt.plot(time_vec, xr_data.loc[sujet, odor, cond, :], label=str(sujet.data), alpha=0.25)

        plt.plot(time_vec, xr_data.loc[:, odor, cond, :].mean('sujet'), label=str(sujet.data), alpha=1, color='r')
        plt.vlines(0, ymin=xr_data.loc[:, odor, cond, :].min(), ymax=xr_data.loc[:, odor, cond, :].max(), color='k')
        plt.title(f'allsujet {cond} {odor}: {sujet_list_erp.shape[0]}')
        # plt.legend()
        plt.show()

        for sujet in sujet_list:

            fig, axs = plt.subplots(ncols=len(odor_list))

            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i]

                for cond in conditions:

                    ax.plot(time_vec, xr_data.loc[sujet, odor, cond, :], label=cond)

            plt.legend()
            plt.suptitle(sujet)
            plt.show()



        plt.plot(xr_data.loc[:, 'o', cond, :].mean('sujet'))
        plt.plot(xr_data.loc[:, '+', cond, :].mean('sujet'))
        plt.plot(xr_data.loc[:, '-', cond, :].mean('sujet'))









################################
######## SLOPE REG ########
################################

def plot_slope_versus_discomfort(xr_lm_data, conditions):

    os.chdir(os.path.join(path_data, 'psychometric'))

    df_mdp = pd.read_excel('OLFADYS_mdp.xlsx')
    df_q = pd.read_excel('OLFADYS_questionnaire.xlsx')

    val = np.zeros((df_q['sujet'].unique().shape[0]*df_q['session'].unique().shape[0]*df_q['cond'].unique().shape[0], 4), dtype='object')
    i = 0
    for sujet_i, sujet in enumerate(df_q['sujet'].unique()):
        for session_i, session in enumerate(df_q['session'].unique()):
            for cond_i, cond in enumerate(df_q['cond'].unique()):
                df_i = df_q.query(f"sujet == '{sujet}' & session == '{session}' & cond == '{cond}'")
                val_p = 300 - df_i[df_i['question'].isin([1, 4, 5])]['val'].sum()
                val_n = df_i[df_i['question'].isin([2, 3, 6])]['val'].sum()
                val_stai = ((val_n + val_p)/600)*100
                val[i, :] = np.array([sujet, session, cond, val_stai])
                i += 1
    df_stai = pd.DataFrame(val, columns=['sujet', 'session', 'cond', 'val'])
    df_stai['val'] = df_stai['val'].astype(np.float64) 

    df_relax = df_q.query(f"question == 7").drop(['question', 'raw_val', 'scale'], axis=1)

    question_A2 = ['A2_1', 'A2_2', 'A2_3', 'A2_4', 'A2_5']
    df_A2 = df_mdp.query(f"question in {question_A2}").groupby(['sujet', 'session', 'cond']).mean().reset_index()

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

    xr_respi = xr.open_dataarray(f'allsujet_ERP_respi.nc')

    os.chdir(os.path.join(path_data, 'respi_detection'))
    df_CO2 = pd.read_excel('OLFADYS_alldata_mean.xlsx').query(f"cond == 'CO2'")[['sujet', 'cond', 'odor', 'PetCO2', 'VAS_S', 'VAS_A']]
    df_CO2 = df_CO2.reset_index(drop=True)
    
    for row_i in range(df_CO2.shape[0]):
        if df_CO2.iloc[row_i]['odor'] == 'p':
            df_CO2['odor'][row_i] = '+'
        if df_CO2.iloc[row_i]['odor'] == 'n':
            df_CO2['odor'][row_i] = '-'

    os.chdir(os.path.join(path_data, 'respi_detection'))
    df_respi_paris = pd.read_excel('OLFADYS_alldata_mean.xlsx').query(f"sujet in {sujet_list_erp.tolist()}").reset_index(drop=True)

    for row_i in range(df_respi_paris.shape[0]):
        if df_respi_paris.iloc[row_i]['odor'] == 'p':
            df_respi_paris['odor'][row_i] = '+'
        if df_respi_paris.iloc[row_i]['odor'] == 'n':
            df_respi_paris['odor'][row_i] = '-'

    #### A2
    reg_slope_dis = {'sujet' : [], 'odor' : [], 'cond' : [], 'nchan' : [], 'slope' : [], 'A2' : [], 'STAI' : [], 'relax' : [], 'rep' : [], 
                     'amplitude' : [], 'amp_mean_diff' : [], 'PCO2' : [], 'VAS_S' : [], 'VAS_A' : [], 'VT' : [], 'Ve' : []}

    #sujet = sujet_list_erp[0]
    for sujet in sujet_list_erp:

        respfeatures = load_respfeatures(sujet)

        for odor in odor_list:

            for cond in conditions:

                if cond in ['FR_CV_1', 'FR_CV_2']:
                    continue

                for nchan in chan_list_eeg:

                    slope = float(xr_lm_data.loc[sujet, cond, odor, nchan, 'slope'].data)

                    sujet_sel = f"{sujet[2:]}{sujet[:2]}"
                    A2 = float(df_A2.query(f"sujet == '{sujet_sel}' and session == '{odor}' and cond == '{cond}'")['value'].values)
                    relax = float(df_relax.query(f"sujet == '{sujet_sel}' and session == '{odor}' and cond == '{cond}'")['val'].values)
                    STAI = float(df_stai.query(f"sujet == '{sujet_sel}' and session == '{odor}' and cond == '{cond}'")['val'].values)

                    if sujet in sujet_best_list_erp:
                        rep = True
                    else:
                        rep = False

                    amplitude = respfeatures[cond][odor].query(f"select == 1")['total_amplitude'].mean()
                    amp_mean_diff = xr_respi.loc[sujet, odor, cond, :].values.max() - xr_respi.loc[sujet, odor, cond, :].values.min()

                    if cond == 'CO2':
                        PCO2 = df_CO2.query(f"sujet == '{sujet}' and odor == '{odor}' and cond == 'CO2'")['PetCO2'].values[0]
                        VAS_A = df_CO2.query(f"sujet == '{sujet}' and odor == '{odor}' and cond == 'CO2'")['VAS_A'].values[0]
                        VAS_S = df_CO2.query(f"sujet == '{sujet}' and odor == '{odor}' and cond == 'CO2'")['VAS_S'].values[0]
                    else:
                        PCO2 = 0
                        VAS_A = 0
                        VAS_S = 0

                    VT = df_respi_paris.query(f"sujet == '{sujet}' and odor == '{odor}' and cond == '{cond}'")['VT'].values[0]
                    Ve = df_respi_paris.query(f"sujet == '{sujet}' and odor == '{odor}' and cond == '{cond}'")['Ve'].values[0]

                    reg_slope_dis['sujet'].append(sujet)
                    reg_slope_dis['odor'].append(odor)
                    reg_slope_dis['cond'].append(cond)
                    reg_slope_dis['nchan'].append(nchan)
                    reg_slope_dis['slope'].append(slope)
                    reg_slope_dis['A2'].append(A2)
                    reg_slope_dis['relax'].append(relax)
                    reg_slope_dis['STAI'].append(STAI)
                    reg_slope_dis['rep'].append(rep)
                    reg_slope_dis['amplitude'].append(amplitude)
                    reg_slope_dis['amp_mean_diff'].append(amp_mean_diff)
                    reg_slope_dis['PCO2'].append(PCO2)
                    reg_slope_dis['VAS_A'].append(VAS_A)
                    reg_slope_dis['VAS_S'].append(VAS_S)
                    reg_slope_dis['VT'].append(VT)
                    reg_slope_dis['Ve'].append(Ve)

    df_reg = pd.DataFrame(reg_slope_dis)

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'reg'))

    #metric = 'Ve'
    for metric in ['A2', 'STAI', 'relax', 'amplitude', 'amp_mean_diff', 'PCO2', 'VAS_S', 'VAS_A', 'VT', 'Ve']:

        print(metric)

        for nchan in chan_list_eeg:

            df_plot = df_reg.query(f"nchan == '{nchan}'")

            g = sns.FacetGrid(df_plot, row='odor', col="cond", hue='rep')
            g.map(sns.scatterplot, "slope", metric)
            plt.gca().invert_xaxis()
            plt.suptitle(f'{nchan}')
            plt.legend()
            # plt.show()

            plt.savefig(f'{metric}_{nchan}.png')
            plt.close('all')

    



################################
######## PPI PROPORTION ########
################################


def plot_PPI_proportion(xr_PPI_count):

    cond_temp = ['VS', 'CO2', 'MECA']

    group_list = ['allsujet', 'rep', 'no_rep']

    sujet_best_list_erp = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list_erp if sujet not in sujet_best_list_erp])

    #### MCN, TS

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'blind_evaluation'))
    df_blind_eva = pd.read_excel('df_PPI_blind_evaluation.xlsx')

    examinateur_list = ['MCN', 'TS']

    xr_dict = {'examinateur' : examinateur_list, 'sujet' : sujet_list_erp, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg}
    _xr_PPI_count = xr.DataArray(data=np.zeros((len(examinateur_list), len(sujet_list_erp), len(conditions), len(odor_list), len(chan_list_eeg))), dims=xr_dict.keys(), coords=xr_dict.values())

    for examinateur in examinateur_list:

        #sujet = sujet_list_erp[0]
        for sujet in sujet_list_erp:

            if sujet in ['28NT']:
                continue

            #cond = 'CO2'
            for cond in conditions:

                if cond in ['FR_CV_1', 'FR_CV_2']:
                    continue

                #odor = odor_list[0]
                for odor in odor_list:

                    #nchan_i, nchan = 0, chan_list_eeg[0]
                    for nchan_i, nchan in enumerate(chan_list_eeg):

                        if nchan in ['Cz', 'Fz']:

                            _eva = df_blind_eva.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}' and nchan == '{nchan}'")[examinateur].values[0]
                            _xr_PPI_count.loc[examinateur, sujet, cond, odor, nchan] = _eva

    df_PPI_count = _xr_PPI_count.to_dataframe(name='PPI').reset_index()

    dict = {'examinateur' : [], 'group' : [], 'cond' : [], 'odor' : [], 'nchan' : [], 'proportion' : []}

    for examinateur in examinateur_list:

        for group in group_list:

            #cond = 'VS'
            for cond in cond_temp:

                for odor in odor_list:

                    for nchan in chan_list_eeg:

                        if group == 'allsujet':
                            df_plot = df_PPI_count.query(f"examinateur == '{examinateur}' and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                        if group == 'rep':
                            df_plot = df_PPI_count.query(f"examinateur == '{examinateur}' and sujet in {sujet_best_list_erp.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                        if group == 'no_rep':
                            df_plot = df_PPI_count.query(f"examinateur == '{examinateur}' and sujet in {sujet_no_respond.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                        
                        if df_plot['PPI'].sum() == 0:
                            prop = 0
                        else:
                            prop = np.round(df_plot['PPI'].sum() / df_plot['sujet'].shape[0], 5)*100

                        dict['examinateur'].append(examinateur)
                        dict['group'].append(group)
                        dict['cond'].append(cond)
                        dict['odor'].append(odor)
                        dict['nchan'].append(nchan)
                        dict['proportion'].append(prop)
            
    df_PPI_plot = pd.DataFrame(dict)

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'proportion'))

    n_sujet_all = df_PPI_count.query(f"examinateur == '{examinateur}' and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]
    n_sujet_rep = df_PPI_count.query(f"examinateur == '{examinateur}' and sujet in {sujet_best_list_erp.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]
    n_sujet_no_rep = df_PPI_count.query(f"examinateur == '{examinateur}' and sujet in {sujet_no_respond.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]

    for examinateur in examinateur_list:

        for nchan in ['Cz', 'Fz']:

            df_plot = df_PPI_plot.query(f"examinateur == '{examinateur}' and nchan == '{nchan}'")
            sns.catplot(data=df_plot, x="odor", y="proportion", hue='group', kind="point", col='cond')
            plt.suptitle(f"{nchan} / all:{n_sujet_all},rep:{n_sujet_rep},no_rep:{n_sujet_no_rep}")
            plt.ylim(0,100)
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"{nchan}_{examinateur}.png")
            plt.close('all')

    #### JG

    df_PPI_count = xr_PPI_count.to_dataframe(name='PPI').reset_index()
    df_PPI_count = df_PPI_count.query(f"examinateur == 'JG'")

    dict = {'group' : [], 'cond' : [], 'odor' : [], 'nchan' : [], 'proportion' : []}

    for group in group_list:

        #cond = 'VS'
        for cond in conditions:

            for odor in odor_list:

                for nchan in chan_list_eeg:

                    if group == 'allsujet':
                        df_plot = df_PPI_count.query(f"cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                    if group == 'rep':
                        df_plot = df_PPI_count.query(f"sujet in {sujet_best_list_erp.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                    if group == 'no_rep':
                        df_plot = df_PPI_count.query(f"sujet in {sujet_no_respond.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                    
                    if df_plot['PPI'].sum() == 0:
                        prop = 0
                    else:
                        prop = np.round(df_plot['PPI'].sum() / df_plot['sujet'].shape[0], 5)*100

                    dict['group'].append(group)
                    dict['cond'].append(cond)
                    dict['odor'].append(odor)
                    dict['nchan'].append(nchan)
                    dict['proportion'].append(prop)
            
    df_PPI_plot = pd.DataFrame(dict)

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'proportion'))

    n_sujet_all = df_PPI_count.query(f"cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]
    n_sujet_rep = df_PPI_count.query(f"sujet in {sujet_best_list_erp.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]
    n_sujet_no_rep = df_PPI_count.query(f"sujet in {sujet_no_respond.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]

    for nchan in chan_list_eeg:

        df_plot = df_PPI_plot.query(f"nchan == '{nchan}'")
        sns.catplot(data=df_plot, x="odor", y="proportion", hue='group', kind="point", col='cond')
        plt.suptitle(f"{nchan} / all:{n_sujet_all},rep:{n_sujet_rep},no_rep:{n_sujet_no_rep}")
        plt.ylim(0,100)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{nchan}_JG.png")
        plt.close('all')




    







################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #### ERP
    # cond_erp = ['VS', 'MECA', 'CO2']

    print(f'#### compute allsujet ####', flush=True)

    xr_data = compute_ERP()
    xr_lm_data, xr_lm_pred, xr_lm_pred_coeff = compute_lm_on_ERP(xr_data)

    xr_PPI_count = get_PPI_count(xr_data)
    
    # shuffle_way = 'inter_cond'
    # shuffle_way = 'intra_cond'
    shuffle_way = 'linear_based'
    xr_surr = compute_surr_ERP(xr_data, shuffle_way)
    
    print(f'#### plot allsujet ####', flush=True)

    plot_ERP(xr_data, xr_lm_data, xr_lm_pred, xr_lm_pred_coeff, xr_surr)

    plot_ERP_CO2(xr_data)
    plot_ERP_rep_norep(xr_data)

    #### plot PPI across subject
    plot_PPI_proportion(xr_PPI_count)

    #### respi
    plot_mean_respi()

    #### manual evaluation
    generate_ppi_evaluation(xr_data)

    #### reg 
    print(f'#### plot discomfort / slope ####', flush=True)
    plot_slope_versus_discomfort(xr_data, xr_lm_data)








