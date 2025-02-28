
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
from sklearn import linear_model
import gc

from n00_config_params import *
from n00bis_config_analysis_functions import *


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

    t_start_ERP = ERP_time_vec[0]
    t_stop_ERP = ERP_time_vec[1]

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
                stretch_point_PPI = int(np.abs(t_start_ERP)*srate + t_stop_ERP*srate)
                data_chunk = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

                x = data[nchan_i,:]

                x_mean, x_std = x.mean(), x.std()
                microV_SD = int(x_std*1e6)

                for start_i, start_time in enumerate(inspi_starts):

                    t_start = int(start_time + t_start_ERP*srate)
                    t_stop = int(start_time + t_stop_ERP*srate)

                    data_chunk[start_i, :] = (x[t_start: t_stop] - x_mean) / x_std

                if debug:

                    time_vec = np.arange(t_start_ERP, t_stop_ERP, 1/srate)

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

                    time_vec = np.arange(t_start_ERP, t_stop_ERP, 1/srate)

                    for inspi_i, _ in enumerate(inspi_starts):

                        plt.plot(time_vec, data_stretch_clean[inspi_i, :], alpha=0.3)

                    plt.vlines(0, ymax=data_stretch_clean.max(), ymin=data_stretch_clean.min(), color='k')
                    plt.hlines([-3, 3], xmax=t_start_ERP, xmin=t_stop_ERP, color='k')
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
            

def compute_normalization_params():

    xr_data = np.zeros((len(sujet_list), len(chan_list_eeg), 4))
    xr_dict = {'sujet' : sujet_list, 'chan_list' : chan_list_eeg, 'params' : ['mean', 'std', 'median', 'mad']}

    for sujet_i, sujet in enumerate(sujet_list):

        print(sujet)

        _respi = np.empty((len(chan_list_eeg),1))

        for odor in odor_list:

            for cond in conditions:

                data = load_data_sujet(sujet, cond, odor)[:len(chan_list_eeg),:]
                data = scipy.signal.detrend(data, type='linear', axis=1)
                data = iirfilt(data, srate, lowcut=0.05, highcut=None, axis=1)
                data = iirfilt(data, srate, lowcut=None, highcut=45, axis=1)
                _respi = np.concat([_respi, data], axis=1)

        xr_data[sujet_i,:,0], xr_data[sujet_i,:,1], xr_data[sujet_i,:,2], xr_data[sujet_i,:,3] = _respi.mean(axis=1), _respi.std(axis=1), np.median(_respi, axis=1), scipy.stats.median_abs_deviation(_respi, axis=1)


    xr_normal_params = xr.DataArray(data=xr_data, dims=xr_dict.keys(), coords=xr_dict.values())

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
    xr_normal_params.to_netcdf('allsujet_ERP_normalization_params.nc')
    

def compute_ERP_stretch(normalization='rscore'):

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'allsujet_ERP_data_stretch.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        xr_data_stretch = xr.open_dataarray('allsujet_ERP_data_stretch.nc')
        xr_data_sem_stretch = xr.open_dataarray('allsujet_ERP_data_sem_stretch.nc')

    else:

        cond_list = ['FR_CV_1', 'MECA', 'CO2']

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_normal_params = xr.open_dataarray('allsujet_ERP_normalization_params.nc')

        xr_dict_stretch = {'sujet' : sujet_list, 'cond' : cond_list, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'phase' : np.arange(stretch_point_TF)}
        
        os.chdir(path_memmap)
        data_stretch_ERP = np.memmap(f'data_stretch_ERP.dat', dtype=np.float64, mode='w+', shape=(len(sujet_list), len(cond_list), len(odor_list),len(chan_list_eeg), stretch_point_TF))
        data_sem_stretch_ERP = np.memmap(f'data_sem_stretch_ERP.dat', dtype=np.float64, mode='w+', shape=(len(sujet_list), len(cond_list), len(odor_list),len(chan_list_eeg), stretch_point_TF))

        #sujet_i, sujet = 0, sujet_list[0]
        def get_stretch_data_for_ERP(sujet_i, sujet):

        #sujet_i, sujet = np.where(sujet_list == '12BD')[0][0], '12BD'
        # for sujet_i, sujet in enumerate(sujet_list):

            print(sujet)

            respfeatures = load_respfeatures(sujet)

            #cond_i, cond = 2, 'CO2'
            for cond_i, cond in enumerate(cond_list):
                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):
                        
                    data = load_data_sujet(sujet, cond, odor)[:len(chan_list_eeg),:]
                    data = scipy.signal.detrend(data, type='linear' ,axis=1)
                    data = iirfilt(data, srate, lowcut=0.05, highcut=None, axis=1)
                    data = iirfilt(data, srate, lowcut=None, highcut=45, axis=1)

                    #nchan_i, nchan = 0, chan_list_eeg[0]
                    for nchan_i, nchan in enumerate(chan_list_eeg):

                        data_stretch, mean_inspi_ratio = stretch_data(respfeatures[cond][odor], stretch_point_TF, data[nchan_i,:], srate)

                        if normalization == 'zscore':
                            data_stretch_norm = (data_stretch - xr_normal_params.loc[sujet, nchan, 'mean'].values) / xr_normal_params.loc[sujet, nchan, 'std'].values
                        if normalization == 'rscore':
                            data_stretch_norm = (data_stretch - xr_normal_params.loc[sujet, nchan, 'median'].values) * 0.6745 / xr_normal_params.loc[sujet, nchan, 'mad'].values

                        if debug:

                            plt.plot(data, label='raw')
                            plt.plot(scipy.signal.detrend(data, type='linear'), label='detrend')
                            plt.legend()
                            plt.show()

                            plt.hist(data, label='raw', bins=100)
                            plt.hist(scipy.signal.detrend(data, type='linear'), label='detrend', bins=100)
                            plt.legend()
                            plt.show()

                            plt.plot(data_stretch_norm[0,:], label='raw')
                            plt.plot(scipy.stats.zscore(data_stretch_norm, axis=1)[0,:], label='stretch')
                            plt.legend()
                            plt.show()

                            fig, ax = plt.subplots()

                            for inspi_i in range(data_stretch_norm.shape[0]):

                                ax.plot(np.arange(stretch_point_TF), data_stretch_norm[inspi_i, :], alpha=0.3)

                            plt.vlines(stretch_point_TF/2, ymax=data_stretch_norm.max(), ymin=data_stretch_norm.min(), color='k')
                            ax.plot(np.arange(stretch_point_TF), data_stretch_norm.mean(axis=0), color='r')
                            plt.title(f'{cond} : {data_stretch_norm.shape[0]}')
                            ax.invert_yaxis()
                            plt.show()

                        if normalization == 'zscore':

                            data_stretch_ERP[sujet_i, cond_i, odor_i, nchan_i, :] = data_stretch_norm.mean(axis=0)
                            data_sem_stretch_ERP[sujet_i, cond_i, odor_i, nchan_i, :] = data_stretch_norm.std(axis=0) / np.sqrt(data_stretch_norm.shape[0])

                        elif normalization == 'rscore':

                            data_stretch_ERP[sujet_i, cond_i, odor_i, nchan_i, :] = np.median(data_stretch_norm, axis=0)
                            data_sem_stretch_ERP[sujet_i, cond_i, odor_i, nchan_i, :] = scipy.stats.median_abs_deviation(data_stretch_norm, axis=0) / np.sqrt(data_stretch_norm.shape[0])

                        if debug:

                            data_stretch_zscore = (data_stretch - xr_normal_params.loc[sujet, nchan, 'mean'].values) / xr_normal_params.loc[sujet, nchan, 'std'].values
                            data_stretch_rscore = (data_stretch - xr_normal_params.loc[sujet, nchan, 'median'].values) * 0.6745 / xr_normal_params.loc[sujet, nchan, 'mad'].values

                            plt.plot(np.arange(stretch_point_TF), np.median(data_stretch, axis=0), label='median_nonorm', color='y')
                            plt.plot(np.arange(stretch_point_TF), np.mean(data_stretch, axis=0), label='mean_nonorm', color='k')
                            plt.plot(np.arange(stretch_point_TF), np.median(data_stretch_rscore, axis=0), label='median', color='b')
                            plt.fill_between(np.arange(stretch_point_TF), np.median(data_stretch_rscore, axis=0) - (scipy.stats.median_abs_deviation(data_stretch_rscore, axis=0) / np.sqrt(data_stretch_rscore.shape[0])), 
                                             np.median(data_stretch_rscore, axis=0) + (scipy.stats.median_abs_deviation(data_stretch_rscore, axis=0) / np.sqrt(data_stretch_rscore.shape[0])), alpha=0.3, color='c')
                            plt.plot(np.arange(stretch_point_TF), np.mean(data_stretch_zscore, axis=0), label='mean', color='r')
                            plt.fill_between(np.arange(stretch_point_TF), np.mean(data_stretch_zscore, axis=0) - (np.std(data_stretch_zscore, axis=0) / np.sqrt(data_stretch_zscore.shape[0])), 
                                             np.mean(data_stretch_zscore, axis=0) + (np.std(data_stretch_zscore, axis=0) / np.sqrt(data_stretch_zscore.shape[0])), alpha=0.3, color='m')
                            plt.legend()
                            plt.show()

                        # inverse to have inspi on the right and expi on the left
                        # data_stretch_ERP[sujet_i, cond_i, nchan_i, :] = np.hstack((data_stretch_load[int(stretch_point_TF/2):], data_stretch_load[:int(stretch_point_TF/2)]))
                        # data_sem_stretch_ERP[sujet_i, cond_i, nchan_i, :] = np.hstack((data_stretch_sem_load[int(stretch_point_TF/2):], data_stretch_sem_load[:int(stretch_point_TF/2)]))

        #### parallelize
        # joblib.Parallel(n_jobs = n_core, prefer = 'threads')(joblib.delayed(get_stretch_data_for_ERP)(sujet_i, sujet) for sujet_i, sujet in enumerate(sujet_list))
        for sujet_i, sujet in enumerate(sujet_list):
            get_stretch_data_for_ERP(sujet_i, sujet)

        #### load data in xr
        xr_data_stretch = xr.DataArray(data=data_stretch_ERP, dims=xr_dict_stretch.keys(), coords=xr_dict_stretch.values())
        xr_data_sem_stretch = xr.DataArray(data=data_sem_stretch_ERP, dims=xr_dict_stretch.keys(), coords=xr_dict_stretch.values())

        os.chdir(path_memmap)
        try:
            os.remove(f'data_stretch_ERP.dat')
            del data_stretch_ERP
        except:
            pass

        os.chdir(path_memmap)
        try:
            os.remove(f'data_sem_stretch_ERP.dat')
            del data_sem_stretch_ERP
        except:
            pass

        #### save data
        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_data_stretch.to_netcdf('allsujet_ERP_data_stretch.nc')
        xr_data_sem_stretch.to_netcdf('allsujet_ERP_data_sem_stretch.nc')

    return xr_data_stretch, xr_data_sem_stretch








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

            time_vec_lm = np.arange(PPI_lm_start, PPI_lm_stop, 1/srate)[:-1]
            lm_data[cond][odor]['Y_pred'] = np.zeros((len(chan_list_eeg),time_vec_lm.shape[0]))

            #nchan_i, nchan = 0, chan_list_eeg[0]
            for nchan_i, nchan in enumerate(chan_list_eeg):
            
                data = data_chunk_allcond[cond][odor][nchan].mean(axis=0)

                time_vec = np.linspace(ERP_time_vec[0], ERP_time_vec[1], data.shape[0])
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

    cond_erp = ['FR_CV_1', 'MECA', 'CO2']
    
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






def get_cluster_stats_manual_prem(stretch=True):

    if stretch:

        if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'cluster_stats_intra_stretch.nc')):

            os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

            print('ALREADY COMPUTED', flush=True)

            cluster_stats_intra = xr.open_dataarray('cluster_stats_intra_stretch.nc')
            cluster_stats_inter = xr.open_dataarray('cluster_stats_inter_stretch.nc')
            cluster_stats_rep_norep = xr.open_dataarray('cluster_stats_rep_norep_stretch.nc')

            return cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep
        
    elif stretch == False:

        if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'cluster_stats_intra.nc')):

            os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

            print('ALREADY COMPUTED', flush=True)

            cluster_stats_intra = xr.open_dataarray('cluster_stats_intra.nc')
            cluster_stats_inter = xr.open_dataarray('cluster_stats_inter.nc')
            cluster_stats_rep_norep = xr.open_dataarray('cluster_stats_rep_norep.nc')

            return cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep

    #### initiate
    conditions_diff = ['MECA', 'CO2']
    cond_sel = ['FR_CV_1', 'MECA', 'CO2']
    odor_diff = ['+', '-']
    sujet_group = ['allsujet', 'rep', 'non_rep']

    if stretch:
        time_vec = np.arange(stretch_point_TF)
    else:      
        time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)

    #### intra
    os.chdir(path_memmap)
    cluster_stats_intra = np.memmap(f'cluster_stats_intra.dat', dtype=np.float64, mode='w+', shape=(len(sujet_group), len(chan_list_eeg), len(odor_list), len(conditions_diff), time_vec.shape[0]))

    def _get_cluster_stat_group(group_i, group):
    #group = sujet_group[0]
    # for group in sujet_group:

        print(group)

        if stretch:
            xr_data, xr_data_sem = compute_ERP_stretch(normalization='rscore')
        else:
            xr_data, xr_data_sem = compute_ERP()

        #nchan = 'C3'
        for nchan_i, nchan in enumerate(chan_list_eeg):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                #cond = conditions_diff[0]
                for cond_i, cond in enumerate(conditions_diff):

                    if group == 'allsujet':
                        data_baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].values
                        data_cond = xr_data.loc[:, cond, odor, nchan, :].values
                    elif group == 'rep':
                        data_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, nchan, :].values
                        data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].values
                    elif group == 'non_rep':
                        data_baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, nchan, :].values
                        data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].values

                    mask = get_permutation_cluster_1d(data_baseline, data_cond, ERP_n_surrogate)
                    # mask = get_permutation_cluster_1d_DEBUG(data_baseline, data_cond, ERP_n_surrogate)

                    if debug:

                        plt.plot(data_baseline.mean(axis=0))
                        plt.plot(data_cond.mean(axis=0))

                        plt.show()

                    cluster_stats_intra[group_i, nchan_i, odor_i, cond_i, :] = mask

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(_get_cluster_stat_group)(group_i, group) for group_i, group in enumerate(sujet_group))

    xr_dict = {'group' : sujet_group, 'chan' : chan_list_eeg, 'odor' : odor_list, 'cond' : conditions_diff, 'time' : time_vec}
    xr_cluster = xr.DataArray(data=cluster_stats_intra, dims=xr_dict.keys(), coords=xr_dict.values())

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

    if stretch:
        xr_cluster.to_netcdf('cluster_stats_intra_stretch.nc')
    else:
        xr_cluster.to_netcdf('cluster_stats_intra.nc')

    os.chdir(path_memmap)
    try:
        os.remove(f'cluster_stats_intra.dat')
        del cluster_stats_intra
    except:
        pass

    #### inter
    os.chdir(path_memmap)
    cluster_stats_inter = np.memmap(f'cluster_stats_inter.dat', dtype=np.float64, mode='w+', shape=(len(sujet_group), len(chan_list_eeg), len(odor_diff), len(cond_sel), time_vec.shape[0]))

    def _get_cluster_stat_group(group_i, group):
    # for group in sujet_group:

        print(group)

        if stretch:
            xr_data, xr_data_sem = compute_ERP_stretch()
        else:
            xr_data, xr_data_sem = compute_ERP()

        for nchan_i, nchan in enumerate(chan_list_eeg):

            #cond_i, cond = 2, cond_sel[2]
            for cond_i, cond in enumerate(cond_sel):

                for odor_i, odor in enumerate(odor_diff):

                    if group == 'allsujet':
                        data_baseline = xr_data.loc[:, cond, 'o', nchan, :].values
                        data_cond = xr_data.loc[:, cond, odor, nchan, :].values
                    elif group == 'rep':
                        data_baseline = xr_data.loc[sujet_best_list_rev, cond, 'o', nchan, :].values
                        data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].values
                    elif group == 'non_rep':
                        data_baseline = xr_data.loc[sujet_no_respond_rev, cond, 'o', nchan, :].values
                        data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].values

                    mask = get_permutation_cluster_1d(data_baseline, data_cond, ERP_n_surrogate)
                    # mask = get_permutation_cluster_1d_DEBUG(data_baseline, data_cond, ERP_n_surrogate)

                    if debug:
                        fig, ax = plt.subplots()
                        ax.plot(np.arange(data_baseline.shape[1]), data_baseline.mean(axis=0))
                        ax.plot(np.arange(data_baseline.shape[1]), data_cond.mean(axis=0))
                        ax.fill_between(np.arange(data_baseline.shape[1]), -0.5, 0.5, where=mask.astype('int'), alpha=0.3, color='r')
                        plt.show()

                    cluster_stats_inter[group_i, nchan_i, odor_i, cond_i, :] = mask

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(_get_cluster_stat_group)(group_i, group) for group_i, group in enumerate(sujet_group))

    xr_dict = {'group' : sujet_group, 'chan' : chan_list_eeg, 'odor' : odor_diff, 'cond' : cond_sel, 'time' : time_vec}
    xr_cluster = xr.DataArray(data=cluster_stats_inter, dims=xr_dict.keys(), coords=xr_dict.values())

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

    if stretch:
        xr_cluster.to_netcdf('cluster_stats_inter_stretch.nc')
    else:
        xr_cluster.to_netcdf('cluster_stats_inter.nc')

    os.chdir(path_memmap)
    try:
        os.remove(f'cluster_stats_inter.dat')
        del cluster_stats_inter
    except:
        pass

    #### rep norep
    os.chdir(path_memmap)
    cluster_stats_rep_norep = np.memmap(f'cluster_stats_rep_norep.dat', dtype=np.float64, mode='w+', shape=(len(chan_list_eeg), len(odor_list), len(cond_sel), time_vec.shape[0]))

    def _get_cluster_stat_rep_norep_chan(chan_i, chan):

        if stretch:
            xr_data, xr_data_sem = compute_ERP_stretch()
        else:
            xr_data, xr_data_sem = compute_ERP()

        print(chan)

        #odor_i, odor = 2, odor_list[2]
        for odor_i, odor in enumerate(odor_list):

            for cond_i, cond in enumerate(cond_sel):

                data_baseline = xr_data.loc[sujet_best_list_rev, cond, odor, chan, :].values
                data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, chan, :].values

                mask = get_permutation_cluster_1d(data_baseline, data_cond, ERP_n_surrogate)
                # mask = get_permutation_cluster_1d_DEBUG(data_baseline, data_cond, ERP_n_surrogate)

                cluster_stats_rep_norep[chan_i, odor_i, cond_i ,:] = mask

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(_get_cluster_stat_rep_norep_chan)(chan_i, chan) for chan_i, chan in enumerate(chan_list_eeg))

    xr_dict = {'chan' : chan_list_eeg, 'odor' : odor_list, 'cond' : cond_sel, 'time' : time_vec}
    xr_cluster = xr.DataArray(data=cluster_stats_rep_norep, dims=xr_dict.keys(), coords=xr_dict.values())

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

    if stretch:
        xr_cluster.to_netcdf('cluster_stats_rep_norep_stretch.nc')
    else:
        xr_cluster.to_netcdf('cluster_stats_rep_norep.nc')

    os.chdir(path_memmap)
    try:
        os.remove(f'cluster_stats_rep_norep.dat')
        del cluster_stats_rep_norep
    except:
        pass
    
    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

    if stretch:
        cluster_stats_intra = xr.open_dataarray('cluster_stats_intra_stretch.nc')
        cluster_stats_inter = xr.open_dataarray('cluster_stats_inter_stretch.nc')
        cluster_stats_rep_norep = xr.open_dataarray('cluster_stats_rep_norep_stretch.nc')
    else:
        cluster_stats_intra = xr.open_dataarray('cluster_stats_intra.nc')
        cluster_stats_inter = xr.open_dataarray('cluster_stats_inter.nc')
        cluster_stats_rep_norep = xr.open_dataarray('cluster_stats_rep_norep.nc')

    return cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep








def get_cluster_stats_manual_prem_subject_wise():

    ######## VERIFY COMPUTATION ########
    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'cluster_stats_manual_perm_allsujet.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('ALREADY COMPUTED', flush=True)
        xr_cluster_based_perm = xr.open_dataarray('cluster_stats_manual_perm_allsujet.nc')

    ######## COMPUTE ########
    else:

        conditions_sel = ['FR_CV_1', 'CO2']
        time_vec = np.arange(stretch_point_TF)
        xr_coords = {'sujet' : sujet_list, 'odor' : odor_list, 'chan' : chan_list_eeg, 'time' : time_vec}

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_zscore_params = xr.open_dataarray('allsujet_ERP_zscore_params.nc')

        ######## INITIATE MEMMAP ########
        os.chdir(path_memmap)
        xr_data = np.memmap(f'cluster_based_perm_allsujet.dat', dtype=np.float32, mode='w+', shape=(len(sujet_list), len(odor_list), len(chan_list_eeg), time_vec.size))

        ######## PARALLELIZATION FUNCTION ########
        #sujet_i, sujet = 3, sujet_list[3]
        def get_cluster_based_perm_one_sujet(sujet_i, sujet):
        # for sujet_i, sujet in enumerate(sujet_list):

            ######## COMPUTE ERP ########
            print(f'{sujet} COMPUTE ERP')

            erp_data = {}

            respfeatures = load_respfeatures(sujet)

            #cond = 'FR_CV_1'
            for cond in conditions_sel:

                erp_data[cond] = {}

                #odor = odor_list[0]
                for odor in odor_list:

                    erp_data[cond][odor] = {}
                    data = load_data_sujet(sujet, cond, odor)[:len(chan_list_eeg),:]

                    #nchan_i, nchan = 0, chan_list_eeg[0]
                    for nchan_i, nchan in enumerate(chan_list_eeg):

                        respfeatures_i = respfeatures[cond][odor]
                        inspi_starts = respfeatures_i.query(f"select == 1")['inspi_index'].values

                        #### chunk
                        stretch_point_PPI = int(np.abs(ERP_time_vec[0])*srate + ERP_time_vec[1]*srate)
                        data_stretch = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

                        #### low pass 45Hz
                        x = scipy.signal.detrend(data[nchan_i,:], type='linear')
                        x = iirfilt(x, srate, lowcut=0.05, highcut=None)
                        x = iirfilt(x, srate, lowcut=None, highcut=45)
                        data_stretch, mean_inspi_ratio = stretch_data(respfeatures[cond][odor], stretch_point_TF, x, srate)
                        data_stretch = (data_stretch - xr_zscore_params.loc[sujet,nchan,'mean'].values) / xr_zscore_params.loc[sujet,nchan,'std'].values

                        erp_data[cond][odor][nchan] = data_stretch
       
            ######## COMPUTE PERMUTATION ########
            print(f'{sujet} COMPUTE PERMUTATION')

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                #chan_i, nchan = 0, chan_list_eeg[0]
                for chan_i, nchan in enumerate(chan_list_eeg):

                    data_baseline = erp_data['FR_CV_1'][odor][nchan]
                    data_cond = erp_data['CO2'][odor][nchan]

                    mask = get_permutation_cluster_1d(data_baseline, data_cond, ERP_n_surrogate)

                    xr_data[sujet_i, odor_i, chan_i, :] = mask

            print(f'{sujet} done')

        ######## PARALLELIZATION COMPUTATION ########
        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_cluster_based_perm_one_sujet)(sujet_i, sujet) for sujet_i, sujet in enumerate(sujet_list))

        ######## SAVE ########
        xr_cluster_based_perm = xr.DataArray(data=xr_data, dims=xr_coords.keys(), coords=xr_coords.values())

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_cluster_based_perm.to_netcdf('cluster_stats_manual_perm_allsujet.nc')

        os.chdir(path_memmap)
        try:
            os.remove(f'cluster_based_perm_allsujet.dat')
            del xr_data
        except:
            pass

    return xr_cluster_based_perm











################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    compute_normalization_params()
    compute_ERP_stretch(normalization='rscore')
    get_cluster_stats_manual_prem(stretch=True)
    get_cluster_stats_manual_prem_subject_wise()
    
    #sujet = sujet_list[0]
    # for sujet in sujet_list:

    #     print(f'#### {sujet} ####', flush=True)
    #     print(f'COMPUTE', flush=True)

        # data_chunk_allcond, data_value_microV = compute_ERP(sujet)
        # lm_data = compute_lm_on_ERP(data_chunk_allcond, cond_erp)

        # shuffle_way = 'inter_cond'
        # shuffle_way = 'intra_cond'
        # shuffle_way = 'linear_based'

        # ERP_surr = compute_surr_ERP(data_chunk_allcond, shuffle_way)
        



    

    






