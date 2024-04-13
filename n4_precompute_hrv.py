

import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
import mne
import scipy.signal
from bycycle.cyclepoints import find_extrema
import respirationtools
import pickle


from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False



############################
######## LOAD ECG ########
############################

def load_ecg(sujet, band_prep):

    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    
    ecg_allcond = {}
    ecg_cR_allcond = {}

    #cond = conditions[0]
    for cond in conditions:

        ecg_allcond[cond] = {}
        ecg_cR_allcond[cond] = {}

        #odor_i = odor_list[0]
        for odor_i in odor_list:

            data = load_data_sujet(sujet, cond, odor_i)
            ecg_allcond[cond][odor_i] = data[chan_list.index('ECG'), :]
            ecg_cR_allcond[cond][odor_i] = data[chan_list.index('ECG_cR'), :]

    return ecg_allcond, ecg_cR_allcond


def load_ecg_cR_corrected(sujet):

    os.chdir(os.path.join(path_results, sujet, 'HRV'))

    with open(f'{sujet}_cR_corrected.pkl', 'rb') as f:

        ecg_cR_allcond = pickle.load(f)

    return ecg_cR_allcond



################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    band_prep = 'wb'

    prms_hrv = {
    'metric_list' : ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S', 'HRV_HF', 'HRV_LF', 'HRV_LFHF', 'HRV_COV', 'HRV_MAD', 'HRV_MEDIAN'],
    'srate' : srate,
    'srate_resample_hrv' : 10,
    'nwind_hrv' : int( 128*srate_resample_hrv ),
    'nfft_hrv' : nwind_hrv,
    'noverlap_hrv' : np.round(nwind_hrv/10),
    'win_hrv' : scipy.signal.windows.hann(nwind_hrv),
    'f_RRI' : (.1, .5)
    }

    
    #### load data and clean
    sujet_list_clean_hrv = []

    #sujet = sujet_list[-1]
    for sujet in sujet_list:

        print(sujet)

        #### load data
        ecg_allcond, ecg_cR_allcond = load_ecg(sujet, band_prep)

        val_verif = []

        for cond in conditions:

            #odor_i = odor_list[0]
            for odor_i in odor_list:

                val_verif.append(ecg_cR_allcond[cond][odor_i].sum())

        if np.array(val_verif).min() > 500:
            sujet_list_clean_hrv.append(sujet)

    #### verif
    if debug: 

        _sujet_list = np.array(['01PD', '02MJ', '03VN', '04GB', '05LV', '06EF', '07PB', '08DM',
       '09TA', '10BH', '11FA', '12BD', '13FP', '14MD', '15LG', '16GM',
       '17JR', '18SE', '19TM', '20TY', '21ZV', '22DI', '23LF', '24TJ',
       '25DF', '26MN', '27BD', '28NT', '29SC', '30AR', '31HJ', '32CM',
       '33MA'], dtype='<U4')
        
        sujet = '33MA'

        ecg_allcond, ecg_cR_allcond = load_ecg(sujet, band_prep)

        ecg_cR_time_allcond = {}

        #### inspect
        #cond = conditions[0]
        for cond in conditions:

            ecg_cR_time_allcond[cond] = {}

            #odor_i = odor_list[0]
            for odor_i in odor_list:

                cR_time = np.where(ecg_cR_allcond[cond][odor_i] == ecg_cR_allcond[cond][odor_i].max())[0]
                ecg = ecg_allcond[cond][odor_i]

                ecg_cR_time_allcond[cond][odor_i] = cR_time

                plt.figure(figsize=(18,7))
                plt.plot(ecg)
                plt.vlines(cR_time, ymin=ecg.min(), ymax=ecg.max(), color='r')
                plt.title(f"{sujet}, {cond}, {odor_i}")
                plt.show()

        #### load bloc to clean
        cond = 'FR_CV_2'
        odor_i = '+'

        cR_time_change = np.where(ecg_cR_allcond[cond][odor_i] == ecg_cR_allcond[cond][odor_i].max())[0]
        ecg = ecg_allcond[cond][odor_i]

        #### redetect
        cR_time_corrected = scipy.signal.find_peaks(ecg, height=2*ecg.std(), distance=np.median(np.diff(cR_time_change))/2)[0]
        # cR_time_corrected = scipy.signal.find_peaks(ecg, height=2*ecg.std())[0]
        # cR_time_corrected = scipy.signal.find_peaks(ecg, height=2*ecg.std(), distance=np.median(np.diff(cR_time_corrected))/2)[0]
        plt.figure(figsize=(18,7))
        plt.plot(ecg)
        plt.vlines(cR_time_corrected, ymin=ecg.min(), ymax=ecg.max(), color='r')
        plt.show()

        #### add
        cR_to_append = np.array([61530])
        cR_time_corrected = np.append(cR_time_corrected, cR_to_append)
        cR_time_corrected.sort()

        plt.plot(ecg)
        plt.vlines(cR_time_corrected, ymin=ecg.min(), ymax=ecg.max(), color='r')
        plt.show()

        plt.plot(cR_time_corrected)
        plt.show()

        #### clean
        exclude_start, exclude_stop = 3250, 3350
        cR_time_mask = (cR_time_change >= exclude_start) & (cR_time_change <= exclude_stop) 
        cR_time_corrected = cR_time_change[~cR_time_mask]
        cR_time_excluded = cR_time_change[cR_time_mask]

        plt.figure(figsize=(18,7))
        plt.plot(ecg)
        plt.vlines(cR_time_corrected, ymin=ecg.min(), ymax=ecg.max(), color='r')
        plt.vlines(cR_time_excluded, ymin=ecg.min(), ymax=ecg.max(), color='g')
        plt.title(f"{sujet}, {cond}, {odor_i}")
        plt.show()

        cR_time_change = cR_time_corrected.copy()

        #### replace
        ecg_cR_time_allcond[cond][odor_i] = cR_time_corrected

        #### verif all
        for cond in conditions:

            #odor_i = odor_list[0]
            for odor_i in odor_list:

                cR_time = ecg_cR_time_allcond[cond][odor_i]
                ecg = ecg_allcond[cond][odor_i]

                plt.figure(figsize=(18,7))
                plt.plot(ecg)
                plt.vlines(cR_time, ymin=ecg.min(), ymax=ecg.max(), color='r')
                plt.title(f"{sujet}, {cond}, {odor_i}")
                plt.show()

        #### save
        os.chdir(os.path.join(path_results, sujet, 'HRV'))
        
        with open(f'{sujet}_cR_corrected.pkl', 'wb') as fp:
            pickle.dump(ecg_cR_time_allcond, fp)




    ### compute & save
    #analysis_time = '3min'
    for analysis_time in ['5min', '3min']:

        #sujet = sujet_list[0]
        for sujet in sujet_list:

            print(sujet)

            if sujet not in sujet_list_clean_hrv:
                os.chdir(os.path.join(path_results, sujet, 'HRV'))

                lines = [f'{sujet}', 'bad detection so far']
                with open(f'{sujet}_bad_detection.txt', 'w') as f:
                    f.writelines(lines)

            #### initiate containers
            df_hrv_dict = {}
            df_hrv_dict['sujet'], df_hrv_dict['cond'], df_hrv_dict['odor'] = [], [], []
            for metric_i in prms_hrv['metric_list']:
                df_hrv_dict[metric_i] = []
            df_hrv = pd.DataFrame(df_hrv_dict)

            #### load data
            ecg_allcond, ecg_cR_allcond = load_ecg(sujet, band_prep)
            ecg_cR_allcond = load_ecg_cR_corrected(sujet)

            #### compute
            #cond = conditions[0]
            for cond in conditions:

                #odor_i = odor_list[0]
                for odor_i in odor_list:

                    cR_time = ecg_cR_allcond[cond][odor_i]
        
                    df = get_hrv_metrics_homemade(cR_time, prms_hrv, analysis_time=analysis_time)

                    dict_res = {}
                    dict_res['sujet'], dict_res['cond'], dict_res['odor'] = [sujet], [cond], [odor_i]
                    
                    #metric_i = prms_hrv['metric_list'][0]
                    for metric_i in prms_hrv['metric_list']:
                        dict_res[metric_i] = df[metric_i].values

                    df_i = pd.DataFrame(dict_res) 

                    df_hrv = pd.concat((df_hrv, df_i))

            #### verif
            sujet_list_clean_hrv

            #### save
            df_hrv = df_hrv.reset_index().drop(columns='index')
            os.chdir(os.path.join(path_results, sujet, 'HRV'))

            if analysis_time == '3min':
                df_hrv.to_excel(f"{sujet}_df_hrv_3min.xlsx")
            else:
                df_hrv.to_excel(f"{sujet}_df_hrv.xlsx")
    