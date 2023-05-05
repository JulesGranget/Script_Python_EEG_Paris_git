

import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
import mne
import scipy.signal
from bycycle.cyclepoints import find_extrema
import respirationtools

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


        


################################
######## EXECUTE ########
################################

if __name__ == '__main__':



    prms_hrv = {
    'metric_list' : ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S', 'HRV_HF', 'HRV_LF', 'HRV_LFHF', 'HRV_COV'],
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

    #band_prep = band_prep_list[0]
    for band_prep in band_prep_list:

        #sujet = sujet_list[0]
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


    ### compute & save
    for band_prep in band_prep_list:

        #sujet = sujet_list[0]
        for sujet in sujet_list:

            print(sujet)

            if sujet not in sujet_list_clean_hrv:
                os.chdir(os.path.join(path_results, sujet, 'HRV'))

                lines = [f'{sujet}', 'bad detection so far']
                with open(f'{sujet}_bad_detection.txt', 'w') as f:
                    f.writelines(lines)

                continue

            #### initiate containers
            df_hrv_dict = {}
            df_hrv_dict['sujet'], df_hrv_dict['cond'], df_hrv_dict['odor'] = [], [], []
            for metric_i in prms_hrv['metric_list']:
                df_hrv_dict[metric_i] = []
            df_hrv = pd.DataFrame(df_hrv_dict)

            #### load data
            ecg_allcond, ecg_cR_allcond = load_ecg(sujet, band_prep)

            #### compute
            #cond = conditions[0]
            for cond in conditions:

                #odor_i = odor_list[0]
                for odor_i in odor_list:
        
                    cR_time = np.where(ecg_cR_allcond[cond][odor_i] == ecg_cR_allcond[cond][odor_i].max())[0]
                    df = get_hrv_metrics_homemade(cR_time, prms_hrv)

                    dict_res = {}
                    dict_res['sujet'], dict_res['cond'], dict_res['odor'] = [sujet], [cond], [odor_i]
                    
                    #metric_i = prms_hrv['metric_list'][0]
                    for metric_i in prms_hrv['metric_list']:
                        dict_res[metric_i] = df[metric_i].values

                    df_i = pd.DataFrame(dict_res) 

                    df_hrv = pd.concat((df_hrv, df_i))

            df_hrv = df_hrv.reset_index().drop(columns='index')
            os.chdir(os.path.join(path_results, sujet, 'HRV'))
            df_hrv.to_excel(f"{sujet}_df_hrv.xlsx")
    