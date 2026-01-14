
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr
import cv2

import pickle
import gc

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *


debug = False







################################################
######## PLOT & SAVE PSD AND COH ########
################################################



def get_stats_topoplots(baseline_values, cond_values, chan_list_eeg):

    data = {'sujet' : [], 'cond' : [], 'chan' : [], 'value' : []}

    for sujet_i in range(baseline_values.shape[-1]):

        for chan_i, chan in enumerate(chan_list_eeg):

            data['sujet'].append(sujet_i)
            data['cond'].append('baseline')
            data['chan'].append(chan)
            data['value'].append(baseline_values[chan_i, sujet_i])

            data['sujet'].append(sujet_i)
            data['cond'].append('cond')
            data['chan'].append(chan)
            data['value'].append(cond_values[chan_i, sujet_i])
    
    df_stats = pd.DataFrame(data)

    mask_signi = np.array((), dtype='bool')

    for chan in chan_list_eeg:

        pval = get_stats_df(df=df_stats.query(f"chan == '{chan}'"), predictor='cond', outcome='value', subject='sujet', design='within')

        if pval < 0.05:
            mask_signi = np.append(mask_signi, True)
        else:
            mask_signi = np.append(mask_signi, False)

    return mask_signi



def get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet):

    source_path = os.getcwd()
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
                
    with open(f'allcond_{sujet}_Pxx.pkl', 'rb') as f:
        Pxx_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_Cxy.pkl', 'rb') as f:
        Cxy_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_surrogates.pkl', 'rb') as f:
        surrogates_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_cyclefreq.pkl', 'rb') as f:
        cyclefreq_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_MVL.pkl', 'rb') as f:
        MVL_allcond = pickle.load(f)

    os.chdir(source_path)

    return Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond





def get_Cxy_Surrogates_allcond(sujet):

    source_path = os.getcwd()
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    with open(f'allcond_{sujet}_Cxy.pkl', 'rb') as f:
        Cxy_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_surrogates.pkl', 'rb') as f:
        surrogates_allcond = pickle.load(f)

    os.chdir(source_path)

    return Cxy_allcond, surrogates_allcond


def compute_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allsujet():

    metric_to_load = ['Cxy', 'MVL', 'Cxy_surr', 'MVL_surr', 'Pxx_theta', 'Pxx_alpha', 'Pxx_beta', 'Pxx_l_gamma', 'Pxx_h_gamma']
    xr_data = np.zeros(( len(sujet_list), len(metric_to_load), len(conditions), len(odor_list), len(chan_list_eeg)))
    xr_dict = {'sujet' : sujet_list, 'metric' : metric_to_load, 'cond' : conditions, 'odor' : odor_list, 'chan' : chan_list_eeg}
    xr_allsujet = xr.DataArray(xr_data, dims=xr_dict.keys(), coords=xr_dict.values())  

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        #### load data
        # Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond = get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet)
        Cxy_allcond, surrogates_allcond = get_Cxy_Surrogates_allcond(sujet)
        prms = get_params()
        respfeatures_allcond = load_respfeatures(sujet)

        #### params
        hzPxx = np.linspace(0,srate/2,int(prms['nfft']/2+1))
        hzCxy = np.linspace(0,srate/2,int(prms['nfft']/2+1))
        mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
        hzCxy = hzCxy[mask_hzCxy]

        #### reduce data
        for cond in conditions:

            for odor_i in odor_list:

                mean_resp = respfeatures_allcond[cond][odor_i]['cycle_freq'].mean()
                hzCxy_mask = (hzCxy > (mean_resp - around_respi_Cxy)) & (hzCxy < (mean_resp + around_respi_Cxy))

                Cxy_allchan_i = Cxy_allcond[cond][odor_i][:,hzCxy_mask].mean(axis=1)
                xr_allsujet.loc[sujet, 'Cxy', cond, odor_i, :] = Cxy_allchan_i
                # MVL_allchan_i = MVL_allcond[cond][odor_i]
                # xr_allsujet.loc[sujet, 'MVL', cond, odor_i, :] = MVL_allchan_i

                Cxy_allchan_surr_i = surrogates_allcond['Cxy'][cond][odor_i][:len(chan_list_eeg),hzCxy_mask].mean(axis=1)
                xr_allsujet.loc[sujet, 'Cxy_surr', cond, odor_i, :] = np.array(Cxy_allchan_i > Cxy_allchan_surr_i)*1
                MVL_allchan_surr_i = np.array([np.percentile(surrogates_allcond['MVL'][cond][odor_i][nchan,:], 99) for nchan, _ in enumerate(chan_list_eeg)])
                # xr_allsujet.loc[sujet, 'MVL_surr', cond, odor_i, :] = np.array(MVL_allchan_i > MVL_allchan_surr_i)*1

        for cond in conditions:

            for odor_i in odor_list:

                #band, freq = 'theta', [4, 8]
                for band, freq in freq_band_fc_analysis.items():

                    hzPxx_mask = (hzPxx >= freq[0]) & (hzPxx <= freq[-1])
                    # Pxx_mean_i = Pxx_allcond[cond][odor_i][:,hzPxx_mask].mean(axis=1)
                    # xr_allsujet.loc[sujet, f'Pxx_{band}', cond, odor_i, :] = Pxx_mean_i

    
    #### get stats
    metric_to_load = ['Pxx_theta', 'Pxx_alpha', 'Pxx_beta', 'Pxx_l_gamma', 'Pxx_h_gamma']
    allsujet_stats_data = np.zeros((2, len(metric_to_load), len(odor_list), len(conditions), len(chan_list_eeg)), dtype='bool')

    #stats_type = 'intra'
    for stats_type_i, stats_type in enumerate(['intra', 'inter']):
        
        #metric = 'Cxy'
        for metric_i, metric in enumerate(metric_to_load):

            #### intra
            if stats_type == 'intra':

                #odor = 'o'
                for odor_i, odor in enumerate(odor_list):

                    #cond = 'MECA'
                    for cond_i, cond in enumerate(conditions):

                        if cond == 'FR_CV_1':

                            continue

                        baseline_values = xr_allsujet.transpose('metric', 'cond', 'odor', 'chan', 'sujet').loc[metric,'FR_CV_1',odor,:,:].values
                        cond_values = xr_allsujet.transpose('metric', 'cond', 'odor', 'chan', 'sujet').loc[metric,cond,odor,:,:].values

                        # print(stats_type,metric,cond,odor, flush=True)

                        allsujet_stats_data[stats_type_i, metric_i, odor_i, cond_i, :] = get_stats_topoplots(baseline_values, cond_values, chan_list_eeg)

            #### inter
            if stats_type == 'inter':

                #odor = 'o'
                for odor_i, odor in enumerate(odor_list):

                    if odor == 'o':
                        
                        continue

                    #cond = 'MECA'
                    for cond_i, cond in enumerate(conditions):

                        baseline_values = xr_allsujet.transpose('metric', 'cond', 'odor', 'chan', 'sujet').loc[metric,cond,'o',:,:].values
                        cond_values = xr_allsujet.transpose('metric', 'cond', 'odor', 'chan', 'sujet').loc[metric,cond,odor,:,:].values

                        # print(stats_type,metric,cond,odor, flush=True)

                        allsujet_stats_data[stats_type_i, metric_i, odor_i, cond_i, :] = get_stats_topoplots(baseline_values, cond_values, chan_list_eeg)

    xr_dict = {'stats_type' : ['intra', 'inter'], 'metric' : ['theta', 'alpha', 'beta', 'l_gamma', 'h_gamma'], 'odor' : odor_list, 'cond' : conditions, 'chan' : chan_list_eeg}
    xr_allsujet_stats = xr.DataArray(allsujet_stats_data, dims=xr_dict.keys(), coords=xr_dict.values())  
    
    return xr_allsujet, xr_allsujet_stats




def compute_Cxy_Surrogates_allsujet():

    metric_to_load = ['Cxy', 'Cxy_surr']
    xr_data = np.zeros(( len(sujet_list), len(metric_to_load), len(conditions), len(odor_list), len(chan_list_eeg)))
    xr_dict = {'sujet' : sujet_list, 'metric' : metric_to_load, 'cond' : conditions, 'odor' : odor_list, 'chan' : chan_list_eeg}
    xr_allsujet = xr.DataArray(xr_data, dims=xr_dict.keys(), coords=xr_dict.values())  

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        #### load data
        Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond = get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet)
        prms = get_params()
        respfeatures_allcond = load_respfeatures(sujet)

        #### params
        hzPxx = np.linspace(0,srate/2,int(prms['nfft']/2+1))
        hzCxy = np.linspace(0,srate/2,int(prms['nfft']/2+1))
        mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
        hzCxy = hzCxy[mask_hzCxy]

        #### reduce data
        for cond in conditions:

            for odor_i in odor_list:

                mean_resp = respfeatures_allcond[cond][odor_i]['cycle_freq'].mean()
                hzCxy_mask = (hzCxy > (mean_resp - around_respi_Cxy)) & (hzCxy < (mean_resp + around_respi_Cxy))

                Cxy_allchan_i = Cxy_allcond[cond][odor_i][:,hzCxy_mask].mean(axis=1)
                xr_allsujet.loc[sujet, 'Cxy', cond, odor_i, :] = Cxy_allchan_i

                Cxy_allchan_surr_i = surrogates_allcond['Cxy'][cond][odor_i][:len(chan_list_eeg),hzCxy_mask].mean(axis=1)
                xr_allsujet.loc[sujet, 'Cxy_surr', cond, odor_i, :] = np.array(Cxy_allchan_i > Cxy_allchan_surr_i)*1

    
    return xr_allsujet



def compute_Cxy_Surrogates_allsujet():

    metric_to_load = ['Cxy', 'Cxy_surr']
    xr_data = np.zeros(( len(sujet_list), len(metric_to_load), len(conditions), len(odor_list), len(chan_list_eeg)))
    xr_dict = {'sujet' : sujet_list, 'metric' : metric_to_load, 'cond' : conditions, 'odor' : odor_list, 'chan' : chan_list_eeg}
    xr_allsujet = xr.DataArray(xr_data, dims=xr_dict.keys(), coords=xr_dict.values())  

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        #### load data
        Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond = get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet)
        prms = get_params()
        respfeatures_allcond = load_respfeatures(sujet)

        #### params
        hzPxx = np.linspace(0,srate/2,int(prms['nfft']/2+1))
        hzCxy = np.linspace(0,srate/2,int(prms['nfft']/2+1))
        mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
        hzCxy = hzCxy[mask_hzCxy]

        #### reduce data
        for cond in conditions:

            for odor_i in odor_list:

                mean_resp = respfeatures_allcond[cond][odor_i]['cycle_freq'].mean()
                hzCxy_mask = (hzCxy > (mean_resp - around_respi_Cxy)) & (hzCxy < (mean_resp + around_respi_Cxy))

                Cxy_allchan_i = Cxy_allcond[cond][odor_i][:,hzCxy_mask].mean(axis=1)
                xr_allsujet.loc[sujet, 'Cxy', cond, odor_i, :] = Cxy_allchan_i

                Cxy_allchan_surr_i = surrogates_allcond['Cxy'][cond][odor_i][:len(chan_list_eeg),hzCxy_mask].mean(axis=1)
                xr_allsujet.loc[sujet, 'Cxy_surr', cond, odor_i, :] = np.array(Cxy_allchan_i > Cxy_allchan_surr_i)*1

    
    return xr_allsujet




    

################################
######## EXTRACT DF PLOT ########
################################


def get_df_TF_allsujet():

    #### create montage
    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')
    phase_list_lmm = ['inspi', 'expi']

    sujet_group_list = ['allsujet', 'rep', 'norep']
    term_list = ['(Intercept)', 'condCO2', 'condFR_CV_2', 'condMECA', 'odor-', 'odor+', 'condCO2:odor-', 'condFR_CV_2:odor-', 'condMECA:odor-', 'condCO2:odor+',
            'condFR_CV_2:odor+', 'condMECA:odor+']
    term_list_repnorep = ['(Intercept)', 'repTRUE', 'condCO2:repTRUE', 'condFR_CV_2:repTRUE', 'condMECA:repTRUE', 'odor-:repTRUE', 'odor+:repTRUE', 'condCO2:odor-:repTRUE', 
                          'condFR_CV_2:odor-:repTRUE', 'condMECA:odor-:repTRUE', 'condCO2:odor+:repTRUE', 'condFR_CV_2:odor+:repTRUE', 'condMECA:odor+:repTRUE']

    #### load data
    os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))
    df_TF = pd.read_excel(f"df_allsujet_TF.xlsx").drop(columns=['Unnamed: 0'])
    prms = get_params()

    os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'lmm'))

    for sujet_group_i, sujet_group in enumerate(sujet_group_list):

        for band_i, band in enumerate(freq_band_dict_lmm):

            print(sujet_group, band)

            for chan_i, chan in enumerate(chan_list_eeg):

                for phase_i, phase in enumerate(phase_list_lmm):

                    if (sujet_group_i + band_i + chan_i + phase_i) == 0:

                        df_stats_lmm = pd.read_excel(f"{sujet_group}_lmm_{chan}_{band}_{phase}_res.xlsx")
                        df_stats_lmm['sujet_group'] = [sujet_group] * df_stats_lmm.shape[0]
                        df_stats_lmm['chan'] = [chan] * df_stats_lmm.shape[0]
                        df_stats_lmm['band'] = [band] * df_stats_lmm.shape[0]
                        df_stats_lmm['phase'] = [phase] * df_stats_lmm.shape[0]

                    else:

                        _df_stats_lmm = pd.read_excel(f"{sujet_group}_lmm_{chan}_{band}_{phase}_res.xlsx")
                        _df_stats_lmm['sujet_group'] = [sujet_group] * _df_stats_lmm.shape[0]
                        _df_stats_lmm['chan'] = [chan] * _df_stats_lmm.shape[0]
                        _df_stats_lmm['band'] = [band] * _df_stats_lmm.shape[0]
                        _df_stats_lmm['phase'] = [phase] * _df_stats_lmm.shape[0]
                        df_stats_lmm = pd.concat([df_stats_lmm, _df_stats_lmm])

    for band_i, band in enumerate(freq_band_dict_lmm):

        print(band)

        for chan_i, chan in enumerate(chan_list_eeg):

            for phase_i, phase in enumerate(phase_list_lmm):

                if (band_i + chan_i + phase_i) == 0:

                    df_stats_lmm_repnorep = pd.read_excel(f"repnorep_lmm_{chan}_{band}_{phase}_res.xlsx")
                    df_stats_lmm_repnorep['chan'] = [chan] * df_stats_lmm_repnorep.shape[0]
                    df_stats_lmm_repnorep['band'] = [band] * df_stats_lmm_repnorep.shape[0]
                    df_stats_lmm_repnorep['phase'] = [phase] * df_stats_lmm_repnorep.shape[0]

                else:

                    _df_stats_lmm = pd.read_excel(f"repnorep_lmm_{chan}_{band}_{phase}_res.xlsx")
                    _df_stats_lmm['chan'] = [chan] * _df_stats_lmm.shape[0]
                    _df_stats_lmm['band'] = [band] * _df_stats_lmm.shape[0]
                    _df_stats_lmm['phase'] = [phase] * _df_stats_lmm.shape[0]
                    df_stats_lmm_repnorep = pd.concat([df_stats_lmm_repnorep, _df_stats_lmm])

    array_Pxx = np.zeros((len(sujet_list), len(freq_band_dict_lmm.keys()), len(phase_list_lmm), len(conditions), len(odor_list), len(chan_list_eeg)))
    array_estimate = np.zeros((len(sujet_group_list), len(freq_band_dict_lmm.keys()), len(phase_list_lmm), len(chan_list_eeg), len(term_list)))
    array_signi = np.zeros((len(sujet_group_list), len(freq_band_dict_lmm.keys()), len(phase_list_lmm), len(chan_list_eeg), len(term_list)))
    array_estimate_repnorep = np.zeros((len(freq_band_dict_lmm.keys()), len(phase_list_lmm), len(chan_list_eeg), len(term_list_repnorep)))
    array_signi_repnorep = np.zeros((len(freq_band_dict_lmm.keys()), len(phase_list_lmm), len(chan_list_eeg), len(term_list_repnorep)))

    sujet_idx = {s: i for i, s in enumerate(sujet_list)}
    band_idx = {b: i for i, b in enumerate(freq_band_dict_lmm.keys())}
    phase_idx = {p: i for i, p in enumerate(phase_list_lmm)}
    cond_idx = {c: i for i, c in enumerate(conditions)}
    odor_idx = {o: i for i, o in enumerate(odor_list)}
    chan_idx = {ch: i for i, ch in enumerate(chan_list_eeg)}

    # Now fill the array
    for _, row in df_TF.iterrows():
        i = sujet_idx[row['sujet']]
        j = band_idx[row['band']]
        k = phase_idx[row['phase']]
        l = cond_idx[row['cond']]
        m = odor_idx[row['odor']]
        n = chan_idx[row['chan']]
        array_Pxx[i, j, k, l, m, n] = row['Pxx']

    for sujet_group_i, sujet_group in enumerate(sujet_group_list):

        for band_i, band in enumerate(freq_band_dict_lmm):

            for phase_i, phase in enumerate(phase_list_lmm):

                for chan_i, chan in enumerate(chan_list_eeg):

                    for term_i, term in enumerate(term_list):

                        array_estimate[sujet_group_i, band_i, phase_i, chan_i, term_i] = df_stats_lmm.query(f"sujet_group == '{sujet_group}' and band == '{band}' and phase == '{phase}' and chan == '{chan}' and term == '{term}'")['estimate'].values[0]
                        array_signi[sujet_group_i, band_i, phase_i, chan_i, term_i] = df_stats_lmm.query(f"sujet_group == '{sujet_group}' and band == '{band}' and phase == '{phase}' and chan == '{chan}' and term == '{term}'")['p.value'].values[0]

    for band_i, band in enumerate(freq_band_dict_lmm):

        for phase_i, phase in enumerate(phase_list_lmm):

            for chan_i, chan in enumerate(chan_list_eeg):

                for term_i, term in enumerate(term_list_repnorep):

                    array_estimate_repnorep[band_i, phase_i, chan_i, term_i] = df_stats_lmm_repnorep.query(f"band == '{band}' and phase == '{phase}' and chan == '{chan}' and term == '{term}'")['estimate'].values[0]
                    array_signi_repnorep[band_i, phase_i, chan_i, term_i] = df_stats_lmm_repnorep.query(f"band == '{band}' and phase == '{phase}' and chan == '{chan}' and term == '{term}'")['p.value'].values[0]
            
    xr_Pxx = xr.DataArray(data=array_Pxx, dims=['sujet', 'band', 'phase', 'cond', 'odor', 'chan'], coords={'sujet' : sujet_list, 'band' : list(freq_band_dict_lmm.keys()), 'phase' : phase_list_lmm, 'cond' : conditions, 'odor' : odor_list, 'chan' : chan_list_eeg})
    xr_estimate = xr.DataArray(data=array_estimate, dims=['sujet_group', 'band', 'phase', 'chan', 'term'], coords={'sujet_group' : sujet_group_list, 'band' : list(freq_band_dict_lmm.keys()), 'phase' : phase_list_lmm, 'chan' : chan_list_eeg, 'term' : term_list})
    xr_signi = xr.DataArray(data=array_signi, dims=['sujet_group', 'band', 'phase', 'chan', 'term'], coords={'sujet_group' : sujet_group_list, 'band' : list(freq_band_dict_lmm.keys()), 'phase' : phase_list_lmm, 'chan' : chan_list_eeg, 'term' : term_list})
    xr_estimate_repnorep = xr.DataArray(data=array_estimate_repnorep, dims=['band', 'phase', 'chan', 'term'], coords={'band' : list(freq_band_dict_lmm.keys()), 'phase' : phase_list_lmm, 'chan' : chan_list_eeg, 'term' : term_list_repnorep})
    xr_signi_repnorep = xr.DataArray(data=array_signi_repnorep, dims=['band', 'phase', 'chan', 'term'], coords={'band' : list(freq_band_dict_lmm.keys()), 'phase' : phase_list_lmm, 'chan' : chan_list_eeg, 'term' : term_list_repnorep})

    #### lmm region
    os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'lmm'))

    for sujet_group_i, sujet_group in enumerate(sujet_group_list):

        for band_i, band in enumerate(freq_band_dict_lmm):

            print(sujet_group, band)

            for region_i, region in enumerate(chan_list_lobes_lmm):

                for phase_i, phase in enumerate(phase_list_lmm):

                    if (sujet_group_i + band_i + region_i + phase_i) == 0:

                        df_stats_lmm_region = pd.read_excel(f"{sujet_group}_lmm_{region}_{band}_{phase}_res.xlsx")
                        df_stats_lmm_region['sujet_group'] = [sujet_group] * df_stats_lmm_region.shape[0]
                        df_stats_lmm_region['region'] = [region] * df_stats_lmm_region.shape[0]
                        df_stats_lmm_region['band'] = [band] * df_stats_lmm_region.shape[0]
                        df_stats_lmm_region['phase'] = [phase] * df_stats_lmm_region.shape[0]

                    else:

                        _df_stats_lmm = pd.read_excel(f"{sujet_group}_lmm_{region}_{band}_{phase}_res.xlsx")
                        _df_stats_lmm['sujet_group'] = [sujet_group] * _df_stats_lmm.shape[0]
                        _df_stats_lmm['region'] = [region] * _df_stats_lmm.shape[0]
                        _df_stats_lmm['band'] = [band] * _df_stats_lmm.shape[0]
                        _df_stats_lmm['phase'] = [phase] * _df_stats_lmm.shape[0]
                        df_stats_lmm_region = pd.concat([df_stats_lmm_region, _df_stats_lmm])

    array_estimate_region = np.zeros((len(sujet_group_list), len(freq_band_dict_lmm.keys()), len(phase_list_lmm), len(chan_list_lobes_lmm), len(term_list)))
    array_signi_region = np.zeros((len(sujet_group_list), len(freq_band_dict_lmm.keys()), len(phase_list_lmm), len(chan_list_lobes_lmm), len(term_list)))
    # array_estimate_repnorep_region = np.zeros((len(freq_band_dict_lmm.keys()), len(phase_list_lmm), len(chan_list_lobes_lmm), len(term_list_repnorep)))

    for sujet_group_i, sujet_group in enumerate(sujet_group_list):

        for band_i, band in enumerate(freq_band_dict_lmm):

            for phase_i, phase in enumerate(phase_list_lmm):

                for region_i, region in enumerate(chan_list_lobes_lmm):

                    for term_i, term in enumerate(term_list):

                        array_estimate_region[sujet_group_i, band_i, phase_i, region_i, term_i] = df_stats_lmm_region.query(f"sujet_group == '{sujet_group}' and band == '{band}' and phase == '{phase}' and region == '{region}' and term == '{term}'")['estimate'].values[0]
                        array_signi_region[sujet_group_i, band_i, phase_i, region_i, term_i] = df_stats_lmm_region.query(f"sujet_group == '{sujet_group}' and band == '{band}' and phase == '{phase}' and region == '{region}' and term == '{term}'")['p.value'].values[0]

    # for band_i, band in enumerate(freq_band_dict_lmm):

    #     for phase_i, phase in enumerate(phase_list_lmm):

    #         for region_i, region in enumerate(chan_list_lobes_lmm):

    #             for term_i, term in enumerate(term_list_repnorep):

    #                 array_estimate_repnorep[band_i, phase_i, region_i, term_i] = df_stats_lmm_repnorep_region.query(f"band == '{band}' and phase == '{phase}' and chan == '{chan}' and term == '{term}'")['estimate'].values[0]
            
    xr_estimate_region = xr.DataArray(data=array_estimate_region, dims=['sujet_group', 'band', 'phase', 'region', 'term'], coords={'sujet_group' : sujet_group_list, 'band' : list(freq_band_dict_lmm.keys()), 'phase' : phase_list_lmm, 'region' : list(chan_list_lobes_lmm.keys()), 'term' : term_list})
    xr_signi_region = xr.DataArray(data=array_signi_region, dims=['sujet_group', 'band', 'phase', 'region', 'term'], coords={'sujet_group' : sujet_group_list, 'band' : list(freq_band_dict_lmm.keys()), 'phase' : phase_list_lmm, 'region' : list(chan_list_lobes_lmm.keys()), 'term' : term_list})
    # xr_stats_repnorep = xr.DataArray(data=array_estimate_repnorep, dims=['band', 'phase', 'region', 'term'], coords={'band' : list(freq_band_dict_lmm.keys()), 'phase' : phase_list_lmm, 'chan' : chan_list_lobes_lmm, 'term' : term_list_repnorep})

    return xr_Pxx, xr_estimate, xr_signi, xr_estimate_repnorep, xr_signi_repnorep, xr_estimate_region, xr_signi_region








################################
######## TOPOPLOT ########
################################



def plot_save_PSD_Cxy_CF_MVL_TOPOPLOT(xr_allsujet, xr_allsujet_stats):

    #### create montage
    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    sujet_respond = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list if sujet not in sujet_respond])

    #### scales
    scales_allband = {}

    #data_type = data_type_list[0]
    for data_type in xr_allsujet['metric'].values:

        scales_allband[data_type] = {}

        val = np.array([])

        for odor_i in odor_list:

            for cond in conditions:

                for sujet_group in ['allsujet', 'rep', 'no_rep']:

                    if data_type.find('surr') == -1:

                        if sujet_group == 'allsujet':
                            data_sel = xr_allsujet.loc[:,data_type, cond, odor_i].median('sujet').values
                        if sujet_group == 'rep':
                            data_sel = xr_allsujet.loc[sujet_respond,data_type, cond, odor_i].median('sujet').values
                        if sujet_group == 'no_rep':
                            data_sel = xr_allsujet.loc[sujet_no_respond,data_type, cond, odor_i].median('sujet').values
                    
                    if data_type.find('surr') != -1:

                        if sujet_group == 'allsujet':
                            data_sel = xr_allsujet.loc[:,data_type, cond, odor_i].sum('sujet').values / sujet_list.shape[0]
                        if sujet_group == 'rep':
                            data_sel = xr_allsujet.loc[sujet_respond,data_type, cond, odor_i].sum('sujet').values / sujet_respond.shape[0]
                        if sujet_group == 'no_rep':
                            data_sel = xr_allsujet.loc[sujet_no_respond,data_type, cond, odor_i].sum('sujet').values / sujet_no_respond.shape[0]
                    
                    val = np.append(val, data_sel)

        scales_allband[data_type]['min'] = val.min()
        scales_allband[data_type]['max'] = val.max()

    #### plot Cxy, MVL
    #sujet_group = 'allsujet'
    for sujet_group in ['allsujet', 'rep', 'no_rep']:

        #data_type = 'Cxy_surr'
        for data_type in xr_allsujet['metric'].values:

            if data_type.find('Pxx') == -1:

                vmin = np.round(scales_allband[data_type]['min'], 2)
                vmax = np.round(scales_allband[data_type]['max'], 2)

                fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
                plt.suptitle(f'{sujet_group}_{data_type}, vmin:{vmin} vmax:{vmax}')
                fig.set_figheight(10)
                fig.set_figwidth(10)

                #c, cond = 0, 'FR_CV_1'
                for c, cond in enumerate(conditions):

                    #r, odor_i = 0, odor_list[0]
                    for r, odor_i in enumerate(odor_list):

                        #### plot
                        ax = axs[r, c]

                        if r == 0:
                            ax.set_title(cond, fontweight='bold', rotation=0)
                        if c == 0:
                            ax.set_ylabel(f'{odor_i}')

                        if sujet_group == 'allsujet':
                            data_sel = xr_allsujet.loc[:, data_type, cond, odor_i].values
                        if sujet_group == 'rep':
                            data_sel = xr_allsujet.loc[sujet_respond, data_type, cond, odor_i].values
                        if sujet_group == 'no_rep':
                            data_sel = xr_allsujet.loc[sujet_no_respond, data_type, cond, odor_i].values

                        if data_type.find('surr') == -1:
                            data_sel = np.median(data_sel, axis=0)

                        elif data_type.find('surr') != -1:
                            if sujet_group == 'allsujet':
                                data_sel = np.sum(data_sel, axis=0) / sujet_list.shape[0]
                            if sujet_group == 'rep':
                                data_sel = np.sum(data_sel, axis=0) / sujet_respond.shape[0]
                            if sujet_group == 'no_rep':
                                data_sel = np.sum(data_sel, axis=0) / sujet_no_respond.shape[0]
                                             
                        mne.viz.plot_topomap(data_sel, info, axes=ax, vlim=(vmin, vmax), show=False)
                
                # plt.show() 

                #### save
                os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot'))
                fig.savefig(f'{data_type}_{sujet_group}_topo.jpeg', dpi=150)
                fig.clf()
                plt.close('all')
                gc.collect()

            elif data_type.find('Pxx') != -1:

                #### plot Pxx
                for stats_type in ['intra', 'inter']:

                    band = data_type[4:]

                    #### plot Pxx
                    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
                    plt.suptitle(f'{sujet_group}_{data_type}_{stats_type}_Pxx')
                    fig.set_figheight(10)
                    fig.set_figwidth(10)

                    #c, cond = 0, 'FR_CV_1'
                    for c, cond in enumerate(conditions):

                        #r, odor = 0, odor_list[0]
                        for r, odor in enumerate(odor_list):

                            #### plot
                            ax = axs[r, c]

                            if r == 0:
                                ax.set_title(cond, fontweight='bold', rotation=0)
                            if c == 0:
                                ax.set_ylabel(f'{odor}')

                            mask = xr_allsujet_stats.loc[stats_type, band, odor, cond,:].values
                            mask_params = dict(markersize=5, markerfacecolor='y')

                            if sujet_group == 'allsujet':
                                data_sel = xr_allsujet.loc[:, data_type, cond, odor_i].median('sujet').values
                            if sujet_group == 'rep':
                                data_sel = xr_allsujet.loc[sujet_respond, data_type, cond, odor_i].median('sujet').values
                            if sujet_group == 'no_rep':
                                data_sel = xr_allsujet.loc[sujet_no_respond, data_type, cond, odor_i].median('sujet').values
                            
                            mne.viz.plot_topomap(data_sel, info, axes=ax, 
                                                vlim=(scales_allband[data_type]['min'], scales_allband[data_type]['max']), 
                                                mask=mask, mask_params=mask_params, show=False)

                    # plt.show() 

                    #### save
                    os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot'))
                    fig.savefig(f'{data_type}_{sujet_group}_{stats_type}_topo.jpeg', dpi=150)
                    fig.clf()
                    plt.close('all')
                    gc.collect()






#n_chan, chan_name = 0, chan_list_eeg[0]
def plot_save_Cxy_TOPOPLOT_allsujet():

    #### create montage
    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    mask_params = dict(markersize=10, markerfacecolor='y')

    montage = mne.channels.make_standard_montage('standard_1020')
    pos = montage.get_positions()['ch_pos']

    #### load data
    # xr_allsujet = compute_Cxy_Surrogates_allsujet()

    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)
    hzCxy = np.linspace(0, srate / 2, int(nfft / 2 + 1))
    mask_hzCxy = (hzCxy >= freq_surrogates[0]) & (hzCxy < freq_surrogates[1])
    hzCxy_respi = hzCxy[mask_hzCxy]
    data_cxy = np.zeros((2, len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg)))

    #sujet = sujet_list[0]
    for sujet_i, sujet in enumerate(sujet_list):

        respfeatures_allcond = load_respfeatures(sujet)

        os.chdir(os.path.join(path_precompute, 'allsujet', 'PSD_Coh'))
        
        _surr_cxy = np.load(f"{sujet}_surr_Cxy.npy")[:,:,:len(chan_list_eeg),:]

        os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

        with open(f'allcond_{sujet}_Cxy.pkl', 'rb') as f:
            Cxy_allcond = pickle.load(f)

        for cond_i, cond in enumerate(conditions):

            for odor_i, odor in enumerate(odor_list):
                
                median_resp = np.median(respfeatures_allcond[cond][odor]['cycle_freq'])
                mask_cxy_hzpxx = (hzCxy_respi > (median_resp - around_respi_Cxy)) & (hzCxy_respi < (median_resp + around_respi_Cxy))

                for chan_i, chan in enumerate(chan_list_eeg):

                    data_cxy[0, sujet_i,cond_i,odor_i,chan_i] = np.median(Cxy_allcond[cond][odor][chan_i,mask_cxy_hzpxx])
                    data_cxy[1, sujet_i,cond_i,odor_i,chan_i] = np.median(_surr_cxy[cond_i,odor_i,chan_i,mask_cxy_hzpxx])

                if debug:

                    plt.plot(hzCxy_respi, Cxy_allcond[cond][odor][0,:], label='Cxy')
                    plt.plot(hzCxy_respi, data_cxy[1,sujet_i,cond_i,odor_i,0,:], label='Cxy_surr', color='r', linestyle='--')
                    plt.ylim(0,1)
                    plt.legend()
                    plt.show()

    xr_dict = {'type' : ['Cxy', 'surr'], 'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'chan' : chan_list_eeg}
    xr_Cxy = xr.DataArray(data=data_cxy, dims=xr_dict.keys(), coords=xr_dict)

    sujet_group_list = {'allsujet' : sujet_list, 'rep' : sujet_best_list_rev, 'norep' : sujet_no_respond_rev}

    cond_sel = ['FR_CV_1', 'CO2']
    vlim = {sujet_group : xr_Cxy.loc['Cxy', sujet_group_list[sujet_group], cond_sel, :, :].median('sujet').max().values for sujet_group in sujet_group_list}
    vlim = np.max(list(vlim.values()))

    #### plot Cxy
    #sujet_group = 'allsujet'
    for sujet_group in sujet_group_list:

        # cond_to_plot = ['FR_CV_1', 'FR_CV_1_surr', 'CO2', 'CO2_surr']
        cond_to_plot = ['FR_CV_1', 'FR_CV_1_nsigni', 'CO2', 'CO2_nsigni']
        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_to_plot))
        plt.suptitle(f'{sujet_group} Cxy n:{sujet_group_list[sujet_group].size}')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(cond_to_plot):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                        ax.set_title(cond, fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')

                if c in [0,2]:

                    topoplot_data = xr_Cxy.loc['Cxy', sujet_group_list[sujet_group], cond, odor, :].median('sujet').values
                    topoplot_data_mask = topoplot_data > xr_Cxy.loc['surr', sujet_group_list[sujet_group], cond, odor, :].median('sujet').values
                    im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                            mask=topoplot_data_mask, mask_params=mask_params, vlim=(0, vlim), cmap='viridis')
                    
                    if r == 2:
                        cbar1 = fig.colorbar(im, ax=ax, orientation='horizontal')
                        cbar1.set_label('Cxy')
                    
                if c in [1,3]:

                    if r == 0:
                        ax.set_title(f"cond surr count", fontweight='bold', rotation=0)

                    topoplot_data = (xr_Cxy.loc['Cxy', sujet_group_list[sujet_group], cond[:-7], odor, :] > xr_Cxy.loc['surr', sujet_group_list[sujet_group], cond[:-7], odor, :]).sum('sujet').values
                    topoplot_data = topoplot_data / sujet_group_list[sujet_group].size * 100
                    im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                            vlim=(0, 100), cmap='viridis')
                    
                    if r == 2:
                        cbar1 = fig.colorbar(im, ax=ax, orientation='horizontal')
                        cbar1.set_label('Cxy count %')


        # plt.show() 

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot_summary'))
        fig.savefig(f'Cxy_{sujet_group}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()




def plot_save_Cxy_TOPOPLOT_allsujet_perm():

    #### create montage
    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    mask_params = dict(markersize=10, markerfacecolor='y')

    sujet_group_list = ['allsujet', 'rep', 'norep']

    #### load data
    os.chdir(os.path.join(path_precompute, 'allsujet', 'PSD_Coh'))
    xr_intra = xr.open_dataarray(f"perm_intra_Cxy.nc")
    xr_inter = xr.open_dataarray(f"perm_inter_Cxy.nc")
    xr_repnorep = xr.open_dataarray(f"perm_repnorep_Cxy.nc")
    prms = get_params()

    #### params
    hzCxy = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #### INTRA plot Cxy
    vlim = {}

    for sujet_group in sujet_group_list:
        vlim[sujet_group] = np.abs(np.array([xr_intra.loc['Cxy_diff', sujet_group,:, :].values.min(), xr_intra.loc['Cxy_diff', sujet_group,:, :].values.max()])).max()

    cond_sel = xr_intra['cond'].values.tolist()
    odor_sel = xr_intra['odor'].values.tolist()

    for sujet_group in sujet_group_list:

        fig, axs = plt.subplots(nrows=len(odor_sel), ncols=len(cond_sel))
        plt.suptitle(f'intra_{sujet_group}_Cxy')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(cond_sel):

            #r, odor = 0, odor_list[0]
            for r, odor in enumerate(odor_sel):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(cond, fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')

                topoplot_data = xr_intra.loc['Cxy_diff', sujet_group, cond, odor, :].values
                mask_signi = xr_intra.loc['cluster', sujet_group, cond, odor, :].values.astype('bool')

                im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                        mask=mask_signi, mask_params=mask_params, vlim=(-vlim[sujet_group], vlim[sujet_group]), cmap='seismic')
                
                cbar = fig.colorbar(im, ax=ax)

        # plt.show() 

        os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot_allsujet_Cxy'))
        fig.savefig(f'intra_{sujet_group}_topo_Cxy.jpeg', dpi=150)
        plt.close('all')

    #### INTER plot Cxy
    vlim = {}

    for sujet_group in sujet_group_list:
        vlim[sujet_group] = np.abs(np.array([xr_inter.loc['Cxy_diff', sujet_group,:, :].values.min(), xr_inter.loc['Cxy_diff', sujet_group,:, :].values.max()])).max()

    cond_sel = xr_inter['cond'].values.tolist()
    odor_sel = xr_inter['odor'].values.tolist()

    for sujet_group in sujet_group_list:

        fig, axs = plt.subplots(nrows=len(odor_sel), ncols=len(cond_sel))
        plt.suptitle(f'inter_{sujet_group}_Cxy')
        fig.set_figheight(10)
        fig.set_figwidth(16)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(cond_sel):

            #r, odor = 0, odor_list[0]
            for r, odor in enumerate(odor_sel):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(cond, fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')

                topoplot_data = xr_inter.loc['Cxy_diff', sujet_group, cond, odor, :].values
                mask_signi = xr_inter.loc['cluster', sujet_group, cond, odor, :].values.astype('bool')

                im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                        mask=mask_signi, mask_params=mask_params, vlim=(-vlim[sujet_group], vlim[sujet_group]), cmap='seismic')
                
                cbar = fig.colorbar(im, ax=ax)

        # plt.show() 

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot_allsujet_Cxy'))
        fig.savefig(f'inter_{sujet_group}_topo_Cxy.jpeg', dpi=150)
        plt.close('all')

    #### REPNOREP plot Cxy
    vlim = np.abs(np.array([xr_repnorep.loc['Cxy_diff', :, :].values.min(), xr_repnorep.loc['Cxy_diff', :, :].values.max()])).max()

    cond_sel = xr_repnorep['cond'].values.tolist()
    odor_sel = xr_repnorep['odor'].values.tolist()

    fig, axs = plt.subplots(nrows=len(odor_sel), ncols=len(cond_sel))
    plt.suptitle(f'repnorep_Cxy')
    fig.set_figheight(10)
    fig.set_figwidth(16)

    #c, cond = 0, 'FR_CV_1'
    for c, cond in enumerate(cond_sel):

        #r, odor = 0, odor_list[0]
        for r, odor in enumerate(odor_sel):

            #### plot
            ax = axs[r, c]

            if r == 0:
                ax.set_title(cond, fontweight='bold', rotation=0)
            if c == 0:
                ax.set_ylabel(f'{odor}')

            topoplot_data = xr_repnorep.loc['Cxy_diff', cond, odor, :].values
            mask_signi = xr_repnorep.loc['cluster', cond, odor, :].values.astype('bool')

            im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim), cmap='seismic')
            
            cbar = fig.colorbar(im, ax=ax)

    # plt.show() 

    #### save
    os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot_allsujet_Cxy'))
    fig.savefig(f'repnorep_topo_Cxy.jpeg', dpi=150)
    plt.close('all')





def plot_save_Cxy_TOPOPLOT_allsujet_lmm():

    #### create montage
    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    mask_params = dict(markersize=10, markerfacecolor='y')

    sujet_group_list = ['allsujet', 'rep', 'norep']
    term_list = ['(Intercept)', 'condCO2', 'condFR_CV_2', 'condMECA', 'odor-', 'odor+', 'condCO2:odor-', 'condFR_CV_2:odor-', 'condMECA:odor-', 'condCO2:odor+',
            'condFR_CV_2:odor+', 'condMECA:odor+']
    term_list_repnorep = ['(Intercept)', 'REPTRUE', 'condCO2:REPTRUE', 'condFR_CV_2:REPTRUE', 'condMECA:REPTRUE', 'odor-:REPTRUE', 'odor+:REPTRUE', 'condCO2:odor-:REPTRUE', 
                          'condFR_CV_2:odor-:REPTRUE', 'condMECA:odor-:REPTRUE', 'condCO2:odor+:REPTRUE', 'condFR_CV_2:odor+:REPTRUE', 'condMECA:odor+:REPTRUE']


    #### load data
    os.chdir(os.path.join(path_precompute, 'allsujet', 'PSD_Coh', 'stats'))
    df_Cxy = pd.read_excel(f"Cxy_allsujet.xlsx").drop(columns=['Unnamed: 0'])
    prms = get_params()

    os.chdir(os.path.join(path_precompute, 'allsujet', 'PSD_Coh', 'stats'))

    for sujet_group_i, sujet_group in enumerate(sujet_group_list):

        for chan_i, chan in enumerate(chan_list_eeg):

            if sujet_group_i == 0 and chan_i == 0:

                df_stats_lmm = pd.read_excel(f"{sujet_group}_lmm_{chan}_res.xlsx")
                df_stats_lmm['sujet_group'] = [sujet_group] * df_stats_lmm.shape[0]
                df_stats_lmm['chan'] = [chan] * df_stats_lmm.shape[0]

            else:

                _df_stats_lmm = pd.read_excel(f"{sujet_group}_lmm_{chan}_res.xlsx")
                _df_stats_lmm['sujet_group'] = [sujet_group] * _df_stats_lmm.shape[0]
                _df_stats_lmm['chan'] = [chan] * _df_stats_lmm.shape[0]
                df_stats_lmm = pd.concat([df_stats_lmm, _df_stats_lmm])

    for chan_i, chan in enumerate(chan_list_eeg):

        if chan_i == 0:

            df_stats_lmm_repnorep = pd.read_excel(f"repnorep_lmm_{chan}_res.xlsx")
            df_stats_lmm_repnorep['chan'] = [chan] * df_stats_lmm_repnorep.shape[0]

        else:

            _df_stats_lmm = pd.read_excel(f"repnorep_lmm_{chan}_res.xlsx")
            _df_stats_lmm['chan'] = [chan] * _df_stats_lmm.shape[0]
            df_stats_lmm_repnorep = pd.concat([df_stats_lmm_repnorep, _df_stats_lmm])

    array_Cxy = np.zeros((len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg)))
    array_stats = np.zeros((len(sujet_group_list), len(chan_list_eeg), len(term_list)))
    array_stats_repnorep = np.zeros((len(chan_list_eeg), len(term_list_repnorep)))

    sujet_idx = {s: i for i, s in enumerate(sujet_list)}
    cond_idx = {c: i for i, c in enumerate(conditions)}
    odor_idx = {o: i for i, o in enumerate(odor_list)}
    chan_idx = {ch: i for i, ch in enumerate(chan_list_eeg)}

    for _, row in df_Cxy.iterrows():
        i = sujet_idx[row['sujet']]
        j = cond_idx[row['cond']]
        k = odor_idx[row['odor']]
        l = chan_idx[row['chan']]
        array_Cxy[i, j, k, l] = row['Cxy']

    for sujet_group_i, sujet_group in enumerate(sujet_group_list):

        for chan_i, chan in enumerate(chan_list_eeg):

            for term_i, term in enumerate(term_list):

                array_stats[sujet_group_i, chan_i, term_i] = df_stats_lmm.query(f"sujet_group == '{sujet_group}' and chan == '{chan}' and term == '{term}'")['estimate'].values[0]
            
    for chan_i, chan in enumerate(chan_list_eeg):

        for term_i, term in enumerate(term_list_repnorep):

            array_stats_repnorep[chan_i, term_i] = df_stats_lmm_repnorep.query(f"chan == '{chan}' and term == '{term}'")['estimate'].values[0]   

    xr_Cxy = xr.DataArray(data=array_Cxy, dims=['sujet', 'cond', 'odor', 'chan'], coords={'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'chan' : chan_list_eeg})
    xr_stats = xr.DataArray(data=array_stats, dims=['sujet_group', 'chan', 'term'], coords={'sujet_group' : sujet_group_list, 'chan' : chan_list_eeg, 'term' : term_list})
    xr_stats_repnorep = xr.DataArray(data=array_stats_repnorep, dims=['chan', 'term'], coords={'chan' : chan_list_eeg, 'term' : term_list_repnorep})

    #### params
    hzCxy = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #### lim
    vlim = {}

    for sujet_group in sujet_group_list:
        if sujet_group == 'allsujet':
            sujet_sel = sujet_list
        elif sujet_group == 'rep':
            sujet_sel = sujet_best_list_rev 
        elif sujet_group == 'norep':
            sujet_sel = sujet_no_respond_rev 

        vlim[sujet_group] = np.array([xr_Cxy.loc[sujet_sel].median('sujet').values.min(), xr_Cxy.loc[sujet_sel].median('sujet').values.max()])

    vlim_stats = {}

    for sujet_group in sujet_group_list:

        vlim_stats[sujet_group] = np.abs(np.array([xr_stats.loc[sujet_group].values.min(), xr_stats.loc[sujet_group].values.max()])).max()

    vlim_stats_repnorep = np.abs(np.array([xr_stats_repnorep.values.min(), xr_stats_repnorep.values.max()])).max()

    #### plot Cxy
    for sujet_group in sujet_group_list:

        if sujet_group == 'allsujet':
            sujet_sel = sujet_list
        elif sujet_group == 'rep':
            sujet_sel = sujet_best_list_rev 
        elif sujet_group == 'norep':
            sujet_sel = sujet_no_respond_rev 

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
        plt.suptitle(f'{sujet_group}_Cxy')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'CO2'
        for c, cond in enumerate(conditions):

            #r, odor = 0, odor_list[1]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(cond, fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')

                topoplot_data = xr_Cxy.loc[sujet_sel, cond, odor].median('sujet').values
                if cond == 'FR_CV_1' and odor == 'o':
                    significance_type = np.zeros(topoplot_data.shape)
                elif cond != 'FR_CV_1' and odor == 'o':
                    mask_signi_cond = []
                    for chan in chan_list_eeg:
                        mask_signi_cond.append(df_stats_lmm.query(f"sujet_group == '{sujet_group}' and chan == '{chan}' and term == 'cond{cond}'")['p.value'].values[0] < 0.05)
                    significance_type = np.array(mask_signi_cond).astype(int)
                elif cond == 'FR_CV_1' and odor != 'o':
                    mask_signi_odor = []
                    for chan in chan_list_eeg:
                        mask_signi_odor.append(df_stats_lmm.query(f"sujet_group == '{sujet_group}' and chan == '{chan}' and term == 'odor{odor}'")['p.value'].values[0] < 0.05)
                    significance_type = np.array(mask_signi_odor).astype(int) * 2
                else:    
                    mask_signi_condodor = []
                    for chan in chan_list_eeg:
                        mask_signi_condodor.append(df_stats_lmm.query(f"sujet_group == '{sujet_group}' and chan == '{chan}' and term == 'cond{cond}:odor{odor}'")['p.value'].values[0] < 0.05)
                    significance_type = np.array(mask_signi_condodor).astype(int) * 3

                im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                        vlim=(vlim[sujet_group][0], vlim[sujet_group][1]), cmap='RdPu')

                # Now manually add the significance markers
                for idx, (chan_name, signi) in enumerate(zip(chan_list_eeg, significance_type)):
                    if signi == 0:
                        continue  # Not significant
                    pos_x, pos_y = info['chs'][idx]['loc'][:2]
                    if signi == 1:
                        color = 'red'  # only cond
                    elif signi == 2:
                        color = 'blue'  # only odor
                    elif signi == 3:
                        color = 'purple'  # both
                    ax.scatter(pos_x, pos_y, color=color, s=100, edgecolor='k', zorder=10)

        # plt.show() 

        os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot_allsujet_Cxy'))
        fig.savefig(f'lmm_{sujet_group}_topo_Cxy.jpeg', dpi=150)
        plt.close('all')

        #### estimate

        mask_params = dict(markersize=12, markerfacecolor='y')

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
        plt.suptitle(f'{sujet_group}_Cxy')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'CO2'
        for c, cond in enumerate(conditions):

            #r, odor = 0, odor_list[1]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(cond, fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')

                if cond == 'FR_CV_1' and odor == 'o':
                    topoplot_data = xr_stats.loc[sujet_group, :, f'(Intercept)'].values
                    significance_type = np.zeros(topoplot_data.shape).astype('bool')

                    im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                        vlim=(-vlim_stats[sujet_group], vlim_stats[sujet_group]), mask=significance_type, 
                                        mask_params=mask_params, cmap='seismic')
                    
                elif cond != 'FR_CV_1' and odor == 'o':
                    topoplot_data = xr_stats.loc[sujet_group, :, f'cond{cond}'].values
                    significance_type = np.zeros(topoplot_data.shape)
                    mask_signi_cond = []
                    for chan in chan_list_eeg:
                        mask_signi_cond.append(df_stats_lmm.query(f"sujet_group == '{sujet_group}' and chan == '{chan}' and term == 'cond{cond}'")['p.value'].values[0] < 0.05)
                    significance_type = np.array(mask_signi_cond)

                    im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                        vlim=(-vlim_stats[sujet_group], vlim_stats[sujet_group]), mask=significance_type, 
                                        mask_params=mask_params, cmap='seismic')
                    

                elif cond == 'FR_CV_1' and odor != 'o':
                    topoplot_data = xr_stats.loc[sujet_group, :, f'odor{odor}'].values
                    mask_signi_odor = []
                    for chan in chan_list_eeg:
                        mask_signi_odor.append(df_stats_lmm.query(f"sujet_group == '{sujet_group}' and chan == '{chan}' and term == 'odor{odor}'")['p.value'].values[0] < 0.05)
                    significance_type = np.array(mask_signi_odor).astype('bool')

                    im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                        vlim=(-vlim_stats[sujet_group], vlim_stats[sujet_group]), mask=significance_type, 
                                        mask_params=mask_params, cmap='seismic')
                    
                else:    
                    topoplot_data = xr_stats.loc[sujet_group, :, f'cond{cond}:odor{odor}'].values
                    mask_signi_condodor = []
                    for chan in chan_list_eeg:
                        mask_signi_condodor.append(df_stats_lmm.query(f"sujet_group == '{sujet_group}' and chan == '{chan}' and term == 'cond{cond}:odor{odor}'")['p.value'].values[0] < 0.05)
                    significance_type = np.array(mask_signi_condodor).astype('bool')

                    im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                        vlim=(-vlim_stats[sujet_group], vlim_stats[sujet_group]), mask=significance_type, 
                                        mask_params=mask_params, cmap='seismic')

        # plt.show() 

        os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot_allsujet_Cxy'))
        fig.savefig(f'lmm_estimate_{sujet_group}_topo_Cxy.jpeg', dpi=150)
        plt.close('all')

    #### estimate repnorep

    mask_params = dict(markersize=12, markerfacecolor='y')

    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
    plt.suptitle(f'repnorep_Cxy')
    fig.set_figheight(10)
    fig.set_figwidth(10)

    #c, cond = 0, 'CO2'
    for c, cond in enumerate(conditions):

        #r, odor = 0, odor_list[1]
        for r, odor in enumerate(odor_list):

            #### plot
            ax = axs[r, c]

            if r == 0:
                ax.set_title(cond, fontweight='bold', rotation=0)
            if c == 0:
                ax.set_ylabel(f'{odor}')

            if cond == 'FR_CV_1' and odor == 'o':
                topoplot_data = xr_stats_repnorep.loc[:, f'REPTRUE'].values
                significance_type = np.zeros(topoplot_data.shape)
                mask_signi_cond = []
                for chan in chan_list_eeg:
                    mask_signi_cond.append(df_stats_lmm_repnorep.query(f"chan == '{chan}' and term == 'REPTRUE'")['p.value'].values[0] < 0.05)
                significance_type = np.array(mask_signi_cond)

                im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    vlim=(-vlim_stats_repnorep, vlim_stats_repnorep), mask=significance_type, 
                                    mask_params=mask_params, cmap='seismic')
                
            elif cond != 'FR_CV_1' and odor == 'o':
                topoplot_data = xr_stats_repnorep.loc[:, f'cond{cond}:REPTRUE'].values
                significance_type = np.zeros(topoplot_data.shape)
                mask_signi_cond = []
                for chan in chan_list_eeg:
                    mask_signi_cond.append(df_stats_lmm_repnorep.query(f"chan == '{chan}' and term == 'cond{cond}:REPTRUE'")['p.value'].values[0] < 0.05)
                significance_type = np.array(mask_signi_cond)

                im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    vlim=(-vlim_stats_repnorep, vlim_stats_repnorep), mask=significance_type, 
                                    mask_params=mask_params, cmap='seismic')
                

            elif cond == 'FR_CV_1' and odor != 'o':
                topoplot_data = xr_stats_repnorep.loc[:, f'odor{odor}:REPTRUE'].values
                mask_signi_odor = []
                for chan in chan_list_eeg:
                    mask_signi_odor.append(df_stats_lmm_repnorep.query(f"chan == '{chan}' and term == 'odor{odor}:REPTRUE'")['p.value'].values[0] < 0.05)
                significance_type = np.array(mask_signi_odor).astype('bool')

                im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    vlim=(-vlim_stats_repnorep, vlim_stats_repnorep), mask=significance_type, 
                                    mask_params=mask_params, cmap='seismic')
                
            else:    
                topoplot_data = xr_stats_repnorep.loc[:, f'cond{cond}:odor{odor}:REPTRUE'].values
                mask_signi_condodor = []
                for chan in chan_list_eeg:
                    mask_signi_condodor.append(df_stats_lmm_repnorep.query(f"chan == '{chan}' and term == 'cond{cond}:odor{odor}:REPTRUE'")['p.value'].values[0] < 0.05)
                significance_type = np.array(mask_signi_condodor).astype('bool')

                im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    vlim=(-vlim_stats_repnorep, vlim_stats_repnorep), mask=significance_type, 
                                    mask_params=mask_params, cmap='seismic')

    # plt.show() 

    os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot_allsujet_Cxy'))
    fig.savefig(f'lmm_estimate_repnorep_topo_Cxy.jpeg', dpi=150)
    plt.close('all')


def plot_save_Pxx_TOPOPLOT_allsujet_lmm():

    #### create montage
    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')
    phase_list_lmm = ['inspi', 'expi']

    plevel = 0.05

    mask_params = dict(markersize=10, markerfacecolor='y')

    sujet_group_list = ['allsujet', 'rep', 'norep']

    xr_Pxx, xr_estimate, xr_signi, xr_estimate_repnorep, xr_signi_repnorep, xr_estimate_region, xr_signi_region = get_df_TF_allsujet()


    #### lim
    vlim_estimate = {}

    for sujet_group in sujet_group_list:

        for band_i, band in enumerate(freq_band_dict_lmm):
            
            for phase_i, phase in enumerate(phase_list_lmm):

                vlim_estimate[sujet_group] = np.abs(np.array([xr_estimate.loc[sujet_group, band, phase].values.min(), xr_estimate.loc[sujet_group, band, phase].values.max()])).max()

    for band_i, band in enumerate(freq_band_dict_lmm):
        
        for phase_i, phase in enumerate(phase_list_lmm):

            vlim_estimate_repnorep = np.abs(np.array([xr_estimate_repnorep.loc[band, phase].values.min(), xr_estimate_repnorep.loc[band, phase].values.max()])).max()


    #### plot Pxx estimate
    mask_params = dict(markersize=12, markerfacecolor='y')

    for sujet_group in sujet_group_list:

        if sujet_group == 'allsujet':
            sujet_sel = sujet_list
        elif sujet_group == 'rep':
            sujet_sel = sujet_best_list_rev 
        elif sujet_group == 'norep':
            sujet_sel = sujet_no_respond_rev 

        for band_i, band in enumerate(freq_band_dict_lmm):
            
            for phase_i, phase in enumerate(phase_list_lmm): 

                fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
                plt.suptitle(f'{sujet_group}_{band}_{phase}_Pxx')
                fig.set_figheight(10)
                fig.set_figwidth(10)

                #c, cond = 0, 'CO2'
                for c, cond in enumerate(conditions):

                    #r, odor = 0, odor_list[1]
                    for r, odor in enumerate(odor_list):

                        #### plot
                        ax = axs[r, c]

                        if r == 0:
                            ax.set_title(cond, fontweight='bold', rotation=0)
                        if c == 0:
                            ax.set_ylabel(f'{odor}')

                        if cond == 'FR_CV_1' and odor == 'o':
                            topoplot_data = xr_estimate.loc[sujet_group, band, phase, :, f'(Intercept)'].values
                            mask_signi = np.zeros(topoplot_data.shape).astype('bool')

                            im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                                vlim=(-vlim_estimate[sujet_group], vlim_estimate[sujet_group]), mask=mask_signi, 
                                                mask_params=mask_params, cmap='seismic')
                            
                        elif cond != 'FR_CV_1' and odor == 'o':
                            topoplot_data = xr_estimate.loc[sujet_group, band, phase, :, f'cond{cond}'].values
                            mask_signi = xr_signi.loc[sujet_group, band, phase, :, f'cond{cond}'].values < plevel

                            im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                                vlim=(-vlim_estimate[sujet_group], vlim_estimate[sujet_group]), mask=mask_signi, 
                                                mask_params=mask_params, cmap='seismic')
                            

                        elif cond == 'FR_CV_1' and odor != 'o':
                            topoplot_data = xr_estimate.loc[sujet_group, band, phase, :, f'odor{odor}'].values
                            mask_signi = xr_signi.loc[sujet_group, band, phase, :, f'odor{odor}'].values < plevel

                            im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                                vlim=(-vlim_estimate[sujet_group], vlim_estimate[sujet_group]), mask=mask_signi, 
                                                mask_params=mask_params, cmap='seismic')
                            
                        else:    
                            topoplot_data = xr_estimate.loc[sujet_group, band, phase, :, f'cond{cond}:odor{odor}'].values
                            mask_signi = xr_signi.loc[sujet_group, band, phase, :, f'cond{cond}:odor{odor}'].values < plevel

                            im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                                vlim=(-vlim_estimate[sujet_group], vlim_estimate[sujet_group]), mask=mask_signi, 
                                                mask_params=mask_params, cmap='seismic')

                # plt.show() 

                os.chdir(os.path.join(path_results, 'allplot', 'TF', 'Pxx', 'topoplot_lmm'))
                fig.savefig(f'{sujet_group}_{band}_{phase}_topo_lmm_Pxx.jpeg', dpi=150)
                plt.close('all')

    #### plot Pxx estimate repnorep
    cond_sel = ['FR_CV_1', 'CO2']

    for band_i, band in enumerate(freq_band_dict_lmm):
        
        for phase_i, phase in enumerate(phase_list_lmm): 

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_sel))
            plt.suptitle(f'repnorep_{band}_{phase}_Pxx')
            fig.set_figheight(10)
            fig.set_figwidth(10)

            #c, cond = 0, 'CO2'
            for c, cond in enumerate(cond_sel):

                #r, odor = 0, odor_list[1]
                for r, odor in enumerate(odor_list):

                    #### plot
                    ax = axs[r, c]

                    if r == 0:
                        ax.set_title(cond, fontweight='bold', rotation=0)
                    if c == 0:
                        ax.set_ylabel(f'{odor}')

                    if cond == 'FR_CV_1' and odor == 'o':
                        topoplot_data = xr_estimate_repnorep.loc[band, phase, :, f'repTRUE'].values
                        mask_signi = xr_signi_repnorep.loc[band, phase, :, f'repTRUE'].values < plevel

                        im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                            vlim=(-vlim_estimate_repnorep, vlim_estimate_repnorep), mask=mask_signi, 
                                            mask_params=mask_params, cmap='seismic')
                        
                    elif cond != 'FR_CV_1' and odor == 'o':
                        topoplot_data = xr_estimate_repnorep.loc[band, phase, :, f'cond{cond}:repTRUE'].values
                        mask_signi = xr_signi_repnorep.loc[band, phase, :, f'cond{cond}:repTRUE'].values < plevel

                        im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                            vlim=(-vlim_estimate_repnorep, vlim_estimate_repnorep), mask=mask_signi, 
                                            mask_params=mask_params, cmap='seismic')
                        

                    elif cond == 'FR_CV_1' and odor != 'o':
                        topoplot_data = xr_estimate_repnorep.loc[band, phase, :, f'odor{odor}:repTRUE'].values
                        mask_signi = xr_signi_repnorep.loc[band, phase, :, f'odor{odor}:repTRUE'].values < plevel

                        im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                            vlim=(-vlim_estimate_repnorep, vlim_estimate_repnorep), mask=mask_signi, 
                                            mask_params=mask_params, cmap='seismic')
                        
                    else:    
                        topoplot_data = xr_estimate_repnorep.loc[band, phase, :, f'cond{cond}:odor{odor}:repTRUE'].values
                        mask_signi = xr_signi_repnorep.loc[band, phase, :, f'cond{cond}:odor{odor}:repTRUE'].values < plevel

                        im, _ = mne.viz.plot_topomap(data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                            vlim=(-vlim_estimate_repnorep, vlim_estimate_repnorep), mask=mask_signi, 
                                            mask_params=mask_params, cmap='seismic')

            # plt.show() 

            os.chdir(os.path.join(path_results, 'allplot', 'TF', 'Pxx', 'topoplot_lmm'))
            fig.savefig(f'repnorep_{band}_{phase}_topo_lmm_Pxx.jpeg', dpi=150)
            plt.close('all')

    #### plot Pxx estimate only CO2
    cond_phase_order = [('FR_CV_1', 'inspi'), ('FR_CV_1', 'expi'), ('CO2', 'inspi'), ('CO2', 'expi')]

    for sujet_group in sujet_group_list:

        if sujet_group == 'allsujet':
            sujet_sel = sujet_list
        elif sujet_group == 'rep':
            sujet_sel = sujet_best_list_rev 
        elif sujet_group == 'norep':
            sujet_sel = sujet_no_respond_rev 

        for band_i, band in enumerate(freq_band_dict_lmm):

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_phase_order))
            plt.suptitle(f'{sujet_group}_{band}_Pxx')
            fig.set_figheight(12)
            fig.set_figwidth(10)

            for c, (cond, phase) in enumerate(cond_phase_order):
                for r, odor in enumerate(odor_list):

                    ax = axs[r, c]

                    if r == 0:
                        ax.set_title(f'{cond}_{phase}', fontweight='bold', rotation=0)
                    if c == 0:
                        ax.set_ylabel(f'{odor}')

                    # Topomap logic
                    if cond == 'FR_CV_1' and odor == 'o':
                        topoplot_data = xr_estimate.loc[sujet_group, band, phase, :, f'(Intercept)'].values
                        mask_signi = np.zeros(topoplot_data.shape).astype('bool')

                        im, _ = mne.viz.plot_topomap(
                            data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            vlim=(-vlim_estimate[sujet_group], vlim_estimate[sujet_group]), mask=mask_signi,
                            mask_params=mask_params, cmap='seismic'
                        )

                    elif cond != 'FR_CV_1' and odor == 'o':
                        topoplot_data = xr_estimate.loc[sujet_group, band, phase, :, f'cond{cond}'].values
                        mask_signi = xr_signi.loc[sujet_group, band, phase, :, f'cond{cond}'].values < plevel

                        im, _ = mne.viz.plot_topomap(
                            data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            vlim=(-vlim_estimate[sujet_group], vlim_estimate[sujet_group]), mask=mask_signi,
                            mask_params=mask_params, cmap='seismic'
                        )

                    elif cond == 'FR_CV_1' and odor != 'o':
                        topoplot_data = xr_estimate.loc[sujet_group, band, phase, :, f'odor{odor}'].values
                        mask_signi = xr_signi.loc[sujet_group, band, phase, :, f'odor{odor}'].values < plevel

                        im, _ = mne.viz.plot_topomap(
                            data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            vlim=(-vlim_estimate[sujet_group], vlim_estimate[sujet_group]), mask=mask_signi,
                            mask_params=mask_params, cmap='seismic'
                        )

                    else:
                        topoplot_data = xr_estimate.loc[sujet_group, band, phase, :, f'cond{cond}:odor{odor}'].values
                        mask_signi = xr_signi.loc[sujet_group, band, phase, :, f'cond{cond}:odor{odor}'].values < plevel

                        im, _ = mne.viz.plot_topomap(
                            data=topoplot_data, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            vlim=(-vlim_estimate[sujet_group], vlim_estimate[sujet_group]), mask=mask_signi,
                            mask_params=mask_params, cmap='seismic'
                        )

            os.chdir(os.path.join(path_results, 'allplot', 'TF', 'Pxx', 'topoplot_lmm', 'only_CO2'))
            fig.savefig(f'{sujet_group}_{band}_topo_lmm_Pxx.jpeg', dpi=150)
            plt.close('all')




def plot_save_Pxx_HIST_allsujet_lmm():

    #### create montage
    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')
    phase_list_lmm = ['inspi', 'expi']

    short_list_chan_eeg = ['F3', 'F4', 'Fz', 'Cz', 'C4', 'C3', 'P4', 'Pz', 'P3']

    pval_thresh = 0.05
    mask_params = dict(markersize=10, markerfacecolor='y')

    sujet_group_list = ['allsujet', 'rep', 'norep']
    term_list = ['(Intercept)', 'condCO2', 'condFR_CV_2', 'condMECA', 'odor-', 'odor+', 'condCO2:odor-', 'condFR_CV_2:odor-', 'condMECA:odor-', 'condCO2:odor+',
            'condFR_CV_2:odor+', 'condMECA:odor+']

    xr_Pxx, xr_estimate, xr_signi, xr_estimate_repnorep, xr_signi_repnorep, xr_estimate_region, xr_signi_region = get_df_TF_allsujet()
    
    df_estimate_lmm = pd.DataFrame()

    for sujet_group in sujet_group_list:

        for band_i, band in enumerate(freq_band_dict_lmm):

            for phase_i, phase in enumerate(phase_list_lmm):

                for chan_i, chan in enumerate(chan_list_eeg):

                    for cond_i, cond in enumerate(conditions):

                        for odor_i, odor in enumerate(odor_list):

                            if cond == 'FR_CV_1' and odor == 'o':
                                val = xr_estimate.loc[sujet_group, band, phase, chan, f'(Intercept)'].values
                                val_signi = False

                            elif cond != 'FR_CV_1' and odor == 'o':
                                val = xr_estimate.loc[sujet_group, band, phase, chan, f'cond{cond}'].values
                                val_signi = xr_signi.loc[sujet_group, band, phase, chan, f'cond{cond}'].values < pval_thresh

                            elif cond == 'FR_CV_1' and odor != 'o':
                                val = xr_estimate.loc[sujet_group, band, phase, chan, f'odor{odor}'].values
                                val_signi = xr_signi.loc[sujet_group, band, phase, chan, f'odor{odor}'].values < pval_thresh

                            else:    
                                val = xr_estimate.loc[sujet_group, band, phase, chan, f'cond{cond}:odor{odor}'].values
                                val_signi = xr_signi.loc[sujet_group, band, phase, chan, f'cond{cond}:odor{odor}'].values < pval_thresh

                            _df = pd.DataFrame({'sujet_group': [sujet_group], 'band': [band], 'phase': [phase], 'chan': [chan], 'cond': [cond], 'odor': [odor], 'estimate': [val], 'significant': [val_signi]})
                            df_estimate_lmm = pd.concat([df_estimate_lmm, _df], ignore_index=True)

    df_estimate_lmm['estimate'] = df_estimate_lmm['estimate'].astype(float)

    # df_estimate_lmm.query(f"sujet_group == 'rep' and band == 'theta' and cond == 'CO2' and odor in ['o', '+'] and chan in {short_list_chan_eeg}")

    #### plot allchan CO2
    os.chdir(os.path.join(path_results, 'allplot', 'TF', 'Pxx', 'histplot'))

    for band_i, band in enumerate(freq_band_dict_lmm):

        df_plot = df_estimate_lmm.query(f"sujet_group in ['rep', 'norep'] and band == '{band}' and cond == 'CO2' and odor in ['o', '+'] and chan in {short_list_chan_eeg}")

        phase_list_plot = ['inspi', 'expi']
        odor_list_plot = ['o', '+']

        custom_palette = {
                'rep': '#1f77b4',
                'norep': '#ff7f0e',
            }
        
        ylim = (df_plot['estimate'].min() - 0.1 * np.abs(df_plot['estimate'].min()), df_plot['estimate'].max() + 0.1 * np.abs(df_plot['estimate'].max()))

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        for col, phase in enumerate(phase_list_plot):

            for row, odor in enumerate(odor_list_plot):

                ax = axs[row, col]
                sub_df = df_plot.query(f"phase == '{phase}' and odor == '{odor}'")

                sub_df_sort = pd.DataFrame()

                for sujet_group in ['rep', 'norep']:
                    for chan in short_list_chan_eeg:
                        _df = sub_df.query(f"sujet_group == '{sujet_group}' and chan == '{chan}'")
                        sub_df_sort = pd.concat([sub_df_sort, _df], ignore_index=True)

                sns.barplot(data=sub_df_sort, x="chan", y="estimate", hue="sujet_group", ax=ax, palette=custom_palette)

                ax.set_ylim(ylim)

                patches = ax.patches

                for patch, (_, row) in zip(patches, sub_df_sort.iterrows()):
                    if row.get("significant", False):  # highlight only if significant is True
                        patch.set_edgecolor("black")
                        patch.set_linewidth(2)
                        patch.set_zorder(10)

                ax.set_title(f'Phase: {phase}, Odor: {odor}')
                ax.set_xlabel('Estimate')

        plt.suptitle(f"CO2_{band}")
        plt.tight_layout()


        fig.savefig(f'CO2_allchan_{band}_HIST_lmm_Pxx.jpeg', dpi=150)
        plt.close('all')

    #### ULTRA SHORT plot allchan CO2
    os.chdir(os.path.join(path_results, 'allplot', 'TF', 'Pxx', 'histplot'))

    for band_i, band in enumerate(freq_band_dict_lmm):

        ultra_short_list_chan_eeg = ['Fz', 'Cz', 'Pz']

        df_plot = df_estimate_lmm.query(f"sujet_group in ['rep', 'norep'] and band == '{band}' and cond == 'CO2' and odor in ['o', '+'] and chan in {ultra_short_list_chan_eeg}")

        phase_list_plot = ['inspi', 'expi']
        odor_list_plot = ['o', '+']

        custom_palette = {
                'rep': '#1f77b4',
                'norep': '#ff7f0e',
            }
        
        ylim = (df_plot['estimate'].min() - 0.1 * np.abs(df_plot['estimate'].min()), df_plot['estimate'].max() + 0.1 * np.abs(df_plot['estimate'].max()))

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        for col, phase in enumerate(phase_list_plot):

            for row, odor in enumerate(odor_list_plot):

                ax = axs[row, col]
                sub_df = df_plot.query(f"phase == '{phase}' and odor == '{odor}'")

                sub_df_sort = pd.DataFrame()

                for sujet_group in ['rep', 'norep']:
                    for chan in ultra_short_list_chan_eeg:
                        _df = sub_df.query(f"sujet_group == '{sujet_group}' and chan == '{chan}'")
                        sub_df_sort = pd.concat([sub_df_sort, _df], ignore_index=True)

                sns.barplot(data=sub_df_sort, x="chan", y="estimate", hue="sujet_group", ax=ax, palette=custom_palette)

                ax.set_ylim(ylim)

                patches = ax.patches

                for patch, (_, row) in zip(patches, sub_df_sort.iterrows()):
                    if row.get("significant", False):  # highlight only if significant is True
                        patch.set_edgecolor("black")
                        patch.set_linewidth(2)
                        patch.set_zorder(10)

                ax.set_title(f'Phase: {phase}, Odor: {odor}')
                ax.set_xlabel('Estimate')

        plt.suptitle(f"CO2_{band}")
        plt.tight_layout()


        fig.savefig(f'SHORT_CO2_allchan_{band}_HIST_lmm_Pxx.jpeg', dpi=150)
        plt.close('all')

    

    #### plot region lmm
    os.chdir(os.path.join(path_results, 'allplot', 'TF', 'Pxx', 'histplot'))

    def barplot_with_highlight(data, x, y, hue, palette, **kwargs):
        ax = plt.gca()
        sns.barplot(data=data, x=x, y=y, hue=hue, palette=palette, ax=ax, **kwargs)

        # Access drawn bars
        patches = ax.patches

        for patch, (_, row) in zip(patches, data.iterrows()):
            if row.get("significant", True):
                patch.set_edgecolor("tab:green")
                patch.set_linewidth(2)
                patch.set_zorder(10)




    for band_i, band in enumerate(freq_band_dict_lmm):

        df_plot = df_estimate_region_lmm.query(f"sujet_group in ['rep', 'norep'] and band == '{band}' and cond == 'CO2'")

        custom_palette = {
            'rep': '#1f77b4',
            'norep': '#ff7f0e',
        }

        g = sns.FacetGrid(df_plot, row='phase', col="region")
        g.map_dataframe(barplot_with_highlight, x="odor", y="estimate", hue="sujet_group", palette=custom_palette)
        g.map(plt.axhline, y=0, linestyle='--', color='gray', linewidth=1)

        plt.suptitle(f"{band}_CO2")
        plt.tight_layout()
        g.add_legend()

        #plt.show()
        
        g.savefig(f'CO2_{band}_HIST_lmm_Pxx.jpeg', dpi=150)
        plt.close('all')


    







########################################
######## PLOT & SAVE TF & ITPC ########
########################################

#tf = tf_plot
def get_tf_stats_no_cluster_thresh(tf, pixel_based_distrib):

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

    return tf_thresh


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





def compilation_compute_TF():

    #chan_i, chan = 0, chan_list_eeg[0]
    for chan_i, chan in enumerate(chan_list_eeg):

        print(f'PLOT {chan}', flush=True)
        
        #### load
        tf_conv = np.zeros((len(sujet_list), len(conditions), len(odor_list), nfrex, stretch_point_TF))

        for sujet_i, sujet in enumerate(sujet_list):

            os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'conv'))
            tf_conv[sujet_i] = np.load(f'{sujet}_tf_conv_allcond.npy')[:,:,chan_i]

        xr_dict = {'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'nfrex' : np.arange(nfrex), 'phase' : np.arange(stretch_point_TF)}
        xr_tf = xr.DataArray(data=tf_conv, dims=xr_dict.keys(), coords=xr_dict)

        os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'stats'))
        xr_intra = xr.open_dataarray(f'tf_STATS_chan{chan}_intra.nc')
        xr_inter = xr.open_dataarray(f'tf_STATS_chan{chan}_inter.nc')
        xr_repnorep = xr.open_dataarray(f'tf_STATS_chan{chan}_repnorep.nc')

        group_list = ['allsujet', 'rep', 'norep']

        #### scale
        vmin = {}
        vmax = {}

        for tf_stats_type in ['inter', 'intra']:

            vmin[tf_stats_type] = {}
            vmax[tf_stats_type] = {}

            for group_i, group in enumerate(group_list):

                vals = np.array([])
            
                for cond in conditions:

                    for odor in odor_list:

                        if group == 'allsujet':
                            if tf_stats_type == 'inter':
                                tf_baseline = xr_tf.loc[:, cond, 'o', :,:].values
                                tf_cond = xr_tf.loc[:, cond, odor, :,:].values
                            elif tf_stats_type == 'intra':
                                tf_baseline = xr_tf.loc[:, 'FR_CV_1', odor, :,:].values
                                tf_cond = xr_tf.loc[:, cond, odor, :,:].values
                        elif group == 'rep':
                            if tf_stats_type == 'inter':
                                tf_baseline = xr_tf.loc[sujet_best_list_rev, cond, 'o', :,:].values
                                tf_cond = xr_tf.loc[sujet_best_list_rev, cond, odor, :,:].values
                            elif tf_stats_type == 'intra':
                                tf_baseline = xr_tf.loc[sujet_best_list_rev, 'FR_CV_1', odor, :,:].values
                                tf_cond = xr_tf.loc[sujet_best_list_rev, cond, odor, :,:].values
                        elif group == 'norep':
                            if tf_stats_type == 'inter':
                                tf_baseline = xr_tf.loc[sujet_no_respond_rev, cond, 'o', :,:].values
                                tf_cond = xr_tf.loc[sujet_no_respond_rev, cond, odor, :,:].values
                            elif tf_stats_type == 'intra':
                                tf_baseline = xr_tf.loc[sujet_no_respond_rev, 'FR_CV_1', odor, :,:].values
                                tf_cond = xr_tf.loc[sujet_no_respond_rev, cond, odor, :,:].values

                        tf_median = np.median(tf_cond - tf_baseline, axis=0)

                        vals = np.append(vals, tf_median.reshape(-1))

                vmin[tf_stats_type][group], vmax[tf_stats_type][group] = vals.min(), vals.max()

                del vals

        #### inspect stats
        if debug:

            tf_stats_type = 'intra'
            n_chan, chan = 0, chan_list_eeg[0]
            r, odor = 0, odor_list[0]
            c, cond = 1, conditions[1] 
            _vmin, _vmax = vmin[n_chan], vmax[n_chan]

            tf_plot = xr_tf.loc['allsujet', cond, odor, :, :].values
            time = xr_tf['times'].values

            os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))
            pixel_based_distrib = np.load(f'allsujet_tf_STATS_nchan{chan}_{cond}_{odor}_intra.npy')[n_chan]

            fig, ax = plt.subplots()
            ax.pcolormesh(time, frex, tf_plot, vmin=_vmin, vmax=_vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
            ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=frex[0], ymax=frex[-1], colors='g')
            ax.contour(time, frex, get_tf_stats(tf_plot, pixel_based_distrib), levels=0, colors='g')
            ax.set_yscale('log')
            ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])
            plt.show()

            plt.pcolormesh(time, frex, tf_plot, vmin=_vmin, vmax=_vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
            plt.yscale('log')
            plt.contour(time, frex, get_tf_stats(tf_plot, pixel_based_distrib), levels=0, colors='g')
            plt.show()


            #wavelet_i = 0
            for wavelet_i in range(tf_plot.shape[0]):

                plt.plot(tf_plot[wavelet_i,:], color='b')
                plt.hlines([pixel_based_distrib[wavelet_i,0], pixel_based_distrib[wavelet_i,1]], xmin=0, xmax=tf_plot.shape[-1] ,color='r')

                plt.title(f'{np.round(wavelet_i/tf_plot.shape[0],2)}')

                plt.show()


        #### plot
        os.chdir(os.path.join(path_results, 'allplot', 'TF'))

        #tf_stats_type = 'inter'
        for tf_stats_type in ['inter', 'intra']:

            #group = group_list[2]
            for group in group_list:

                print(f"{tf_stats_type} {group}", flush=True)

                if tf_stats_type == 'inter':
                    cond_sel = conditions
                    odor_sel = ['+', '-']
                elif tf_stats_type == 'intra':
                    cond_sel = ['MECA', 'CO2', 'FR_CV_2']
                    odor_sel = odor_list

                #### plot
                fig, axs = plt.subplots(nrows=len(odor_sel), ncols=len(cond_sel))

                plt.suptitle(f'{group}_{chan}')

                fig.set_figheight(10)
                fig.set_figwidth(15)

                time = range(stretch_point_TF)

                #r, odor_i = 0, odor_list[0]
                for r, odor in enumerate(odor_sel):

                    #c, cond = 1, conditions[1]
                    for c, cond in enumerate(cond_sel):

                        if group == 'allsujet':
                            if tf_stats_type == 'inter':
                                tf_baseline = xr_tf.loc[:, cond, 'o', :,:].values
                                tf_cond = xr_tf.loc[:, cond, odor, :,:].values
                            elif tf_stats_type == 'intra':
                                tf_baseline = xr_tf.loc[:, 'FR_CV_1', odor, :,:].values
                                tf_cond = xr_tf.loc[:, cond, odor, :,:].values
                        elif group == 'rep':
                            if tf_stats_type == 'inter':
                                tf_baseline = xr_tf.loc[sujet_best_list_rev, cond, 'o', :,:].values
                                tf_cond = xr_tf.loc[sujet_best_list_rev, cond, odor, :,:].values
                            elif tf_stats_type == 'intra':
                                tf_baseline = xr_tf.loc[sujet_best_list_rev, 'FR_CV_1', odor, :,:].values
                                tf_cond = xr_tf.loc[sujet_best_list_rev, cond, odor, :,:].values
                        elif group == 'norep':
                            if tf_stats_type == 'inter':
                                tf_baseline = xr_tf.loc[sujet_no_respond_rev, cond, 'o', :,:].values
                                tf_cond = xr_tf.loc[sujet_no_respond_rev, cond, odor, :,:].values
                            elif tf_stats_type == 'intra':
                                tf_baseline = xr_tf.loc[sujet_no_respond_rev, 'FR_CV_1', odor, :,:].values
                                tf_cond = xr_tf.loc[sujet_no_respond_rev, cond, odor, :,:].values

                        tf_plot = np.median(tf_cond - tf_baseline, axis=0)
                    
                        ax = axs[r,c]

                        if r == 0 :
                            ax.set_title(cond, fontweight='bold', rotation=0)

                        if c == 0:
                            ax.set_ylabel(odor)

                        ax.pcolormesh(time, frex, tf_plot, vmin=vmin[tf_stats_type][group], vmax=vmax[tf_stats_type][group], shading='gouraud', cmap=plt.get_cmap('seismic'))
                        ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=frex[0], ymax=frex[-1], colors='g')
                        ax.set_yscale('log')

                        if tf_stats_type == 'intra':
                            
                            ax.contour(time, frex, xr_intra.loc[group, cond, odor, :, :].values, levels=0, colors='g')

                        if tf_stats_type == 'inter':

                            ax.contour(time, frex, xr_inter.loc[group, cond, odor, :, :].values, levels=0, colors='g')

                        ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])

                #plt.show()

                #### save
                os.chdir(os.path.join(path_results, 'allplot', 'TF'))
                fig.savefig(f'{group}_{chan}_{tf_stats_type}.jpeg', dpi=150)

                os.chdir(os.path.join(path_results, 'allplot', 'TF', tf_stats_type))
                fig.savefig(f'{group}_{chan}_{tf_stats_type}.jpeg', dpi=150)
                    
                fig.clf()
                plt.close('all')
                gc.collect()

        #### scale repnorep
        vals = np.array([])
    
        for cond in conditions:

            for odor in odor_list:

                tf_cond = xr_tf.loc[sujet_best_list_rev, cond, odor, :,:].median('sujet').values
                tf_baseline = xr_tf.loc[sujet_no_respond_rev, cond, odor, :,:].median('sujet').values

                tf_med = tf_cond - tf_baseline

                vals = np.append(vals, tf_med.reshape(-1))

        vmin, vmax = vals.min(), vals.max()

        del vals

        #### plot repnorep
        tf_stats_type = 'repnorep'

        print(f"{tf_stats_type}", flush=True)

        cond_sel = conditions
        odor_sel = odor_list

        #### plot
        fig, axs = plt.subplots(nrows=len(odor_sel), ncols=len(cond_sel))

        plt.suptitle(f'{chan}')

        fig.set_figheight(10)
        fig.set_figwidth(15)

        time = range(stretch_point_TF)

        #r, odor_i = 0, odor_list[0]
        for r, odor in enumerate(odor_sel):

            #c, cond = 1, conditions[1]
            for c, cond in enumerate(cond_sel):

                tf_baseline = xr_tf.loc[sujet_no_respond_rev, cond, odor, :,:].median('sujet').values
                tf_cond = xr_tf.loc[sujet_best_list_rev, cond, odor, :,:].median('sujet').values

                tf_plot = tf_cond - tf_baseline
            
                ax = axs[r,c]

                if r == 0 :
                    ax.set_title(cond, fontweight='bold', rotation=0)

                if c == 0:
                    ax.set_ylabel(odor)

                ax.pcolormesh(time, frex, tf_plot, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=frex[0], ymax=frex[-1], colors='g')
                ax.set_yscale('log')
                    
                ax.contour(time, frex, xr_repnorep.loc[cond, odor, :, :].values, levels=0, colors='g')

                ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])

        #plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'TF'))
        fig.savefig(f'{group}_{chan}_{tf_stats_type}.jpeg', dpi=150)

        os.chdir(os.path.join(path_results, 'allplot', 'TF', tf_stats_type))
        fig.savefig(f'{group}_{chan}_{tf_stats_type}.jpeg', dpi=150)
            
        fig.clf()
        plt.close('all')
        gc.collect()




def plot_TF_median():

    #chan_i, chan = 0, chan_list_eeg[0]
    for chan_i, chan in enumerate(chan_list_eeg):

        print(f'PLOT {chan}', flush=True)
        
        #### load
        tf_conv = np.zeros((len(sujet_list), len(conditions), len(odor_list), nfrex, stretch_point_TF))

        for sujet_i, sujet in enumerate(sujet_list):

            os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'conv'))
            tf_conv[sujet_i] = np.load(f'{sujet}_tf_conv_allcond.npy')[:,:,chan_i]

        xr_dict = {'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'nfrex' : np.arange(nfrex), 'phase' : np.arange(stretch_point_TF)}
        xr_tf = xr.DataArray(data=tf_conv, dims=xr_dict.keys(), coords=xr_dict)

        group_list = ['allsujet', 'rep', 'norep']

        #### scale
        vmin = {}
        vmax = {}

        for group_i, group in enumerate(group_list):

            vals = np.array([])
        
            for cond in conditions:

                for odor in odor_list:

                    if group == 'allsujet':
                        tf_cond = xr_tf.loc[:, cond, odor, :,:].values
                    elif group == 'rep':
                        tf_cond = xr_tf.loc[sujet_best_list_rev, cond, odor, :,:].values
                    elif group == 'norep':
                        tf_cond = xr_tf.loc[sujet_no_respond_rev, cond, odor, :,:].values

                    tf_median = np.median(tf_cond, axis=0)

                    vals = np.append(vals, tf_median.reshape(-1))

            vmin[group], vmax[group] = vals.min(), vals.max()

            del vals

        #### plot
        os.chdir(os.path.join(path_results, 'allplot', 'TF'))

        #group = group_list[2]
        for group in group_list:

            print(f"{group}", flush=True)

            #### plot
            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

            plt.suptitle(f'{group}_{chan}')

            fig.set_figheight(10)
            fig.set_figwidth(15)

            time = range(stretch_point_TF)

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #c, cond = 1, conditions[1]
                for c, cond in enumerate(conditions):

                    if group == 'allsujet':
                        tf_cond = xr_tf.loc[:, cond, odor, :,:].values
                    elif group == 'rep':
                        tf_cond = xr_tf.loc[sujet_best_list_rev, cond, odor, :,:].values
                    elif group == 'norep':
                        tf_cond = xr_tf.loc[sujet_no_respond_rev, cond, odor, :,:].values

                    tf_plot = np.median(tf_cond, axis=0)
                
                    ax = axs[r,c]

                    if r == 0 :
                        ax.set_title(cond, fontweight='bold', rotation=0)

                    if c == 0:
                        ax.set_ylabel(odor)

                    ax.pcolormesh(time, frex, tf_plot, vmin=vmin[group], vmax=vmax[group], shading='gouraud', cmap=plt.get_cmap('seismic'))
                    ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=frex[0], ymax=frex[-1], colors='g')
                    ax.set_yscale('log')

                    ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])

            #plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'allplot', 'TF', 'TF_median'))
            fig.savefig(f'{group}_{chan}.jpeg', dpi=150)
                
            fig.clf()
            plt.close('all')
            gc.collect()

        



########################################
######## COMPILATION FUNCTION ########
########################################

def compilation_compute_Pxx_Cxy_Cyclefreq_MVL():

    xr_allsujet, xr_allsujet_stats = compute_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allsujet()

    print('######## PLOT & SAVE TOPOPLOT ########', flush=True)
    plot_save_PSD_Cxy_CF_MVL_TOPOPLOT(xr_allsujet, xr_allsujet_stats)

    print('done', flush=True)









################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #### Pxx Cxy CycleFreq
    # compilation_compute_Pxx_Cxy_Cyclefreq_MVL()
    # execute_function_in_slurm_bash_mem_choice('n19_res_allsujet_power', 'compilation_compute_Pxx_Cxy_Cyclefreq_MVL', [nchan, nchan_name, band_prep], 15)

    #### Cxy TOPOPLOT
    # plot_save_Cxy_TOPOPLOT_allsujet()
    plot_save_Cxy_TOPOPLOT_allsujet_perm()
    plot_save_Cxy_TOPOPLOT_allsujet_lmm()

    #### TF & ITPC
    plot_save_Pxx_TOPOPLOT_allsujet_lmm()
    plot_save_Pxx_HIST_allsujet_lmm()
    plot_TF_median()
    compilation_compute_TF()
    # execute_function_in_slurm_bash_mem_choice('n19_res_allsujet_power', 'compilation_compute_TF', [nchan, nchan_name, band_prep], 15)



