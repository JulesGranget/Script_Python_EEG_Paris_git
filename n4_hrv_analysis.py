

import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
import mne
import scipy.signal
from bycycle.cyclepoints import find_extrema
import respirationtools

from n0_config import *

debug = False



############################
######## LOAD ECG ########
############################


os.chdir(os.path.join(path_prep, sujet, 'sections'))
dirlist_subject = os.listdir()

cond_keep = []
for cond in conditions_allsubjects:

    for file in dirlist_subject:

        if file.find(cond) != -1 : 
            cond_keep.append(cond)
            break

conditions = cond_keep

ecg_allcond = {}
ecg_stim_allcond = {}

for session_EEG in range(3):

    ecg_allcond_session = {}
    ecg_stim_allcond_session = {}

    for cond in conditions:

        load_i = []
        for session_i, session_name in enumerate(os.listdir()):
            if np.sum([(session_name.find(cond) > 0), (session_name.find(f's{session_EEG+1}') > 0), (session_name.find('lf') != -1 or session_name.find('wb') != -1)]) == 3:                    
                load_i.append(session_i)
            else:
                continue

        load_list = [os.listdir()[i] for i in load_i]

        data_ecg = []
        data_ecg_stim = []
        for load_name in load_list:
            raw = mne.io.read_raw_fif(load_name, preload=True)
            srate = int(raw.info['sfreq'])
            chan_list = raw.info['ch_names']
            ecg_i = chan_list.index('ECG')
            stim_i = chan_list.index('ECG_cR')
            ecg = raw.get_data()[ecg_i,:]
            ecg_stim = raw.get_data()[stim_i,:]

            if sujet_ecg_adjust.get(sujet) == 'inverse':
                ecg = ecg*-1
            
            data_ecg.append(ecg)
            data_ecg_stim.append(ecg_stim)

        ecg_allcond_session[cond] = data_ecg
        ecg_stim_allcond_session[cond] = data_ecg_stim

    ecg_allcond[f's{session_EEG+1}'] = ecg_allcond_session
    ecg_stim_allcond[f's{session_EEG+1}'] = ecg_stim_allcond_session





########################################
######## HRV ANALYSIS HOMEMADE ########
########################################

#### RRI, IFR

#cond = 'RD_CV'
#session_i = 0
#ecg, cR_stim, condition, srate, srate_resample = ecg_allcond[cond][session_i], ecg_stim_allcond[cond][session_i], cond, srate, srate_resample_hrv
def get_RRI_IFR(ecg, cR_stim, condition, srate, srate_resample) :

    cR = np.where(cR_stim == 10)[0]
    cR_sec = cR/srate # cR in sec
    times = np.arange(0,len(ecg))/srate # in sec

    # RRI computation
    RRI = np.diff(cR_sec)
    RRI = np.insert(RRI, 0, np.median(RRI))
    IFR = (1/RRI)


    # interpolate
    f = scipy.interpolate.interp1d(cR_sec, RRI, kind='quadratic')
    cR_sec_resample = np.arange(cR_sec[0], cR_sec[-1], 1/srate_resample)
    RRI_resample = f(cR_sec_resample)

    #plt.plot(cR_sec, RRI, label='old')
    #plt.plot(cR_sec_resample, RRI_resample, label='new')
    #plt.legend()
    #plt.show()


    # figure
    fig, ax = plt.subplots()
    fig.suptitle(condition, fontsize=16) 
    ax = plt.subplot(411)
    plt.plot(times, ecg)
    plt.title('ECG')
    plt.ylabel('a.u.')
    plt.xlabel('s')
    plt.vlines(cR_sec, ymin=min(ecg), ymax=max(ecg), colors='k')
    plt.subplot(412, sharex=ax)
    plt.plot(cR_sec, RRI)
    plt.title('RRI')
    plt.ylabel('s')
    plt.subplot(413, sharex=ax)
    plt.plot(cR_sec_resample, RRI_resample)
    plt.title('RRI_resampled')
    plt.ylabel('Hz')
    plt.subplot(414, sharex=ax)
    plt.plot(cR_sec, IFR)
    plt.title('IFR')
    plt.ylabel('Hz')
    #plt.show()

    # in this plot one RRI point correspond to the difference value between the precedent RR
    # the first point of RRI is the median for plotting consideration

    return RRI, RRI_resample, IFR, fig
    


#### LF / HF

#RRI_resample, srate, nwind, nfft, noverlap, win, condition = result_struct[keys_result[0]][1], srate_resample, nwind, nfft, noverlap, win, cond
def get_PSD_LF_HF(RRI_resample, srate_resample, nwind, nfft, noverlap, win, condition) :

    # DETREND
    RRI_detrend = RRI_resample-np.median(RRI_resample)

    # FFT WELCH
    hzPxx, Pxx = scipy.signal.welch(RRI_detrend, fs=srate_resample, window=win, nperseg=nwind, noverlap=noverlap, nfft=nfft)

    # PLOT
    VLF, LF, HF = .04, .14, .4
    fig = plt.figure()
    plt.plot(hzPxx,Pxx)
    plt.ylim(0, np.max(Pxx[hzPxx>0.01]))
    plt.xlim([0,.6])
    plt.vlines([VLF, LF, HF], ymin=min(Pxx), ymax=max(Pxx))
    plt.title('PSD HRV : '+condition)
    #plt.show()
    
    AUC_LF = np.trapz(Pxx[(hzPxx>VLF) & (hzPxx<LF)])
    AUC_HF = np.trapz(Pxx[(hzPxx>LF) & (hzPxx<HF)])
    LF_HF_ratio = AUC_LF/AUC_HF

    return AUC_LF, AUC_HF, LF_HF_ratio, fig, hzPxx, Pxx



#### SDNN, RMSSD, NN50, pNN50
# RR_val = RRI
def get_stats_descriptors(RR_val) :
    SDNN = np.std(RR_val)

    RMSSD = np.sqrt(np.mean((np.diff(RR_val)*1e3)**2))

    NN50 = []
    for RR in range(len(RR_val)) :
        if RR == len(RR_val)-1 :
            continue
        else :
            NN = abs(RR_val[RR+1] - RR_val[RR])
            NN50.append(NN)

    NN50 = np.array(NN50)*1e3
    pNN50 = np.sum(NN50>50)/len(NN50)

    return SDNN, RMSSD, NN50, pNN50

#SDNN_CV, RMSSD_CV, NN50_CV, pNN50_CV = get_stats_descriptors(RRI_CV)


#### Poincarré

def get_poincarre(RRI, condition):
    RRI_1 = RRI[1:]
    RRI_1 = np.append(RRI_1, RRI[-1]) 

    fig = plt.figure()
    plt.scatter(RRI, RRI_1)
    plt.xlabel('RR (ms)')
    plt.ylabel('RR+1 (ms)')
    plt.title('Poincarré '+condition)
    plt.xlim(.600,1.)
    plt.ylim(.600,1.)

    SD1_val = []
    SD2_val = []
    for RR in range(len(RRI)) :
        if RR == len(RRI)-1 :
            continue
        else :
            SD1_val_tmp = (RRI[RR+1] - RRI[RR])/np.sqrt(2)
            SD2_val_tmp = (RRI[RR+1] + RRI[RR])/np.sqrt(2)
            SD1_val.append(SD1_val_tmp)
            SD2_val.append(SD2_val_tmp)

    SD1 = np.std(SD1_val)
    SD2 = np.std(SD2_val)
    Tot_HRV = SD1*SD2*np.pi

    return SD1, SD2, Tot_HRV, fig

    
#### DeltaHR

#RRI, srate_resample, f_RRI, condition = result_struct[keys_result[0]][1], srate_resample, f_RRI, cond 
def get_dHR(RRI_resample, srate_resample, f_RRI, condition):
    
    times = np.arange(0,len(RRI_resample))/srate_resample

        # stairs method
    #RRI_stairs = np.array([])
    #len_cR = len(cR) 
    #for RR in range(len(cR)) :
    #    if RR == 0 :
    #        RRI_i = cR[RR+1]/srate - cR[RR]/srate
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(cR[RR+1]))])
    #    elif RR != 0 and RR != len_cR-1 :
    #        RRI_i = cR[RR+1]/srate - cR[RR]/srate
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(cR[RR+1] - cR[RR]))])
    #    elif RR == len_cR-1 :
    #        RRI_stairs = np.append(RRI_stairs, [RRI_i*1e3 for i in range(int(len(ecg) - cR[RR]))])


    peaks, troughs = find_extrema(RRI_resample, srate_resample, f_RRI)
    peaks_RRI, troughs_RRI = RRI_resample[peaks], RRI_resample[troughs]
    peaks_troughs = np.stack((peaks_RRI, troughs_RRI), axis=1)

    fig_verif = plt.figure()
    plt.plot(times, RRI_resample)
    plt.vlines(peaks/srate_resample, ymin=min(RRI_resample), ymax=max(RRI_resample), colors='b')
    plt.vlines(troughs/srate_resample, ymin=min(RRI_resample), ymax=max(RRI_resample), colors='r')
    #plt.show()

    dHR = np.diff(peaks_troughs/srate_resample, axis=1)*1e3

    fig_dHR = plt.figure()
    plt.suptitle(condition)
    ax = plt.subplot(211)
    plt.plot(times, RRI_resample*1e3)
    plt.title('RRI')
    plt.ylabel('ms')
    plt.subplot(212, sharex=ax)
    plt.plot(troughs/srate_resample, dHR)
    plt.hlines(np.median(dHR), xmin=min(times), xmax=max(times), colors='m', label='median = {:.3f}'.format(np.median(dHR)))
    plt.legend()
    plt.title('dHR')
    plt.ylabel('ms')
    plt.vlines(peaks/srate_resample, ymin=0, ymax=0.01, colors='b')
    plt.vlines(troughs/srate_resample, ymin=0, ymax=0.01, colors='r')
    plt.tight_layout()
    #plt.show()


    return fig_verif, fig_dHR

#session_eeg, session_i, cond, odor = 's1', 0, 'RD_CV', '+'
def ecg_analysis_homemade(session_eeg, session_i, cond, odor):

    
    res_list = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_SD1', 'HRV_SD2']

    #### RRI
    RRI, RRI_resample, IFR, fig_RRI = get_RRI_IFR(ecg_allcond[session_eeg][cond][session_i], ecg_stim_allcond[session_eeg][cond][session_i], cond, srate, srate_resample_hrv)

    HRV_MeanNN = np.mean(RRI)
    
    #### PSD
    AUC_LF, AUC_HF, LF_HF_ratio, fig_PSD, hzPxx, Pxx = get_PSD_LF_HF(RRI_resample, srate_resample_hrv, nwind_hrv, nfft_hrv, noverlap_hrv, win_hrv, cond)

    #### descriptors
    SDNN, RMSSD, NN50, pNN50 = get_stats_descriptors(RRI)

    #### poincarré
    SD1, SD2, Tot_HRV, fig_poincarre = get_poincarre(RRI, cond)

    #### dHR
    fig_verif, fig_dHR = get_dHR(RRI_resample, srate_resample_hrv, f_RRI, cond)

    #### fig
    fig_list = [fig_RRI, fig_PSD, fig_poincarre, fig_verif, fig_dHR]

    #### df
    res_tmp = [HRV_MeanNN*1e3, SDNN*1e3, RMSSD, pNN50*100, AUC_LF, AUC_HF, SD1*1e3, SD2*1e3]
    data_df = {}
    for i, dv in enumerate(res_list):
        data_df[dv] = [res_tmp[i]]

    hrv_metrics_homemade = pd.DataFrame(data=data_df)
    hrv_metrics_homemade.insert(0,'Subject',[sujet_list.index(sujet)])
    hrv_metrics_homemade.insert(1,'Name',[sujet])
    hrv_metrics_homemade.insert(2,'RDorFR',[cond[:2]])
    hrv_metrics_homemade.insert(3,'Condition',[cond])
    hrv_metrics_homemade.insert(4,'Odor',odor)


    return hrv_metrics_homemade, fig_list





########################################
######## ECG ANALYSIS NK ########
########################################

#session_eeg, session_i, cond, odor = 's1', 0, 'RD_CV', '+'
def ecg_analysis(session_eeg, session_i, cond, odor):

    #### load cR
    times = np.arange(len(ecg_allcond[session_eeg][cond][session_i]))
    ecg_stim = ecg_stim_allcond[session_eeg][cond][session_i]
    ecg_stim = np.where(ecg_stim == 10, 1, ecg_stim) 
    peaks_dict = {'ECG_R_Peaks' : ecg_stim.astype(int)}
    ecg_peaks = pd.DataFrame(peaks_dict)

    #### verif cR
    ecg = ecg_allcond[session_eeg][cond][session_i]
    ecg_cR = np.where(ecg_stim == 1)[0]
    fig_verif = plt.figure(figsize=(60,40))
    plt.plot(ecg)
    plt.vlines(ecg_cR, ymax=np.max(ecg), ymin=np.min(ecg), colors='r')
    #fig_verif.show()

    #### compute metrics
    hrv_metrics = nk.hrv(ecg_peaks, sampling_rate=srate, show=False)

    total_hrv = np.array([hrv_metrics.iloc[0]['HRV_HF'] + hrv_metrics.iloc[0]['HRV_LF']])
    
    #### export 
    total_hrv = pd.DataFrame(data=total_hrv, index=None, columns=['HRV_Total'])

    hrv_metrics = pd.concat([hrv_metrics,total_hrv],axis=1)
    hrv_metrics.insert(0,'Subject',[sujet_list.index(sujet)])
    hrv_metrics.insert(1,'Name',[sujet])
    hrv_metrics.insert(2,'RDorFR',[cond[:2]])
    hrv_metrics.insert(3,'Condition',[cond])
    hrv_metrics.insert(4,'Odor',odor)

    col_to_drop = []
    col_hrv = list(hrv_metrics.columns.values) 
    for metric_name in col_hrv :
        if metric_name == 'Subject' or metric_name == 'Name' or metric_name == 'RDorFR' or metric_name == 'Condition' or metric_name == 'Odor':
            continue
        elif (metric_name in hrv_metrics_short_name) == False :
            col_to_drop.append(metric_name)

    hrv_metrics_short = hrv_metrics.copy()
    hrv_metrics_short = hrv_metrics_short.drop(col_to_drop, axis=1)

    return hrv_metrics, hrv_metrics_short, fig_verif



if __name__ == '__main__':

    #### compute hrv NK
    hrv_metrics_allcond = pd.DataFrame()
    hrv_metrics_allcond_short = pd.DataFrame()
    fig_verif_list= []
    cond_name = []
    for session_eeg in range(3):

        odor = odor_order[sujet][session_eeg]

        for cond in conditions:

            if len(ecg_allcond[f's{session_eeg+1}'][cond]) == 1:
                hrv_metrics, hrv_metrics_short, fig_verif = ecg_analysis(f's{session_eeg+1}', 0, cond, odor)
                hrv_metrics_allcond = pd.concat([hrv_metrics_allcond, hrv_metrics])
                hrv_metrics_allcond_short = pd.concat([hrv_metrics_allcond_short, hrv_metrics_short])
                fig_verif_list.append(fig_verif)
                cond_name.append(f's{session_eeg+1}_' + cond + '_1')

            else:
                for session_i in range(len(ecg_allcond[f's{session_eeg+1}'][cond])):
                    hrv_metrics, hrv_metrics_short, fig_verif = ecg_analysis(f's{session_eeg+1}', session_i, cond, odor)
                    hrv_metrics_allcond = pd.concat([hrv_metrics_allcond, hrv_metrics])
                    hrv_metrics_allcond_short = pd.concat([hrv_metrics_allcond_short, hrv_metrics_short])
                    fig_verif_list.append(fig_verif)
                    cond_name.append(f's{session_eeg+1}_' + cond + '_' + str(session_i+1))

    #### compute hrv homemade
    hrv_metrics_allcond_homemade = pd.DataFrame()
    fig_allcond_list = []

    for session_eeg in range(3):

        odor = odor_order[sujet][session_eeg]

        for cond in conditions:
            if len(ecg_allcond[f's{session_eeg+1}'][cond]) == 1:
                hrv_metrics_homemade, fig_list = ecg_analysis_homemade(f's{session_eeg+1}', 0, cond, odor)
                hrv_metrics_allcond_homemade = pd.concat([hrv_metrics_allcond_homemade, hrv_metrics_homemade])
                fig_allcond_list.append(fig_list)

            else:
                for session_i in range(len(ecg_allcond[f's{session_eeg+1}'][cond])):
                    hrv_metrics_homemade, fig_list = ecg_analysis_homemade(f's{session_eeg+1}', session_i, cond, odor)
                    hrv_metrics_allcond_homemade = pd.concat([hrv_metrics_allcond_homemade, hrv_metrics_homemade])
                    fig_allcond_list.append(fig_list)



    #### save fig
    os.chdir(os.path.join(path_results, sujet, 'HRV'))
    for fig_i, fig in enumerate(fig_verif_list):
        fig.savefig(sujet + '_' + cond_name[fig_i] + '.jpeg')


    #### compute for mean cond
    hrv_metrics_allcond_mean_cond = pd.DataFrame()
    hrv_metrics_allcond_short_mean_cond = pd.DataFrame()
    hrv_metrics_allcond_homemade_mean_cond = pd.DataFrame()

    #odor = '-'
    for odor in odor_order[sujet]:
        
        #### NK long
        hrv_metrics_allcond_mean_cond_odor = hrv_metrics_allcond[hrv_metrics_allcond['Odor'] == odor].groupby('Condition').mean().reset_index()
        hrv_metrics_allcond_mean_cond_odor = hrv_metrics_allcond_mean_cond_odor.drop(columns=['Subject'])

        df_to_concat = hrv_metrics_allcond[hrv_metrics_allcond['Odor'] == odor][['Subject', 'Name', 'Odor']].iloc[:len(hrv_metrics_allcond_mean_cond_odor)].reset_index().drop(['index'], axis=1)

        hrv_metrics_allcond_mean_cond_odor = pd.concat([df_to_concat, hrv_metrics_allcond_mean_cond_odor], axis=1)
        hrv_metrics_allcond_mean_cond = pd.concat([hrv_metrics_allcond_mean_cond, hrv_metrics_allcond_mean_cond_odor], axis=0)

        #### NK short
        hrv_metrics_allcond_short_mean_cond_odor = hrv_metrics_allcond_short[hrv_metrics_allcond_short['Odor'] == odor].groupby('Condition').mean().reset_index()
        hrv_metrics_allcond_short_mean_cond_odor = hrv_metrics_allcond_short_mean_cond_odor.drop(columns=['Subject'])

        df_to_concat = hrv_metrics_allcond[hrv_metrics_allcond['Odor'] == odor][['Subject', 'Name', 'Odor']].iloc[:len(hrv_metrics_allcond_short_mean_cond_odor)].reset_index().drop(['index'], axis=1)

        hrv_metrics_allcond_short_mean_cond_odor = pd.concat([df_to_concat, hrv_metrics_allcond_short_mean_cond_odor], axis=1)
        hrv_metrics_allcond_short_mean_cond = pd.concat([hrv_metrics_allcond_short_mean_cond, hrv_metrics_allcond_short_mean_cond_odor], axis=0)

        #### Homemade short
        hrv_metrics_allcond_homemade_mean_cond_odor = hrv_metrics_allcond_homemade[hrv_metrics_allcond_homemade['Odor'] == odor].groupby('Condition').mean().reset_index()
        hrv_metrics_allcond_homemade_mean_cond_odor = hrv_metrics_allcond_homemade_mean_cond_odor.drop(columns=['Subject'])

        df_to_concat = hrv_metrics_allcond[hrv_metrics_allcond['Odor'] == odor][['Subject', 'Name', 'Odor']].iloc[:len(hrv_metrics_allcond_homemade_mean_cond_odor)].reset_index().drop(['index'], axis=1)
        
        hrv_metrics_allcond_homemade_mean_cond_odor = pd.concat([df_to_concat, hrv_metrics_allcond_homemade_mean_cond_odor], axis=1)
        hrv_metrics_allcond_homemade_mean_cond = pd.concat([hrv_metrics_allcond_homemade_mean_cond, hrv_metrics_allcond_homemade_mean_cond_odor], axis=0)

    hrv_metrics_allcond_mean_cond.reset_index()
    hrv_metrics_allcond_short_mean_cond.reset_index()
    hrv_metrics_allcond_homemade_mean_cond.reset_index()

    #### save hrv metrics
    os.chdir(os.path.join(path_results, sujet, 'HRV'))
    hrv_metrics_allcond.to_excel(sujet + '_HRV_Metrics.xlsx')
    hrv_metrics_allcond_short.to_excel(sujet + '_HRV_Metrics_Short.xlsx')
    hrv_metrics_allcond_mean_cond.to_excel(sujet + '_HRV_Metrics_mean_cond.xlsx')
    hrv_metrics_allcond_short_mean_cond.to_excel(sujet + '_HRV_Metrics_Short_mean_cond.xlsx')
    hrv_metrics_allcond_homemade.to_excel(sujet + '_HRV_Metrics_homemade.xlsx')
    hrv_metrics_allcond_homemade_mean_cond.to_excel(sujet + '_HRV_Metrics_homemade_mean_cond.xlsx')


    