
import numpy as np
import scipy.signal

################################
######## MODULES ########
################################

# anaconda (numpy, scipy, pandas, matplotlib, glob2, joblib, xlrd)
# neurokit2 as nk
# respirationtools
# mne
# neo
# bycycle
# pingouin

################################
######## GENERAL PARAMS ######## 
################################

enable_big_execute = False
perso_repo_computation = False

#sujet = 'DEBUG'

conditions = ['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']

sujet_list = np.array(['01PD','03VN','04GB','05LV','06EF','07PB','08DM','09TA','10BH','11FA','12BD','13FP',
'14MD','15LG','16GM','17JR','18SE','19TM','20TY','21ZV','22DI','23LF','24TJ','DF25','26MN','27BD','28NT','29SC',
'30AR','HJ31', '32CM','33MA'])

# 02MJ signal problems
# DF25 has no session 2
# HJ31 has no session 4

sujet_best_list = np.array(['BD12', 'CM32', 'DI22', 'FA11', 'GM16', 'HJ31', 'JR17', 'MA33',
       'MJ02', 'MN26', 'PD01', 'SC29', 'TA09', 'TJ24', 'TM19', 'VN03',
       'ZV21'])

sujet_list_hyperventilation = ['20TY']

band_prep_list = ['wb']

freq_band_dict = {'wb' : {'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120], 'whole' : [2,50]},
                'lf' : {'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40], 'whole' : [2,50]},
                'hf' : {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]} }

freq_band_list_precompute = {'wb' : {'theta_1' : [2,10], 'theta_2' : [4,8], 'alpha_1' : [8,12], 'alpha_2' : [8,14], 
                                    'beta_1' : [12,40], 'beta_2' : [10,40], 'whole_1' : [2,50], 'l_gamma_1' : [50, 80], 
                                    'h_gamma_1' : [80, 120]} }

freq_band_dict_FC = {'wb' : {'theta' : [4,8], 'alpha' : [8,12], 'beta' : [12,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]},
                'lf' : {'theta' : [4,8], 'alpha' : [8,12], 'beta' : [12,40]},
                'hf' : {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]} }

odor_list = ['o', '+', '-']

phase_list = ['whole', 'inspi', 'expi']

srate = 500

chan_list = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 
            'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 
            'PRESS', 'ECG', 'ECG_cR']
            
chan_list_eeg = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 
            'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2']

chan_list_eeg_fc = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8','Fz', 'FC1', 'FC2', 'FC5', 'FC6', 'FT9', 'FT10', 'Cz', 'C3', 'C4',
                    'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8', 'T7', 'T8', 'TP9', 'TP10', 'Oz', 'O1', 'O2']



################################
######## ODOR ORDER ########
################################

odor_order = {

'01PD' : {'ses02' : 'o', 'ses03' : '+', 'ses04' : '-'},   '02MJ' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   '03VN' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   
'04GB' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   '05LV' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   '06EF' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   
'07PB' : {'ses02' : 'o', 'ses03' : '+', 'ses04' : '-'},   '08DM' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},   '09TA' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   
'10BH' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   '11FA' : {'ses02' : 'o', 'ses03' : '+', 'ses04' : '-'},   '12BD' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   
'13FP' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   '14MD' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   '15LG' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},
'16GM' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   '17JR' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   '18SE' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   
'19TM' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   '20TY' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},   '21ZV' : {'ses02' : 'o', 'ses03' : '+', 'ses04' : '-'},   
'22DI' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   '23LF' : {'ses02' : '+', 'ses03' : '-', 'ses04' : 'o'},   '24TJ' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   
'25DF' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   '26MN' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   '27BD' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   
'28NT' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},   '29SC' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   '30AR' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},
'31HJ' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   '32CM' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   '33MA' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'}
}




########################################
######## PATH DEFINITION ########
########################################

import socket
import os
import platform
 
PC_OS = platform.system()
PC_ID = socket.gethostname()

if PC_ID == 'LAPTOP-EI7OSP7K':

    PC_working = 'Jules_VPN'
    if perso_repo_computation:
        path_main_workdir = '/home/jules/Bureau/perso_repo_computation/Script_Python_EEG_Paris_git'
    else:    
        path_main_workdir = 'Z:\\multisite\\DATA_MANIP\\EEG_Paris_J\\Script_Python_EEG_Paris_git'
    path_general = 'Z:\\multisite\\DATA_MANIP\\EEG_Paris_J'
    path_memmap = 'Z:\\multisite\\DATA_MANIP\\EEG_Paris_J\\Mmap'
    n_core = 4


elif PC_ID == 'DESKTOP-3IJUK7R':

    PC_working = 'Jules_Labo_Win'
    if perso_repo_computation:
        path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    else:    
        path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_general = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_memmap = 'D:\\LPPR_CMO_PROJECT\\Lyon\\Mmap'
    n_core = 2

elif PC_ID == 'pc-jules' or PC_ID == 'LAPTOP-EI7OSP7K':

    PC_working = 'Jules_Labo_Linux'
    if perso_repo_computation:
        path_main_workdir = '/home/jules/Bureau/perso_repo_computation/Script_Python_EEG_Paris_git'
    else:    
        path_main_workdir = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J/Script_Python_EEG_Paris_git'
    path_general = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J'
    path_memmap = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J/Mmap'
    n_core = 4

elif PC_ID == 'pc-valentin':

    PC_working = 'Valentin_Labo_Linux'
    if perso_repo_computation:
        path_main_workdir = '/home/valentin/Bureau/perso_repo_computation/Script_Python_EEG_Paris_git'
    else:    
        path_main_workdir = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J/Script_Python_EEG_Paris_git'
    path_general = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J'
    path_memmap = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J/Mmap'
    n_core = 6

elif PC_ID == 'nodeGPU':

    PC_working = 'nodeGPU'
    path_main_workdir = '/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J/Script_Python_EEG_Paris_git'
    path_general = '/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J'
    path_memmap = '/mnt/data/julesgranget'
    n_core = 15

else:

    PC_working = 'crnl_cluster'
    path_main_workdir = '/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J/Script_Python_EEG_Paris_git'
    path_general = '/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J'
    path_memmap = '/mnt/data/julesgranget'
    n_core = 10
    

path_data = os.path.join(path_general, 'Data')
path_prep = os.path.join(path_general, 'Analyses', 'preprocessing')
path_precompute = os.path.join(path_general, 'Analyses', 'precompute') 
path_results = os.path.join(path_general, 'Analyses', 'results') 
path_respfeatures = os.path.join(path_general, 'Analyses', 'results') 
path_anatomy = os.path.join(path_general, 'Analyses', 'anatomy')
path_slurm = os.path.join(path_general, 'Script_slurm')

#### slurm params
mem_crnl_cluster = '10G'
n_core_slurms = 10







################################
######## RESPI PARAMS ########
################################ 

#### INSPI DOWN
sujet_respi_adjust = {
'01PD':'inverse',   '02MJ':'inverse',   '03VN':'inverse',   '04GB':'inverse',   '05LV':'inverse',
'06EF':'inverse',   '07PB':'inverse',   '08DM':'inverse',   '09TA':'inverse',   '10BH':'inverse',
'11FA':'inverse',   '12BD':'inverse',   '13FP':'inverse',   '14MD':'inverse',   '15LG':'inverse',
'16GM':'inverse',   '17JR':'inverse',   '18SE':'inverse',   '19TM':'inverse',   '20TY':'inverse',
'21ZV':'inverse',   '22DI':'inverse',   '23LF':'inverse',   '24TJ':'inverse',   '25DF':'inverse',
'26MN':'inverse',   '27BD':'inverse',   '28NT':'inverse',   '29SC':'inverse',   '30AR':'inverse',
'31HJ':'inverse',   '32CM':'inverse',   '33MA':'inverse'
}


cycle_detection_params = {
'exclusion_metrics' : 'med',
'metric_coeff_exclusion' : 3,
'inspi_coeff_exclusion' : 2,
'respi_scale' : [0.1, 0.35], #Hz
}



################################
######## ECG PARAMS ########
################################ 

sujet_ecg_adjust = {
'01PD':'inverse',   '02MJ':'inverse',   '03VN':'inverse',   '04GB':'inverse',   '05LV':'inverse',
'06EF':'inverse',   '07PB':'inverse',   '08DM':'inverse',   '09TA':'inverse',   '10BH':'inverse',
'11FA':'inverse',   '12BD':'inverse',   '13FP':'inverse',   '14MD':'inverse',   '15LG':'inverse',
'16GM':'inverse',   '17JR':'inverse',   '18SE':'inverse',   '19TM':'inverse',   '20TY':'inverse',
'21ZV':'inverse',   '22DI':'inverse',   '23LF':'inverse',   '24TJ':'inverse',   '25DF':'inverse',
'26MN':'inverse',   '27BD':'inverse',   '28NT':'inverse',   '29SC':'inverse',   '30AR':'inverse',
'31HJ':'inverse',   '32CM':'inverse',   '33MA':'inverse'
}


hrv_metrics_short_name = ['HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_SD1', 'HRV_SD2']




################################
######## PREP PARAMS ########
################################ 


prep_step_debug = {
'reref' : {'execute': True, 'params' : ['TP9']}, #chan = chan to reref
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': True},
'csd_computation' : {'execute': True},
}

prep_step_wb = {
'reref' : {'execute': True, 'params' : ['TP9', 'TP10']}, #chan = chan to reref
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'csd_computation' : {'execute': False},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': False},
}

prep_step_lf = {
'reref' : {'execute': False, 'params' : ['chan']}, #chan = chan to reref
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': False},
'csd_computation' : {'execute': True},
}

prep_step_hf = {
'reref_mastoide' : {'execute': False},
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : 55, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : None, 'h_freq': None}},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': False},
'csd_computation' : {'execute': True},
}



segment_to_clean = {

'04GB' : {'-' : ['O1', 'O2', 'Oz']}

}


sujet_extra_ICA = ['05LV']

################################
######## ERP PARAMS ########
################################



PPI_time_vec = [-3, 1] #seconds
PPI_lm_time = [-1.5, -.5]

# '13FP' signal bad

sujet_list_erp = np.array(['01PD','03VN','04GB','05LV','06EF','07PB','08DM','09TA','10BH','11FA','12BD',
'14MD','15LG','16GM','17JR','18SE','19TM','20TY','21ZV','22DI','23LF','24TJ','26MN','27BD','28NT','29SC',
'30AR','32CM','33MA'])



sujet_best_list_erp = np.array(['12BD', '32CM', '22DI', '11FA', '16GM', '17JR', '33MA',
       '26MN', '01PD', '29SC', '09TA', '24TJ', '19TM', '03VN',
       '21ZV'])


########################################
######## PARAMS SURROGATES ########
########################################

#### Pxx Cxy

zero_pad_coeff = 15

def get_params_spectral_analysis(srate):
    nwind = int( 20*srate ) # window length in seconds*srate
    nfft = nwind*zero_pad_coeff # if no zero padding nfft = nwind
    noverlap = np.round(nwind/2) # number of points of overlap here 50%
    hannw = scipy.signal.windows.hann(nwind) # hann window

    return nwind, nfft, noverlap, hannw

#### plot Pxx Cxy  
if zero_pad_coeff - 5 <= 0:
    remove_zero_pad = 0
remove_zero_pad = zero_pad_coeff - 5

#### stretch
stretch_point_surrogates = 500

#### coh
n_surrogates_coh = 500
freq_surrogates = [0, 2]
percentile_coh = .95

#### cycle freq
n_surrogates_cyclefreq = 500
percentile_cyclefreq_up = .99
percentile_cyclefreq_dw = .01






################################
######## PRECOMPUTE TF ########
################################


#### stretch
stretch_point_TF = 500
stretch_TF_auto = False
ratio_stretch_TF = 0.5

#### TF & ITPC
nfrex = 150
ncycle_list = [7, 41]
freq_list = [2, 150]
srate_dw = 10
wavetime = np.arange(-3,3,1/srate)
frex = np.logspace(np.log10(freq_list[0]), np.log10(freq_list[1]), nfrex) 
cycles = np.logspace(np.log10(ncycle_list[0]), np.log10(ncycle_list[1]), nfrex).astype('int')
Pxx_wavelet_norm = 1000


#### STATS
n_surrogates_tf = 500
tf_percentile_sel_stats_dw = 5 
tf_percentile_sel_stats_up = 95 
tf_stats_percentile_cluster = 95
norm_method = 'rscore'# 'zscore', 'dB'
exclude_frex_range = [48, 52]

#### plot
tf_plot_percentile_scale = 99 #for one side




################################
######## POWER ANALYSIS ########
################################

#### analysis
coh_computation_interval = .02 #Hz around respi


################################
######## FC ANALYSIS ########
################################

nfrex_fc = 50

#### band to remove
freq_band_fc_analysis = {'theta' : [4, 8], 'alpha' : [9,12], 'beta' : [15,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}

percentile_thresh = 90

#### for DFC
slwin_dict = {'theta' : 5, 'alpha' : 3, 'beta' : 1, 'l_gamma' : .3, 'h_gamma' : .3} # seconds
slwin_step_coeff = .1  # in %, 10% move

band_name_fc_dfc = ['theta', 'alpha', 'beta', 'l_gamma', 'h_gamma']

#### cond definition
cond_FC_DFC = ['FR_CV', 'AL', 'SNIFF', 'AC']

#### down sample for AL
dw_srate_fc_AL = 10

#### down sample for AC
dw_srate_fc_AC = 50

#### n points for AL interpolation
n_points_AL_interpolation = 10000
n_points_AL_chunk = 1000

#### for df computation
percentile_graph_metric = 25



################################
######## TOPOPLOT ########
################################

around_respi_Cxy = 0.025


################################
######## HRV ANALYSIS ########
################################



srate_resample_hrv = 10
nwind_hrv = int( 128*srate_resample_hrv )
nfft_hrv = nwind_hrv
noverlap_hrv = np.round(nwind_hrv/10)
win_hrv = scipy.signal.windows.hann(nwind_hrv)
f_RRI = (.1, .5)




################################
######## HRV TRACKER ########
################################

cond_label_tracker = {'FR_CV_1' : 1, 'MECA' : 2, 'CO2' : 3, 'FR_CV_2' : 1}








