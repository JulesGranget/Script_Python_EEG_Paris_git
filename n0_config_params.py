
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

#### subjects
sujet = 'pilote_sub01'
sujet = 'PD01'

#sujet = 'DEBUG'

conditions = ['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']

conditions_allsubjects = ['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']
sujet_list = np.array(['PD01','MJ02','VN03','GB04','LV05','EF06','PB07','DM08','TA09','BH10','FA11','BD12','FP13',
'MD14','LG15','GM16','JR17','SE18','TM19','TY20','ZV21','DI22','LF23','TJ24','DF25','MN26','BD27','NT28','SC29',
'AR30','HJ31','CM32','MA33'])

band_prep_list = ['wb']

freq_band_dict = {'wb' : {'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]},
                'lf' : {'theta' : [2,10], 'alpha' : [8,14], 'beta' : [10,40], 'whole' : [2,50]},
                'hf' : {'l_gamma' : [50, 80], 'h_gamma' : [80, 120]} }


odor_list = ['o', '+', '-']

srate = 500


################################
######## ODOR ORDER ########
################################

odor_order = {

'PD01' : {'ses02' : 'o', 'ses03' : '+', 'ses04' : '-'}

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

    PC_working = 'Jules_Home'
    if perso_repo_computation:
        path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon\\EEG'
    else:    
        path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon\\EEG'
    path_general = 'D:\\LPPR_CMO_PROJECT\\Lyon\\EEG'
    path_memmap = 'D:\\LPPR_CMO_PROJECT\\Lyon\\EEG\\Mmap'
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

elif PC_ID == 'pc-jules':

    PC_working = 'Jules_Labo_Linux'
    if perso_repo_computation:
        path_main_workdir = '/home/jules/Bureau/perso_repo_computation/Script_Python_EEG_Paris_git'
    else:    
        path_main_workdir = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J/Script_Python_EEG_Paris_git'
    path_general = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J'
    path_memmap = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J/Mmap'
    n_core = 6

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








################################################
######## ELECTRODES REMOVED BEFORE LOCA ######## 
################################################

electrodes_to_remove = {

'Pilote' : [],

}



################################
######## PREP INFO ######## 
################################



sujet_adjust_trig = {
'Pilote' : False

}





################################
######## RESPI PARAMS ########
################################ 

#### INSPI DOWN
sujet_respi_adjust = {
'PD01':'inverse',   'MJ02':'',   'VN03':'',   'GB04':'',   'LV05':'',
'EF06':'',   'PB07':'',   'DM08':'',   'TA09':'',   'BH10':'',
'FA11':'',   'BD12':'',   'FP13':'',   'MD14':'',   'LG15':'',
'GM16':'',   'JR17':'',   'SE18':'',   'TM19':'',   'TY20':'',
'ZV21':'',   'DI22':'',   'LF23':'',   'TJ24':'',   'DF25':'',
'MN26':'',   'BD27':'',   'NT28':'',   'SC29':'',   'AR30':'',
'HJ31':'',   'CM32':'',   'MA33':''
}





################################
######## ECG PARAMS ########
################################ 

sujet_ecg_adjust = {
'PD01':'inverse',   'MJ02':'',   'VN03':'',   'GB04':'',   'LV05':'',
'EF06':'',   'PB07':'',   'DM08':'',   'TA09':'',   'BH10':'',
'FA11':'',   'BD12':'',   'FP13':'',   'MD14':'',   'LG15':'',
'GM16':'',   'JR17':'',   'SE18':'',   'TM19':'',   'TY20':'',
'ZV21':'',   'DI22':'',   'LF23':'',   'TJ24':'',   'DF25':'',
'MN26':'',   'BD27':'',   'NT28':'',   'SC29':'',   'AR30':'',
'HJ31':'',   'CM32':'',   'MA33':''
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
stretch_point_surrogates = 1000

#### coh
n_surrogates_coh = 1000
freq_surrogates = [0, 2]
percentile_coh = .95

#### cycle freq
n_surrogates_cyclefreq = 1000
percentile_cyclefreq_up = .99
percentile_cyclefreq_dw = .01






################################
######## PRECOMPUTE TF ########
################################

#### stretch
stretch_point_TF = 1000
stretch_TF_auto = False
ratio_stretch_TF = 0.45

#### TF & ITPC
nfrex_hf = 50
nfrex_lf = 50
nfrex_wb = 50
ncycle_list_lf = [7, 15]
ncycle_list_hf = [20, 30]
ncycle_list_wb = [7, 30]
srate_dw = 10



################################
######## POWER ANALYSIS ########
################################

#### analysis
coh_computation_interval = .02 #Hz around respi


################################
######## FC ANALYSIS ########
################################

#### band to remove
freq_band_fc_analysis = {'theta' : [4, 8], 'alpha' : [9,12], 'beta' : [15,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}




################################
######## HRV ANALYSIS ########
################################



srate_resample_hrv = 10
nwind_hrv = int( 128*srate_resample_hrv )
nfft_hrv = nwind_hrv
noverlap_hrv = np.round(nwind_hrv/10)
win_hrv = scipy.signal.windows.hann(nwind_hrv)
f_RRI = (.1, .5)



