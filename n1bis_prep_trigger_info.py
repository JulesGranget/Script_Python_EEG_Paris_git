

import matplotlib.pyplot as plt

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False


#### to correct error
raw_aux = []
srate = 500
session_i = []
ecg_events_time = []

#### verif trig with raw_aux from preproc
if debug:
    plt.plot(zscore(raw_aux.get_data()[0,:]), label='PRESS')
    plt.plot(zscore(raw_aux.get_data()[2,:]), label='TRIG')
    plt.legend()
    plt.show()


#### trigger value
dict_trig_sujet = {

'PD01' :    {'ses02' :  {'FR_CV_1_start' : int(1473), 'FR_CV_1_stop' : int(1473 + (srate*300)), 
                        'MECA_start' : int(5.249e5), 'MECA_stop' : int(5.249e5 + (srate*300)), 
                        'CO2_start' : int(1.4398e6 - (srate*300)), 'CO2_stop' : int(1.4398e6), 
                        'FR_CV_2_start' : int(1.5931e6), 'FR_CV_2_stop' : int(1.5931e6 + (srate*300))},
            }

}


#### verif trig 
if debug:
    plt.plot(zscore(raw_aux.get_data()[0,:]), label='PRESS')
    plt.plot(zscore(raw_aux.get_data()[2,:]), label='TRIG')
    plt.vlines(dict_trig_sujet[sujet][f'ses0{session_i+1}'].values(), ymin=zscore(raw_aux.get_data()[0,:]).min(), ymax=zscore(raw_aux.get_data()[0,:]).max(), color='g')
    plt.legend()
    plt.show()



#### ecg correction
dict_ecg_correction = {

'PD01' :    {'ses02' :  {'add' : [], 'remove' : []},
            'ses03' :  {'add' : [], 'remove' : []},
            'ses04' :  {'add' : [], 'remove' : []},
            }

}

#### add
ecg_events_corrected = [2799780, 2802240, 2803050, 2808680, 2811270, 2809297, 2812524, 2813138, 2816806, 2816992, 2817652, 2818294, 2818923, 2819521, 2820121, 2820693, 2821269, 2821804, 2940129, 2973382, 2973777, 3065220, 3065711]
ecg_events_time += ecg_events_corrected
ecg_events_time.sort()
#### remove
ecg_events_to_remove = []
[ecg_events_time.remove(i) for i in ecg_events_to_remove]






