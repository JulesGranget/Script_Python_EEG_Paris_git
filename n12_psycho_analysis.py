


import numpy as np
import pandas as pd
import os

from n0_config_params import *
from n0bis_config_analysis_functions import *


########################################
######## COMPUTE PSYCHO METRICS ########
########################################


#session_eeg = 0
def STAI_short_process(sujet, session_eeg):

    #### get to logtrig
    os.chdir(os.path.join(path_data, sujet))
    os.chdir(f'ses_0{session_eeg+2}')

    #### compute
    df = pd.read_excel(f'{sujet}_ses0{session_eeg+2}_STAI_short_raw.xlsx')
    df[df.columns[-7:]] = ( df[df.columns[-7:]] *100 / 150 ).values.astype(int)
    df.to_excel(f'{sujet}_ses0{session_eeg+2}_STAI_short_processed.xlsx')

    conditions = df['cond'].values

    etats = {}
    #cond = conditions[0]
    for cond in conditions:
        etat = 100 - df[df['cond'] == cond]['calme'].values[0] + df[df['cond'] == cond]['tendu'].values[0] + df[df['cond'] == cond]['emu'].values[0] + 100 - df[df['cond'] == cond]['decontracte'].values[0] + 100 - df[df['cond'] == cond]['satisfait'].values[0] + df[df['cond'] == cond]['inquiet'].values[0]
        etats[cond] = etat
        
    df = df[df.columns[:-7]]
    
    df.insert(df.shape[1], 'etat', etats.values())

    df.to_excel(f'{sujet}_ses0{session_eeg+2}_STAI_short_processed.xlsx')




def processing_odor_cotation(sujet):

    #### get to data
    os.chdir(os.path.join(path_data, sujet))
    os.chdir(f'ses_0{session_eeg+1}')

    #### compute
    df = pd.read_excel(f'{sujet}_odor_absolute_raw.xlsx')
    df['appreciation_absolue'] = ( df['appreciation_absolue'] *100 / 150 ).values.astype(int) - 50
    df.to_excel(f'{sujet}_odor_absolute_processed.xlsx')

    df = pd.read_excel(f'{sujet}_odor_relative_raw.xlsx')
    df['appreciation_relative'] = ( df['appreciation_relative'] *100 / 150 ).values.astype(int) - 50
    df[df.columns[-4:]] = ( df[df.columns[-4:]] *100 / 150 ).values.astype(int) 
    df.to_excel(f'{sujet}_odor_relative_processed.xlsx')





def processing_stai_longform(sujet):

    #### get to data
    os.chdir(os.path.join(path_data, sujet))
    os.chdir(f'ses_0{session_eeg+1}')

    file_to_open = f'{sujet}_stai_longform_raw.xlsx'

    raw_stai = pd.read_excel(file_to_open)

    list_scores = list(raw_stai['score'].values)
    list_corrections = list(raw_stai['correction'].values)

    score_corrected = []
    for score, correction in zip(list_scores, list_corrections):
        if correction == '-':
            score_c = 5-score
        else : 
            score_c = score
            
        score_corrected.append(score_c)

    etat_score = np.sum(score_corrected[0:20])
    trait_score = np.sum(score_corrected[20:None])

    mean_etat = 35.4
    mean_trait = 24.8
    std_etat = 10.5
    std_trait = 9.2

    etat_result = ''
    trait_result = ''

    if etat_score > (mean_etat+1.96*std_etat):
        etat_result = 'etat_anxieux'
    elif etat_score < (mean_etat-1.96*std_etat):
        etat_result = 'etat_moins_que_anxieux'
    else :
        etat_result = 'etat_dans_les_normes'
        
    if trait_score > (mean_trait+1.96*std_trait): 
        trait_result = 'trait_anxieux'
    elif trait_score < (mean_trait-1.96*std_trait):
        trait_result = 'trait_moins_que_anxieux'
    else : 
        trait_result = 'trait_dans_les_normes'


    dict_results = {'sujet':sujet, 'etat_score': etat_score, 'etat_result': etat_result, 'trait_score': trait_score, 'trait_result':trait_result}
    ser = pd.Series(dict_results)
    df = pd.DataFrame(ser).T.set_index('sujet')
    
    df.to_excel(f'{sujet}_stai_longform_processed.xlsx')



def processing_maia(sujet):

    #### get data
    os.chdir(os.path.join(path_data, sujet))
    os.chdir(f'ses_0{session_eeg+1}')

    file_to_open = f'{sujet}_maia_raw.xlsx'

    raw_maia = pd.read_excel(file_to_open)

    labels = ['sujet','noticing','not_distracting','not_worrying','attention_regulation','emotional_awareness','self_regulation','body_listening','trusting','awareness_of_body_sensations','emotional_reaction','capicity_regulation_attention','awareness_of_mind_body','trusting_body_sensations','global_mean']
    idx_labels = [(0,4),(4,7),(7,10),(10,17),(17,22),(22,26),(26,29),(29,None),(0,4),(4,10),(10,17),(17,22),(22,29),(29,None),(None,None)]

    dict_means = {}
    for label, idxs in zip(labels, idx_labels):
        if label == 'sujet':
            dict_means[label] = sujet
        else:
            dict_means[label] = np.mean(raw_maia['score'][idxs[0]:idxs[1]])


    ser = pd.Series(dict_means)
    df = pd.DataFrame(ser).T.set_index('sujet')
    
    df.to_excel(f'{sujet}_maia_processed.xlsx')
    








################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    for sujet in sujet_list:

        processing_stai_longform(sujet)
        processing_maia(sujet)
        processing_odor_cotation(sujet)

        
        for session_eeg in range(3):
            STAI_short_process(sujet, session_eeg)
        