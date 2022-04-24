


import numpy as np
import pandas as pd
import glob
import os

from n0_config import *
from n0bis_analysis_functions import *


########################################
######## COMPUTE PSYCHO METRICS ########
########################################


#session_eeg = 0
def logtrigg_to_df(sujet, session_eeg):

    #### get to logtrig
    os.chdir(os.path.join(path_data, sujet))
    os.chdir(f'./{os.listdir()[0]}')
    os.chdir(f'ses_0{session_eeg+2}')

    logtrig_file = [file_i for file_i in os.listdir() if file_i.find('LogTrigger') != -1][0]
    random_bloc_file = [file_i for file_i in os.listdir() if file_i.find('random') != -1][0]

    logtrigg = pd.read_fwf(logtrig_file, colspecs = [(0,1000000)]).values
    random_bloc = pd.read_fwf(random_bloc_file, colspecs = [(0,1000000)]).values
    bloc_order = random_bloc.reshape(random_bloc.shape[0],)[0].rsplit(sep = " ")

    bloc_order.insert(0, 'entrainement')
    bloc_order.insert(0, 'baseline')
    bloc_order.insert(len(bloc_order), 'baseline')
    
    bloc_types = list(set(bloc_order))
    bloc_labeled = []
    bloc_pos = {}
    for bloc in bloc_types:
        bloc_pos[bloc] = np.where(np.array(bloc_order) == bloc)[0]
        
    bloc_nums = bloc_order.copy()
    trial_nums = bloc_order.copy()
    for bloc in bloc_order:
        for i, idx in enumerate(list(bloc_pos[bloc])):
            bloc_nums[idx] = f'{bloc}{i+1}'
            trial_nums[idx] = f'trial{i+1}'
            
    items = [
        'calme',
        'crispé',
        'ému',
        'décontracté',
        'satisfait',
        'inquiet',
        'attention',
        'relaxé'
    ]



    df = pd.DataFrame(columns=items)
    
    for item in items:
        value_item = []
        for line in logtrigg:
            if item in line[0]:
                value = int(line[0][len(line[0]) - 2 :])
                value_item.append(value)
        for i, value in enumerate(value_item):
            df.loc[i+1, item] = value

    df.index = bloc_nums
    
    odor = odor_order[sujet][f's{session_eeg+1}'] # here session_eeg +1 and not +2 because for the rest of the analysis its +1 that is used

    etats = []
    for index in bloc_nums:
        etat = 100 - df.loc[index,'calme'] + df.loc[index,'crispé'] + df.loc[index,'ému'] + 100 - df.loc[index,'décontracté'] + 100 - df.loc[index,'satisfait'] + df.loc[index,'inquiet']
        etats.append(etat)
        
    df = df.reset_index().rename(columns = {"index":"trial"})
    df['trial'] = trial_nums
    
    session_eeg_col = [f'{session_eeg+1}' for i in range(df.shape[0])]
    odeur_col = [f'odor_{odor}' for i in range(df.shape[0])]
    sujet_col = [f'{sujet}' for i in range(df.shape[0])]
    
    df.insert(df.shape[1], 'état', etats)
    df.insert(0, 'blocs', bloc_order)
    df.insert(0, 'odeur', odeur_col)
    df.insert(0, 'session', session_eeg_col)
    df.insert(0, 'participant', sujet_col)

    df.to_excel(f'{sujet}_{session_eeg+2}_stai_shortform.xlsx')



def processing_stai_longform(sujet):

    #### get data
    os.chdir(os.path.join(path_data, sujet))
    os.chdir(f'./{os.listdir()[0]}')
    os.chdir(f'ses_01')

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
    os.chdir(f'./{os.listdir()[0]}')
    os.chdir(f'ses_01')

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

    processing_stai_longform(sujet)
    processing_maia(sujet)
    
    for session_eeg in range(3):
        logtrigg_to_df(sujet, session_eeg)
    