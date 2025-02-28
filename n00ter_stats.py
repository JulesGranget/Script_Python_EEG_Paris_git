
import os
from n00_config_params import *
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import statsmodels.formula.api as smf

def mad(data, constant = 1.4826):
    median = np.median(data)
    return np.median(np.abs(data - median)) * constant

def normality(df, predictor, outcome):
    df = df.reset_index(drop=True)
    groups = list(set(df[predictor]))
    n_groups = len(groups)

    normalities = pg.normality(data = df , dv = outcome, group = predictor)['normal']
    
    if sum(normalities) == normalities.size:
        normal = True
    else:
        normal = False
    
    return normal

def sphericity(df, predictor, outcome, subject):
    spher, W , chi2, dof, pval = pg.sphericity(data = df, dv = outcome, within = predictor, subject = subject)
    return spher

def homoscedasticity(df, predictor, outcome):

    homoscedasticity = pg.homoscedasticity(data = df, dv = outcome, group = predictor)['equal_var'].values[0] # Levene test

    return homoscedasticity

def parametric(df, predictor, outcome, subject = None):
    
    df = df.reset_index(drop=True)
    groups = list(set(df[predictor]))
    n_groups = len(groups)
    
    normal = normality(df, predictor, outcome)

    if subject is None:
        equal_var = homoscedasticity(df, predictor, outcome)
    else:
        equal_var = sphericity(df, predictor, outcome, subject)
    
    if normal and equal_var:
        parametricity = True
    else:
        parametricity = False
        
    return parametricity


def guidelines(df, predictor, outcome, design, parametricity):
        
    n_groups = len(list(set(df[predictor])))
    
    if parametricity:
        if n_groups <= 2:
            if design == 'between':
                tests = {'pre':'t-test_ind','post':None, 'post_name_corrected':None}
            elif design == 'within':
                tests = {'pre':'t-test_paired','post':None, 'post_name_corrected':None}
        else:
            if design == 'between':
                tests = {'pre':'anova','post':'pairwise_tukey', 'post_name_corrected':'pairwise_tukey'}
            elif design == 'within':
                tests = {'pre':'rm_anova','post':'pairwise_tests_paired_paramTrue', 'post_name_corrected':'pairwise_ttests_paired'}
    else:
        if n_groups <= 2:
            if design == 'between':
                tests = {'pre':'Mann-Whitney','post':None, 'post_name_corrected':None}
            elif design == 'within':
                tests = {'pre':'Wilcoxon','post':None, 'post_name_corrected':None}
        else:
            if design == 'between':
                tests = {'pre':'Kruskal','post':'pairwise_tests_ind_paramFalse', 'post_name_corrected':'mwu'}
            elif design == 'within':
                tests = {'pre':'friedman','post':'pairwise_tests_paired_paramFalse', 'post_name_corrected':'pairwise_wilcoxon_paired'}
                
    return tests

def pg_compute_pre(df, predictor, outcome, test, subject=None, show = False):
    
    pval_labels = {'t-test_ind':'p-val','t-test_paired':'p-val','anova':'p-unc','rm_anova':'p-unc','Mann-Whitney':'p-val','Wilcoxon':'p-val', 'Kruskal':'p-unc', 'friedman':'p-unc'}
    esize_labels = {'t-test_ind':'cohen-d','t-test_paired':'cohen-d','anova':'np2','rm_anova':'np2','Mann-Whitney':'CLES','Wilcoxon':'CLES', 'Kruskal':None, 'friedman':None}
    
    if test == 't-test_ind':
        groups = list(set(df[predictor]))
        pre = df[df[predictor] == groups[0]][outcome]
        post = df[df[predictor] == groups[1]][outcome]
        res = pg.ttest(pre, post, paired=False)
        
    elif test == 't-test_paired':
        groups = list(set(df[predictor]))
        pre = df[df[predictor] == groups[0]][outcome]
        post = df[df[predictor] == groups[1]][outcome]
        res = pg.ttest(pre, post, paired=True)
        
    elif test == 'anova':
        res = pg.anova(dv=outcome, between=predictor, data=df, detailed=False, effsize = 'np2')
    
    elif test == 'rm_anova':
        res = pg.rm_anova(dv=outcome, within=predictor, data=df, detailed=False, effsize = 'np2', subject = subject)
        
    elif test == 'Mann-Whitney':
        groups = list(set(df[predictor]))
        x = df[df[predictor] == groups[0]][outcome]
        y = df[df[predictor] == groups[1]][outcome]
        res = pg.mwu(x, y)
        
    elif test == 'Wilcoxon':
        groups = list(set(df[predictor]))
        x = df[df[predictor] == groups[0]][outcome]
        y = df[df[predictor] == groups[1]][outcome]
        res = pg.wilcoxon(x, y)
        
    elif test == 'Kruskal':
        res = pg.kruskal(data=df, dv=outcome, between=predictor)
        
    elif test == 'friedman':
        res = pg.friedman(data=df, dv=outcome, within=predictor, subject=subject)
    
    pval = res[pval_labels[test]].values[0]
    es_label = esize_labels[test]
    if es_label is None:
        es = None
    else:
        es = res[es_label].values[0]
    
    es_interp = es_interpretation(es_label, es)
    results = {'p':pval, 'es':es, 'es_label':es_label, 'es_interp':es_interp}
      
    return results

def es_interpretation(es_label , es_value):

    if es_label == 'cohen-d' or es_label == 'CLES':
        if es_value < 0.2:
            interpretation = 'VS'
        elif es_value >= 0.2 and es_value < 0.5:
            interpretation = 'S'
        elif es_value >= 0.5 and es_value < 0.8:
            interpretation = 'M'
        elif es_value >= 0.8 and es_value < 1.3:
            interpretation = 'L'
        else:
            interpretation = 'VL'
    
    elif es_label == 'np2':
        if es_value < 0.01:
            interpretation = 'VS'
        elif es_value >= 0.01 and es_value < 0.06:
            interpretation = 'S'
        elif es_value >= 0.06 and es_value < 0.14:
            interpretation = 'M'
        else:
            interpretation = 'L'
            
    elif es_label is None:
        interpretation = None
                
    return interpretation


def homemade_post_hoc(df, predictor, outcome, design = 'within', subject = None, parametric = True):
    pairs = pg.pairwise_tests(data=df, dv = outcome, within = predictor, subject = subject, parametric = False).loc[:,['A','B']]
    pvals = []
    for i, pair in pairs.iterrows():
        x = df[df[predictor] == pair[0]][outcome]
        y = df[df[predictor] == pair[1]][outcome]

        if design == 'within':
            if parametric:
                p = pg.ttest(x, y, paired= True)['p-val']
            else:
                p = pg.wilcoxon(x, y)['p-val']
        elif design == 'between':
            if parametric:
                p = pg.ttest(x, y, paired= False)['p-val']
            else:
                p = pg.mwu(x, y)['p-val']
        pvals.append(p.values[0])
        
    pairs['p-unc'] = pvals
    _, pvals_corr = pg.multicomp(pvals)
    pairs['p-corr'] = pvals_corr
    return pairs
        
#test = post_test
def pg_compute_post_hoc(df, predictor, outcome, test, subject=None):

    if not subject is None:
        n_subjects = df[subject].unique().size
    else:
        n_subjects = df[predictor].value_counts()[0]
    
    if test == 'pairwise_tukey':
        res = pg.pairwise_tukey(data = df, dv=outcome, between=predictor)
        res['p-corr'] = pg.multicomp(res['p-tukey'])[1]

    elif test == 'pairwise_tests_paired_paramTrue':
        res = pg.pairwise_tests(data = df, dv=outcome, within=predictor, subject=subject, parametric=True, padjust = 'holm', return_desc=True)
        # res = homemade_post_hoc(df = df, outcome=outcome, predictor=predictor, design = 'within', subject=subject, parametric=True)
        
    elif test == 'pairwise_tests_ind_paramFalse':
        if n_subjects >= 15:
            res = pg.pairwise_tests(data = df, dv=outcome, between=predictor, parametric=False, padjust = 'holm', return_desc=True)
        else:
            res = permutation(df = df, outcome=outcome, predictor=predictor, design = 'between')

    elif test == 'pairwise_tests_paired_paramFalse':
        if n_subjects >= 15:
            res = pg.pairwise_tests(data = df, dv=outcome, within=predictor, subject=subject, parametric=False, padjust = 'holm', return_desc=True)
        else:
            res = permutation(df = df, outcome=outcome, predictor=predictor, design = 'within')
     
    return res


def auto_annotated_stats(df, predictor, outcome, test):
    
    x = predictor
    y = outcome

    order = list(set(df[predictor]))

    ax = sns.boxplot(data=df, x=x, y=y, order=order, showfliers=False)
    pairs=[(order[0],order[1])]
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
    annotator.configure(test=test, text_format='star', loc='inside')
    annotator.apply_and_annotate()
    # plt.show()

def custom_annotated_two(df, predictor, outcome, order, pval, ax=None, plot_mode = 'box'):
        
    stars = pval_stars(pval)
    
    x = predictor
    y = outcome

    order = order
    formatted_pvalues = [f"{stars}"]
    if plot_mode == 'box':
        ax = sns.boxplot(data=df, x=x, y=y, order=order, ax=ax, showfliers = False)
    elif plot_mode == 'violin':
        ax = sns.violinplot(data=df, x=x, y=y, order=order, bw = 0.08)
    pairs=[(order[0],order[1])]
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order, verbose = False)
    annotator.configure(test='test', text_format='star', loc='inside')
    annotator.set_custom_annotations(formatted_pvalues)
    annotator.annotate()
    return ax

def custom_annotated_ngroups(df, predictor, outcome, post_hoc, order, ax=None, plot_mode = 'box'):
        
    pvalues = list(post_hoc['p-corr'])

    x = predictor
    y = outcome

    order = order
    pairs = [tuple(post_hoc.loc[i,['A','B']]) for i in range(post_hoc.shape[0])]
    formatted_pvalues = [f"{pval_stars(pval)}" for pval in pvalues]
    if plot_mode == 'box':
        ax = sns.boxplot(data=df, x=x, y=y, order=order, ax=ax, showfliers = False)
    elif plot_mode == 'violin':
        ax = sns.violinplot(data=df, x=x, y=y, order=order, bw= 0.08)
    
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order, verbose = False)
    annotator.configure(test='test', text_format='star', loc='inside')
    annotator.set_custom_annotations(formatted_pvalues)
    annotator.annotate()
    return ax

             
def pval_stars(pval):
    if pval < 0.05 and pval >= 0.01:
        stars = '*'
    elif pval < 0.01 and pval >= 0.001:
        stars = '**'
    elif pval < 0.001 and pval >= 0.0001:
        stars = '***'
    elif pval < 0.0001:
        stars = '****'
    elif pval >= 0.05:
        stars = 'ns'
    return stars


def transform_data(df, outcome):
    df_transfo = df.copy()
    df_transfo[outcome] = np.log(df[outcome])
    return df_transfo  




def auto_stats(df, predictor, outcome, ax=None, subject=None, design='within', mode='box', transform=False, verbose=True, order=None, extra_title=None, estimator='mean'):
    
    """
    Automatically compute statistical tests chosen based on normality & homoscedasticity of data and plot it
    ------------
    Inputs =
    - df : tidy dataframe
    - predictor : str or list of str of the column name of the predictive variable (if list --> N-way anova)
    - outcome : column name of the outcome/target/dependent variable
    - ax : ax on which plotting the subplot, created if None (default = None)
    - subject : column name of the subject variable = the within factor variable
    - design : 'within' or 'between' for repeated or independent stats , respectively
    - mode : 'box' or 'violin' for mode of plotting
    - transform : log transform data if True and if data are non-normally distributed & heteroscedastic , to try to do a parametric test after transformation (default = False)
    - verbose : print idea of successfull or unsucessfull transformation of data, if transformed, acccording to non-parametric to parametric test feasable after transformation (default = True)
    - order : order of xlabels (= of groups) if the plot, default = None = default order
    
    Output = 
    - ax : subplot
    
    """
    
    if isinstance(predictor, str):

        if ax is None:
            fig, ax = plt.subplots()

        N = df[predictor].value_counts()[0]
        groups = list(df[predictor].unique())
        ngroups = len(groups)
        
        parametricity_pre_transfo = parametric(df, predictor, outcome, subject)
        
        if transform:
            if not parametricity_pre_transfo:
                df = transform_data(df, outcome)
                parametricity_post_transfo = parametric(df, predictor, outcome, subject)
                parametricity = parametricity_post_transfo
                if verbose:
                    if parametricity_post_transfo:
                        print('Successfull transformation')
                    else:
                        print('Un-successfull transformation')
            else:
                parametricity = parametricity_pre_transfo
        else:
            parametricity = parametricity_pre_transfo
        
        tests = guidelines(df, predictor, outcome, design, parametricity)
        
        pre_test = tests['pre']
        post_test = tests['post']
        results = pg_compute_pre(df, predictor, outcome, pre_test, subject)
        pval = round(results['p'], 4)
        
        if not results['es'] is None:
            es = round(results['es'], 3)
        else:
            es = results['es']
        es_label = results['es_label']
        es_inter = results['es_interp']
        
        if order is None:
            order = list(df[predictor].unique())
        else:
            order = order

        if estimator == 'mean':
            estimators = pd.concat([df.groupby(predictor).mean(outcome)[outcome].reset_index(), df[[predictor, outcome]].groupby(predictor).std()[outcome].reset_index()[outcome].rename('sd')], axis = 1).round(2).set_index(predictor)
        elif estimator == 'median':
            df_mad = {predictor : [], outcome : []}
            for _predictor in df[predictor].unique():
                df_mad[predictor].append(_predictor)
                df_mad[outcome].append(mad(df[[predictor, outcome]].query(f"{predictor} == '{_predictor}'")[outcome].values))
            df_mad = pd.DataFrame(df_mad)[outcome].rename('sd')
            estimators = pd.concat([df.groupby(predictor).median(outcome)[outcome].reset_index(), df_mad], axis = 1).round(2).set_index(predictor)
        
        cis = [f'[{round(confidence_interval(x)[0],3)};{round(confidence_interval(x)[1],3)}]' for x in [df[df[predictor] == group][outcome] for group in groups]]
        ticks_estimators = [f"{cond} \n {estimators.loc[cond,outcome]} ({estimators.loc[cond,'sd']}) \n {ci} " for cond, ci in zip(order,cis)]

        if mode == 'box':
            if not post_test is None:
                post_hoc = pg_compute_post_hoc(df, predictor, outcome, post_test, subject)
                ax = custom_annotated_ngroups(df, predictor, outcome, post_hoc, order, ax=ax)
            else:
                ax = custom_annotated_two(df, predictor, outcome, order, pval, ax=ax)
            ax.set_xticks(range(ngroups))
            ax.set_xticklabels(ticks_estimators)
            
        elif mode == 'distribution':
            # ax = sns.histplot(df, x=outcome, hue = predictor, kde = True, ax=ax)
            ax = sns.kdeplot(data=df, x=outcome, hue = predictor, ax=ax, bw_adjust = 0.6)
        
        if design == 'between':
            if es_label is None:
                ax.set_title(f'Effect of {predictor} on {outcome} : {pval_stars(pval)} \n N = {N} values/group * {ngroups} groups \n {pre_test} : p-{pval}')
            else:
                ax.set_title(f'Effect of {predictor} on {outcome} : {pval_stars(pval)} \n N = {N} values/group * {ngroups} groups \n {pre_test} : p-{pval}, {es_label} : {es} ({es_inter})')
        elif design == 'within':
            n_subjects = df[subject].unique().size
            if es_label is None:
                ax.set_title(f'Effect of {predictor} on {outcome} : {pval_stars(pval)} \n N = {n_subjects} subjects * {ngroups} groups (*{int(N/n_subjects)} trial/group) \n {pre_test} : p-{pval}')
            else:
                ax.set_title(f'Effect of {predictor} on {outcome} : {pval_stars(pval)} \n  N = {n_subjects} subjects * {ngroups} groups (*{int(N/n_subjects)} trial/group) \n {pre_test} : p-{pval}, {es_label} : {es} ({es_inter})')

        return ax
    
    elif isinstance(predictor, list):
        
        if design == 'within':
            test_type = 'rm_anova'
            test = pg.rm_anova(data=df, dv=outcome, within = predictor, subject = subject, effsize = 'np2').set_index('Source').round(3)
            pval = test.loc[f'{predictor[0]} * {predictor[1]}','p-GG-corr']
            pstars = pval_stars(pval)
            es_label = test.columns[-2]
            es = test.loc[f'{predictor[0]} * {predictor[1]}','np2']
            es_inter = es_interpretation(es_label=es_label, es_value=es)
            ppred_0 = test.loc[f'{predictor[0]}', 'p-GG-corr']
            ppred_1 = test.loc[f'{predictor[1]}', 'p-GG-corr']
            
        elif design == 'between':
            test_type = 'anova'
            test = pg.anova(data=df, dv=outcome, between = predictor).set_index('Source').round(3)
            pval = test.loc[f'{predictor[0]} * {predictor[1]}','p-unc']
            pstars = pval_stars(pval)
            es_label = test.columns[-1]
            es = test.loc[f'{predictor[0]} * {predictor[1]}','np2']
            es_inter = es_interpretation(es_label=es_label, es_value=es)
            ppred_0 = test.loc[f'{predictor[0]}', 'p-unc']
            ppred_1 = test.loc[f'{predictor[1]}', 'p-unc']
            
        if len(df[predictor[0]]) >= len(df[predictor[1]]):
            x = predictor[0]
            hue = predictor[1]
        else:
            x = predictor[1]
            hue = predictor[0]
        
        # sns.pointplot(data = df , x = x, y = outcome, hue = hue, ax=ax, dodge=True)
        if extra_title != None:
            title = f'{extra_title} \n Effect of {predictor[0]} * {predictor[1]} on {outcome} : {pstars} \n {test_type} : pcorr-{pval}, {es_label} : {es} ({es_inter}) \n p-{predictor[0]}-{ppred_0} , p-{predictor[1]}-{ppred_1}'
        else:
            title = f'Effect of {predictor[0]} * {predictor[1]} on {outcome} : {pstars} \n {test_type} : pcorr-{pval}, {es_label} : {es} ({es_inter}) \n p-{predictor[0]}-{ppred_0} , p-{predictor[1]}-{ppred_1}'
        
        sns.catplot(data=df, kind='bar', x=x, y=outcome, hue=hue, order=order).set(title=title)
        # plt.title(f'Effect of {predictor[0]} * {predictor[1]} on {outcome} : {pstars} \n {test_type} : pcorr-{pval}, {es_label} : {es} ({es_inter}) \n p-{predictor[0]}-{ppred_0} , p-{predictor[1]}-{ppred_1}')        # title = f'Effect of {predictor[0]} * {predictor[1]} on {outcome} : {pstars} \n {test_type} : pcorr-{pval}, {es_label} : {es} ({es_inter}) \n p-{predictor[0]}-{ppred_0} , p-{predictor[1]}-{ppred_1}'
        # ax.set_title(title)
        # ax.set_title(f'Effect of {predictor[0]} * {predictor[1]} on {outcome} : {pstars} \n {test_type} : pcorr-{pval}, {es_label} : {es} ({es_inter}) \n p-{predictor[0]}-{ppred_0} , p-{predictor[1]}-{ppred_1}')        # title = f'Effect of {predictor[0]} * {predictor[1]} on {outcome} : {pstars} \n {test_type} : pcorr-{pval}, {es_label} : {es} ({es_inter}) \n p-{predictor[0]}-{ppred_0} , p-{predictor[1]}-{ppred_1}'
        
        return ax



def get_auto_stats_df(df, predictor, outcome, subject=None, design='within', transform=False, verbose=True):
    
    if isinstance(predictor, str):
        
        parametricity_pre_transfo = parametric(df, predictor, outcome, subject)
        
        if transform:
            if not parametricity_pre_transfo:
                df = transform_data(df, outcome)
                parametricity_post_transfo = parametric(df, predictor, outcome, subject)
                parametricity = parametricity_post_transfo
                if verbose:
                    if parametricity_post_transfo:
                        print('Successfull transformation')
                    else:
                        print('Un-successfull transformation')
            else:
                parametricity = parametricity_pre_transfo
        else:
            parametricity = parametricity_pre_transfo
        
        tests = guidelines(df, predictor, outcome, design, parametricity)
        
        pre_test = tests['pre']
        post_test = tests['post']
        results = pg_compute_pre(df, predictor, outcome, pre_test, subject)
        pval = round(results['p'], 4)

        if not post_test is None:
            post_hoc = pg_compute_post_hoc(df, predictor, outcome, post_test, subject)

    post_hoc['pre_test'] = np.array([pre_test] * post_hoc.shape[0])
    post_hoc['pre_test_pval'] = np.array([pval] * post_hoc.shape[0])
    post_hoc['Contrast'] = np.array([predictor] * post_hoc.shape[0])

    df_post_hoc = post_hoc.reindex(columns=['pre_test', 'pre_test_pval', 'Contrast', 'A', 'B', 'p-unc', 'p-corr', 'p-adjust'])

    df_post_hoc = df_post_hoc.rename(columns={'p-unc': 'p_unc'})

    return df_post_hoc


def virer_outliers(df, predictor, outcome, deviations = 5):
    
    groups = list(df[predictor].unique())
    
    group1 = df[df[predictor] == groups[0]][outcome]
    group2 = df[df[predictor] == groups[1]][outcome]
    
    outliers_trop_hauts_g1 = group1[(group1 > group1.std() * deviations) ]
    outliers_trop_bas_g1 = group1[(group1 < group1.std() * -deviations) ]
    
    outliers_trop_hauts_g2 = group2[(group2 > group1.std() * deviations) ]
    outliers_trop_bas_g2 = group2[(group2 < group1.std() * -deviations) ]
    
    len_h_g1 = outliers_trop_hauts_g1.size
    len_b_g1 = outliers_trop_bas_g1.size
    len_h_g2 = outliers_trop_hauts_g2.size
    len_b_g2 = outliers_trop_bas_g2.size
    
    return len_b_g2


def outlier_exploration(df, predictor, labels, outcome, figsize = (16,8)):
                 
    g1 = df[df[predictor] == labels[0]][outcome]
    g2 = df[df[predictor] == labels[1]][outcome]

    fig, axs = plt.subplots(ncols = 2, figsize = figsize, constrained_layout = True)
    fig.suptitle('Outliers exploration', fontsize = 20)

    ax = axs[0]
    ax.scatter(g1 , g2)
    ax.set_title(f'Raw {labels[0]} vs {labels[1]} scatterplot')
    ax.set_ylabel(f'{outcome} in condition {labels[0]}')
    ax.set_xlabel(f'{outcome} in condition {labels[1]}')

    g1log = np.log(g1)
    g2log = np.log(g2)

    ax = axs[1]
    ax.scatter(g1log, g2log)
    ax.set_title(f'Log-log {labels[0]} vs {labels[1]} scatterplot')
    ax.set_ylabel(f'{outcome} in condition {labels[0]}')
    ax.set_xlabel(f'{outcome} in condition {labels[1]}')

    plt.show()
    
    
def qqplot(df, predictor, outcome, figsize = (10,15)):
    
    labels = list(df[predictor].unique())
    ngroups = len(labels) 
    
    groupe = {}
    
    for label in labels: 
        groupe[label] = {
                         'log' : np.log(df[df[predictor] == label][outcome]), 
                         'inverse' : 1 / (df[df[predictor] == label][outcome]),
                         'none' : df[df[predictor] == label][outcome]
                        }
     
    fig, axs = plt.subplots(nrows = 3, ncols = ngroups, figsize = figsize, constrained_layout = True)
    fig.suptitle(f'QQ-PLOT', fontsize = 20)
    
    for col, label in enumerate(labels): 
        for row, transform in enumerate(['none','log','inverse']):
            ax = axs[row, col]
            ax = pg.qqplot(groupe[label][transform], ax=ax)
            ax.set_title(f'Condition : {label} ; data are {transform} transformed')
        
    plt.show()

def permutation_test_homemade(x,y, design = 'within', n_resamples=999):
    def statistic(x, y):
        return np.mean(x) - np.mean(y)
    if design == 'within':
        permutation_type = 'samples'
    elif design == 'between':
        permutation_type = 'independent'
    res = stats.permutation_test(data=[x,y], statistic=statistic, permutation_type=permutation_type, n_resamples=n_resamples, batch=None, alternative='two-sided', axis=0, random_state=None)
    return res.pvalue

def permutation(df, predictor, outcome , design = 'within' , subject = None, n_resamples=999):
    pairs = list((itertools.combinations(df[predictor].unique(), 2)))
    pvals = []
    for pair in pairs:
        x = df[df[predictor] == pair[0]][outcome].values
        y = df[df[predictor] == pair[1]][outcome].values
        p = permutation_test_homemade(x=x,y=y, design=design, n_resamples=n_resamples)
        pvals.append(p)
    df_return = pd.DataFrame(pairs, columns = ['A','B'])
    df_return['p-unc'] = pvals
    rej , pcorrs = pg.multicomp(pvals, method = 'holm')
    df_return['p-corr'] = pcorrs
    return df_return

def reorder_df(df, colname, order):
    concat = []
    for cond in order:
        concat.append(df[df[colname] == cond])
    return pd.concat(concat)


def lmm(df, predictor, outcome, subject, order=None):

    if isinstance(predictor, str):
        formula = f'{outcome} ~ {predictor}' 
    elif isinstance(predictor, list):
        if len(predictor) == 2:
            formula = f'{outcome} ~ {predictor[0]}*{predictor[1]}' 
        elif len(predictor) == 3:
            formula = f'{outcome} ~ {predictor[0]}*{predictor[1]}*{predictor[2]}' 

    if not order is None:
        df = reorder_df(df, predictor, order)

    order = list(df[predictor].unique())

    md = smf.mixedlm(formula, data=df, groups=df[subject])
    mdf = md.fit()
    print(mdf.summary())

    pvals = mdf.pvalues.to_frame(name = 'p')
    coefs = mdf.fe_params.to_frame(name = 'coef').round(3)
    dict_pval_stars = {idx.split('.')[1][:-1]:pval_stars(pvals.loc[idx,'p']) for idx in pvals.index if not idx in ['Intercept','Group Var']}
    dict_coefs = {idx.split('.')[1][:-1]:coefs.loc[idx,'coef'] for idx in coefs.index if not idx in ['Intercept','Group Var']}

    fig, ax = plt.subplots()
    if isinstance(predictor, str):
        sns.boxplot(data=df, x = predictor, y = outcome, ax=ax, showfliers = False)
    elif isinstance(predictor, list):
        sns.pointplot(data=df, x = predictor[0], y = outcome, hue = predictor[1],ax=ax)
    ax.set_title(formula)
    ticks = []
    for i, cond in enumerate(order):
        if i == 0:
            tick = cond
        else:
            tick = f"{cond} \n {dict_pval_stars[cond]} \n {dict_coefs[cond]}"
        ticks.append(tick)
    ax.set_xticks(range(df[predictor].unique().size))
    ax.set_xticklabels(ticks)
    plt.show()
    
    return mdf


def confidence_interval(x, confidence = 0.95, verbose = False):
    m = x.mean() 
    s = x.std() 
    dof = x.size-1 
    t_crit = np.abs(stats.t.ppf((1-confidence)/2,dof))
    ci = (m-s*t_crit/np.sqrt(len(x)), m+s*t_crit/np.sqrt(len(x))) 
    if verbose:
        print(f'm : {round(m, 3)} , std : {round(s,3)} , ci : [{round(ci[0],3)};{round(ci[1],3)}]')
    return ci




def get_stats_df(df, predictor, outcome, subject=None, design='within'):

    N = df[predictor].value_counts()[0]
    groups = list(df[predictor].unique())
    ngroups = len(groups)
    
    parametricity_pre_transfo = parametric(df, predictor, outcome, subject)
    parametricity = parametricity_pre_transfo
    
    tests = guidelines(df, predictor, outcome, design, parametricity)
    
    pre_test = tests['pre']
    post_test = tests['post']
    results = pg_compute_pre(df, predictor, outcome, pre_test, subject)
    pval = round(results['p'], 4)

    return pval



def pg_compute_pre_full_res(df, predictor, outcome, test, subject=None, show = False):
    
    pval_labels = {'t-test_ind':'p-val','t-test_paired':'p-val','anova':'p-unc','rm_anova':'p-unc','Mann-Whitney':'p-val','Wilcoxon':'p-val', 'Kruskal':'p-unc', 'friedman':'p-unc'}
    esize_labels = {'t-test_ind':'cohen-d','t-test_paired':'cohen-d','anova':'np2','rm_anova':'np2','Mann-Whitney':'CLES','Wilcoxon':'CLES', 'Kruskal':None, 'friedman':None}
    
    if test == 't-test_ind':
        groups = list(set(df[predictor]))
        pre = df[df[predictor] == groups[0]][outcome]
        post = df[df[predictor] == groups[1]][outcome]
        res = pg.ttest(pre, post, paired=False)
        
    elif test == 't-test_paired':
        groups = list(set(df[predictor]))
        pre = df[df[predictor] == groups[0]][outcome]
        post = df[df[predictor] == groups[1]][outcome]
        res = pg.ttest(pre, post, paired=True)
        
    elif test == 'anova':
        res = pg.anova(dv=outcome, between=predictor, data=df, detailed=False, effsize = 'np2')
    
    elif test == 'rm_anova':
        res = pg.rm_anova(dv=outcome, within=predictor, data=df, detailed=False, effsize = 'np2', subject = subject)
        
    elif test == 'Mann-Whitney':
        groups = list(set(df[predictor]))
        x = df[df[predictor] == groups[0]][outcome]
        y = df[df[predictor] == groups[1]][outcome]
        res = pg.mwu(x, y)
        
    elif test == 'Wilcoxon':
        groups = list(set(df[predictor]))
        x = df[df[predictor] == groups[0]][outcome]
        y = df[df[predictor] == groups[1]][outcome]
        res = pg.wilcoxon(x, y)
        
    elif test == 'Kruskal':
        res = pg.kruskal(data=df, dv=outcome, between=predictor)
        
    elif test == 'friedman':
        res = pg.friedman(data=df, dv=outcome, within=predictor, subject=subject)
    
    pval = res[pval_labels[test]].values[0]
    es_label = esize_labels[test]
    if es_label is None:
        es = None
    else:
        es = res[es_label].values[0]
    
    es_interp = es_interpretation(es_label, es)
    results = {'p':pval, 'es':es, 'es_label':es_label, 'es_interp':es_interp}
      
    return res


def get_df_stats_pre(df, predictor, outcome, subject=None, design='within', transform=False, verbose=True):

    if isinstance(predictor, str):

        parametricity_pre_transfo = parametric(df, predictor, outcome, subject)

        if transform:
            if not parametricity_pre_transfo:
                df = transform_data(df, outcome)
                parametricity_post_transfo = parametric(df, predictor, outcome, subject)
                parametricity = parametricity_post_transfo
                if verbose:
                    if parametricity_post_transfo:
                        print('Successfull transformation')
                    else:
                        print('Un-successfull transformation')
            else:
                parametricity = parametricity_pre_transfo
        else:
            parametricity = parametricity_pre_transfo

        tests = guidelines(df, predictor, outcome, design, parametricity)

        pre_test = tests['pre']
        post_test = tests['post']
        results = pg_compute_pre_full_res(df, predictor, outcome, pre_test, subject)
        # pval = round(results['p'], 4)

        if not post_test is None:
            post_hoc = pg_compute_post_hoc(df, predictor, outcome, post_test, subject)

    results = results.reset_index()
    results = results.rename(columns={'index' : 'test'})
    results = results[['test', 'alternative', 'p-val']]
    results['outcome'] = outcome

    predictor_group = df[predictor].unique()

    for predictor_i in predictor_group:

        results[predictor_i] = df.query(f"{predictor} == '{predictor_i}'")[outcome].mean()

    return results


# df = load_respi_stat_df()
# df = load_respi_stat_df().query(f"cond == 'MECA'")
#df, predictor, outcome, subject, design, transform, verbose, export_param_test =  df, 'session', 'BF', 'sujet', 'within', False
def get_summary_stats(df, predictor, outcome, subject=None, design='within', export_param_test=False):

    #### one independant variable

    if isinstance(predictor, str):

        #### clean pred and outcome for formula
        for _str in ['+', '-', '/']:
            if outcome.find(_str) != -1:
                raise ValueError('rename outcome to supress + / - string')
        for _str in ['+', '-', '/']:
            if predictor.find(_str) != -1:
                raise ValueError('rename outcome to supress + / - string')

        #### parametricity

        n_group = df[predictor].unique().size

        if n_group <= 2:

            #### identify normality of every groups
            df_parametricity = pg.normality(data=df, dv=outcome, group=predictor, method='normaltest') # Shapiro Wilk
            df_parametricity.insert(0, 'param_test', ['shapiro_wilk']*df_parametricity.shape[0], True)
            
            if sum(df_parametricity['normal']) == df_parametricity['normal'].size:
                parametricity = True
            else:
                parametricity = False

        if n_group > 2:

            #### 1) identify normality of residuals
            formula = f'{outcome} ~ {predictor}'

            model = smf.ols(formula, data=df).fit()
            residuals = model.resid
            df_parametricity = pg.normality(residuals, method='normaltest') # Shapiro Wilk
            df_parametricity.insert(0, 'param_test', ['shapiro']*df_parametricity.shape[0], True)
            df_parametricity = df_parametricity.rename(columns={'normal' : 'parametric'})

            #### 2) identify homoscedasticity
            _df_parametricity = pg.homoscedasticity(data=df, dv=outcome, group=predictor, method='levene').reset_index(drop=True) # Levene test
            _df_parametricity.insert(0, 'param_test', ['levene']*df_parametricity.shape[0], True)
            _df_parametricity = _df_parametricity.rename(columns={'equal_var' : 'parametric'})

            df_parametricity = pd.concat([df_parametricity, _df_parametricity])

            #### 3) idependance, to check on the protocol

            #### Final decision

            if sum(df_parametricity['parametric']) == 2:
                parametricity = True
            else:
                parametricity = False 
        
        # parametricity_pre_transfo = parametric(df, predictor, outcome, subject)
        
        # if transform:
        #     if not parametricity_pre_transfo:
        #         df = transform_data(df, outcome)
        #         parametricity_post_transfo = parametric(df, predictor, outcome, subject)
        #         parametricity = parametricity_post_transfo
        #         if verbose:
        #             if parametricity_post_transfo:
        #                 print('Successfull transformation')
        #             else:
        #                 print('Un-successfull transformation')
        #     else:
        #         parametricity = parametricity_pre_transfo
        # else:
        #     parametricity = parametricity_pre_transfo

        #### test identification

        tests = guidelines(df, predictor, outcome, design, parametricity)

        pre_test = tests['pre']
        post_test = tests['post']
        post_name_test = tests['post_name_corrected']

        # pval_labels = {'t-test_ind':'p-val','t-test_paired':'p-val','anova':'p-unc','rm_anova':'p-unc','Mann-Whitney':'p-val','Wilcoxon':'p-val', 'Kruskal':'p-unc', 'friedman':'p-unc'}
        # esize_labels = {'t-test_ind':'cohen-d','t-test_paired':'cohen-d','anova':'np2','rm_anova':'np2','Mann-Whitney':'CLES','Wilcoxon':'CLES', 'Kruskal':None, 'friedman':None}

        #### tests < 2 groups

        groups = list(set(df[predictor]))

        if len(groups) <= 2:
            A_data = df[df[predictor] == groups[0]][outcome]
            B_data = df[df[predictor] == groups[1]][outcome]

            if pre_test == 't-test_ind':
                res = pg.ttest(A_data, B_data, paired=False)

                stat_test = f"T_{np.round(res['T'].values[0], 5)}"
                alternative = res['alternative'].values[0]
                pval = np.round(res['p-val'].values[0], 5)
                cohen_d = np.round(res['cohen-d'].values[0], 5)

            elif pre_test == 't-test_paired':
                res = pg.ttest(A_data, B_data, paired=True)

                stat_test = f"T_{np.round(res['T'].values[0], 5)}"
                alternative = res['alternative'].values[0]
                pval = np.round(res['p-val'].values[0], 5)
                cohen_d = np.round(res['cohen-d'].values[0], 5)

            elif pre_test == 'Mann-Whitney':
                res = pg.mwu(A_data, B_data)

                stat_test = f"U_{np.round(res['U-val'].values[0], 5)}"
                alternative = res['alternative'].values[0]
                pval = np.round(res['p-val'].values[0], 5)
                cohen_d = None
                
            elif pre_test == 'Wilcoxon':
                res = pg.wilcoxon(A_data, B_data)

                stat_test = f"W_{np.round(res['W-val'].values[0], 5)}"
                alternative = res['alternative'].values[0]
                pval = np.round(res['p-val'].values[0], 5)
                cohen_d = None

            stats_descriptives = {'A' : [groups[0]], 'B' : [groups[1]], 
                        'A_N' : [A_data.size], 'B_N' : [B_data.size],           
                        'mean(A)' : [A_data.mean()], 'std(A)' : [A_data.std()], 'mean(B)' : [B_data.mean()], 'std(B)' : [B_data.std()],
                        'median(A)' : [A_data.median()], 'mad(A)' : [stats.median_abs_deviation(A_data)], 'A_Q1' : [np.percentile(A_data, 25)], 'A_Q3' : [np.percentile(A_data, 75)], 
                        'median(B)' : [B_data.median()], 'mad(B)' : [stats.median_abs_deviation(B_data)], 'B_Q1' : [np.percentile(B_data, 25)], 'B_Q3' : [np.percentile(B_data, 75)]}
            
            res['predictor'] = predictor
            res['outcome'] = outcome
            res['pre_test_name'] = pre_test
            res['stat_test'] = stat_test
            res['alternative'] = alternative
            res['pre_pval'] = pval
            res['cohen_d'] = cohen_d
            res['param'] = parametricity
            df_res = res.reindex(columns=['predictor', 'outcome', 'param', 'pre_test_name', 'stat_test', 'alternative', 'pre_pval', 'cohen_d'])
            df_res = pd.concat([df_res.reset_index(drop=True), pd.DataFrame(stats_descriptives)], axis=1)

        #### more than 2 groups

        if len(groups) > 2:

            #### pre tests
            if pre_test == 'anova':
                res = pg.anova(dv=outcome, between=predictor, data=df, detailed=False, effsize = 'np2')

                stat_test = f"F_{np.round(res['F'].values[0], 5)}"
                alternative = None
                pval = np.round(res['p-unc'].values[0], 5)
                cohen_d = None

            elif pre_test == 'rm_anova':
                res = pg.rm_anova(dv=outcome, within=predictor, data=df, detailed=False, effsize = 'np2', subject = subject)

                stat_test = f"F_{np.round(res['F'].values[0], 5)}"
                alternative = None
                pval = np.round(res['p-unc'].values[0], 5)
                cohen_d = None
                
            elif pre_test == 'Kruskal':
                res = pg.kruskal(data=df, dv=outcome, between=predictor)

                stat_test = f"H_{np.round(res['H'].values[0], 5)}"
                alternative = None
                pval = np.round(res['p-unc'].values[0], 5)
                cohen_d = None
                
            elif pre_test == 'friedman':
                res = pg.friedman(data=df, dv=outcome, within=predictor, subject=subject)

                stat_test = f"W_{np.round(res['W'].values[0], 5)}"
                alternative = None
                pval = np.round(res['p-unc'].values[0], 5)
                cohen_d = None

            #### post tests
            post_hoc = pg_compute_post_hoc(df, predictor, outcome, post_test, subject)

            post_hoc['outcome'] = outcome
            post_hoc['pre_test_name'] = np.array([pre_test] * post_hoc.shape[0])
            post_hoc['stat_test'] = np.array([stat_test] * post_hoc.shape[0])
            post_hoc['alternative'] = np.array([alternative] * post_hoc.shape[0])
            post_hoc['pre_pval'] = np.array([pval] * post_hoc.shape[0])
            post_hoc['cohen_d'] = np.array([cohen_d] * post_hoc.shape[0])
            post_hoc['post_test_name'] = np.array([post_name_test] * post_hoc.shape[0])
            post_hoc['param'] = np.array([parametricity] * post_hoc.shape[0])

            post_hoc = post_hoc.rename(columns={'Contrast' : 'predictor'})

            df_post_hoc = post_hoc.reindex(columns=['predictor', 'outcome', 'param', 'pre_test_name', 'stat_test', 'alternative', 'pre_pval', 'cohen_d', 'post_test_name', 'A', 'B', 'mean(A)', 'std(A)', 'mean(B)', 'std(B)', 'p-unc', 'p-corr', 'p-adjust'])

            _dict_median = {'median(A)' : [], 'median(B)' : [], 'mad(A)' : [], 'mad(B)' : [], 'A_Q1' : [], 'B_Q1' : [], 'A_Q3' : [], 'B_Q3' : [], 'A_N' : [], 'B_N' : []}

            for row_i in range(df_post_hoc.shape[0]):
                A_data, B_data = df[df[predictor] == df_post_hoc.iloc[row_i,:]['A']][outcome], df[df[predictor] == df_post_hoc.iloc[row_i,:]['B']][outcome]
                A_median, B_median, A_mad, B_mad = A_data.median(), B_data.median(), stats.median_abs_deviation(A_data), stats.median_abs_deviation(B_data)
                A_Q1, B_Q1, A_Q3, B_Q3 = np.percentile(A_data, 25), np.percentile(B_data, 25), np.percentile(A_data, 75), np.percentile(B_data, 75)
                _dict_median['median(A)'].append(A_median), _dict_median['median(B)'].append(B_median), _dict_median['mad(A)'].append(A_mad), _dict_median['mad(B)'].append(B_mad)
                _dict_median['A_Q1'].append(A_Q1), _dict_median['B_Q1'].append(B_Q1), _dict_median['A_Q3'].append(A_Q3), _dict_median['B_Q3'].append(B_Q3)
                _dict_median['A_N'].append(A_data.size), _dict_median['B_N'].append(B_data.size)

            df_median = pd.DataFrame(_dict_median)
            df_post_hoc = pd.concat([df_post_hoc, df_median], axis=1)

            df_post_hoc = df_post_hoc.reindex(columns=['predictor', 'outcome', 'param', 'pre_test_name', 'stat_test', 'alternative', 'pre_pval','cohen_d', 'post_test_name', 'A', 'B', 
                                                        'A_N', 'mean(A)', 'std(A)', 'B_N', 'mean(B)', 'std(B)',
                                                        'median(A)', 'mad(A)', 'A_Q1', 'A_Q3', 'median(B)', 'mad(B)', 'B_Q1',  'B_Q3', 
                                                        'p-unc', 'p-corr', 'p-adjust'])

            df_res = df_post_hoc.rename(columns={'p-unc': 'post_p_unc', 'p-corr': 'post_p_corr', 'p-adjust': 'post_p_adjust'})
            df_res = df_res.rename(columns={})
            
    #### several independant variable

    elif isinstance(predictor, list):

        #### parametricity

        #### 1) identify normality of residuals
        formula = f'{outcome} ~ {predictor[0]} + {predictor[1]} + {predictor[0]}:{predictor[1]}'

        model = smf.ols(formula, data=df).fit()
        residuals = model.resid
        df_parametricity = pg.normality(residuals, method='normaltest') # Shapiro Wilk
        df_parametricity.insert(0, 'param_test', ['shapiro']*df_parametricity.shape[0], True)
        df_parametricity = df_parametricity.rename(columns={'normal' : 'parametric'})

        #### 2) identify homoscedasticity
        df['interaction'] = df[predictor[0]] + df[predictor[1]]

        _df_parametricity = pg.homoscedasticity(data=df, dv=outcome, group='interaction', method='levene').reset_index(drop=True)
        _df_parametricity.insert(0, 'param_test', ['levene']*df_parametricity.shape[0], True)
        _df_parametricity = _df_parametricity.rename(columns={'equal_var' : 'parametric'})

        df_parametricity = pd.concat([df_parametricity, _df_parametricity])

        #### 3) idependance, to check on the protocol

        #### Final decision

        if sum(df_parametricity['parametric']) == 2:
            parametricity = True
        else:
            parametricity = False 

        if parametricity == False:
            print(df_parametricity)
            raise ValueError('Non parametric test for 2 factors, two way ANOVA not possible')

        #### tests

        if design == 'within':
            test_type = 'two_way_rm_anova'
            test = pg.rm_anova(data=df, dv=outcome, within = predictor, subject = subject, effsize = 'np2').set_index('Source').round(3)
            
        elif design == 'between':
            test_type = 'two_way_anova'
            test = pg.anova(data=df, dv=outcome, between = predictor).set_index('Source').round(3)

        df_res = test.reset_index(drop=False)
        df_res['predictor'] = np.array([f"{predictor[0]}_{predictor[1]}"] * df_res.shape[0])
        df_res['outcome'] = np.array([outcome] * df_res.shape[0])
        df_res['test_name'] = np.array([test_type] * df_res.shape[0])
        df_res['param'] = np.array([parametricity] * df_res.shape[0])
        df_res = df_res.reindex(columns=np.concatenate([df_res.columns[-3:], df_res.columns[:-3]]))

    if export_param_test:
        return df_res, df_parametricity
    else:
        return df_res



def load_respi_stat_df():

    os.chdir(os.path.join(path_data, 'respi_detection'))
    df_respi_paris = pd.read_excel('OLFADYS_alldata_mean.xlsx').query(f"sujet in {sujet_list.tolist()}").reset_index(drop=True)

    for row_i in range(df_respi_paris.shape[0]):
        if df_respi_paris.iloc[row_i]['odor'] == 'p':
            df_respi_paris['odor'][row_i] = '+'
        if df_respi_paris.iloc[row_i]['odor'] == 'n':
            df_respi_paris['odor'][row_i] = '-'

    df_respi_paris = df_respi_paris.rename(columns={"odor": "session"})

    sujet_sel_mask = []

    for row_i in range(df_respi_paris.shape[0]):

        if df_respi_paris['sujet'].iloc[row_i] in sujet_best_list_rev:

            sujet_sel_mask.append('YES')

        else:

            sujet_sel_mask.append('NO')

    df_respi_paris['select_best'] = sujet_sel_mask

    nan_count = 0

    for respi_metric in ['TI', 'Te', 'Ttot', 'BF', 'VT', 'Ve', 'VT_Ti', 'Ti_Ttot', 'PRESS', 'PetCO2']:

        for nan_i in np.where(df_respi_paris[respi_metric].isnull().values)[0]:

            odor = df_respi_paris['session'][nan_i]
            cond = df_respi_paris['cond'][nan_i]
            nan_replace = df_respi_paris.query(f"session == '{odor}' and cond == '{cond}'")[respi_metric].median()
            df_respi_paris[respi_metric][nan_i] = nan_replace 

            nan_count += 1

    print(f"{nan_count} replaced")
    df_respi_paris

    return df_respi_paris