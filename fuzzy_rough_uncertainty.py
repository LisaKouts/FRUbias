import numpy as np
import re
import os
import pandas as pd
import pickle

def fru_single_feature(full, prot, decision_class):
    '''
    Parameters
    ----------
    full: membership values of instances to the regions using the full set of features,
    prot: membership values of instances to the regions using the set of features without including the protected feature
    decision_class: index of the decision class
    Returns
    -------
    FRU-value attached to the specified decision class for the suppressed protected attribute
    '''
    
    _, _, BND_full = full
    _, _, BND_prot = prot

    # older approach
    #diff_prot = BND_prot[decision_class] - BND_full[decision_class]
    #diff_prot = np.where(diff_prot < 0, 0, diff_prot)
    #from numpy import linalg as la
    #round((float(la.norm(diff_prot) / la.norm(BND_prot[decision_class]))),2)

    return (np.sum((BND_prot[decision_class] - BND_full[decision_class])**2))**0.5

def fru_all_feature(FRU_gt, decision_class, columns, prots, path = '.', l1_norm = False,  
                    setting = '', prot = None, save = False):
    '''
    Parameters
    ----------
    FRU_gt : pandas DataFrame
        Either empty dataframe or dataframe to append 
    prots : list of strings
        list of protected features
    
    Returns
    -------
    FRU-value attached to the specified decision class for the protected attribute that was removed
    '''
    
    # load membership values
    mem_dic = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if 'pickle' in name:
                att_name = re.findall(r"(.*?)_mem.pickle", name)[0] # there is an issue when there _ in name, only picks first word, yikes
                file_path_name = os.path.join(root, name) 
                with open(file_path_name, 'rb') as handle: 
                    mem_dic_att = pickle.load(handle)
                    mem_dic[att_name] = mem_dic_att

    for col in columns:
        # l2 -----------------------
        
        BND_diff = mem_dic[col][-1][decision_class] - mem_dic['full'][-1][decision_class]
        FRU_gt.loc[col,'FRU_raw_l2'+setting] = ((np.sum(BND_diff**2))**0.5)
        FRU_gt.loc[col,'FRU_l2_N'+setting] = ((np.sum(BND_diff**2))**0.5)/(mem_dic[col][-1][0].shape[0]**0.5) # l2 norm
        # l1 -----------------------
        if l1_norm:
            BND_abs_diff = np.abs(mem_dic[col][-1][decision_class] - mem_dic['full'][-1][decision_class])
            FRU_gt.loc[col,'FRU_raw_l1'+setting] = np.sum(BND_abs_diff)
            FRU_gt.loc[col,'FRU_l1_N'+setting] = np.sum(BND_abs_diff)/(mem_dic[col][-1][0].shape[0]) # l1 norm

    FRU_gt.loc[:,'FRU_l2_norm'+setting] = (FRU_gt['FRU_raw_l2'+setting] / FRU_gt['FRU_raw_l2'+setting].sum())
    
    if l1_norm:
        FRU_gt.loc[:,'FRU_l1_norm'+setting] = (FRU_gt['FRU_raw_l1'+setting] / FRU_gt['FRU_raw_l1'+setting].sum())

    # save
    if save:
        with pd.ExcelWriter('results_synth.xlsx',mode='w') as writer:
            FRU_gt.T.to_excel(writer, sheet_name='FRU_feature')


    # Group FRU
    # for prot in prots:


    # return mem_dic, FRU_gt