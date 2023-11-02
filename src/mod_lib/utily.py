# -*- coding: utf-8 -*-
import pandas as pd
import pickle


def train_test_split_panel(panel,vId='KGB',test_size=0.01,random_state=100):
    
    """
     Split panel data in training and test set while keeping all observations
     from one id together 

     Argument:
         panel -- pandas dataframe with multi index year and vId
         test_size -- Fraction of test size 
         vId -- name of id series
         random_state -- random number seed
         
     Returns:
         df_train -- training data
         df_test -- Test data
    """
    #%% TODO this is not unit tested
    
    # Get all ids
    ids_all = pd.DataFrame(panel.index.get_level_values(vId).unique())
    # Obtain a sample of ids
    ids_test = ids_all.sample(frac=test_size,random_state=random_state)
    
    
    # split test and train data
    df_test = panel.loc[(slice(None),list(ids_test[vId].values)),:]
    df_train = panel[~panel.index.isin(df_test.index)]
    
    # Check that kgb_test and kgb_train do not intersect
    assert pd.concat([df_test, df_train], axis=1, join='inner').empty == True
        
    return df_train, df_test
    
def save_obj(obj, name):
    """
    Save object to pickle
    """
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """
    Load object from pickle
    """
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def findIndex(searchList, elem):
    """
    Find index of multiple items in a list in another list
    searchList: List in which indices should be found
    elem: List of elements for which the index is desired
    """
    
    
    a = []
    for e in elem:
        for i, x in enumerate(searchList):
            if x == e:
                a.append(i)
    return a


