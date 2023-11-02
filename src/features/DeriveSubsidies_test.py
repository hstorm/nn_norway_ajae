#%%
import matplotlib
matplotlib.use('Agg') # This forces matplotlib to not open windows for figures
from numpy.random import seed
seed(1) # set seed at very beginning, best practice

import unittest
import sys
import numpy as np
from importlib import reload

sys.path.append("src/models")
sys.path.append("src/features")

import luigi_features
reload(luigi_features)
from luigi_features import TestTrainSplit_unitTest

from multiprocessing import Pool
#from multiprocessing.dummy import Pool

import time
import pandas as pd


from calc_dpay import SubsidyScheme

from calc_dpay import loadSubsidySchemeFiles
from calc_dpay import DeriveSubsidies

import pickle

def nan_equal(a,b):
    """
    Comparing numpy arrays containing NaN
    Based on https://stackoverflow.com/questions/10710328/comparing-numpy-arrays-containing-nan
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.testing.assert_array_equal.html
    """
    try:
        np.testing.assert_equal(a,b)
    except AssertionError:
        return False
    return True

def load_obj(name):
        with open(name, 'rb') as f:
            return pickle.load(f)
#%%
class DeriveSubsidies_test(unittest.TestCase):
    
       
    def test_subDUnit(self):
        
        #%% Load unitTest data
        df, df_test = load_obj("C:/temp/storm/nn_norway/data/processed/train_test_split_unit_test.pkl")  
        
        #%% Load subsidy scheme data
        bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()
        
        pool = Pool()
        #%%
        t0 = 2004
        t1 = t0+1
        sattrint0 =  sattrin[t0]
        bunnt0 = bunn[bunn.year == t0]
        makst0 = maks[maks.year == t0]
        grovt0 = grov[grov.year == t0]
        
        kgb = df.index.get_level_values('KGB').unique()[0]
        dd = df.loc[(str(t0),slice(kgb,kgb)),:]
        
        
        
        #%% get subsidy scheme
        deriveSub = DeriveSubsidies(t0,
                                sattrin[t0],
                                bunn[bunn.year == t0],
                                maks[maks.year == t0],
                                grov[grov.year == t0],
                                t1,
                                sattrin[t1],
                                bunn[bunn.year == t1],
                                maks[maks.year == t1],
                                grov[grov.year == t1]
                                )
         
        deriveSub.pool = pool
        
        #%% get subsidy scheme
        subScheme = SubsidyScheme(t0,
                          sattrint0,
                          bunnt0,
                          makst0,
                          grovt0
                          )
        
        #%% Consider unit increase by 1
        pay = deriveSub.subDUnit(dd, t0,
                 sattrint0, bunnt0, makst0, grovt0,
                 unitIncrease=1,
                 setZero = False,
                 currentSub=pd.DataFrame(),
                 usePool=True)
        
        for vAct in deriveSub.vDiffAct:
            dd0 = dd.copy()
            dd1 = dd.copy()
            dd1[vAct] += 1.
    
            sub0 = subScheme.sub(dd0,onlyDpay=True)
            sub1 = subScheme.sub(dd1,onlyDpay=True)
            
            self.assertTrue(np.isclose(pay[vAct],sub1-sub0, equal_nan=True))
            
        #%% Consider unit increase by 10
        pay = deriveSub.subDUnit(dd, t0,
                 sattrint0, bunnt0, makst0, grovt0,
                 unitIncrease=10,
                 setZero = False,
                 currentSub=pd.DataFrame(),
                 usePool=True)
        
        for vAct in deriveSub.vDiffAct:
            dd0 = dd.copy()
            dd1 = dd.copy()
            dd1[vAct] += 10.
    
            sub0 = subScheme.sub(dd0,onlyDpay=True)
            sub1 = subScheme.sub(dd1,onlyDpay=True)
            
            self.assertTrue(np.isclose(pay[vAct],sub1-sub0, equal_nan=True))
        #%% Consider  set zero
        pay = deriveSub.subDUnit(dd, t0,
                 sattrint0, bunnt0, makst0, grovt0,
                 unitIncrease=0,
                 setZero = True,
                 currentSub=pd.DataFrame(),
                 usePool=True)
        
        for vAct in deriveSub.vDiffAct:
            dd0 = dd.copy()
            dd1 = dd.copy()
            dd1[vAct] = 0
    
            sub0 = subScheme.sub(dd0,onlyDpay=True)
            sub1 = subScheme.sub(dd1,onlyDpay=True)
            
            self.assertTrue(np.isclose(pay[vAct],sub1-sub0, equal_nan=True))
            
        pool.close() 
        pool.join()
        
    def test_avgSub(self):
        
        #%% Load unitTest data
        df, df_test = load_obj("C:/temp/storm/nn_norway/data/processed/train_test_split_unit_test.pkl")  
        
        #%% Load subsidy scheme data
        bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()
        
        pool = Pool()
        #%%
        t0 = 2004
        t1 = t0+1
        sattrint0 =  sattrin[t0]
        bunnt0 = bunn[bunn.year == t0]
        makst0 = maks[maks.year == t0]
        grovt0 = grov[grov.year == t0]
        
        kgb = df.index.get_level_values('KGB').unique()[0]
        dd = df.loc[(str(t0),slice(kgb,kgb)),:]
        
        
        #%% get subsidy scheme
        deriveSub = DeriveSubsidies(t0,
                                sattrin[t0],
                                bunn[bunn.year == t0],
                                maks[maks.year == t0],
                                grov[grov.year == t0],
                                t1,
                                sattrin[t1],
                                bunn[bunn.year == t1],
                                maks[maks.year == t1],
                                grov[grov.year == t1]
                                )
         
        deriveSub.pool = pool
        
        #%% get subsidy scheme
        subScheme = SubsidyScheme(t0,
                          sattrint0,
                          bunnt0,
                          makst0,
                          grovt0
                          )
        
        #%% Consider set zero
        
        vAct = 'SAU'
        dd0 = dd.copy()
        dd0[vAct] = 10
        dd1 = dd.copy()
        dd1[vAct] = 0
        
        pay = deriveSub.avgSub(dd0,t0,sattrint0,bunnt0,makst0,grovt0,currentSub=pd.DataFrame())
        
        sub0 = subScheme.sub(dd0,onlyDpay=True)
        sub1 = subScheme.sub(dd1,onlyDpay=True)
        
        self.assertTrue(np.isclose(pay[vAct],(sub0-sub1)/10, equal_nan=True))
        
        #%% Check change in davg subsidiy
        pay = deriveSub.get_dAvgSub(dd0,currentSub=pd.DataFrame())

        subSchemet0 = SubsidyScheme(t0,
                                sattrin[t0],
                                bunn[bunn.year == t0],
                                maks[maks.year == t0],
                                grov[grov.year == t0],
                          )
        subSchemet1 = SubsidyScheme(t1,
                                sattrin[t1],
                                bunn[bunn.year == t1],
                                maks[maks.year == t1],
                                grov[grov.year == t1]
                                )

        sub0_t0 = subSchemet0.sub(dd0,onlyDpay=True)
        sub1_t0 = subSchemet0.sub(dd1,onlyDpay=True)
        
        sub0_t1 = subSchemet1.sub(dd0,onlyDpay=True)
        sub1_t1 = subSchemet1.sub(dd1,onlyDpay=True)
        
        dsubt0 = (sub0_t0-sub1_t0)/10
        dsubt1 = (sub0_t1-sub1_t1)/10
        
        self.assertTrue(np.isclose(pay[vAct],dsubt1-dsubt0, equal_nan=True))


        #%%            
        pool.close() 
        pool.join()
        
       #%%
if __name__ == '__main__':
    unittest.main()