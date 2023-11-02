#%%
import matplotlib
matplotlib.use('Agg') # This forces matplotlib to not open windows for figures
from numpy.random import seed
seed(1) # set seed at very beginning, best practice

import unittest
import sys
import numpy as np
from importlib import reload

#from keras import backend as K

sys.path.append("src/models")
sys.path.append("src/features")

import luigi_features
reload(luigi_features)
from luigi_features import TestTrainSplit_unitTest



#import Model
#reload(Model)
#from Model import Model

#import Predict
#reload(Predict)
#from Predict import Predict

import time
#import run_unitTest
#reload(run_unitTest)
#from run_unitTest import run_unitTest

import pandas as pd

import Model
reload(Model)
from Model import Model

from calc_dpay import SubsidyScheme

from calc_dpay import loadSubsidySchemeFiles
from calc_dpay import DeriveSubsidies

#from make_dataset import BlancePanel
#from make_dataset import ExtractFeatures

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

#%%
class SubsidyScheme_test(unittest.TestCase):
    
       
    def test_dpay_TPROD(self):
        """
        Test TPROD
        """
        
        #%% Load unitTest data
        model = Model()
        model.random_state = 42
        df, df_test = model.load_data(TestTrainSplit_unitTest.targetFileName)  
        
        #%% Load subsidy scheme data
        bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()
        
        #%%
        t0 = 2004
        sattrint0 =  sattrin[t0]
        bunnt0 = bunn[bunn.year == t0]
        makst0 = maks[maks.year == t0]
        grovt0 = grov[grov.year == t0]
        
        
        #%% get subsidy scheme
        subScheme = SubsidyScheme(t0,
                          sattrint0,
                          bunnt0,
                          makst0,
                          grovt0
                          )
        #%% get single farm
        kgb = df.index.get_level_values('KGB').unique()[0]
        dd = df.loc[(str(t0),slice(kgb,kgb)),:]
        
        #%%
        dd.reset_index(inplace=True)
        dd.set_index(['year','KGB'],inplace=True)
        
        #%% create farm with nothing except Set Zones
        ftest = dd*0
        ftest['zoneTAKTL'] = 2.0000
        ftest['zoneTDISE'] = 1.0000
        ftest['zoneTDISG'] = 2.0000
        ftest['zoneTDISK'] = 1.0000
        ftest['zoneTDISM'] = 1.0000
        ftest['zoneTDMLK'] = 1.0000
        ftest['zoneTPROD'] = 2.0000
        
        #%% Test steps in TPROD SAU
        # Satser
        #2004.TPROD.SAU .1  0      591   591   591
        #2004.TPROD.SAU .2  0      134   134   134
        #2004.TPROD.SAU .3  0      35    35    35
        #2004.TPROD.SAU .4  0      0     0     0
        # Trin
        #2004.TPROD.SAU     75     200    300    INF  

        ftest['SAU'] = 10 #%% Set Sau
        TPROD_SAU_2004 = 591*ftest['SAU']/1000
        pay = subScheme.sub(ftest)
        self.assertTrue(np.isclose(TPROD_SAU_2004, pay['TPROD'], equal_nan=True))
        
        #%%
        ftest['SAU'] = 100 #%% Set Sau
        TPROD_SAU_2004 = 591*75/1000
        TPROD_SAU_2004 += 134*25/1000
        pay = subScheme.sub(ftest)
        self.assertTrue(np.isclose(TPROD_SAU_2004, pay['TPROD'], equal_nan=True))
        
        #%%
        ftest['SAU'] = 500 #%% Set Sau
        TPROD_SAU_2004 = 591*75/1000
        TPROD_SAU_2004 += 134*125/1000
        TPROD_SAU_2004 += 35*100/1000
        pay = subScheme.sub(ftest)
        self.assertTrue(np.isclose(TPROD_SAU_2004, pay['TPROD'], equal_nan=True))
        
        #%%
        ftest['SAU'] = 500 #%% Set Sau
        TPROD_SAU_2004 = 591*75/1000
        TPROD_SAU_2004 += 134*125/1000
        TPROD_SAU_2004 += 35*100/1000
        pay = subScheme.sub(ftest)
        self.assertTrue(np.isclose(TPROD_SAU_2004, pay['TPROD'], equal_nan=True))
        
        #%% Check that TPROD x120 is correct
        
        #2004.TPROD.120 .1  0      3330  3330  3330
        #2004.TPROD.120 .2  0      2000  2000  2000
        #2004.TPROD.120 .3  0      1000  1000  1000
        #2004.TPROD.120 .4  0      0     0     0
        # Trin
        #2004.TPROD.120     16     25     50     INF  
        # MAKSSAT limit
        #2004.TPROD     150000  
        
        ftest['SAU'] = 0 
        ftest['x120'] = 500

        TPROD_x120_2004 = 3330*16/1000
        TPROD_x120_2004 += 2000*9/1000
        TPROD_x120_2004 += 1000*25/1000
        
        pay = subScheme.sub(ftest)
       
        self.assertTrue(np.isclose(TPROD_x120_2004, pay['TPROD'], equal_nan=True))
        
        #%% Test Maksstat
        ftest['SAU'] = 500 
        ftest['x120'] = 500 
        TPROD_SAU_2004 = 591*75/1000
        TPROD_SAU_2004 += 134*125/1000
        TPROD_SAU_2004 += 35*100/1000
        
        TPROD_x120_2004 = 3330*16/1000
        TPROD_x120_2004 += 2000*9/1000
        TPROD_x120_2004 += 1000*25/1000
        
        pay = subScheme.sub(ftest)
        
        maksCap = 150000./1000
        
        aa = np.minimum(maksCap,TPROD_SAU_2004+TPROD_x120_2004)
        self.assertTrue(np.isclose(aa, pay['TPROD'], equal_nan=True))
        
        
        #%% Test TPROD zones 
        # Satser
        #2004.TPROD.155 .1  0      832   832   1122
        #2004.TPROD.155 .2  0      832   832   832
        # Trin
        #2004.TPROD.155     25     70     INF   
        ftest['SAU'] = 0 
        ftest['x120'] = 0 
        ftest['x155'] = 50 
        ftest['zoneTPROD'] = 2.0000

        TPROD_x155_2004 = 832*25/1000
        TPROD_x155_2004 += 832*25/1000
        
        pay = subScheme.sub(ftest)
        
        aa = np.minimum(maksCap,TPROD_x155_2004)
        self.assertTrue(np.isclose(aa, pay['TPROD'], equal_nan=True))
        
        # Test zone 3
        ftest['zoneTPROD'] = 3.0000

        TPROD_x155_2004 = 1122*25/1000
        TPROD_x155_2004 += 832*25/1000
        
        pay = subScheme.sub(ftest)
        
        aa = np.minimum(maksCap,TPROD_x155_2004)
        self.assertTrue(np.isclose(aa, pay['TPROD'], equal_nan=True))
    
    def test_dpay_TVELF(self):
        """
        Test TVELF
        """
        
        #%% Load unitTest data
        model = Model()
        model.random_state = 42
        df, df_test = model.load_data(TestTrainSplit_unitTest.targetFileName)  
        
        #%% Load subsidy scheme data
        bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()
        
        #%%
        t0 = 2004
        sattrint0 =  sattrin[t0]
        bunnt0 = bunn[bunn.year == t0]
        makst0 = maks[maks.year == t0]
        grovt0 = grov[grov.year == t0]
        
        
        #%% get subsidy scheme
        subScheme = SubsidyScheme(t0,
                          sattrint0,
                          bunnt0,
                          makst0,
                          grovt0
                          )
        #%% get single farm
        kgb = df.index.get_level_values('KGB').unique()[0]
        dd = df.loc[(str(t0),slice(kgb,kgb)),:]
        
        #%%
        dd.reset_index(inplace=True)
        dd.set_index(['year','KGB'],inplace=True)
        
        #%%------------------------------------------------------------------
        #  Test TVELF
        #  -----------------------------------------------------------------
        # create farm with nothing except Set Zones
        ftest = dd*0
        ftest['zoneTAKTL'] = 2.0000
        ftest['zoneTDISE'] = 1.0000
        ftest['zoneTDISG'] = 2.0000
        ftest['zoneTDISK'] = 1.0000
        ftest['zoneTDISM'] = 1.0000
        ftest['zoneTDMLK'] = 1.0000
        ftest['zoneTPROD'] = 2.0000

        #%%
        """
        Satser
        2004.TVELF.HEST.1  800
        2004.TVELF.193 .1  800
        2004.TVELF.STOR.1  405
        2004.TVELF.120 .1  2800
        2004.TVELF.120 .2  1900
        2004.TVELF.121 .1  518
        2004.TVELF.VSAU.1  352
        2004.TVELF.142 .1  352
        2004.TVELF.140 .1  665
        2004.TVELF.140 .2  445
        2004.TVELF.155 .1  800
        2004.TVELF.157 .1  28
        * Fjoerfe er angitt i 1000 dyr
        2004.TVELF.160 .1  7000
        2004.TVELF.186 .1  300
        
        Trin
        2004.TVELF.HEST    INF
        2004.TVELF.193     INF
        2004.TVELF.120     8      INF
        2004.TVELF.STOR    INF
        2004.TVELF.140     40     INF
        2004.TVELF.121     INF
        2004.TVELF.VSAU    INF
        2004.TVELF.142     INF
        2004.TVELF.155     INF
        2004.TVELF.157     INF
        2004.TVELF.160     INF
        2004.TVELF.186     INF
        
        Maks Limit
        2004.TVELF     50000 
        
        """
        ftest['x120'] = 8
         
        TVELF_x120_2014 = 2800*8/1000
        #TVELF_x120_2014 += 1900*12/1000
        
        
        pay = subScheme.sub(ftest)
        
        maksCap_TVELF = 50000/1000
        
        aa = np.minimum(maksCap_TVELF,TVELF_x120_2014)
        
        self.assertTrue(np.isclose(aa, pay['TVELF'], equal_nan=True))
        #%% Test TVELF maks limit
        
        ftest['x120'] = 20
        ftest['x160'] = 5 

        TVELF_x120_2014 = 2800*8/1000
        TVELF_x120_2014 += 1900*12/1000
        
        TVELF_x160_2014 = 7000*5/1000
        
        pay = subScheme.sub(ftest)
        
        maksCap_TVELF = 50000/1000
        
        aa = np.minimum(maksCap_TVELF,TVELF_x120_2014+TVELF_x160_2014)
        self.assertTrue(np.isclose(aa, pay['TVELF'], equal_nan=True))
        
        #%%------------------------------------------------------------------
        #  Test TVELF
        #  -----------------------------------------------------------------
        # create farm with nothing except Set Zones
        ftest = dd*0
        ftest['zoneTAKTL'] = 2.0000
        ftest['zoneTDISE'] = 1.0000
        ftest['zoneTDISG'] = 2.0000
        ftest['zoneTDISK'] = 1.0000
        ftest['zoneTDISM'] = 1.0000
        ftest['zoneTDMLK'] = 1.0000
        ftest['zoneTPROD'] = 2.0000
        
        
    def test_dpay_TDMLK(self):
        """
        Test TDMLK
        """
        #%% Load unitTest data
        model = Model()
        model.random_state = 42
        df, df_test = model.load_data(TestTrainSplit_unitTest.targetFileName)  
        
        #%% Load subsidy scheme data
        bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()
        
        #%%
        t0 = 2004
        sattrint0 =  sattrin[t0]
        bunnt0 = bunn[bunn.year == t0]
        makst0 = maks[maks.year == t0]
        grovt0 = grov[grov.year == t0]
        
        
        #%% get subsidy scheme
        subScheme = SubsidyScheme(t0,
                          sattrint0,
                          bunnt0,
                          makst0,
                          grovt0
                          )
        #%% get single farm
        kgb = df.index.get_level_values('KGB').unique()[0]
        dd = df.loc[(str(t0),slice(kgb,kgb)),:]
        #%%
        dd.reset_index(inplace=True)
        dd.set_index(['year','KGB'],inplace=True)
        

        # create farm with nothing except Set Zones
        ftest = dd*0
        ftest['zoneTAKTL'] = 2.0000
        ftest['zoneTDISE'] = 1.0000
        ftest['zoneTDISG'] = 2.0000
        ftest['zoneTDISK'] = 1.0000
        ftest['zoneTDISM'] = 1.0000
        ftest['zoneTDMLK'] = 1.0000
        ftest['zoneTPROD'] = 2.0000
        
        
        #%%
        """
        Satser
        2004.TDMLK.120 .1  0      12800 11800 13000
        2004.TDMLK.120 .2  0      0     0     0
        2004.TDMLK.121 .1  0      1125  1125  1125
        2004.TDMLK.121 .2  0      0     0     0
        2004.TDMLK.140 .1  0      2407  2407  2407
        2004.TDMLK.140 .2  0      0     0     0
        
        TRIN
        2004.TDMLK.120     5      INF
        2004.TDMLK.121     40     INF
        2004.TDMLK.140     27     INF
        
        """
        
        ftest['x120'] = 20
        ftest['x121'] = 5 

        # In 2004 only TDMLK for x121 when no x120 
        TDMLK_x120_2004 = 12800*5/1000
        
        pay = subScheme.sub(ftest)
        
        self.assertTrue(np.isclose(TDMLK_x120_2004, pay['TDMLK'], equal_nan=True))
        #%% Check same in 2007
        """
        Satser
        2007.TDMLK.120 .1  0      11740 10760 11920
        2007.TDMLK.120 .2  0      0     0     0
        2007.TDMLK.121 .1  0      1420  1420  1420
        2007.TDMLK.121 .2  0      0     0     0
        2007.TDMLK.140 .1  0      2207  2207  2207
        2007.TDMLK.140 .2  0      0     0     0
        
        TRIN
        2007.TDMLK.120     5      INF
        2007.TDMLK.121     50     INF
        2007.TDMLK.140     27     INF
        """
        t0 = 2007
        sattrint0 =  sattrin[t0]
        bunnt0 = bunn[bunn.year == t0]
        makst0 = maks[maks.year == t0]
        grovt0 = grov[grov.year == t0]
                
        #%% get subsidy scheme
        subScheme = SubsidyScheme(t0,
                          sattrint0,
                          bunnt0,
                          makst0,
                          grovt0
                          )
        #%%
        ftest['x120'] = 20
        ftest['x121'] = 20 

        # In 2007 only TDMLK for x121 when no x120 
        TDMLK_x120_2008 = 11740*5/1000
        
        TDMLK_x121_2008 = 1420*20/1000
        
        pay = subScheme.sub(ftest)
        
        self.assertTrue(np.isclose(TDMLK_x120_2008, pay['TDMLK'], equal_nan=True))
        
        #%% Check same in 2008
        """
        Satser
        2008.TDMLK.120 .1  0      11960 11960 12940 13120
        2008.TDMLK.120 .2  0      0     0     0     0
        2008.TDMLK.121 .1  0      1640  1640  1640  1640
        2008.TDMLK.121 .2  0      0     0     0     0
        2008.TDMLK.140 .1  0      2430  2430  2430  2430
        2008.TDMLK.140 .2  0      0     0     0     0
        
        TRIN
        2008.TDMLK.120     5      INF
        2008.TDMLK.121     50     INF
        2008.TDMLK.140     27     INF
        """
        t0 = 2008
        sattrint0 =  sattrin[t0]
        bunnt0 = bunn[bunn.year == t0]
        makst0 = maks[maks.year == t0]
        grovt0 = grov[grov.year == t0]
                
        #%% get subsidy scheme
        subScheme = SubsidyScheme(t0,
                          sattrint0,
                          bunnt0,
                          makst0,
                          grovt0
                          )
        #%%
        ftest['x120'] = 20
        ftest['x121'] = 20 

        # In 2008 TDMLK is paid for both x121 and x120 
        TDMLK_x120_2008 = 11960*5/1000
        
        TDMLK_x121_2008 = 1640*20/1000
        
        pay = subScheme.sub(ftest)
        
        self.assertTrue(np.isclose(TDMLK_x120_2008+TDMLK_x121_2008, pay['TDMLK'], equal_nan=True))
        
        
        #%% Check 6 x121 limit in 2009
        # In all years payment for 121 is only for farms with
        #  more then 5 Suckler cows. Set number of suckler cows to zero for
        #  all farms with less then 6 ammekyr (121)
        """
        Satser
        2009.TDMLK.120 .1  0      13160 13160 14140 14320
        2009.TDMLK.120 .2  0      0     0     0     0
        2009.TDMLK.121 .1  0      1640  1640  1640  1640
        2009.TDMLK.121 .2  0      0     0     0     0
        2009.TDMLK.140 .1  0      2652  2652  2652  2652
        2009.TDMLK.140 .2  0      0     0     0     0
        
        TRIN
        2009.TDMLK.120     5      INF
        2009.TDMLK.121     50     INF
        2009.TDMLK.140     27     INF
        """
        t0 = 2009
        sattrint0 =  sattrin[t0]
        bunnt0 = bunn[bunn.year == t0]
        makst0 = maks[maks.year == t0]
        grovt0 = grov[grov.year == t0]
                
        #%% get subsidy scheme
        subScheme = SubsidyScheme(t0,
                          sattrint0,
                          bunnt0,
                          makst0,
                          grovt0
                          )
        #%%
        ftest['x120'] = 20
        ftest['x121'] = 5 
        ftest['x140'] = 25 

        # Only x120 because of 6 cows limit
        TDMLK_x120_2009 = 13160*5/1000
        
        # But x140 still added
        TDMLK_x140_2009 = 2652*25/1000
        
                
        pay = subScheme.sub(ftest)
        
        self.assertTrue(np.isclose(TDMLK_x120_2009+TDMLK_x140_2009, pay['TDMLK'], equal_nan=True))
    
   
    #%%TAKTL
    def test_dpay_TAKTL(self):
        """
        Test TAKTL
        """
        #%% Load unitTest data
        model = Model()
        model.random_state = 42
        df, df_test = model.load_data(TestTrainSplit_unitTest.targetFileName)  
        
        #%% Load subsidy scheme data
        bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()
        
        #%%
        t0 = 2004
        sattrint0 =  sattrin[t0]
        bunnt0 = bunn[bunn.year == t0]
        makst0 = maks[maks.year == t0]
        grovt0 = grov[grov.year == t0]
        
        
        #%% get subsidy scheme
        subScheme = SubsidyScheme(t0,
                          sattrint0,
                          bunnt0,
                          makst0,
                          grovt0
                          )
        #%% get single farm
        kgb = df.index.get_level_values('KGB').unique()[0]
        dd = df.loc[(str(t0),slice(kgb,kgb)),:]
        
        #%%
        dd.reset_index(inplace=True)
        dd.set_index(['year','KGB'],inplace=True)
        

        # create farm with nothing except Set Zones
        ftest = dd*0
        ftest['zoneTAKTL'] = 2.0000
        ftest['zoneTDISE'] = 1.0000
        ftest['zoneTDISG'] = 2.0000
        ftest['zoneTDISK'] = 1.0000
        ftest['zoneTDISM'] = 1.0000
        ftest['zoneTDMLK'] = 1.0000
        ftest['zoneTPROD'] = 2.0000
        
        
        #%%
        """
        Satser
        2004.TAKTL.FODD.1  0      287   212   311   311   403   438   468
        2004.TAKTL.FODD.2  0      200   200   200   200   200   200   200
        2004.TAKTL.CERE.1  0      260   316   316   410   410   410   410
        2004.TAKTL.230 .1  0      279   279   279   279   279   1100  1400
        2004.TAKTL.GRON.1  0      800   800   800   800   800   1900  1900
        2004.TAKTL.GRON.2  0      300   300   300   300   300   300   300
        2004.TAKTL.FRUK.1  0      800   800   800   800   800   1400  1400
        2004.TAKTL.FRUK.2  0      300   300   300   300   300   300   300
        2004.TAKTL.BAER.1  0      800   800   800   800   800   1400  1400
        2004.TAKTL.BAER.2  0      300   300   300   300   300   300   300
        
        TRIN
        2004.TAKTL.CERE    INF
        2004.TAKTL.FODD    200    INF
        2004.TAKTL.230     INF
        2004.TAKTL.GRON    30     INF
        2004.TAKTL.FRUK    30     INF
        2004.TAKTL.BAER    30     INF
        
        Bunn
        2004.TAKTL     6000  
        
        """
        
        ftest['x230'] = 300
        ftest['FRUK'] = 35 

        
        TAKTL_230_2004 = 279*300/1000
        
        
        TAKTL_FRUK_2004 = 800*30/1000
        TAKTL_FRUK_2004 += 300*5/1000
        
        pay = subScheme.sub(ftest)
        
        bunn = 6000/1000
        
        self.assertTrue(np.isclose(TAKTL_230_2004+TAKTL_FRUK_2004-bunn, pay['TAKTL'], equal_nan=True))
        
        
        #%% Check eligable FOOD
        # create farm with nothing except Set Zones
        ftest = dd*0
        ftest['zoneTAKTL'] = 2.0000
        ftest['zoneTDISE'] = 1.0000
        ftest['zoneTDISG'] = 2.0000
        ftest['zoneTDISK'] = 1.0000
        ftest['zoneTDISM'] = 1.0000
        ftest['zoneTDMLK'] = 1.0000
        ftest['zoneTPROD'] = 2.0000
        
        ftest['FODD'] = 300
        ftest['FRUK'] = 35 

        
        TAKTL_FODD_2004 = 212*200/1000
        TAKTL_FODD_2004 += 200*100/1000
        
        TAKTL_FRUK_2004 = 800*30/1000
        TAKTL_FRUK_2004 += 300*5/1000
        
        pay = subScheme.sub(ftest)
        
        bunn = 6000/1000
        # Check that  TAKTL_FODD_2004 are not considered because not animals
        self.assertTrue(np.isclose(TAKTL_FRUK_2004-bunn, pay['TAKTL'], equal_nan=True))
        
        #%% Check eligable FOOD
        """
        2014.GROVFO.HEST 0      12      12      12      12      13      15      15
        2014.GROVFO.STOR 0      5       5       5       5       5.5     6.5     6.5
        2014.GROVFO.120  0      14      14      14      14      15      18      18
        2014.GROVFO.121  0      14      14      14      14      15      18      18
        2014.GROVFO.140  0      1.75    1.75    1.75    1.75    2       2.25    2.25
        2014.GROVFO.142  0      1.75    1.75    1.75    1.75    2       2.25    2.25
        2014.GROVFO.VSAU 0      1.9     1.9     1.9     1.9     2.2     2.5     2.5
        2014.GROVFO.155  0      0.75    0.75    0.75    0.75    1       1.25    1.25
        2014.GROVFO.180  0      0.75    0.75    0.75    0.75    1       1.25    1.25
        2014.GROVFO.181  0      5       5       5       5       5.5     6.5     6.5
        2014.GROVFO.183  0      0.75    0.75    0.75    0.75    1       1.25    1.25
        2014.GROVFO.192  0      5       5       5       5       5.5     6.5     6.5
        2014.GROVFO.193  0      5       5       5       5       5.5     6.5     6.5
        2014.GROVFO.196  0      5       5       5       5       5.5     6.5     6.5
        2014.GROVFO.197  0      5       5       5       5       5.5     6.5     6.5
        2014.GROVFO.832  0      0.2     0.2     0.2     0.2     0.25    0.3     0.3
        2014.GROVFO.841  0      0.02    0.02    0.02    0.02    0.02    0.02    0.02
        2014.GROVFO.521  0      0.0025  0.0025  0.0025  0.0025  0.00308 0.00364 0.00364
        2014.GROVFO.522  0      0.001   0.001   0.001   0.001   0.00105 0.00125 0.00125
        2014.GROVFO.523  0      0.002   0.002   0.002   0.002   0.0025  0.00308 0.00308
        
        
        
        2014.INMARK.HEST 0      5.5     5.5     5.5     5.5     6       7.5     7.5
        2014.INMARK.STOR 0      3       3       3       3       4       5       5
        2014.INMARK.120  0      7       7       7       7       7.5     9       9
        2014.INMARK.121  0      7       7       7       7       7.5     9       9
        2014.INMARK.140  0      2       2       2       2       2.5     3       3
        2014.INMARK.142  0      2       2       2       2       2.5     3       3
        2014.INMARK.VSAU 0      2.2     2.2     2.2     2.2     2.8     3.3     3.3
        2014.INMARK.181  0      2       2       2       2       2.5     3       3
        2014.INMARK.192  0      3       3       3       3       4       5       5
        2014.INMARK.193  0      5.5     5.5     5.5     5.5     6       7.5     7.5
        2014.INMARK.196  0      3       3       3       3       4       5       5
        2014.INMARK.197  0      3       3       3       3       4       5       5
        """
        
        #---------- Test GROVFO
        # create farm with nothing except Set Zones
        ftest = dd*0
        ftest['zoneTAKTL'] = 2.0000
        ftest['zoneTDISE'] = 1.0000
        ftest['zoneTDISG'] = 2.0000
        ftest['zoneTDISK'] = 1.0000
        ftest['zoneTDISM'] = 1.0000
        ftest['zoneTDMLK'] = 1.0000
        ftest['zoneTPROD'] = 2.0000
        
        ftest['x210'] = 300.
        ftest['FRUK'] = 35. 
        ftest['x120'] = 15. 

        # Calc GROVFO
        eligableFood = ftest['x120']*14
        FODD = np.minimum(ftest['x210'],eligableFood)
        
        TAKTL_FODD_2004 = 212*np.minimum(200,FODD)/1000
        TAKTL_FODD_2004 += 200*np.maximum(0,FODD-200)/1000
        
        TAKTL_FRUK_2004 = 800*30/1000
        TAKTL_FRUK_2004 += 300*5/1000
        
        pay = subScheme.sub(ftest)
        
        bunn = 6000/1000
        
        self.assertTrue(np.isclose(TAKTL_FRUK_2004+TAKTL_FODD_2004-bunn, pay['TAKTL'], equal_nan=True))
        
        #%%---------- Test GROVFO AND INMARK
        # create farm with nothing except Set Zones
        ftest = dd*0
        ftest['zoneTAKTL'] = 2.0000
        ftest['zoneTDISE'] = 1.0000
        ftest['zoneTDISG'] = 2.0000
        ftest['zoneTDISK'] = 1.0000
        ftest['zoneTDISM'] = 1.0000
        ftest['zoneTDMLK'] = 1.0000
        ftest['zoneTPROD'] = 2.0000
        
        ftest['x210'] = 300.
        ftest['x211'] = 300.
        ftest['x212'] = 300.
        ftest['FRUK'] = 35. 
        ftest['x120'] = 15. 

        # Calc Inmark GROVFO
        inmark = ftest['x120'] * 7
        foodInmark = np.minimum(inmark,ftest['x212'])
        
        # Calc GROVFO
        eligableFood = ftest['x120']*14
        
        foodArea = ftest['x210']+ftest['x211']+0.6*foodInmark
        
        FODD = np.minimum(foodArea,eligableFood)
        
        TAKTL_FODD_2004 = 212*np.minimum(200,FODD)/1000
        TAKTL_FODD_2004 += 200*np.maximum(0,FODD-200)/1000
        
        TAKTL_FRUK_2004 = 800*30/1000
        TAKTL_FRUK_2004 += 300*5/1000
        
        pay = subScheme.sub(ftest)
        
        bunn = 6000/1000
        
        self.assertTrue(np.isclose(TAKTL_FRUK_2004+TAKTL_FODD_2004-bunn, pay['TAKTL'], equal_nan=True))
        
       #%%
if __name__ == '__main__':
    unittest.main()