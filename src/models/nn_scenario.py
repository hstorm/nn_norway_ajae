# %%
"""
This file is use to run the scenario simulations, post estimation 
"""
import os
import sys
import pathlib
import re 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from tensorflow.keras import backend as K
wd = "/nn_norway"#
os.chdir(wd)
sys.path.append(wd)
from src.features.luigi_features import TestTrainSplit_unitTest
from src.features.calc_dpay import loadSubsidySchemeFiles
from src.features.calc_dpay import DeriveSubsidies
from src.features.calc_dpay import sattrinToDict
from src.lib.utily import findIndex


# Import model
from src.models.nn_model import nn_model
# %%
#===========================================================================        
class nn_predict(nn_model):
    """
    Class to perform predictions and scenario analysis
    """

    def __init__(self,wd=""):
        super(nn_predict, self).__init__(wd=wd)

    #%%
    # =============================================================================
    # Functions to prepare data
    # =============================================================================
    def derive_sub(self,X,
                   t0, t1,
                   sattrint0, bunnt0, makst0, grovt0,
                   sattrint1, bunnt1, makst1, grovt1,
                   subType='CsubD',
                   unitIncrease=1,
                   vDiffAct=['CERE', 'GRON','SAU','STOR','HEST','VSAU','GEIT',
                             'USAU', 'FRUK','BAER',
                             'x410', 'x420','x440',
                             'x521', 'x522',
                             'x210', 'x212','x230',
                             'x120', 'x121','x136','x140', 'x142',
                             'x155', 'x157','x160','x180']
                   ):
        """
        Function to derive variables
        """
        #%%
        print('Derive '+subType)
        print('Test in derive_sub')
        # Create Subsidy scheme
        deriveSub = DeriveSubsidies(t0,
                                    sattrint0,
                                    bunnt0,
                                    makst0,
                                    grovt0,
                                    t1,
                                    sattrint1,
                                    bunnt1,
                                    makst1,
                                    grovt1
                                    )
        print('after DeriveSubsidies in derive_sub')
        #deriveSub.pool = self.pool
        #%%
        currentSub = deriveSub.get_sub(X)

        #%%
        print('after get_sub in derive_sub')
        if subType=='CsubD':
            print(1)
            sub =  deriveSub.get_CsubDUnit(X,unitIncrease,currentSub=currentSub)
            suffix = ('CsubD{}_').format(unitIncrease)
            sub.columns = [suffix + str(col) for col in sub.columns]
        elif subType=='subD1':
            print(2)
            sub =  deriveSub.get_subDUnit(X,unitIncrease,currentSub=currentSub)
            suffix = ('subD{}_').format(unitIncrease)
            sub.columns = [suffix + str(col) for col in sub.columns]
        elif subType == 'dAvgSub':
            print(3)
            sub =  deriveSub.get_dAvgSub(X,currentSub=currentSub)
            sub.columns = ['DAvgSub_' + str(col) for col in sub.columns]
        elif subType == 'dSub':
            sub =  deriveSub.get_dSub(X,currentSub=currentSub)
            sub.columns = ['dSub' + str(col) for col in sub.columns]
        else:
            sub = []
            print("subType not known...")
        #%%   

        print('after Ifs in derive_sub') 
        return sub
    
    #%%   
    def extract_features(self,X,
                         t0, t1,
                         sattrint0, bunnt0, makst0, grovt0,
                         sattrint1, bunnt1, makst1, grovt1
                         ):
        """
        Add CsubD variable to dataframe
        """
        #%%
        X.reset_index(inplace=True)
        X.set_index(['year','KGB'], inplace=True)
    
        #%% Derive CsubD1
        CsubD1 = self.derive_sub(X,
                   t0, t1,
                   sattrint0, bunnt0, makst0, grovt0,
                   sattrint1, bunnt1, makst1, grovt1,
                   subType='CsubD',
                   unitIncrease=1,
                   )
    
        #%% Derive CsubD10
        CsubD10 = self.derive_sub(X,
                   t0, t1,
                   sattrint0, bunnt0, makst0, grovt0,
                   sattrint1, bunnt1, makst1, grovt1,
                   subType='CsubD',
                   unitIncrease=10,
                   )
    
        # Derive CsubD50
        CsubD50 = self.derive_sub(X,
                   t0, t1,
                   sattrint0, bunnt0, makst0, grovt0,
                   sattrint1, bunnt1, makst1, grovt1,
                   subType='CsubD',
                   unitIncrease=50,
                   )
    
        #%% Derive dAvgSub
        dAvgSub = self.derive_sub(X,
                   t0, t1,
                   sattrint0, bunnt0, makst0, grovt0,
                   sattrint1, bunnt1, makst1, grovt1,
                   subType='dAvgSub'
                   )
    
        #%% Derive dSub
        dSub = self.derive_sub(X,
                   t0, t1,
                   sattrint0, bunnt0, makst0, grovt0,
                   sattrint1, bunnt1, makst1, grovt1,
                   subType='dSub'
                   )
        #%%
        df_sub = X
        # Check if columns already exists in X
        if all(x in X.columns  for x in CsubD1.columns):
            X.loc[:,CsubD1.columns] = CsubD1
        else:
            df_sub = pd.concat([df_sub,CsubD1],axis=1)
        
        # Check if columns already exists in X
        if all(x in X.columns  for x in CsubD10.columns):
            X.loc[:,CsubD10.columns] = CsubD10
        else:
            df_sub = pd.concat([df_sub,CsubD10],axis=1)
            
        # Check if columns already exists in X
        if all(x in X.columns  for x in CsubD50.columns):
            X.loc[:,CsubD50.columns] = CsubD50
        else:
            df_sub = pd.concat([df_sub,CsubD50],axis=1)

        # Check if columns already exists in X
        if all(x in X.columns  for x in dAvgSub.columns):
            X.loc[:,dAvgSub.columns] = dAvgSub
        else:
            df_sub = pd.concat([df_sub,dAvgSub],axis=1)

        # Check if columns already exists in X
        if all(x in X.columns  for x in dSub.columns):
            X.loc[:,dSub.columns] = dSub
        else:
            df_sub = pd.concat([df_sub,dSub],axis=1)

            
        #%%
        #df_sub = pd.concat([X,CsubD1,CsubD10,CsubD50,dAvgSub,dSub],axis=1)
    
        return df_sub

    #%%
    def features_year(self,t0,t1,df,bunn, maks, sattrin, grov):
    
        """
        Wrapper function to obtain features for one particular year
        """
        #%%
        sattrint0 = sattrin[t0]
        bunnt0 = bunn[bunn.year == t0]
        makst0 = maks[maks.year == t0]
        grovt0 = grov[grov.year == t0]
        print('use sattrin t0=',t0)
        sattrint1 = sattrin[t1]
        bunnt1 = bunn[bunn.year == t1]
        makst1= maks[maks.year == t1]
        grovt1 = grov[grov.year == t1]
        print('use sattrin t1=',t1)
        #%%  Get data for year t0
        X = df.loc[str(t0)].copy()
        #%%
        df_feat = self.extract_features(X, t0, t1,
                                   sattrint0, bunnt0, makst0, grovt0,
                                   sattrint1, bunnt1, makst1, grovt1
                                  )
    
    
        #%% Fill nan, -inf and inf with 0
        df_feat.replace(np.inf, 0,inplace=True)
        df_feat.replace(-np.inf, 0,inplace=True)
        df_feat.fillna(0,inplace=True)
        df_feat.fillna(0,inplace=True)
        #%%
        return df_feat
    
    #%%
        
    def load_spec_pred(self):
        """
        Load specifications of the model
        """

        #%%
        self.specTrain_from_file()
        #%%
        mod_e = self.model()
        #%%
        
        # set Tx to 1, for prediction only one time steps is 
        # predicted at once, such that the cell states can be outputed and the 
        # exogeouns features (i.e. the subsidy variables) can be updated based
        # on the last prediction
        self.Tx = 1 
        
        mod_p = self.model_pred() # why i here Model Predict to Deafault = False ?
        #%%
        #load latest weights imprivement to ensure best version of Model
        mod_p.load_weights(self.fullId+"weights-improvement") 
        #mod_p.load_weights(self.model_path+self.id+"_estimator.h5") 
        #%%
        self.estimator = mod_p


 
    def make_one_prediction(self,X_pred,minmaxscale = True):
        """
        Make a prediction for one single year
        
        Inputs:
        minmaxscale = specifiy if X_pred should be scaled 
        
        """
        if minmaxscale:
            #%% apply minmax transformation
            print("minmax scaling")
            X_scaled, _ = self.data_transform(X_pred[0], None)
        
        X_scaled[1] = np.squeeze(X_pred[1])
        X_scaled[2] = np.squeeze(X_pred[2])

        # perform prediction
        yhat_dict = self.est_predict(X_scaled)

        #%% invert scaling of Y
        _, yhat_orig = self.data_inverse_transform(None,yhat_dict["regression"])


        #%%
        return yhat_orig, yhat_dict 
    
    def one_step_prediction(self,X_pred,a0,c0,minmaxscale=True):
        print("one_step_prediction")
        """
        Make a prediction over several years but always for just one year 
        
        Inputs:
            X_pred (I,T,vAct) = 3D array with values to make prediction
            a0 = inital state
            c0 = inital state
            minmaxscale = specifiy if X_pred should be scaled 
        
        Return:
            yhat_orig (T,I,vTag) = 3D prediction of each year in X_pred
            a = last state
            c = last state
        """
        
        #%%
        T = X_pred.shape[1]
        a = a0
        c = c0
        
        yhat_orig = np.zeros((X_pred.shape[0],T,len(self.vTag)))*np.nan
        #%%-----------------------------------------------------------
        for t in range(0, T):
            
            X_pred_t = X_pred[:,t:t+1,:]
            #% Make one prediction
            In_List = [X_pred_t,a,c]

            yhat_orig_t, Y_hat_minmax = self.make_one_prediction(In_List,minmaxscale=minmaxscale)
            
            yhat_orig[:,t:t+1,:] = yhat_orig_t
            
            a = Y_hat_minmax["a"]
            c = Y_hat_minmax["c"]
            
            
        #%%  
        return yhat_orig, a, c
        
        
        #%%
    
    
    def integrate_prediction(self,dat,vDat, yhat_orig,vTag):
        """
        Intergrate prediction of one year into dat with all variables necessary
        to calcualte subsidies. appends dat by copying last values and 
        updating yhat_orig values in this last year
        
        dat: 3d-array with all variables 
        yhat_orig 3d-array with prediction of last next year
        """       
        #%% make a copy of last observed year
        dat_next = dat[:,-1,:].copy().reshape(dat.shape[0],1,dat.shape[2])
        #%%
        print("yhat_orig.shape in integrate_prediction",yhat_orig.shape)
        print("dat_next.shape in integrate_prediction",dat_next.shape)
        print("dat_next[:,-1,findIndex(vDat,vTag)].shape in integrate_prediction",dat_next[:,-1,findIndex(vDat,vTag)].shape)

        dat_next[:,-1,findIndex(vDat,vTag)] = yhat_orig.reshape(yhat_orig.shape[0],yhat_orig.shape[2])
        
        #%%
        #TODO: check if FODD is in vTag if not sum up x210,x211,x212,x213 and replace FODD

        #%%
        dat_append = np.concatenate((dat,dat_next),axis=1)

        #%%
        return dat_append


    def derive_explantory(self,dat_pred_year,vDat,year_pred,idv_index, bunn, maks, sattrin, grov):
        """
        
        dat_pred_year: Predicted values for one particular year (2d array)
        vDat: Variable names in dat_pred_year
        year_pred: year for which the prediction is made 
        idv_index: KGB for observations in dat_pred_year (the rows)
        
        """
         
        #%% Transfor numpy array to pandas df
        df = pd.DataFrame(dat_pred_year,columns=vDat)
        
        #% specify year
        t0 = year_pred
        t1 = t0+1
        
        # Set year and KGB index
        df['year'] = pd.to_datetime(str(year_pred))
        df['KGB'] = idv_index
    
        # Set index
        df.set_index(['year','KGB'],inplace=True)
        
        #%% Derive features
        feat = self.features_year(t0,t1,df,bunn, maks, sattrin, grov)
        
        #%%
        return feat
    #%%
    # =============================================================================
    # Functions to define scenarios
    # =============================================================================
    def append_last_year(self,df,steps):
        """
        Append datafram by copying entries for last year
        This function takes variables like bunn, maks, satdf, trin, grov ... etc
        and elongates the Dataframe corresponding to the years in advance(steps),
        one wants to predict
        it is only used in scen const
        """
        #%%
        old_index = list(df.index.names) # Save names of old index
        df.reset_index(inplace=True)
        df.set_index(['year'], inplace=True)
        for t in range(0,steps):
            dfNext = df.loc[[max(df.index)]].copy()
            dfNext.set_index(dfNext.index+1,inplace=True)
            df = df.append(dfNext)
        df.reset_index(inplace=True)

        if old_index!=[None]:
            df.set_index(old_index, inplace=True)
            df.sort_index(inplace=True)
            
        if  'index' in list(df.columns):
            df.drop(columns='index',inplace=True)
          
        #%%
        return df
    
   
    def scen_original(self):
        """
        Create a constant scenario where no changes occure
        """
        #%%
        
        bunn, maks, sattrin, trin, grov, satdf  = loadSubsidySchemeFiles()
        
        #%%
        return bunn, maks, sattrin, grov, satdf
    
    def scen_const(self,steps=3):
        """
        Create a constant scenario where no changes occure
        """
        #%%
        bunn, maks, _, trin, grov, satdf  = loadSubsidySchemeFiles()
        #%%
        bunn = self.append_last_year(bunn,steps)
        maks = self.append_last_year(maks,steps)
        satdf = self.append_last_year(satdf,steps)
        grov = self.append_last_year(grov,steps)
                
        # transform sattrin df to sattrin dict
        sattrin = sattrinToDict(satdf,grov)
        #%%
        return bunn, maks, sattrin, grov, satdf
    
    def scen_increase_tprod_SAU(self):
        """
        Create a scenario where TPROD paymetns for SAU
        are increased for lower size classes and decreased for larger farms
        Set equal to 1500 NOK/head up to sheep <=100  and
        equal to 0 NOK/head sheep>100
        """
        #%%
        bunn, maks, _, grov, satdf  = self.scen_const(3)
        #%%
        satdf.loc[(2016,'TPROD',slice(None),1),('rate','SAU')] = 1500
        satdf.loc[(2016,'TPROD',slice(None),2),('rate','SAU')] = 0
        satdf.loc[(2016,'TPROD',slice(None),3),('rate','SAU')] = 0
        satdf.loc[(2016,'TPROD',slice(None),4),('rate','SAU')] = 0
        satdf.loc[(2016,'TPROD',slice(None),5),('rate','SAU')] = 0
               
        #%% transform sattrin df to sattrin dict
        sattrin = sattrinToDict(satdf,grov)
        #%%
        return bunn, maks, sattrin, grov, satdf
    
    def scen_flat_tprod_SAU(self):
        """
        Create a scenario where TPROD paymetns for SAU
        are flat (decrease for small farm, increase for large farms).
        Set equal to 600 NOK/head

        """
        #%%
        bunn, maks, _, grov, satdf  = self.scen_const(3)
        #%%
        satdf.loc[(2016,'TPROD'),('rate','SAU')] = 600
        satdf.loc[(2017,'TPROD'),('rate','SAU')] = 600
        satdf.loc[(2018,'TPROD'),('rate','SAU')] = 600
               
        #%% transform sattrin df to sattrin dict
        sattrin = sattrinToDict(satdf,grov)
        #%%

        #print("sattrin in scen_flat_tprod_SAU",sattrin)
        #sys.exit()
        return bunn, maks, sattrin, grov, satdf
    
    def scen_increase_tprod_x120(self):
        """
        Create a scenario where TPROD paymetns for x120
        are increased for lower size classes
        """
        #%%
        bunn, maks, _, grov, satdf  = self.scen_const(3)
        #%%
        satdf.loc[(2016,'TPROD',slice(None),1),('rate','x120')] = 6000
        satdf.loc[(2016,'TPROD',slice(None),2),('rate','x120')] = 3000
        satdf.loc[(2016,'TPROD',slice(None),3),('rate','x120')] = 0
        satdf.loc[(2016,'TPROD',slice(None),4),('rate','x120')] = 0
               
        #%% transform sattrin df to sattrin dict
        sattrin = sattrinToDict(satdf,grov)
        #%%
        return bunn, maks, sattrin, grov, satdf
    
    
    def scen_flat_tprod_x120(self):
        """
        Create a scenario where TPROD paymetns for x120
        are falt equal to 1500 NOK irrespectivly of farm size
        """
        #%%
        bunn, maks, _, grov, satdf  = self.scen_const(3)
        #%%
        
        
        satdf.loc[(2016,'TPROD'),('rate','x120')] = 1500
        satdf.loc[(2017,'TPROD'),('rate','x120')] = 1500
        satdf.loc[(2018,'TPROD'),('rate','x120')] = 1500
               
        #%% transform sattrin df to sattrin dict
        sattrin = sattrinToDict(satdf,grov)
        #%%
        return bunn, maks, sattrin, grov, satdf
    def scen_crazy_trin(self):
        """
        Create a scenario where TPROD paymetns for x120
        are falt equal to 1500 NOK irrespectivly of farm size
        """
        #%%
        bunn, maks, _, grov, satdf  = self.scen_const(3)
        #%%
        
        
        satdf.loc[(2016,'TPROD'),('rate','x120')] = 15000
        satdf.loc[(2017,'TPROD'),('rate','x120')] = 15000
        satdf.loc[(2018,'TPROD'),('rate','x120')] = 15000
               
        #%% transform sattrin df to sattrin dict
        sattrin = sattrinToDict(satdf,grov)
        #%%
        return bunn, maks, sattrin, grov, satdf

    def scen_regional_TDISM(self):
        """
        Create a scenario where TDISM paymetns for x120
        are increased in only one zone =7
        """
        #%%
        bunn, maks, _, grov, satdf  = self.scen_const(3)
        #%%
        satdf.loc[(2016,'TDISM',7),('rate','x120')] = 15000
        satdf.loc[(2017,'TDISM',7),('rate','x120')] = 15000
        satdf.loc[(2018,'TDISM',7),('rate','x120')] = 15000
               
        # transform sattrin df to sattrin dict
        sattrin = sattrinToDict(satdf,grov)
        #%%
        return bunn, maks, sattrin, grov, satdf
        
        
    def plot_pred(self,Y_true, vYTrue,Y_hat, vYHat,vAct,t_start,t_end,nIdv = 50):
        """
        Inputs:
            Y_true: 3d Array with true values
            vYTrue: variables names for third dimension of Y_true
            vYHat: variables names for third dimension of Y_hat
            vTag : name of variable to plot
            t_start: specify start of series
            t_end: specify end of series
        """
        #%%
        for i in range(0,nIdv):
            
            yIndex = np.arange(t_start,t_end+1)
            
            ytrue = Y_true[:,i,findIndex(vYTrue,vAct)]
            yhat = Y_hat[:,i,findIndex(vYHat,vAct)]
            
            
            if sum(ytrue[0]) :
                plt.figure();
                plt.plot(yIndex,ytrue,'r--',yIndex,yhat,'b--');
                plt.show()
                
        #%%
        
    
    def make_prediction(self,year_start,year_end,df,df_idv_index,bunn,maks,sattrin,grov):
        print("make_prediction")
        """
        Make prediction over multiple years
        
        year_start: year for which the first prediction is made (i.e if equal
                    to 2000, X values for 1999 are taken to make prediction 
                    for 2000)
        year_end:   year for which last prediction is made (i.e if equal to 
                    2017)
        """
        

        a0_test = np.zeros((df.shape[0],self.n_a))
        c0_test = np.zeros((df.shape[0],self.n_a))

        a = a0_test
        c = c0_test
       
        listyear = [y.year for y in self.t_index]

        #% Check if out of sample prediction should be made an extent listyear
        if year_end > listyear[-1]:
            listyear = list(np.append(listyear,list(np.arange(listyear[-1]+1,year_end+1))))
        
        t0 = listyear.index(year_start)
        t_last = listyear.index(year_end)

        #for base and scen alike:
       
        #%% make prediction up to year_start to build up LSTM cell states
        #   i.e. if year_start = 2010 we make a prediction up to 2009 here
        yhat_orig, a_last, c_last = self.one_step_prediction(df[:,:,findIndex(self.col_index,self.vFeat)]
                                                ,a, c,minmaxscale=True)

        #%% Get names of all variables relevant to calc subs
        self.allvAct = sattrin['allvAct']
        self.allvActFeat = list(set(self.allvAct+self.vFeat))
        
        #% Get all variables required to clac sub for all year up to year_pred
        # i.e. if year_pred= 2016 then values up to 2015 are selected
        dat_pred = df[:,0:t0,findIndex(self.col_index,self.allvActFeat)].copy()
        
        ### !!! Recalcualte subsidies for for last year (in example i.e. 2015) 
        # because this requires Subsidies scheme from 2016 that might 
        # be altered for the simulation. The original df subsidy explanatory 
        # variables are all zero because the scheme for 2016 was not existing
        # when deriving the exaplantory variables 
        # This needs to be done such that one-time step ahead prediction can be
        # made where the subsidy scheme is varied in the prediction year
        t_i = t0-1
        year_pred = self.t_index[t_i].year    
        
        feat = self.derive_explantory(dat_pred_year=df[:,t_i,findIndex(self.col_index,self.allvAct)],
                                       vDat=self.allvAct,
                                       year_pred = year_pred,
                                       idv_index = df_idv_index, 
                                       bunn = bunn, 
                                       maks = maks, 
                                       sattrin = sattrin, 
                                       grov = grov)
        print("feat.shape in ",feat.shape)
       
        # transform to 3d array
        dat_feat_next, t_index, idv_index, col_index = self.resphape_3d_panel(feat)
        
        #
        setIntersect = list(set(self.vFeat).intersection(col_index))
        print("Are subsidies same ? :",np.all(dat_pred[:,-1,findIndex(self.allvActFeat,setIntersect)] == dat_feat_next[:,0,findIndex(col_index,setIntersect)]))
        dat_pred[:,-1,findIndex(self.allvActFeat,setIntersect)] = dat_feat_next[:,0,findIndex(col_index,setIntersect)]
        
        #%% Assert dimensions
        # Check that all vTag are included in vFeat
        assert(len(self.vTag)==len(findIndex(self.vFeat,self.vTag)))
        # Check that col_index inludes all variabales necessary to calc subs (in allvAct)
        assert(len(self.allvAct)==len(findIndex(self.col_index,self.allvAct)))

        #%% Get explanatory variables for year_start-1
        #i.e. if year_start=2010 get values for 2009
        X_pred_orig = dat_pred[:,-1,findIndex(self.allvActFeat,self.vFeat)]
        # Reshape to 3d slicing -1 remove time dimension and returns 2d array
        X_pred_orig = X_pred_orig.reshape((X_pred_orig.shape[0],1,X_pred_orig.shape[1]))
        
        pred_t_index = self.t_index[:t0].copy()
        
        for ti in range(t0,t_last+1):
            
            year_expl = listyear[ti-1:ti][0]
            year_pred = listyear[ti]
            
            pred_t_index.append(pd.Timestamp(year=year_pred, month=6, day=15))
            
            #% Make one prediction
            
            yhat_orig, Y_hat_minmax = self.make_one_prediction([X_pred_orig,a_last,c_last])

            
            # Update cell states
            #[regr,ar,cr, activity,ac,cc] Model Output order, select accordingly
            
            a_last = Y_hat_minmax["a"]
            c_last = Y_hat_minmax["c"]
           
            #Integrate prdiction in dat_pred, i.e. append at the end based on 
            # last year and replacing/updating predicted values (while keeping
            # variables that are not predicted constant)  
            dat_pred_append = self.integrate_prediction(dat_pred,self.allvActFeat,yhat_orig,self.vTag)
            
            if ti < t_last:
                print("Trigger Warning ti < t_last")

                # For all years except last year of prediction update features

                # Derive subsidies 
                year = year_pred
                
                print('Derive features based on prediction for year:',year)
                print('(used for next predicion)')
                
                feat = self.derive_explantory(dat_pred_year=dat_pred_append[:,-1,findIndex(self.allvActFeat,self.allvAct)],
                                               vDat=self.allvAct,
                                               year_pred = year,
                                               idv_index = df_idv_index, 
                                               bunn = bunn, 
                                               maks = maks, 
                                               sattrin = sattrin, 
                                               grov = grov)
                
                # transform to 3d array
                dat_feat_next, t_index, idv_index, col_index = self.resphape_3d_panel(feat)
                
                
                
                #
                setIntersect = list(set(self.vFeat).intersection(col_index))
                dat_pred_append[:,-1,findIndex(self.allvActFeat,setIntersect)] = dat_feat_next[:,0,findIndex(col_index,setIntersect)]
                #
                
                dat_pred = dat_pred_append.copy()
                #% Get explanatory variables for year_start-1
                X_pred_orig = dat_pred[:,-1,findIndex(self.allvActFeat,self.vFeat)]
                # Reshape to 3d slicing -1 remove time dimension and returns 2d array
                X_pred_orig = X_pred_orig.reshape((X_pred_orig.shape[0],1,X_pred_orig.shape[1]))
            else:
                print('There')
                # If last year of prediction, done update features
                # Replace features with nan to make sure that is this
                # not accidently used in oder places 
                dat_pred_append[:,-1,findIndex(self.allvActFeat,set(self.allvActFeat)-set(self.vTag))] = np.nan
                
                dat_pred = dat_pred_append.copy()
                    

        #%% transform to panel
        panel = self.revert_resphape_3d_panel(dat_pred,df_idv_index,pred_t_index,self.allvActFeat)

        #%%
        return dat_pred, panel

    
    
    def scen_comp(self,t_start,t_end,df,df_idv_index,base_func, scen_func):
        print("scen_comp")
        """
        Run a scenario comparison
        
        t_start: year for which the first prediction is made (i.e if equal
                    to 2000, X values for 1999 are taken to make prediction 
                    for 2000)
        t_end:   year for which last prediction is made (i.e if equal to 
                    2017, last prediction is made for 2017 used 2016 values)
        df:     3d array with data
        df_idv_index    : index of idviduals in 3d df
        
        base_func: function for the baseline scenario
        scen_func: function for the sceario that sould be the comparison
        
        """
        bunn_base, maks_base, sattrin_base, grov_base, _ = base_func()
        #%%
        
        self.where = "base"
        _, df_base = self.make_prediction(t_start,t_end,df,
                                       df_idv_index,
                                       bunn_base,
                                       maks_base,
                                       sattrin_base,
                                       grov_base)
                                    

        #%% Make scenario prediction
        bunn_scen, maks_scen, sattrin_scen, grov_scen, _ = scen_func()


        #
        self.where = "scen"
        _, df_scen = self.make_prediction(t_start,t_end,df,
                                       df_idv_index,
                                       bunn_scen,
                                       maks_scen,
                                       sattrin_scen,
                                       grov_scen)
        

        print("df_base.columns",df_base.columns)
        print("df_scen.columns",df_scen.columns)
        print("df_base.equals(df_scen)",df_base.loc[self.vTag].equals(df_scen.loc[self.vTag]))
        
        #%% Add knr to df_scen and df_base for ploting maps        
        df_knr = pd.DataFrame(df[:,0,findIndex(self.col_index,['knr'])],columns=['knr'])
        df_knr['KGB']= df_idv_index
        df_knr.set_index('KGB',inplace=True)
        
        #%% Merge knr on df_base
        firsts = df_base.index.get_level_values('KGB')
        df_base['knr'] = df_knr.loc[firsts].values
        
        #%% Merge knr on df_scen
        firsts = df_scen.index.get_level_values('KGB')
        df_scen['knr'] = df_knr.loc[firsts].values
        
        print("df_base.columns",df_base.columns)
        print("df_scen.columns",df_scen.columns)
        
        #%%
        return df_base, df_scen
    
    def plot_scen_map(self,vVar,year,df_base,df_scen):    
        """
        Plot a map with scenario results
        
        Inputs: 
        year = 2016    -> year as int
        vVar = ['SAU']  -> name of variable as list
        """
    
        import plotmap

        #%%
        aa = df_scen.loc[str(year),('knr',vVar[0])]
        dvVar = 'd'+vVar[0]
        aa[dvVar] = df_scen.loc[str(year),vVar[0]].values-df_base.loc[str(year),vVar[0]].values

        
        fig, plt, ax, aggKnr = plotmap.plot_agg_map(aa,dvVar,func='mean',saveName='map')
        
#% -----------------------------------------------------------------------
def prep_df_heat(df_base,df_scen,vVar,vVarPrev,iYear, 
                 filter_prev_positiv=True):
    """
    """
    #%%
    df_com_base = df_base.loc[iYear,[vVar]]
    df_com_scen = df_scen.loc[iYear,[vVar]]
    df_com_previous = df_scen.loc[str(int(iYear)-1),[vVarPrev]]
    
    df_com_base.index = df_com_base.index.droplevel('year')
    df_com_scen.index = df_com_scen.index.droplevel('year')
    df_com_previous.index = df_com_previous.index.droplevel('year')

    df_com_base.columns = ['base']
    df_com_scen.columns = ['scen']
    df_com_previous.columns = ['previous']
    
    df_concat = pd.concat([df_com_base,df_com_scen,df_com_previous],axis=1)
    
    if filter_prev_positiv:
        df_concat = df_concat.loc[df_concat['previous']>0,:].copy()
    
    
    df_concat['diff'] = df_concat['scen']-df_concat['base']
    #%%
    return df_concat

#% function to plot head map historgram
def plot_heat_hist(xVal,yVal,xlable, ylable,title,legendPosition = 3,
                   ylim_min = None, ylim_max = None):
    #%%
    # Prepare figure using a gridSpec
    fig = plt.figure()
    gspec = gridspec.GridSpec(5,5)
    lower_right = plt.subplot(gspec[1:,0:-1])
    sub_hist = plt.subplot(gspec[1:,-1],sharey=lower_right)
    act_hist = plt.subplot(gspec[0,:-1],sharex=lower_right)
    
    
    fig.suptitle(title, fontsize=14)
    # Create the activtiy historgram
    act_hist.hist(xVal,bins=50,density=True, color='gray')
    act_hist.tick_params(labelbottom=False)  
    
    act_hist.get_yaxis().set_visible(False)
    act_hist.spines['top'].set_visible(False)
    act_hist.spines['right'].set_visible(False)
    act_hist.spines['left'].set_visible(False)
    
    # Create the change historgram
    sub_hist.hist(yVal,bins=50,orientation='horizontal',
                  density=True, color='gray')
    sub_hist.get_xaxis().set_visible(False)
    sub_hist.tick_params(labelleft=False)
    
    sub_hist.spines['top'].set_visible(False)
    sub_hist.spines['bottom'].set_visible(False)
    sub_hist.spines['right'].set_visible(False)
    
    # Plot scatter as a 2d histogram
    scat = lower_right.hist2d(xVal,yVal,bins=50,
                              norm=mpl.colors.LogNorm(1,1000),
                            #   norm=mpl.colors.LogNorm(),
                              cmap=mpl.cm.cool,
                            #   cmap=mpl.cm.get_cmap('Greys')
                            # range=[[-np.inf, np.inf], [ylim_min, ylim_max]]
                              )
    
    if (ylim_min is not None)|(ylim_max is not None):
        lower_right.set_ylim(ylim_min,ylim_max)
    # Hider frame
    lower_right.spines['top'].set_visible(False)
    lower_right.spines['right'].set_visible(False)
    lower_right.set_xlabel(xlable, fontsize=14)
    lower_right.set_ylabel(ylable, fontsize=14)
    lower_right.tick_params(axis='both', which='major', labelsize=14)
    
    # Add color bar
    # cbaxes = inset_axes(lower_right,width="30%",height="3%",loc=7)
    ymin, ymax = lower_right.get_ylim()
    xmin, xmax = lower_right.get_xlim()
    yRange = ymax-ymin
    xRange = xmax-xmin

    cbaxes = inset_axes(lower_right,
                    width="30%",  
                    height="3.5%",
                    loc='upper right',
                    borderpad=-5.2
                   )
    cbar = plt.colorbar(scat[3], 
                        cax=cbaxes, 
                        orientation="horizontal",
                        label='#farms',
                        )

    cbar.set_ticks([1, 10,  100, 1000])
    cbar.set_ticklabels([1, 10, 100, 1000])
    cbar.minorticks_off()
    
    plt.tight_layout()

    #%%
    return fig
#%%
def make_plots(modelId = 'fcdf8418-b26a-11ea-9be8-42010aa4001a',
               scen_name = 'increase_tprod_SAU',
               vVarPrev = 'SAU',
               reports_path='reports'):
    #%%
    print(f'Creating plots for {scen_name}')
    #%
    # Create folder to hold all figures for model
    figPath = os.path.join(reports_path,'figures',modelId,'scenarios')
    pathlib.Path(figPath).mkdir(parents=True, exist_ok=True)
    #%%
    # Load model specAttr
    model = nn_predict(wd=wd)
    model.fullId = modelId
    model.model_load(mod_predict=False)
    specAttr = model.load_obj(model.savePath+"_paramStore.pkl" )
    
    #%%
    # Read csv
    df_base = pd.read_csv(os.path.join(reports_path,'scenarios',modelId,
                                       f'{modelId}_scen_{scen_name}_base.csv'),sep=';')
    df_scen = pd.read_csv(os.path.join(reports_path,'scenarios',modelId,
                                       f'{modelId}_scen_{scen_name}_scen.csv'),sep=';')
    df_base['year'] = pd.to_datetime(df_base['year'])
    df_scen['year'] = pd.to_datetime(df_scen['year'])
    df_base.set_index(['year','knr'],inplace=True)
    df_scen.set_index(['year','knr'],inplace=True)
    
    # Define cut-off values such that scales of x-axis is always the same
    xScaleMapping = {'SAU':600,
                    'x120':120}
    
    # Loop over all model target and prepare head map histogram
    iYear = '2016'
    #%%
    for vVar in model.vTag: 
        #%%
        # vVar = 'SAU'
        # xVar = 'SAU'
        xVar = vVarPrev
        
        df_concat = prep_df_heat(df_base,df_scen,vVar=vVar,vVarPrev=xVar,iYear=iYear)
        
        #%%
        # Translate codes to plain english
        import re 
        xVarClearName = xVar
        vVarClearName = vVar
        translation = {'x120':'dairy cows (head)',
                        'x121':'suckler cows (head)',
                        'x140':'female goats (head)',
                        'x155':'sows (head)',
                        'x160': 'hens (head)',
                        'x210': 'fodder arable land\n(daa=1/10ha)',
                        'x211': 'pasture arable land\n(daa=1/10ha)',
                        'x212': 'fodder non-arable land\n(daa=1/10ha)',
                        'x230': 'potatoes (daa=1/10ha)',
                        'CERE': 'crops (daa=1/10ha)',
                        'GEIT': 'male goats (head)',
                        'GRON': 'vegetables\n(daa=1/10ha)',
                        'SAU':  'sheep (head)',
                        'STOR': 'other cattle (head)'}
        for key in translation.keys():
            # Use regex in order to match exact work
            xVarClearName = re.sub(r'\b'+key+r'\b', translation[key], xVarClearName)
            vVarClearName = re.sub(r'\b'+key+r'\b', translation[key], vVarClearName)

        #%%
        print('!!! Cut outlier !!!')
        if df_concat['diff'].abs().quantile(0.99)>0:
            df_concat = df_concat.loc[df_concat['diff'].abs()<df_concat['diff'].abs().quantile(0.99),:]
        #%%
        # Apply cut-off values such that scales of x-axis is always the same
        if xVar in xScaleMapping:
            mask = df_concat['previous']<xScaleMapping[xVar]
            df_concat = df_concat.loc[mask,:]
        
        # Create plots    
        fig = plot_heat_hist(xVal=df_concat['previous'],
                    yVal=df_concat['diff'],
                    xlable='#'+xVarClearName+' in t-1', 
                    ylable= '$\Delta$ '+vVarClearName,
                    title=vVar)

        #%%
        fig.savefig(os.path.join(figPath,f"{modelId}_{scen_name}_prev_{xVar}_{vVar}.png"))
        plt.close('all')

    #%%
    # Get plots change in subsidy against number of vVarPrev
    for vVar, ylable in [
        (f'CsubD1_{vVarPrev}','$\Delta$ MS (in 1000 NOK)'),
        (f'DAvgSub_{vVarPrev}','$\Delta$ AS (in 1000 NOK)'),
        ('dSubTPROD','$\Delta$ TS (in 1000 NOK)'),
        ]:
        # xVar = 'SAU'
        xVar = vVarPrev
        selIdx = (df_scen.loc[:,xVar ]!=0) & (df_scen.loc[:,vVar ]!=0)
        df_sel= df_scen[selIdx]

        # Get data for one year
        df_sel_year = df_sel.loc['2015',:].copy()

        # Apply cut-off values such that scales of x-axis is always the same
        if xVar in xScaleMapping:
            mask = (df_sel_year.loc['2015',xVar ]<xScaleMapping[xVar]).array
            df_sel_year = df_sel_year.loc[mask,:]
        
        #
        fig = plot_heat_hist(xVal=df_sel_year.loc[:,xVar ],
                    yVal=df_sel_year.loc[:,vVar],
                    xlable=f'# {translation[vVarPrev]} in t-1', 
                    ylable= ylable,
                    title=vVar)
        #%%
        fig.savefig(os.path.join(figPath,f"{modelId}_{scen_name}_prev_{xVar}_{vVar}.png"))
        plt.close('all')
    
    #%%
    # Get changes in SAU againt changes in subsidies
    for vVar in [f'DAvgSub_{vVarPrev}','dSubTPROD',f'CsubD1_{vVarPrev}' ]:
        # yVar = 'SAU'
        yVar = vVarPrev
        iYear = '2016'
        df_diff = df_scen.loc[iYear,yVar]-df_base.loc[iYear,yVar]
        df_vVar = df_scen.loc[str(int(iYear)-1),vVar]
        df_diff.index = df_diff.index.droplevel('year')
        df_vVar.index = df_vVar.index.droplevel('year')
        df_comp = pd.concat([df_diff,df_vVar],axis=1)
        selIdx = (df_comp[yVar]!=0) | (df_comp[vVar]!=0)
        df_comp= df_comp[selIdx]
        
        fig = plot_heat_hist(xVal=df_comp[vVar],
                    yVal=df_comp[yVar],
                    xlable=vVar, 
                    ylable= '$\Delta$ '+yVar,
                    title=vVar)

        fig.show()
        fig.savefig(os.path.join(figPath,f"{modelId}_{scen_name}_diff_{yVar}_{vVar}.png"))
        plt.close('all')
    #
#%
def run_sim(scen_name='scen_increase_tprod_SAU',
            model_id ="fcdf8418-b26a-11ea-9be8-42010aa4001a",
            smoketest=False,reports_path='reports'):
    #%% Create new model and load spec
    model = nn_predict(wd=wd)
    #model.subId = '146'
    model.fullId = model_id
    
    # Get the scen function by name of scen
    scen_func = getattr(model,scen_name)
    
    #%%
    model.model_load(mod_predict=True)

    # Set Tx to 1 such that only one step head prediction are made
    model.Tx = 1
    model.Ty = 1
    model.mod_predict = True

    #%% Clear Graph in tensorflow backend
    K.clear_session()

    # Load estimator
    model.estimator = model.model(mod_predict=True)
    model.estimator.load_weights(model.savePath+"_weights-improvement.hdf5") 

    #%% Get data
    model.smoketest = smoketest
    if model.smoketest == True:
        
        print('--- Smoke test') 

        (X_orig_train, Y_orig_train, dat_train,
                    X_orig_dev, Y_orig_dev, dat_dev, 
                    X_orig_test, Y_orig_test, dat_test) \
            = model.data_load(file=TestTrainSplit_unitTest().targetFileName,
                            dev_size=0.4,getMinMax=False)
    else:

        (X_orig_train, Y_orig_train, dat_train,
                    X_orig_dev, Y_orig_dev, dat_dev, 
                    X_orig_test, Y_orig_test, dat_test) \
            = model.data_load(getMinMax=False)
        
    #%%specify start/end year
    year_start = 2016
    year_end = 2016  

    listyear = [y.year for y in model.t_index]

    #% Check if out of sample prediction should be made an extent listyear
    if year_end > listyear[-1]:
        listyear = list(np.append(listyear,list(np.arange(listyear[-1]+1,year_end+1))))

    t0 = listyear.index(year_start)

    #%%
    t_start = year_start
    t_end = year_end

    base_func = model.scen_const

    scen_name = model.fullId+'_'+scen_name

    df = dat_train[:,0:t0,:]
    df_idv_index = model.idv_index_train 

    #%% make prediction up to year_start to build up LSTM cell states
    #   i.e. if year_start = 2010 we make a prediction up to 2009 here
    df_base, df_scen = model.scen_comp(t_start,t_end,df,df_idv_index,base_func, scen_func)

    #%% Save results to csv for Klaus
    selVarSave = ['knr']+model.vTag+['dSubTPROD']
    selVarSave += list(df_base.columns[df_base.columns.str.contains('CsubD')])
    selVarSave += list(df_base.columns[df_base.columns.str.contains('DAvgSub_')])

    df_base.reset_index(inplace=True)
    df_base['year'] = pd.to_datetime(df_base['year'])
    df_base.set_index('year',inplace=True)
    df_scen.reset_index(inplace=True)
    df_scen['year'] = pd.to_datetime(df_scen['year'])
    df_scen.set_index('year',inplace=True)
    
    if smoketest:
        scen_name += '_smoke'
    
    # Create folder if not exists
    savePath = os.path.join(reports_path,'scenarios',model_id)
    pathlib.Path(savePath).mkdir(parents=True, exist_ok=True)
    
    df_base.loc[slice('2015-01-01','2017-01-01'),
                selVarSave].to_csv(
                    os.path.join(savePath, scen_name+'_base.csv'),
                    sep=';')
    df_scen.loc[slice('2015-01-01','2017-01-01'),
                selVarSave].to_csv(
                    os.path.join(savePath, scen_name+'_scen.csv'),
                    sep=';')
# %%
def translateVarName(var):
    """Translate variable names to clear names

    Args:
        var (str): variable name in code

    Returns:
        str: clear name
    """
    
    translation = {'x120':'dairy cows (head)',
                    'x121':'suckler cows (head)',
                    'x140':'female goats (head)',
                    'x155':'sows (head)',
                    'x160': 'hens (head)',
                    'x210': 'fodder arable land\n(daa=1/10ha)',
                    'x211': 'pasture arable land\n(daa=1/10ha)',
                    'x212': 'fodder non-arable land\n(daa=1/10ha)',
                    'x230': 'potatoes (daa=1/10ha)',
                    'CERE': 'crops (daa=1/10ha)',
                    'GEIT': 'male goats (head)',
                    'GRON': 'vegetables\n(daa=1/10ha)',
                    'SAU':  'sheep (head)',
                    'STOR': 'other cattle (head)'}
    for key in translation.keys():
        # Use regex in order to match exact work
        VarClearName = re.sub(r'\b'+key+r'\b', translation[key], var)
        
    return VarClearName


#%%
if __name__ == "__main__":
    # %%
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f","--figures", action="store_true", help="create figures")
    args, _ = parser.parse_known_args()
    # %%
    model_id = "15283bd4-51d2-11ec-8639-0242ac110003" # R2: 0.9329754728365416 
    # %%
    smoketest = False
    for scen_name in ['scen_increase_tprod_SAU',
                      'scen_flat_tprod_SAU',
                      'scen_increase_tprod_x120', 
                      'scen_flat_tprod_x120'
                      ]:
    
        run_sim(scen_name=scen_name, model_id=model_id,
                smoketest=smoketest)
  
    # %%
    if args.figures:
        # %%
        make_plots(modelId = model_id,
                scen_name = 'increase_tprod_SAU',
                vVarPrev = 'SAU')
        make_plots(modelId = model_id,
                scen_name = 'flat_tprod_SAU',
                vVarPrev = 'SAU')
        make_plots(modelId = model_id,
                scen_name = 'increase_tprod_x120',
                vVarPrev = 'x120')
        make_plots(modelId = model_id,
                scen_name = 'flat_tprod_x120',
                vVarPrev = 'x120')
        # %%
    else:
        print('No figures')

