# %%
# User tensorflow:2.3.0-gpu
# and pip install shap==0.36.0

# %%
from distutils.log import error
import sys
import os
from pathlib import Path
#%
wd = "/nn_norway"#
os.chdir(wd)
sys.path.append(wd)
#%
import numpy as np
from numpy.random import  choice
import shap
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tensorflow.keras import backend as K
import tensorflow as tf

# Import model
from src.models.nn_model import nn_model
from src.lib.utily import findIndex

# %%
def SHAP(model, Xbackground, Xtrail, n_samples = 500):

    #%%
    # Sample Background data (this is used to integate out SHAP values, see SHAP paper)
    random_ind = np.random.choice(Xbackground.shape[0],n_samples, replace=False)
    backgroundData = Xbackground[random_ind,:,:]
    
    Xtrail_raw, _ = model.data_inverse_transform(Xtrail,None)
    #%%
    # Setup GradientExplainer (Currently DeepExplainer does not work for tf 2.x)
    e = shap.GradientExplainer(model.estimator,backgroundData, batch_size=model.batch_size)
    #%%
    # Calculate SHAP values
    shp_val = e.shap_values(Xtrail)
    # Transform to array
    shp_val = np.array(shp_val)

    #%%
    # shp_val shape = (#outputs,#examples,#yearsInput,#featurs)
    aa = shp_val.swapaxes(0,3).reshape(shp_val.shape[3]*shp_val.shape[1]*shp_val.shape[2],1,shp_val.shape[0])
    _, aat = model.data_inverse_transform(None,aa)
    shp_val_raw = aat.reshape(shp_val.shape[3],shp_val.shape[1],shp_val.shape[2],shp_val.shape[0]).swapaxes(3,0)

    #%%
    return shp_val_raw, Xtrail_raw
# %%
# Function to create and save shape value plots
def plot_shape(model,shp_val_raw,
               Xte_raw,
            #    Xtr_raw, Ytr_raw,
               strOutput = 'SAU',
               lstExplVar = ['SAU','CsubD1_SAU','CsubD10_SAU','CsubD50_SAU', 'dSubTPROD','DAvgSub_SAU'],
               lstYearInput = [13,14,15]
               ):

    #%%
    for yearInput in lstYearInput:
        for strInput in lstExplVar:

            y_min = y_shp = np.min(shp_val_raw[
                findIndex(model.vTag,[strOutput])[0],:,:,
                findIndex(model.vFeat,[strInput])[0]])
            y_max = y_shp = np.max(shp_val_raw[
                findIndex(model.vTag,[strOutput])[0],:,:,
                findIndex(model.vFeat,[strInput])[0]])
            #%%
            y_shp = shp_val_raw[
            # y_shp = shp_val[
                findIndex(model.vTag,[strOutput])[0],:,yearInput,
                findIndex(model.vFeat,[strInput])[0]]
            x_var = Xte_raw[:,yearInput,findIndex(model.vFeat,[strInput])[0]]
            hasOuputVar = Xte_raw[:,-1,findIndex(model.vFeat,[strOutput])[0]]==0 	
        
            # Create and save scatter plot
            fig,ax = plt.subplots(tight_layout=True)
            scatter = ax.scatter(x_var,y_shp,s=0.2,c=hasOuputVar)

            handles, labels = scatter.legend_elements()
            labels = [f'Has {strOutput}',f'Does not have {strOutput}']
            ax.legend(handles, labels, loc="upper left",frameon=False)
            ax.set_ylabel(f'SHAP Value ($\Delta$ {strOutput})')
            ax.set_xlabel(f'{strInput}: t {yearInput}')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(([y_min,y_max]))
            
            fig.savefig(os.path.join('reports',
                            'figures', 'shap',
                            f"shp_{model.fullId}_{strOutput}_{strInput}_{yearInput}.png"),dpi=300)
                

# %%
# Function to create and save shape value plots
def plot_shape_featimport(model,shp_val_raw,
                strOutput = 'SAU',
                lstYearInput = [-3,-2,-1],
                sampleType = '',
                lstNumFeat = [15,120]
               ):
    
    #%%
    for numFeat in lstNumFeat:
        # Plot feature importance 
        for yearInput in lstYearInput:
            meanShap = np.abs(shp_val_raw[findIndex(model.vTag,[strOutput])[0],:,yearInput,:]).mean(axis=0)
            dfMeanShap = pd.DataFrame(meanShap,index=model.vFeat,columns=[f'MeanSHAP_{strOutput}'])
            dfMeanShap = dfMeanShap.sort_values(f'MeanSHAP_{strOutput}')

            strTitle = f'Feature Importance (input year: t{yearInput})'

            fig,ax = plt.subplots(tight_layout=True)
            if dfMeanShap.iloc[-1,:].name == strOutput:
                lstIdx = list(dfMeanShap.iloc[-numFeat:-1,:].index)
                print(f'Excluding laged {strOutput} as the most important variable')
                strTitle += f'\n(Excluding lagged {strOutput} as the most important variable)'
            else:
                lstIdx = list(dfMeanShap.iloc[-numFeat:,:].index)
            ax.barh(lstIdx,dfMeanShap.loc[lstIdx,f'MeanSHAP_{strOutput}'])
        
            ax.set_xlabel('mean(|SHAP value|)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(strTitle)	
            # Note: The absolute value of the SHAP values gives in indication
            # 	about feature importance but beyond this the absolute 
            # 	value does not really has a direct interpreation 
            
            if numFeat==120:
                # Set the tick labels font
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(3)
  
            fig.savefig(os.path.join('reports',
                            'figures', 'shap',
                            f"featureImportance_{model.fullId}_{sampleType}_{strOutput}_{yearInput}_{numFeat}.png"),dpi=300)
    
# %%
# Function to create and save shape value plots
def plot_shape_featimport_time(model,shp_val_raw,
                strOutput = 'SAU',
                lstYearInput = [-1,-2,-3,-4],
                sampleType = '',
                numFeat = 15,
                strFileSuffix = ''
               ):
    #%%
    translation = {'x120':'dairy cows',
            'x121':'suckler cows',
            'x136':'lamb',
            'x140':'female goats',
            'x155':'sows',
            'x160': 'hens',
            'x210': 'fodder (arable land)',
            'x211': 'pasture (arable land)',
            'x212': 'fodder (non-arable land)',
            'x230': 'potatoes',
            'CERE': 'crops',
            'GEIT': 'male goats',
            'GRON': 'vegetables',
            'SAU':  'sheep',
            'STOR': 'other cattle',
            'DAvgSub_':'$\Delta Avg$ ',
            'dSub':'$\Delta tot$ ',
            'CsubD1_':'$\Delta M_{\Delta 1}$ ',
            'CsubD10_':'$\Delta M_{\Delta 10}$ ',
            'CsubD50_':'$\Delta M_{\Delta 50}$ ',
            'DPAY':'payments',
            'TAKTL':'Acreage pay.',
            'TBEIT':'Animal pay. meadows',
            'TDISE':'Output pay. egg',
            'TDISG':'Output pay. veg./fruits',
            'TDISK':'Output pay. meat',
            'TDISM':'Output pay. milk',
            'TDMLK':'Income support milk',
            'TGRUN':'Nat./reg. output pay.',
            'TPROD':'Animal pay.',
            'TUTMK':'Animal pay. outlying fields',
            'TVELF':'Welfare payments'
            }
    
    #%%
    meanShap = np.abs(shp_val_raw[findIndex(model.vTag,[strOutput])[0],:,lstYearInput,:]).squeeze().mean(axis=1)
    dfMeanShap = pd.DataFrame(meanShap,columns=model.vFeat,index=lstYearInput).transpose()
    
    dfMeanShap = dfMeanShap.loc[dfMeanShap.index.str.contains('|'.join(['DAvgSub','Csub','dSub'])),:]
    
    dfMeanShap = dfMeanShap.sort_values(-1)
    #%
    for k,v in translation.items():
        dfMeanShap.index = dfMeanShap.index.str.replace(k,v)
    dfMeanShap
    #%%
    lstIdx = list(dfMeanShap.iloc[-numFeat:,:].index)

    plt.style.use('seaborn')

    fig,ax = plt.subplots(tight_layout=True)
    lstHatch = ['///','**','xxx',None]
    for year in lstYearInput:
        ax.barh(lstIdx,dfMeanShap.loc[lstIdx,year],hatch = lstHatch[year], label=f"year: t{year}")
    #%
    # Set title

    strTitle = f'Feature Importance for "{translation[strOutput]}"'
    # Show legend
    ax.set_title(strTitle)
    ax.legend(frameon=False)
    ax.set_xlabel('mean(|SHAP value|)')
    
    if numFeat>15:
        # Set the tick labels font
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(8)
    #%
    savePath = os.path.join('reports','figures', model.fullId,'shap')
    Path(savePath).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(savePath,
                    f"featureImportance_time_{model.fullId}_{sampleType}_{strOutput}_numFeat{numFeat}{strFileSuffix}.png"),dpi=300)


# %%
def plot_shape_featimport_AbsMarg(model,
                                  sampleType,
                                  Xtrail_raw,
                                  shp_val_raw,
                                  strFileSuffix= ''):
    #%
    translation = {'x120':'dairy\ncows',
        'x121':'suckler\ncows',
        'x140':'female\ngoats',
        'x155':'sows',
        'x160': 'hens',
        'x210': 'fodder\n(arable land)',
        'x211': 'pasture\n(arable land)',
        'x212': 'fodder\n(non-arable\nland)',
        'x230': 'potatoes',
        'CERE': 'crops',
        'GEIT': 'male\ngoats',
        'GRON': 'vegeta.',
        'SAU':  'sheep',
        'STOR': 'other\ncattle',
        'DAvgSub':'$\Delta Avg$',
        'CsubD1':'$\Delta M$',
        }    
    resShap = []
    for strOutput in model.vTag:
        # Using only those observation that have the activity
        maskAct = Xtrail_raw[:,-2,findIndex(model.vFeat,[strOutput])[0]]>0
        shp_val_raw_mask =  shp_val_raw[:,maskAct,:,:].copy()
        
        meanShap = np.abs(shp_val_raw_mask[findIndex(model.vTag,[strOutput])[0],:,-1,
                    findIndex(model.vFeat,['DAvgSub_'+strOutput,'CsubD1_'+strOutput])]).mean(axis=1)
        dfShap_i = pd.DataFrame(meanShap,index=['DAvgSub','CsubD1'],columns=[strOutput]).transpose()    
        resShap.append(dfShap_i)
    dfShap = pd.concat(resShap)

    dfsum = dfShap.sum(axis=1).copy()
    dfShap['DAvgSub'] = dfShap['DAvgSub']/dfsum
    dfShap['CsubD1'] = dfShap['CsubD1']/dfsum
    dfShap = dfShap.sort_values('DAvgSub')
    dfShap.rename(columns={c:translation[c] for c in dfShap.columns},inplace=True)
    dfShap.rename(index={c:translation[c] for c in dfShap.index},inplace=True)
    #%
    plt.style.use('seaborn')
    fig,ax = plt.subplots(tight_layout=True)

    dfShap.plot.bar(ax=ax,color=['gray','darkgray'])
    strTitle = 'Relative Feature Importance of Average/Marginal subsidy changes'
    ax.set_title(strTitle)
    ax.set_ylabel('Relative Feature importance')
    
    bars = ax.patches
    hatches = ['///']*dfShap.shape[0]+[None]

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.legend()
    
    savePath = os.path.join('reports','figures', model.fullId,'shap')
    Path(savePath).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(savePath,
                    f"featureImportance_relative_{model.fullId}_{sampleType}_{strFileSuffix}.png"),dpi=300)


# %%
def inspect_short_time_model():
    
    # %%
    modelA = nn_model()
    modelB = nn_model()
 
    modelA.Shapely = True
    modelB.Shapely = True
    modelA.fullId = model_id
    modelB.fullId = model_id
    modelA.model_load(mod_predict=True, Shapely=True)
    modelB.model_load(mod_predict=True, Shapely=True)
    
    yhatA = modelA.model_predict(Xde_raw)

    # %%
    # Xde_short = Xde
    # Xde_raw_short = Xde_raw
    Xde_short = Xde[:,11:16,:]
    Xde_raw_short = Xde_raw[:,11:16,:]
    
    modelB.Tx = Xde_short.shape[1]
    modelB.estimator = modelB.model(mod_predict=True)
    modelB.estimator.load_weights(modelB.savePath+"_weights-improvement.hdf5") 
    modelB.estimator.summary()
    
    yhatB = modelB.model_predict(Xde_raw_short)
    
    print('Array equal?', np.allclose(yhatA,yhatB))
    # %%
    for k in range(0,len(modelA.vTag)):
        # %%
        k = 12
        plt.scatter(yhatA[0,:,k],yhatB[0,:,k])
        print(modelA.vTag[k])
        aa = pd.DataFrame([yhatA[0,:,k],yhatB[0,:,k]]).transpose()
        aa.loc[aa.sum(axis=1)>0,:]

    # %%
    # %%
    # Compare model performance in train/test set
    modelA = nn_model()
    model_id = "2a16da08-4630-11ec-ac68-0242ac110003" # newly trained model  (R2: 0.9355343020921835, Seed: 16)
    modelA.fullId = model_id
    modelA.model_load()
    # Load data
    (Xtr_raw, Ytr_raw, dat_train,
        Xde_raw, Yde_raw,dat_dev, 
        Xte_raw, Yte_raw,dat_test) = modelA.data_load(getMinMax=False)
    # %%
    yhat_tr = modelA.model_predict(Xtr_raw)
    yhat_te = modelA.model_predict(Xte_raw)
    # %%
    modelA.model_performance(Ytr_raw,yhat_tr)
    # %%
    modelA.model_performance(Yte_raw,yhat_te)
#%%
def inspect_shap():
    # %%
    # Create two differnt models with different input time
    modelA = nn_model()
    modelB = nn_model()
    model_id = "2a16da08-4630-11ec-ac68-0242ac110003" # newly trained model  (R2: 0.9355343020921835, Seed: 16)
   
    modelA.Shapely = True
    modelB.Shapely = True
    modelA.fullId = model_id
    modelB.fullId = model_id
    modelA.model_load(mod_predict=True, Shapely=True)
    modelB.model_load(mod_predict=True, Shapely=True)
    
    # %%
    # Load data
    (Xtr_raw, Ytr_raw, dat_train,
        Xde_raw, Yde_raw,dat_dev, 
        Xte_raw, Yte_raw,dat_test) = model.data_load(getMinMax=False)
    # %%
    # Xtrail_A = Xde[:,11:16,:]
    Xtrail_A = Xde[:50,11:16,:]
    Xbackground_A = Xtr[:,11:16,:]
    # Xtrail_B = Xde[:,11:15,:]
    Xtrail_B = Xde[:50,11:15,:]
    Xbackground_B = Xtr[:,11:15,:]
    
    modelA.Tx = Xtrail_A.shape[1]
    modelB.Tx = Xtrail_B.shape[1]
    modelA.estimator = modelA.model(mod_predict=True)
    modelB.estimator = modelB.model(mod_predict=True)
    modelA.estimator.load_weights(modelA.savePath+"_weights-improvement.hdf5") 
    modelB.estimator.load_weights(modelB.savePath+"_weights-improvement.hdf5") 
    modelA.estimator.summary()
    modelB.estimator.summary()

    
    # %%
    n_samples = 100
    # Sample Background data (this is used to integate out SHAP values, see SHAP paper)
    random_ind_A = np.random.choice(Xbackground_A.shape[0],n_samples, replace=False)
    backgroundData_A = Xbackground_A[random_ind_A,:,:]
    random_ind_B = np.random.choice(Xbackground_B.shape[0],n_samples, replace=False)
    backgroundData_B = Xbackground_B[random_ind_B,:,:]
    
    Xtrail_raw_A, _ = modelA.data_inverse_transform(Xtrail_A,None)
    Xtrail_raw_B, _ = modelB.data_inverse_transform(Xtrail_B,None)
    #%%
    # Setup GradientExplainer (Currently DeepExplainer does not work for tf 2.x)
    e_A = shap.GradientExplainer(modelA.estimator,backgroundData_A, batch_size=modelA.batch_size)
    e_B = shap.GradientExplainer(modelB.estimator,backgroundData_B, batch_size=modelB.batch_size)
    # %%
    # Calculate SHAP values
    shp_val_A = e_A.shap_values(Xtrail_A)
    # %%
    shp_val_B = e_B.shap_values(Xtrail_B)

    # %%
    shp_val_A = np.array(shp_val_A)
    shp_val_B = np.array(shp_val_B)
    
    aa_A = shp_val_A.swapaxes(0,3).reshape(shp_val_A.shape[3]*shp_val_A.shape[1]*shp_val_A.shape[2],1,shp_val_A.shape[0])
    _, aat_A = model.data_inverse_transform(None,aa_A)
    shp_val_A_raw = aat_A.reshape(shp_val_A.shape[3],shp_val_A.shape[1],shp_val_A.shape[2],shp_val_A.shape[0]).swapaxes(3,0)
    
    aa_B = shp_val_B.swapaxes(0,3).reshape(shp_val_B.shape[3]*shp_val_B.shape[1]*shp_val_B.shape[2],1,shp_val_B.shape[0])
    _, aat_B = model.data_inverse_transform(None,aa_B)
    shp_val_B_raw = aat_B.reshape(shp_val_B.shape[3],shp_val_B.shape[1],shp_val_B.shape[2],shp_val_B.shape[0]).swapaxes(3,0)
    
    # %%
    np.array_equal(shp_val_A_raw, shp_val_B_raw)
    # %%
    np.array_equal(shp_val_A, shp_val_B)


#%%
if __name__ == "__main__":
    # %%
    
    # smoke_test = True
    smoke_test = False
    if smoke_test:
        print("!!!! Runing a smoke test !!!! ")    
        
    lstSampleType = ['shortTimeSuffel']    
    
    for sampleType in lstSampleType:
        # %%
        print(f"Running: {sampleType}")
        model = nn_model()
    
        model_id = "15283bd4-51d2-11ec-8639-0242ac110003" # R2: 0.9329754728365416
        # %%
        # Setup and load model
        model.Shapely = True
        model.fullId = model_id
        model.model_load(mod_predict=True, Shapely=True)
        # %%
        # Load data
        (Xtr_raw, Ytr_raw, dat_train,
            Xde_raw, Yde_raw,dat_dev, 
            Xte_raw, Yte_raw,dat_test) = model.data_load(getMinMax=False)
        # %%
        # Get transformed train/test data
        Xtr, _ =model.data_transform(Xtr_raw, None)
        Xde, _ =model.data_transform(Xde_raw, None)
        Xte, _ =model.data_transform(Xte_raw, None)
        Xtr = Xtr[0]
        Xde = Xde[0]
        Xte = Xte[0]
        # %%
        # =================
        # Prepare data to calculate shap values
        # =================
        if 'shortTime' in sampleType:
            
            if sampleType == 'shortTimeSuffel':
                # %%
                # Create a artificual sample by 
                # 1) limiting the time diminsions to 5 years
                # 2) getting random draws (for 5 year slices) from dev set 
                #    from differnt time period
                # Note: This approach ensure that their is enough variation in the policy variable.
                #       The alternative approach (see below, simply using the dev set) has the
                #       potential disadvantage there their might not be a polciy change in the 
                #       last year (or not enough variation) impacting the feature importance 
                #       calcuation. In the extrem case, if there is no change in policy, 
                #       policy variables should not have an effect and hence not feature importance.        # 
                n_Xb_samples = 1000
        
                Xb_N = Xtr.shape[0]
                Xbackground = np.concatenate((
                                Xtr[choice(Xb_N,n_Xb_samples, replace=False),0:5,:],
                                Xtr[choice(Xb_N,n_Xb_samples, replace=False),1:6,:],
                                Xtr[choice(Xb_N,n_Xb_samples, replace=False),2:7,:],
                                Xtr[choice(Xb_N,n_Xb_samples, replace=False),3:8,:],
                                Xtr[choice(Xb_N,n_Xb_samples, replace=False),4:9,:],
                                Xtr[choice(Xb_N,n_Xb_samples, replace=False),5:10,:],
                                Xtr[choice(Xb_N,n_Xb_samples, replace=False),6:11,:],
                                Xtr[choice(Xb_N,n_Xb_samples, replace=False),7:12,:],
                                Xtr[choice(Xb_N,n_Xb_samples, replace=False),8:13,:],
                                Xtr[choice(Xb_N,n_Xb_samples, replace=False),9:14,:],
                                Xtr[choice(Xb_N,n_Xb_samples, replace=False),10:15,:],
                                Xtr[choice(Xb_N,n_Xb_samples, replace=False),11:16,:]
                                ))
                Xde_N = Xde.shape[0]
                n_samples = 500
                Xtrail = np.concatenate((
                                Xde[choice(Xde_N,n_samples, replace=False),0:5,:],
                                Xde[choice(Xde_N,n_samples, replace=False),1:6,:],
                                Xde[choice(Xde_N,n_samples, replace=False),2:7,:],
                                Xde[choice(Xde_N,n_samples, replace=False),3:8,:],
                                Xde[choice(Xde_N,n_samples, replace=False),4:9,:],
                                Xde[choice(Xde_N,n_samples, replace=False),5:10,:],
                                Xde[choice(Xde_N,n_samples, replace=False),6:11,:],
                                Xde[choice(Xde_N,n_samples, replace=False),7:12,:],
                                Xde[choice(Xde_N,n_samples, replace=False),8:13,:],
                                Xde[choice(Xde_N,n_samples, replace=False),9:14,:],
                                Xde[choice(Xde_N,n_samples, replace=False),10:15,:],
                                Xde[choice(Xde_N,n_samples, replace=False),11:16,:]
                                ))
                # %%
                # Adjuste the size of the input to the model by changing self.Tx
                model.Tx = Xtrail.shape[1]
                # %%
                # Create new model with new input size
                model.estimator = model.model(mod_predict=True)
                # Load trained weights in the new model
                model.estimator.load_weights(model.savePath+"_weights-improvement.hdf5") 
                model.estimator.summary()
                
                # %% 
                # ---------------------------------
                # Manipulating background values
                # ---------------------------------
                # Set all policy variables to zero for background
                lstVarSub = [f for f in model.vFeat if (('Csub' in f) or ('DAvgSub' in f) or ('dSub' in f))]
                Xbackground[:,:,findIndex(model.vFeat,lstVarSub)] = 0
                # %% 
                # Set all price variable equal to the last year for the background
                lstVarPrice = ['pWheat','pRye','pBarley','pOat','pOtherCereals','pRape',
                    'pPeas','pPotatoes','pVegGreenhouse','pVegFields',
                    'pApplesPeaches','pOtherFruits','pFlowers',
                    'pCowMilk',
                    'pGoatMilk','pBeef','pVeal','pPork','pSheepMeat',
                    'pPoultry','pEgg']
                Xbackground[:,:,findIndex(model.vFeat,lstVarPrice)] = Xbackground[0,-1,findIndex(model.vFeat,lstVarPrice)] 
                
            else:
                # %%
                timeSlice = [int(n) for n in sampleType.replace('shortTime','').split('_')]
                s = timeSlice[0] 
                e = timeSlice[1] 
                # %%
                Xbackground = Xtr[:,s:e,:]
                Xtrail = Xde[:,s:e,:]
                #%%
                # Adjuste the size of the input to the model by changing self.Tx
                model.Tx = Xtrail.shape[1]
                # Create new model with new input size
                model.estimator = model.model(mod_predict=True)
                # Load trained weights in the new model
                model.estimator.load_weights(model.savePath+"_weights-improvement.hdf5") 
                model.estimator.summary()
        
        elif sampleType == 'regular':
            
            # Simple use dev set to calculate shape values
            Xbackground = Xtr # Use train set to sample background
            Xtrail = Xde # Use dev set for shap values
        else:
            raise Warning('sample Type not correct')
        # %%
        if smoke_test:
            Xtrail = Xtrail[choice(Xtrail.shape[0],50, replace=False)]
            
        # %%
        trainShap=True
        if trainShap:
            # %%
            n_samples = 1000
            shp_val_raw, Xtrail_raw = SHAP(model, 
                                Xbackground = Xbackground, 
                                Xtrail = Xtrail, 
                                n_samples = n_samples,
            )
            #%%
            # print("shp_val_raw.shape",shp_val_raw.shape) 
            print(f"shp_val_raw_{model.fullId}_{sampleType}.npy")
            np.save(os.path.join('reports','shap',f"shp_val_raw_{model.fullId}_{sampleType}.npy"),
                    shp_val_raw,allow_pickle=True)
            np.save(os.path.join('reports','shap',f"Xtrail_raw_{model.fullId}_{sampleType}.npy"),
                    Xtrail_raw,allow_pickle=True)
        # %%
        else: 
            # %%
            shp_val_raw = np.load(os.path.join('reports','shap',f"shp_val_raw_{model.fullId}_{sampleType}.npy"))
            Xtrail_raw = np.load(os.path.join('reports','shap',f"Xtrail_raw_{model.fullId}_{sampleType}.npy"))

        # %%
        # Plot relative importance of abs/marginal policy variables
        plot_shape_featimport_AbsMarg(model,
                                  sampleType,
                                  Xtrail_raw,
                                  shp_val_raw,
                                  strFileSuffix= '')

        
        
        
        # %%
        for strOutput in model.vTag: 

            maskAct = Xtrail_raw[:,-2,findIndex(model.vFeat,[strOutput])[0]]>0
            shp_val_raw_mask =  shp_val_raw[:,maskAct,:,:].copy()
            plot_shape_featimport_time(model,shp_val_raw_mask,
                strOutput = strOutput,
                lstYearInput = [-1,-2,-3,-4],
                sampleType = sampleType,
                # numFeat = 15
                numFeat = 40,
                strFileSuffix = 'mask'
               )
        

    # %%
    # =========================
    # Plot Total variation across all years
    # =========================
    plotTotalVariation = True
    if plotTotalVariation:
        # %%
        from nn_scenario import plot_heat_hist
        savePath = os.path.join('reports','figures', model.fullId,'fullVariation')
        Path(savePath).mkdir(parents=True, exist_ok=True)
        
        translation = {'x120':'dairy cows',
            'x121':'suckler cows',
            'x136':'lamb',
            'x140':'female goats',
            'x155':'sows',
            'x160': 'hens',
            'x210': 'fodder (arable land)',
            'x211': 'pasture (arable land)',
            'x212': 'fodder (non-arable land)',
            'x230': 'potatoes',
            'CERE': 'crops',
            'GEIT': 'male goats',
            'GRON': 'vegetables',
            'SAU':  'sheep',
            'STOR': 'other cattle',
            'DAvgSub_':'$\Delta Avg$ ',
            'dSub':'$\Delta tot$ ',
            'CsubD1_':'$\Delta M_{\Delta 1}$ ',
            'CsubD10_':'$\Delta M_{\Delta 10}$ ',
            'CsubD50_':'$\Delta M_{\Delta 50}$ ',
            'DPAY':'payments',
            'TAKTL':'Acreage pay.',
            'TBEIT':'Animal pay. meadows',
            'TDISE':'Output pay. egg',
            'TDISG':'Output pay. veg./fruits',
            'TDISK':'Output pay. meat',
            'TDISM':'Output pay. milk',
            'TDMLK':'Income support milk',
            'TGRUN':'Nat./reg. output pay.',
            'TPROD':'Animal pay.',
            'TUTMK':'Animal pay. outlying fields',
            'TVELF':'Welfare payments'
            }
        
        for strX in model.vTag:
            
            for policyType in ['DAvgSub','CsubD1']:
                
                strY = f'{policyType}_{strX}'

                yVal = Xtr_raw[:,:,findIndex(model.vFeat,[strY])].flatten()
                xVal = Xtr_raw[:,:,findIndex(model.vFeat,[strX])].flatten()
                
                xlable = translation[strX]
                
                if policyType == 'DAvgSub':
                    ylable = f"$\Delta Avg$ {translation[strX]}"
                elif policyType == 'CsubD1':
                    ylable = f"$\Delta M$ {translation[strX]}"
                else:
                    raise Exception()
                
                
                mask = (xVal!=0)
                fig = plot_heat_hist(xVal[mask],yVal[mask],
                                     xlable = xlable, 
                                     ylable = ylable, title = '',legendPosition = 3)

                fig.savefig(os.path.join(savePath,
                    f"totalVariance_{model_id}_{strX}_{strY}.png"),dpi=300)
                

# %%
