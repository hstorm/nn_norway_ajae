#%%
"""
Model class for nn_norway
"""
import os
import sys
import uuid
import random

# Set working dir
wd = "/nn_norway"#
os.chdir(wd)
sys.path.append(wd)

#from importlib import reload
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras as ke
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Lambda 
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers
import tensorflow as tf

from src.features.luigi_features import TestTrainSplit_unitTest
from src.lib.utily import train_test_split_panel
from src.lib.utily import findIndex

# %%
# =============================================================================
#  Train model
# =============================================================================

class nn_model():

    def __init__(self, smoketest=False, mod_predict=False, wd=""):
        
        self.project_path = wd
        self.dataPath = os.path.join(self.project_path, "data")
        self.model_path = os.path.join(self.project_path,'models')
        
        self.log_path = os.path.join(self.model_path,'Logs')
        if not os.path.isdir(self.log_path):
            os.mkdir(self.log_path)  
            
        # Define target variables
        self.vTag = ['x120',
                    'x121',
                    'x140',
                    'x155',
                    'x160',
                    'x210',
                    'x211',
                    'x212',
                    'x230',
                    'CERE',
                    'GEIT',
                    'GRON',
                    'SAU',
                    'STOR',
         ]
        self.vTag_description = {
            'x120':"Dairy cows",
            'x121':"Suckler cows for special meat production",
            'x140':"Female goat over 1 year /Milkgoat",
            'x155':"Sows for breeding with minimum one litter",
            'x160':"Laying hens at counting date / over 20 weeks",
            'x210':"Fodder",
            'x211':"Fodder",
            'x212':"Fodder",
            'x230':"Potatoes",
            'CERE':"Ceral",
            'GEIT':"male goat",
            'GRON':"Vegetables outside",
            'SAU':"Sheep",
            'STOR':"Cattle",
        }
  
        # Define feature variables
        self.vFeat = (self.vTag 
            + ['age']
            + ['CsubD1_CERE', 'CsubD1_GRON', 'CsubD1_SAU', 'CsubD1_STOR',
                'CsubD1_GEIT',
                'CsubD1_x210','CsubD1_x211', 'CsubD1_x212', 'CsubD1_x230', 'CsubD1_x120',
                'CsubD1_x121', 'CsubD1_x140',  'CsubD1_x155',
                'CsubD1_x136',  'CsubD1_x160']
            + ['DAvgSub_CERE', 'DAvgSub_GRON', 'DAvgSub_SAU', 'DAvgSub_STOR',
                'DAvgSub_GEIT',
                'DAvgSub_x210','DAvgSub_x211', 'DAvgSub_x212', 'DAvgSub_x230', 'DAvgSub_x120',
                'DAvgSub_x121', 'DAvgSub_x140', 'DAvgSub_x155',
                'DAvgSub_x136',  'DAvgSub_x160']
            + ['dSubDPAY','dSubTAKTL','dSubTBEIT','dSubTDISE','dSubTDISG',
                'dSubTDISK','dSubTDISM','dSubTDMLK','dSubTGRUN','dSubTPROD',
                'dSubTUTMK','dSubTVELF']
            + ['pWheat','pRye','pBarley','pOat','pOtherCereals','pRape',
                'pPotatoes','pVegGreenhouse','pVegFields',
                'pCowMilk',
                'pGoatMilk','pBeef','pVeal','pPork','pSheepMeat',
                'pPoultry','pEgg']
                )
        
        self.smoketest = smoketest
        self.fullId = ''
        self.Xmin = []
        self.Xmax = []
        self.vVarMinMax = []
        self.t_index = []
        
        # Needs to be set here because it is required in data_reshaper
        self.savePath = os.path.join(self.model_path,self.fullId)
        self.mod_predict = mod_predict
        self.use_checkpoint=True
        self.load_weights = False
        self.Shapely = False
        self.testing_losses = True
        self.n_a = 512
        self.batch_size = 512
        self.input_dropout_LSTM = 0.2
        self.recurrent_dropout_LSTM = 0.2
        
        # These settings come for best model found with hyperparameter tuning using RayTune 
        self.l2_activity_LSTM = 0.
        self.l1_activity_LSTM = 0.
        self.l2_kernel_LSTM = 0.
        self.l1_kernel_LSTM = 0.
        self.l1_bias_LSTM = 0.008456211707985446
        self.l2_bias_LSTM = 0.06804052560676127

        self.l1_l2_activity_LSTM = regularizers.l1_l2(self.l1_activity_LSTM,self.l2_activity_LSTM)
        self.l1_l2_kernel_LSTM = regularizers.l1_l2(self.l1_kernel_LSTM,self.l2_kernel_LSTM)
        self.l1_l2_bias_LSTM = regularizers.l1_l2(self.l1_bias_LSTM,self.l2_bias_LSTM)
        self.reduce_LR_OnPlateu = True
        self.activation_output_regression = 'relu'
        self.activation_LSTM_regression = "sigmoid"
        self.activation_output_binary = "sigmoid"
        self.activation_LSTM_binary = "sigmoid"
        self.loss = 'mean_absolute_error'
        self.binary_loss = 'hinge'
        self.lr =  1e-05
        self.opt = Nadam(learning_rate=self.lr)
        self.use_BN = True
        if self.smoketest== False:
            self.epochs = 1200
        else:
            self.epochs = 1
        self.verbose = 1
        self.random_state = 42

           
    def save_obj(self,obj, name):
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self,name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def model_inputs(self):
        """
        Setup model inputs and initial states. Used in model and model_pred
        """
        #%% Define the input of the model with shape
        # defining Batchsize as None is neccecary for variable batchsize in prediction etc.
        print('call model_inputs: self.Tx = ',self.Tx)
        batch_size = None
        X = Input(shape=(self.Tx,self.kx), batch_size=batch_size, name="Input")
        # a for output and c for hiddenstate
        a0 = Input(shape=self.n_a, batch_size=batch_size,name='a0')
        c0 = Input(shape=self.n_a, batch_size=batch_size,name='c0')
        
        return X, a0, c0



    def model_setup (self,mod_predict=False):

        """
        Create a LSTM model with two layers in the the output
        n_a : Number of hidden_units used in the model
        """
        
        # Setup model inputs and initial states in respect to the model mode
        X, a0, c0= self.model_inputs()
        
        # computing with shapely values does not handle initial states well,
        # fortunatly the initial states are by default the same as constructet
        # by .model_inputs(); so a simple ommision suffices
        # shap woould compute the relatonship between the initial state and the prediction
        # this would cost ressources while producing garbage data
        if not self.Shapely :	
            initialstates = [a0,c0]
        else:
            initialstates = None

        a, _, c = LSTM(units=self.n_a, 
                              dropout=self.input_dropout_LSTM,
                              recurrent_dropout=self.recurrent_dropout_LSTM,
                              activation = self.activation_LSTM_regression,
                              activity_regularizer=regularizers.l1_l2(self.l1_activity_LSTM,self.l2_activity_LSTM),
                              kernel_regularizer=regularizers.l1_l2(self.l1_kernel_LSTM,self.l2_kernel_LSTM),
                              bias_regularizer=regularizers.l1_l2(self.l1_bias_LSTM,self.l2_bias_LSTM),
                              kernel_initializer = ke.initializers.Orthogonal(gain=2.0, seed=None),
                              recurrent_initializer = ke.initializers.Orthogonal(gain=2.0, seed=None),
                              return_state = True,
                              return_sequences=True) (X,initial_state= initialstates)
                                
        a_out = tf.identity(a, name="a_out")

        # sends the Lstm output through BatchNormalisation
        if self.use_BN:
            a = ke.layers.BatchNormalization()(a)

        # a finals dense layer to serve as postion for the predited variables, has length ky == number of predicted variables
        out = Dense(self.ky,
                    activation=self.activation_output_regression,
                    name="regression")(a)
        
        # depending on wether or not the model is predicting or training different outputs are provided
        print("mod_predict",mod_predict)
        if mod_predict ==False:
            #the case of training does not require the internal states, to be exported
            outputs = {"regression":out} 
            
        else:
            # while predicting, the internal states are exportet, to make a timestepwise prediction possible
            # the in t exportet states will be fed as intial states in t+1
            outputs = {'regression':out,
                               'a':a_out,
                               'c':c }
            
        inputs = [X,a0,c0]
                   
        # shapely requieres the time dimension to be flattened into timesteps x variables
        if self.Shapely:
            
            outputs = Lambda(lambda x: x[:,-1,:], 
                    output_shape=(tf.shape(out)[0],len(self.vTag)),
                    name='Flatten')(out)
   
            inputs = X

        model = Model(inputs=inputs, outputs=outputs)
        print(model.summary())
        model = self.model_compile(model)

        return model

    def model_compile(self,model):

        model_type = "regression"
        if not self.mod_predict:
            loss_weights_dict={model_type:1}		
        else:
            loss_weights_dict={
                        model_type:1,
                        "a":None,
                        "c":None}	
            
        model.compile(loss="mean_absolute_error",
                      loss_weights=loss_weights_dict,
                      optimizer= self.opt
                      )
        return model

    def model(self,mod_predict=False,from_load=False):
        """
        Build complet model and compile it
        """
        #Setup model achitecture
        model = self.model_setup(mod_predict=mod_predict)

        return model
    
    def model_callbacks(self):

        """
        Create model callbacks
        """
        #% specify log directory
        log_dir = os.path.join(self.log_path , self.fullId)
        
        # Tensorboard is a visualisation and gathering tool for training metrics
        tbCallBack =ke.callbacks.TensorBoard(log_dir=log_dir,
                                             histogram_freq=10,
                                             write_grads=False,
                                             write_graph=False)
        
        min_delta=0.0001 # original value from Alex
        # min_delta=0.00001
        patience=10
        
        # Define a EarlyStopping monitoring callback
        earlyStopping = ke.callbacks.EarlyStopping(monitor='val_loss', 
                                   min_delta=min_delta, 
                                   patience=patience, 
                                   verbose=1, 
                                   mode='min', 
                                   )
        
        #this callback multiplies the learning rate by a difined faktor
        # if the validation_loss has not changed for patience Epochs
        ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='auto', min_delta=0.00001, cooldown=5, min_lr=0)

        # Checkpoint saved improvements of the model to an own weights file, wich is overwritten with each improvement
        filepath=os.path.join(self.model_path,self.fullId+"_weights-improvement.hdf5")
        checkpoint = ke.callbacks.ModelCheckpoint(filepath, monitor='val_loss', 
                                verbose=1, save_weights_only=True,save_best_only=True, mode='min')

        if self.smoketest:
            callbacks = []
        else:
            if self.use_checkpoint:
                callbacks = [tbCallBack,earlyStopping, checkpoint]
                print("Caution: Using Checkpoints")
            elif not self.use_checkpoint: 
                callbacks = [tbCallBack,earlyStopping]
                print("Caution: NOT Using Checkpoints")
            
        # !!! Exclude tensorboard callback for performance reasons
        callbacks = callbacks[1:]
        print("Caution: tensorboard callback is excluded")
  
        if self.reduce_LR_OnPlateu:
            callbacks = callbacks+[ReduceLR]
        
        return callbacks
    
    
    def est_fit(self,X_train,Y_train,X_dev, Y_dev):
        
        # Initialize and complie model
        print("--- Constructing Model")
        estimator = self.model(mod_predict=False)
        
        #%% Load weights
        if self.load_weights:
            estimator.load_weights(self.weights_path) 
            
        #%% Get model callbacks

        callbacks = self.model_callbacks()
        # Fitting the Estimator to the Training Data
        print("--- Fit Estimator")
        self.loss_history = estimator.fit(X_train, Y_train,
                                  validation_data=(X_dev, Y_dev),
                                  callbacks=callbacks,
                                  epochs=self.epochs,
                                  batch_size=self.batch_size,
                                  verbose=self.verbose)

        return estimator
    
    def model_fit(self,X_orig_train, Y_orig_train, X_orig_dev, Y_orig_dev):
        """ Fit an estimator. First transform orginal scale data in ndarray 
            fromat to required format and minmax scale.
        
        """
        #%%-----------------------------------------------------------
        # Transform data
        X_train, Y_train = self.data_transform(X_orig_train, Y_orig_train)
        X_dev, Y_dev = self.data_transform(X_orig_dev, Y_orig_dev)

        #%%-----------------------------------------------------------
        # Estimate the model and set self.estimator
        self.estimator = self.est_fit(X_train,Y_train,X_dev, Y_dev)
        
        return X_train, Y_train, X_dev, Y_dev 
    
    def run_train(self,dat = False):
        
        #%%-----------------------------------------------------------
        if not dat:
            print("loading Data from Pipeline")
            (X_orig_train, Y_orig_train, dat_train,
                    X_orig_dev, Y_orig_dev,dat_dev, 
                    X_orig_test, Y_orig_test,dat_test) \
                    = self.data_load()
        else:
            (X_orig_train, Y_orig_train, dat_train,
                    X_orig_dev, Y_orig_dev,dat_dev, 
                    X_orig_test, Y_orig_test,dat_test) = dat

        #fits the Model and returns some of the transformed Data. Since Data Standardisation its performed inside .model_fit()
        X_train, Y_train, X_dev, Y_dev = self.model_fit(X_orig_train.copy(), Y_orig_train.copy(), X_orig_dev.copy(), Y_orig_dev.copy())
        
        if not self.smoketest:
            self.model_save()
    
        if not self.smoketest:
            #this is to ensure, best weights are used for prediction
            try:
                self.model_load(False,self.fullId+"_weights-improvement")
            except:
                pass
    
    def est_predict(self,X):
        # predicting with the fitted estimator with some XData
        Yhat_dict = self.estimator.predict(X)

        model_type = "regression"

        # Sometimes a Dict, dometimes a List
        # will be predicted; this section it to ensure the desired format
        if isinstance(Yhat_dict,list):
            print("Trigger Warning List")
            Yhat_dict = {model_type:Yhat_dict[2],"c":Yhat_dict[1],"a":Yhat_dict[0]}
        elif isinstance(Yhat_dict,dict):
            print("Trigger Warning Dict")
            Yhat_dict = Yhat_dict
        else:
            print("Trigger Warning Else")
            Yhat_dict = {model_type:Yhat_dict}
        
        return Yhat_dict
    
    def model_save(self):
        #print('--- Save model')

        # Save estimator
        self.estimator.save_weights(os.path.join(self.model_path,self.fullId+"_estimator.h5"), overwrite=True)
        # Filter out tensorflow.keras objects that can not be pickeled
        #%%
        selfDict = {}
        for k,v in self.__dict__.items():
     
            if isinstance(v, str) or isinstance(v, int) \
                or isinstance(v, float) or isinstance(v, list) \
                or isinstance(v, np.ndarray) or isinstance(v, dict):
                    print(k)
                    selfDict.update({k:v})
        #%%    
        # Save attr of object
        self.save_obj(selfDict, os.path.join(self.model_path,self.fullId+"_paramStore.pkl") )
        
    def model_load(self,specific_model_id=False,mod_predict=False,
                Shapely=False):
        print('--- Load Model')
        print("self.model_path in model_load", self.model_path)
        #%%
        # Load attr of object
        if specific_model_id:
            self.savePath = os.path.join(self.model_path,specific_model_id)
            print("Loading specific Model:", specific_model_id)
        else:
            self.savePath = os.path.join(self.model_path,self.fullId)
            print("Loading Skript Model:", self.fullId)
        #%%
        specAttr = self.load_obj(self.savePath+"_paramStore.pkl" )
        keep_attribs = ["project_path",
                        "dataPath",
                        "model_path","reports_path","log_path","savePath"]
        for attr, value in specAttr.items():
            if not attr in keep_attribs:
                setattr(self, attr, value)
        #%%
        self.mod_predict = mod_predict
  
        if Shapely:
            self.Shapely = Shapely

        # recustruct estimator with loaded attributes
        self.estimator = self.model(mod_predict=mod_predict,from_load=True)
        
        #ensures, that the latest improvemnt of the model is loaded; 
        self.estimator.load_weights(self.savePath+"_weights-improvement.hdf5") 

    
    def data_load(self, file="processed/train_test_split.pkl",dev_size=0.05,
                  getMinMax=True):
        file = os.path.join(self.dataPath,file)
            
        """ 1) Load data, 2) split in train,dev,test, 
            3) Exctract X and Y, 4) get minmax scaling
        
        Keyword Arguments:
            file {str} -- Dataset file name 
            dev_size {float} -- share of N allocated to  dev set (default: {0.05})
        
        """
        
        # Load file
        df, df_test = self.__load_file(file)

        # Sort and set index
        df = self.__sort_df_set_index(df)
        df_test = self.__sort_df_set_index(df_test)
        # Fill nan, -inf and inf with 0
        df.replace(np.inf, 0,inplace=True)
        df_test.replace(-np.inf, 0,inplace=True)
        df.fillna(0,inplace=True)
        df_test.fillna(0,inplace=True)  

        #Split in train,dev,test
        df_train, df_dev = train_test_split_panel(df,vId='KGB',
                            test_size=dev_size,
                            random_state=self.random_state)


        #Reshape to 3darray (N,T,K)
        dat_train, t_index_train, idv_index_train, col_index_train = \
                                            self.resphape_3d_panel(df_train)
        dat_dev, t_index_dev, idv_index_dev, col_index_dev = \
                                            self.resphape_3d_panel(df_dev)
        dat_test, t_index_test, idv_index_test, col_index_test = \
                                            self.resphape_3d_panel(df_test)
        
        assert np.array_equal(col_index_train,col_index_test)
        assert np.array_equal(col_index_train,col_index_dev)
        
        assert np.array_equal(t_index_train,t_index_test)
        assert np.array_equal(t_index_train,t_index_dev)
       
        # Save index of columns
        self.col_index = col_index_train
        
        # Save time index
        self.t_index = t_index_train

        # Save individual index (KGB)
        self.idv_index_train = idv_index_train
        self.idv_index_dev = idv_index_dev
        self.idv_index_test = idv_index_test
        
        X_orig_train, Y_orig_train = self.__extract_XY(dat_train,
                                                    col_index_train,
                                                    self.vTag,self.vFeat)
        X_orig_dev, Y_orig_dev = self.__extract_XY(dat_dev,
                                                    col_index_dev,
                                                    self.vTag,self.vFeat)
        X_orig_test, Y_orig_test = self.__extract_XY(dat_test,
                                                    col_index_test,
                                                    self.vTag,self.vFeat)

        # Filter outlier
        if not self.smoketest:
            X_orig_train, Y_orig_train = self.__filter_outlier(X_orig_train, 
                                                            Y_orig_train)

        # Get minmax scale
        # Scaling is only based on X_orig_train such that no information is 
        # taken from dev or test set. Y_orig_train does not need to be
        # considered as all variables of Y_orig_train are included in 
        # X_orig_train (except for the last year due to the shift)
        if getMinMax:
            self.__get_minmax_scale(X_orig_train,self.vFeat)

        # Get dimensions		
        self.kx = X_orig_train.shape[2]
        self.ky = Y_orig_train.shape[2]
        self.T = X_orig_train.shape[1]
        self.N = X_orig_train.shape[0]

        return (X_orig_train, Y_orig_train, dat_train,
                X_orig_dev, Y_orig_dev,dat_dev, 
                X_orig_test, Y_orig_test,dat_test) 

    def resphape_3d_panel(self,panel):
        """
        Reshape df panel into a 3d numpy array with shape 
        (N,T,K)
        
        """
        #%% Getting the dimensions right
        n_idv = len(panel.index.get_level_values(1).unique())	# Number of individuals
        T = len(panel.index.get_level_values(0).unique())  # Length of time sequence 
    
        t_index = list(panel.index.get_level_values(0).unique())
        idv_index = list(panel.index.get_level_values(1).unique())
        col_index = list(panel.columns)

        dat = panel.values.reshape((T, n_idv, panel.shape[1]))
        
        # Swap axis such that number of individuls comes frist and time second
        dat = np.swapaxes(dat,0,1)
       
        #%%	 
        return dat, t_index, idv_index, col_index
    
    def revert_resphape_3d_panel(self,dat,idv_index,t_index,col_index):
        """
        Reshape df panel into a 3d numpy array with shape 
        (N,T,K)
        
        """
        #%% Getting the dimensions right
        n_idv = len(idv_index)	# Number of individuals
        T = len(t_index)  # Length of time sequence 
        K = len(col_index)
        #%%
        dat_s = np.swapaxes(dat,0,1)
        rDat = dat_s.reshape(n_idv*T,K)
        
        #%%	   
        panel = pd.DataFrame(rDat,columns=col_index)
        
        #%%
        list_idx = []
        list_t = []
        for t in t_index:
            list_idx += idv_index
            list_t +=  ([t]*n_idv)
        
        panel['KGB'] = list_idx
        panel['year'] = list_t
        
        panel.set_index(['year', 'KGB'], inplace=True)
        
        return panel
    
    def __filter_outlier(self,X_orig, Y_orig):
        """
        Fitler outlier: Determine outliere by finding 0.0001 percent 
        of larges values for each activtiy (excluding zero observations)
        """

        cut_max = np.ndarray(shape=(1,1,Y_orig.shape[2]), dtype=float)
        
        for k in range(0,Y_orig.shape[2]):
            aa = Y_orig[:,:,k].flatten()
            # Exclude zero observation for the calculation of the quantile
            aa = aa[aa>0]
            cut_max[:,:,k] = np.quantile(aa,0.9999)

        sel_col = np.sum(Y_orig>cut_max,axis=(1,2))==0

        X_orig = X_orig[sel_col]
        Y_orig = Y_orig[sel_col]
                
        return X_orig, Y_orig
    
    def __extract_XY(self,dat,col_index,vTag,vFeat):
        """
        Extract X and Y from 3d-array and shift Y appropriatly
        
        dat: 3d numpy array
        col_index: list with column names in dat
        vTag: list with target names
        vFeat: list with feature names		
        """
        #% Select relevant columns
        idxTag = findIndex(col_index,vTag)
        idxFeat = findIndex(col_index,vFeat)
        #%
        Y_raw = dat[:,:,idxTag]
        X_raw = dat[:,:,np.array(idxFeat)]

        #% Shift Y forward and adjust dimensions
        Y = Y_raw[:,1:,:]
        X = X_raw[:,:-1,:]
        
        Ty = Y.shape[1]
        Tx = X.shape[1]
        
        assert (Ty==Tx)
    
        return X, Y
    
    def __get_minmax_scale(self,X,col_index):
        """
        Derive min max for scaling
        
        X: 3D array with dimensions (ids,time,variables)
        col_index: list specify that last (i.e. variables) dimension in X
        """
        self.Xmin = np.amin(X,axis=(0,1),keepdims=True)
        self.Xmax = np.amax(X,axis=(0,1),keepdims=True)
        self.vVarMinMax = col_index

        
    def __load_file(self,file):
        # Import individual dataframes split into training/dev (df) set and a seperate
        # test set (df_test)
        df, df_test = self.load_obj(file)

        return df, df_test
    
    def __sort_df_set_index(self,df,idxYear='year',idxId='KGB'):
        """
        Reset index from df, set index to year and KGB and sort df
        """
        df.reset_index(inplace=True)
        df[idxYear] = pd.to_datetime(df[idxYear])  # transform to datatime
        df.set_index([idxYear, idxId], inplace=True)  # set index to year
        
        # Sorting the index is important otherwise slicing e.g. by year
        # is not possible
        df.sort_index(inplace=True)
        
        return df
    
    def data_transform(self, X_orig, Y_orig):
        X_data=X_orig
        Y_data=Y_orig
        
        """Scale data to minmax scale and transform data to specific format 
            required for estimation. 
            For each transformation an corresponding inverse transformation 
            needs to be defined in data_inverse_transform().
        
        Arguments:
            X_orig {ndarray(N,T,K)} -- Features
            Y_orig {ndarray(N,T,K)} -- Targets
        """
        
        # Min Max Scaling
        if X_data is not None:
            if not isinstance(X_data,list):
                X = X_data
            else:
                X = X_data[0]	
            if not self.smoketest:
                X = self.__apply_minmax_scale(X,self.vFeat)
        else:
            X = None
        
        if Y_data is not None:
            if not self.smoketest:
                Y = self.__apply_minmax_scale(Y_data,self.vTag)
            else:
                Y = Y_data
                self.ky = Y.shape[-1]
    
            Y_out={"regression":Y}

        else:
            Y_out = None
    
        # Reshape from 3darray (N,T,K) to specific format
        # Need to implement matching data_inverse_transform()
        if isinstance(X_data, list): 
            a = X_data[1]
            c = X_data[2]
        elif not isinstance(X_data, list):
            self.Tx = X.shape[1]
            self.kx = X.shape[2]
            # create initial states
            a0 = np.zeros((X.shape[0],self.n_a))
            c0 = np.zeros((X.shape[0],self.n_a))
            a=a0.copy()
            c=c0.copy()
        
        X= [X,a,c]
        
        return X,Y_out
            
    def __apply_minmax_scale(self,X_orig,col_index):
        """
        Apply minmax scaling 
        
        X_orig: variales to be scaled 
        col_index: list of variable names in X_orig
        """
        Xmin = self.Xmin[:,:,findIndex(self.vVarMinMax,col_index)]
        Xmax = self.Xmax[:,:,findIndex(self.vVarMinMax,col_index)]

        
        # Check if Xmin and Xmax are both equal to zero i.e. all values 
        # equal to zero. If so set Xmax to 1, which means that scaling has
        # no effect
        if np.sum(Xmax-Xmin ==0)>0:
            Xmax[Xmax-Xmin ==0] = 1
        # Scale to 0-1   
        X = (X_orig - Xmin) / (Xmax - Xmin)

        return X
    
    def invert_minmax_scale(self,X_scal,col_index):
        
        Xmin = self.Xmin[:,:,findIndex(self.vVarMinMax,col_index)]
        Xmax = self.Xmax[:,:,findIndex(self.vVarMinMax,col_index)]
        
        # Check if Xmin and Xmax are both equal to zero i.e. all values 
        # equal to zero. If so set Xmax to 1, which means that scaling has
        # no effect
        if np.sum(Xmax-Xmin ==0)>0:
            Xmax[Xmax-Xmin ==0] = 1
                
        # Scale to 0-1

        X_orig = X_scal * (Xmax - Xmin) + Xmin
             
        return X_orig
     
    def data_inverse_transform(self, X, Y):
        """ Call inverse data reshaper and scale data back to original format
        
        Returns:
            X_orig {ndarray(N,T,K)} -- Features in original scale and ndarray format
            Y_orig {ndarray(N,T,K)} -- Targets in original scale and ndarray format
        """
        # Reshape from specific format to 3darray (N,T,K) 
        
        # Invert Min Max Scaling, Rescale to original
        if X is not None:
            if not self.smoketest:
                X_orig = self.invert_minmax_scale(X,self.vFeat)
            else :
                X_orig = X
        else:
            X_orig = None
        if isinstance(Y,dict):
            Y = Y['regression']
            
        if Y is not None:
            if not self.smoketest:
                Y_orig = self.invert_minmax_scale(Y,self.vTag)
            else :
                Y_orig = Y
            
        else:
            Y_orig = None

        return X_orig,Y_orig

    def model_predict(self,X_orig):
        """ Perform prediction 
        Arguments:
            X_orig {ndarray(N,T,K)} -- Features in original scale
        Returns:
            yhat_orig {ndarray(N,T,K)} -- Predicted Targets in original scale 
        """
        # Transform data
        X, _ =self.data_transform(X_orig, None)

        # Predict	
        Yhat = self.est_predict(X)
        
        # Rescale to original
        _, yhat_orig = self.data_inverse_transform(None,Yhat)
        
        return yhat_orig
    
    def model_performance(self,y_orig,yh_orig):
        """ Interface to provide performance evaluation across different 
            model types
        
        Arguments:
            nd_X {ndarray(n,t,x_k)} -- Feature ndarray used for prediction
            nd_y {ndarray(n,t,y_k)} -- Target ndarray with observed values
            nd_yhat {ndarray(n,t,y_k)} -- Predicted Target ndarray with observed values
        
        Keyword Arguments:
            scale_to_orig {bool} -- Set to True if x,y are scaled, set to False if 
                                    they are already scaled to original scale
                                    (default: {True})
        """
        assert yh_orig.shape==y_orig.shape, "Shape of yhat not correct"
        assert yh_orig.shape[2]==len(self.vTag), "k dimension of yhat not correct"
        
        y = y_orig.copy()
        yh = yh_orig.copy() 
        
        # Exlcude first two time periods
        yNaiv = y[:,1:-1,:].copy() # for naiv prediction simply shift y by one year
        y = y[:,2:,:]
        yh = yh[:,2:,:]

        # Calculate the overall R2 across time and output
        r2Model_overall = self.r2(y.flatten(),yh.flatten())		
        # print('r2Model:',r2Model)
        r2Naive_overall = self.r2(y.flatten(),yNaiv.flatten())		
        # print('r2Naive:',r2Naive)
        
        # Calculate the R2 for each year
        r2Time = pd.DataFrame(columns=['r2 Model','r2 Naive'])
        for t in range(0,y.shape[1]):
            r2Model = self.r2(y[:,t,:].flatten(),yh[:,t,:].flatten())
            r2Time.loc[self.t_index[t+3],'r2 Model'] = r2Model
            r2Naive = self.r2(y[:,t,:].flatten(),yNaiv[:,t,:].flatten())
            r2Time.loc[self.t_index[t+3],'r2 Naive'] = r2Naive
        r2Time
        
        # Calculate the R2 for each k 
        r2Tag = pd.DataFrame(columns=['r2 Model','r2 Naive'])
        for k in range(0,len(self.vTag)):
            r2Model = self.r2(y[:,:,k].flatten(),yh[:,:,k].flatten())
            r2Tag.loc[self.vTag[k],'r2 Model'] = r2Model
            r2Naive = self.r2(y[:,:,k].flatten(),yNaiv[:,:,k].flatten())
            r2Tag.loc[self.vTag[k],'r2 Naive'] = r2Naive
        r2Tag

        # Calculate the R2 for each k across years
        index = pd.MultiIndex.from_product([self.t_index[2:], ['Model','Naive']],
                                           names=['year', 'type'])
        r2TimeTag = pd.DataFrame(index=index, columns=self.vTag)
        r2TimeTag
        for k in range(0,len(self.vTag)):
            for t in range(0,y.shape[1]):
                r2Model = self.r2(y[:,t,k].flatten(),yh[:,t,k].flatten())
                r2TimeTag.loc[(self.t_index[t+3],'Model'),self.vTag[k]] = r2Model
                r2Naive = self.r2(y[:,t,k].flatten(),yNaiv[:,t,k].flatten())
                r2TimeTag.loc[(self.t_index[t+3],'Naive'),self.vTag[k]] = r2Naive
        
        # Calcualte Number of predicted Entries
        yf = y.flatten()
        yhf = yh.flatten()
        yNaivf = yNaiv.flatten()
        
        entryHat = yhf[yNaivf==0]>0
        numEntry = np.sum(entryHat)		
        # print('Number of predicted Entries',numEntry)

        entryTrue = yf[yNaivf==0]>0
        numTrueEntry = np.sum(entryTrue)		
        # print('Number of true Entries',numTrueEntry)
            
        correctPredictEntry = np.sum(entryHat[entryTrue==1]== \
                                         entryTrue[entryTrue==1])
        # print('Number of Correctly predicted entries',correctPredictEntry)

        exitHat = yhf[yNaivf>0]<=0
        numExit = np.sum(exitHat)
        # print('Number of predicted Exits',numExit)

        exitTrue = yf[yNaivf>0]<=0
        numTrueExit = np.sum(exitTrue)
        # print('Number of True Exits',numTrueExit)

        correctPredictExit = np.sum(exitHat[exitTrue==1]== \
                                        exitTrue[exitTrue==1])
        # print('Number of Correctly predicted Exits',correctPredictExit)

        return (r2Model_overall, r2Time, r2Tag, r2TimeTag, 
               numEntry,numTrueEntry,correctPredictEntry,
               numExit,numTrueExit,correctPredictExit 
               )
    
   
    def r2(self,y_true, y_pred):
        """Calcualte and return R2.

        y_true -- the observed values
        y_pred -- the prediced values
        """
        if not self.smoketest:
            SS_res =  np.sum(np.square(y_true - y_pred))
            SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
            r2_score = ( 1 - SS_res/SS_tot )
        else:
            r2_score = -4
        
        return r2_score


    def load_evaluate(self, dat = False):
        # loads a specific model and evaluates its performance
        if not dat:
            print("loading Data from Pipeline")
            (X_orig_train, Y_orig_train, dat_train,
                    X_orig_dev, Y_orig_dev,dat_dev, 
                    X_orig_test, Y_orig_test,dat_test) \
                    = self.data_load()
        else:
            (X_orig_train, Y_orig_train, dat_train,
                    X_orig_dev, Y_orig_dev,dat_dev, 
                    X_orig_test, Y_orig_test,dat_test) = dat

        # make predictions with validation Data
        yhat_orig_train = self.model_predict(X_orig_train)
        yhat_orig_dev = self.model_predict(X_orig_dev)
        yhat_orig_test = self.model_predict(X_orig_test)
        
        (r2Model_train, r2Time_train, r2Tag_train, r2TimeTag_train, 
        numEntry_train,numTrueEntry_train,correctPredictEntry_train,
        numExit_train,numTrueExit_train,correctPredictExit_train 
               ) = self.model_performance(
             y_orig = Y_orig_train,
             yh_orig = yhat_orig_train)
               
        (r2Model_dev, r2Time_dev, r2Tag_dev, r2TimeTag_dev, 
        numEntry_dev,numTrueEntry_dev,correctPredictEntry_dev,
        numExit_dev,numTrueExit_dev,correctPredictExit_dev 
               ) = self.model_performance(
             y_orig = Y_orig_dev,
             yh_orig = yhat_orig_dev)
        
        
        (r2Model_test, r2Time_test, r2Tag_test, r2TimeTag_test, 
        numEntry_test,numTrueEntry_test,correctPredictEntry_test,
        numExit_test,numTrueExit_test,correctPredictExit_test 
               ) = self.model_performance(
             y_orig = Y_orig_test,
             yh_orig = yhat_orig_test)

        
        print('r2Model Train:',r2Model_train)
        print('r2Model Dev:',r2Model_dev)
        print('r2Model Test:',r2Model_test)
        
        r2TagRes = r2Tag_train.copy()
        r2TagRes['r2 Dev'] = r2Tag_dev['r2 Model']
        r2TagRes['r2 Test'] = r2Tag_test['r2 Model']
        r2TagRes.rename(columns={'r2 Model':'r2 Train'},inplace=True)
        r2TagRes = r2TagRes.loc[:,['r2 Train', 'r2 Dev', 'r2 Test']]
        r2TagRes.loc['Overall','r2 Train'] = r2Model_train
        r2TagRes.loc['Overall','r2 Dev'] = r2Model_dev
        r2TagRes.loc['Overall','r2 Test'] = r2Model_test
        
        r2TagRes.loc['Farms','r2 Train'] = Y_orig_train.shape[0]
        r2TagRes.loc['Farms','r2 Dev'] = Y_orig_dev.shape[0]
        r2TagRes.loc['Farms','r2 Test'] = Y_orig_test.shape[0]
        
        r2TagRes.loc['Times','r2 Train'] = Y_orig_train.shape[1]
        r2TagRes.loc['Times','r2 Dev'] = Y_orig_dev.shape[1]
        r2TagRes.loc['Times','r2 Test'] = Y_orig_test.shape[1]
        
        r2TagRes.loc['Total Obs','r2 Train'] = Y_orig_train.shape[0]*Y_orig_train.shape[1]
        r2TagRes.loc['Total Obs','r2 Dev'] = Y_orig_dev.shape[0]*Y_orig_dev.shape[1]
        r2TagRes.loc['Total Obs','r2 Test'] = Y_orig_test.shape[0]*Y_orig_test.shape[1]
        
        
        r2TagRes
        
        # Get number of vegetable (GRON) farm in test set
        aa = np.sum(Y_orig_test[:,:,findIndex(self.vTag,['GRON'])]>0,axis=1)    
        bin, counts = np.unique(aa, return_counts=True)
        np.sum(counts[1:])
        
        return r2TagRes

#%
def test_model_save_load():
    #==============================
    # Test if model save/load works as expected
    #==============================
    new_model = nn_model()
    new_model_id = 'TestModelId'
    new_model.fullId = new_model_id
    new_model.epochs = 2
    new_model.smoketest = False
    # Load data and get MinMax 
    dat = new_model.data_load()
    new_model.run_train(dat=dat)
    #% 
    # Load the model
    load_model = nn_model()
    load_model.model_load(specific_model_id = new_model_id)
    #%
    # Make prediction with new_model and load_model 
    (_, _, _,X_orig_dev, _,_,_,_,_) = dat
    yhat_new_model = new_model.model_predict(X_orig_dev)
    yhat_load_model = load_model.model_predict(X_orig_dev)

    # Check if results are equal
    print('Array equal?', np.array_equal(yhat_new_model,yhat_load_model))

# %%
def check_load_modpredict():
    #==============================
    # Test if model save/load works as expected
    # using mod_predict=True and Shapely=True
    #==============================
    #%% 
    # Load the model
    load_model = nn_model()
    model_id = 'TestModelId'
    load_model.model_load(specific_model_id = model_id)
    #%%
    # Load data and get MinMax 
    dat = load_model.data_load()
    #%%
    load_predmodel = nn_model()
    load_predmodel.Shapely = True
    load_predmodel.fullId = model_id
    load_predmodel.model_load(mod_predict=True, Shapely=True)
    #%%
    # Make prediction with new_model and load_model 
    (_, _, _,X_orig_dev, _,_,_,_,_) = dat
    yhat_loadmodel = load_model.model_predict(X_orig_dev)
    yhat_predmodel = load_predmodel.model_predict(X_orig_dev)
    # Check if results are equal
    print('Array equal?', np.array_equal(yhat_loadmodel[:,-1,:],yhat_predmodel[0,:,:]))
    
# %%
def loadModelEval():
    # %%
    #==============================
    # Load an existing model
    # #==============================
    model_id = "15283bd4-51d2-11ec-8639-0242ac110003" # (R2: 0.9329754728365416) 
  
    #==============================
    # Evaluate loaded model 
    #==============================
    eval_model_id = model_id

    eval_model = nn_model()
    # load model 
    eval_model.model_load(specific_model_id = eval_model_id)
   
    # Load data
    dat = eval_model.data_load()
    #%%
    # eval_model.use_specified_cutoff = True
    r2TagRes = eval_model.load_evaluate(dat=dat)

    print('Evaluation Done!')
    print('Model ID: ',eval_model.fullId)
    #%%
    translation = {'x120':'dairy cows',
                'x121':'suckler cows',
                'x140':'female goats',
                'x155':'sows',
                'x160': 'hens',
                'x210': 'fodder arable land',
                'x211': 'pasture arable land',
                'x212': 'fodder non-arable land',
                'x230': 'potatoes',
                'CERE': 'crops',
                'GEIT': 'male goats',
                'GRON': 'vegetables',
                'SAU':  'sheep',
                'STOR': 'other cattle'}
    r2TagRes.rename(index=translation, inplace=True)
    r2TagRes
    #%%
    savePath = os.path.join('reports','model_stats')
    Path(savePath).mkdir(parents=True, exist_ok=True)
    r2TagRes.to_csv(os.path.join(savePath,f'{eval_model.fullId}_model_stats.csv'),sep=';')


    
# %%
if __name__ == "__main__":
    
    # 
    # %%
    #==============================
    # Load an existing model
    # #==============================
    # modelTest = nn_model()
    # model_id = "15283bd4-51d2-11ec-8639-0242ac110003" # R2: 0.9329754728365416 
    
    # modelTest.load_weights = True
    # modelTest.model_load(specific_model_id = model_id)
    # # Load data
    # dat = modelTest.data_load()
    # msodelTest.load_evaluate(dat=dat)
    
    # %%
    #==============================
    # Train model
    #==============================
    new_model = nn_model()
    new_model_id = str(uuid.uuid1())
    new_model.fullId = new_model_id
    
    # new_model.epochs = 2000
    new_model.epochs = 2
    new_model.smoketest = False
    # Load data
    dat = new_model.data_load()
    
    new_model.run_train(dat=dat)
    print('Training Done!')
    print('Model ID: ',new_model.fullId)
    # %%
    #==============================
    # Evaluate model trained model
    #==============================
    # new_model.use_specified_cutoff = True
    new_model.load_evaluate(dat=dat)
    # %%
    #==============================
    # Evaluate loaded model 
    #==============================
    eval_model_id = new_model_id

    eval_model = nn_model()
    # load model 
    eval_model.model_load(specific_model_id = eval_model_id)
    # %%
    # eval_model.use_specified_cutoff = True
    eval_model.load_evaluate(dat=dat)

    print('Evaluation Done!')
    print('Model ID: ',eval_model.fullId)
   
# %%
