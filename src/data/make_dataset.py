

"""The file is used to preprocess the raw data. 

    This relies in parts of Matlab code for the initial preprocessing
 
    The data needs to be located under 'Z:/ptFarmDataNorway/PT/300119/'
    if this path is changed then it is not(!) sufficient to just change
    raw_data_path. It would also required to change pathes in the Matlab code!
    

    The pipeline is build using luigi https://github.com/spotify/luigi
    
    1. Step
    To run the pipeline start an anaconda promt and run a terminal using: "luigid"
    Then one can open in browser http://agpcom3:8082 to see the luigi terminal

    2. Step 
    To use the matlab parts one needs a working matlab instalation and the python
    matlab libaries installed. The libaries are already set up in the conda enviroment 
    from conda promt: "activate matlab_py3_5" 
    (Enviroment text file: cond_enviroment_matlab_py3_5.txt )
        To setup a new matlab lib installation follow info under
        https://de.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html

        Specifically do the following because user has no permission on the agpcom3 server
        cd "matlabroot\extern\engines\python" 
        python setup.py build --build-base=/temp/storm/matlab install --prefix=/User/Storm/AppData/Local/Coninuum/Anaconda3/envs/matlab_py3_5
 
            
    3. Step
    Set up a temporary folder "C:\temp\storm\nn_norway\data\raw". To store data. 
    Ideally under C: directly (which is an SSD) and not on a network drive 
    to have minimum read/write time. The folder containing "data/raw" must be
    specified under "data_path"

    4. Step
    From directory nn_norway/src/data run: (important to start for this directory)
    "python make_dataset.py"



"""

#%%

import luigi
from luigi import configuration
import sys
data_path = sys.path[0]+"/"
#data_path = 'N:/agpo/work2/SpatialNorway/5_NeuralNetworkPolicy/Model/nn_norway/'#'C:/temp/storm/nn_norway/'  # define path to data
raw_data_path = 'Z:/ptFarmDataNorway/PT/300119/'  # define path to raw data
# %%
data_path = "/nn_norway/"#
# os.chdir(wd)

import os
"""if not os.path.isdir('C:/temp/storm/'):
    os.mkdir('C:/temp/storm/')
if not os.path.isdir('C:/temp/storm/nn_norway/'):
    os.mkdir('C:/temp/storm/nn_norway/')
if not os.path.isdir('C:/temp/storm/nn_norway/data/'):
    os.mkdir('C:/temp/storm/nn_norway/data/')
if not os.path.isdir('C:/temp/storm/nn_norway/data/raw/'):
    os.mkdir('C:/temp/storm/nn_norway/data/raw')
if not os.path.isdir('C:/temp/storm/nn_norway/data/processed/'):
    os.mkdir('C:/temp/storm/nn_norway/data/processed')
"""


if not os.path.isdir(data_path):
    os.mkdir(data_path)
if not os.path.isdir(data_path+'/data/'):
    os.mkdir(data_path+'/data/')
if not os.path.isdir(data_path+'/data/raw/'):
    os.mkdir(data_path+'/data/raw')
if not os.path.isdir(data_path+'/data/processed/'):
    os.mkdir(data_path+'/data/processed')


 


#%%
#-----------------------
#--- Step one -> Join raw files
#-----------------------
class joinRawFiles(luigi.Task):

    #date_interval = luigi.DateIntervalParameter()
    targetFileName = raw_data_path+"PTRK_1995_2015.mat"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)


    def run(self):
        #%% Call matlab to join raw files
        import matlab.engine
        eng = matlab.engine.start_matlab()
        #%%
        eng.addpath(eng.genpath('Z:/5_NeuralNetworkPolicy/Model/MatReg'))

        #%%
        dat = eng.DataPT2();
        #%%
        eng.joinDataFiles(dat);

        #%%
        eng.quit()


#-----------------------
#--- Step two -> Clean PT2 Data
#-----------------------
class cleanPT(luigi.Task):

    #date_interval = luigi.DateIntervalParameter()
    targetFileName = raw_data_path+"clean_PTRK_1995_2015cNames.mat"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return joinRawFiles()

    def run(self):
        #%% Call matlab to join raw files
        import matlab.engine
        eng = matlab.engine.start_matlab()
        #%%
        eng.addpath(eng.genpath('Z:/5_NeuralNetworkPolicy/Model/MatReg'))
        eng.addpath(eng.genpath('C:/gams24.9'))

        #%%
        dat = eng.DataPT2();
        #%%
        eng.load(dat);
        eng.subsidiesToCleanDat(dat,1);

        #%%
        eng.quit()



#-----------------------
#--- create Bunn file
#-----------------------
class deriveBunn(luigi.Task):

    #date_interval = luigi.DateIntervalParameter()
    targetFileName = data_path+"data/raw/bunn.csv"
    
    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def run(self):
        #%% Call matlab to join raw files
        import matlab.engine
        eng = matlab.engine.start_matlab()
        #%%
        eng.addpath(eng.genpath('Z:/5_NeuralNetworkPolicy/Model/MatReg'))
        eng.addpath(eng.genpath('C:/gams24.9'))

        #%%
        p = eng.PaperNeuralNetPolicy();
        #%%
        eng.exportBunn(p,self.targetFileName)

        #%%
        eng.quit()


#-----------------------
#--- create Makssats file
#-----------------------
class deriveMaks(luigi.Task):

    #date_interval = luigi.DateIntervalParameter()
    targetFileName = data_path+"data/raw/maks.csv"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def run(self):
        #%% Call matlab to join raw files
        import matlab.engine
        eng = matlab.engine.start_matlab()
        #%%
        eng.addpath(eng.genpath('Z:/5_NeuralNetworkPolicy/Model/MatReg'))
        eng.addpath(eng.genpath('C:/gams24.9'))

        #%%
        p = eng.PaperNeuralNetPolicy();
        #%%
        eng.exportMaks(p,self.targetFileName)

        #%%
        eng.quit()

#-----------------------
#--- create Satser file
#-----------------------
class deriveSatser(luigi.Task):

    #date_interval = luigi.DateIntervalParameter()
    targetFileName = data_path+"data/raw/satser.csv"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def run(self):
        #%% Call matlab to join raw files
        import matlab.engine
        eng = matlab.engine.start_matlab()
        #%%
        eng.addpath(eng.genpath('Z:/5_NeuralNetworkPolicy/Model/MatReg'))
        eng.addpath(eng.genpath('C:/gams24.9'))

        #%%
        p = eng.PaperNeuralNetPolicy();
        #%%
        eng.exportSatser(p,self.targetFileName)

        #%%
        eng.quit()

#-----------------------
#--- create Trin file
#-----------------------
class deriveTrin(luigi.Task):

    #date_interval = luigi.DateIntervalParameter()
    targetFileName = data_path+"data/raw/trin.csv"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def run(self):
        #%% Call matlab to join raw files
        import matlab.engine
        eng = matlab.engine.start_matlab()
        #%%
        eng.addpath(eng.genpath('Z:/5_NeuralNetworkPolicy/Model/MatReg'))
        eng.addpath(eng.genpath('C:/gams24.9'))

        #%%
        p = eng.PaperNeuralNetPolicy();
        #%%
        eng.exportTrin(p,self.targetFileName)

        #%%
        eng.quit()

 #-----------------------
#--- create Grocfact file
#-----------------------
class deriveGrovfact(luigi.Task):

    #date_interval = luigi.DateIntervalParameter()
    targetFileName = data_path+"data/raw/grovfac.csv"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def run(self):
        #%% Call matlab to join raw files
        import matlab.engine
        eng = matlab.engine.start_matlab()
        #%%
        eng.addpath(eng.genpath('Z:/5_NeuralNetworkPolicy/Model/MatReg'))
        eng.addpath(eng.genpath('C:/gams24.9'))

        #%%
        p = eng.PaperNeuralNetPolicy();
        #%%
        eng.exportGrov(p,self.targetFileName)

        #%%
        eng.quit()



#-----------------------
#--- Step Three -> Clean PTSuper Data
#-----------------------
class cleanPTSuper(luigi.Task):

    #date_interval = luigi.DateIntervalParameter()
    targetFileName = "P:/Research/ptRawData/clean_PTSupercNames.mat"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return cleanPT()

    def run(self):
        #%% Call matlab to join raw files
        import matlab.engine
        eng = matlab.engine.start_matlab()
        #%%
        eng.addpath(eng.genpath('Z:/5_NeuralNetworkPolicy/Model/MatReg'))
        eng.addpath(eng.genpath('C:/gams24.9'))


        #%%
        dat = eng.DataPTSuper();
        #%%
        eng.load(dat);

        #%%
        eng.quit()




#-----------------------
#--- Step Four -> Extract features and targets for NN
#-----------------------
class ExtractFeatures(luigi.Task):

    #date_interval = luigi.DateIntervalParameter()
    targetFileName = data_path+"data/raw/featureTarget.csv"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return [cleanPT(),
                deriveBunn(),
                deriveMaks(),
                deriveSatser(),
                deriveTrin(),
                deriveGrovfact()]

    def run(self):
        #%% Call matlab to join raw files
        import matlab.engine
        eng = matlab.engine.start_matlab()
        #%%
        eng.addpath(eng.genpath('Z:/5_NeuralNetworkPolicy/Model/MatReg'))
        eng.addpath(eng.genpath('C:/gams24.9'))

        #%%
        p = eng.PaperNeuralNetPolicy();
        #%%
        eng.exportNNDataToFile(p,self.targetFileName);

        #%%
        eng.quit()

#-----------------------
#--- Step Five -> Transfrom to balanced panel
#-----------------------
class BlancePanel(luigi.Task):

    #date_interval = luigi.DateIntervalParameter()
    targetFileName = data_path+"data/raw/balancedPanel_dummy.csv"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return ExtractFeatures()

    def run(self):
        import os
        os.system("python balance_panel.py")


#%%
# Note: To run this open in anaconda promt "luigid"
# Then open in browser http://agpcom3:8082

# Need to be run using environment with Matlab libary installed 
# activate matlab_py3_5

# Need to be run from folder src/data otherwise fiel balance_panel.py is not found
if __name__ =='__main__':
    luigi.run(main_task_cls=BlancePanel)



