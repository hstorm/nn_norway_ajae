
""" Data pipeline to create features

    The pipeline is build using luigi https://github.com/spotify/luigi
    
    Adjuste data_path to location where output for data preprocessing 
    form "make_dataset.py" are stored: (Adjust below)
    data_path = 'C:/temp/storm/nn_norway/' 

    Requires to create a folder: 'C:\temp\storm\nn_norway\data\processed'


    1. Step
    To run the pipeline start an anaconda promt and run a terminal using: "luigid"
    Then one can open in browser http://agpcom3:8082 to see the luigi terminal

    2. Step
    Run pipeline using 
    "python src/features/luigi_features.py"


# For some very very strange reasons. Execution of the file least to an error. 
While a exact copy for the file "copy_luigi_features.py" runs without 
problems...???????



"""

import luigi
import pandas as pd

import sys
import os
#print("In Luigi",sys.path)
# sys.path.append("src/lib")
# %%
wd = "/nn_norway"#
os.chdir(wd)
# %%
sys.path.append(os.path.join("src","features"))
sys.path.append(os.path.join("src","models"))
sys.path.append(os.path.join("src","mod_lib"))
sys.path.append(os.path.join("src","lib"))
sys.path.append(os.path.join("src","data"))

from utily import save_obj
from utily import load_obj
from utily import train_test_split_panel

"""
 import sys
 sys.path.append('src/data')
 sys.path.append('src/features')
 from make_dataset import BlancePanel
 from calc_dpay import loadSubsidySchemeFiles
 from calc_dpay import DeriveSubsidies
"""
# from luigi import configuration

from calc_dpay import loadSubsidySchemeFiles
from calc_dpay import DeriveSubsidies

from multiprocessing import Pool
#from multiprocessing.dummy import Pool

from sklearn.model_selection import train_test_split

from make_dataset import BlancePanel


# %%
data_path = "/nn_norway/"#



# %%
# =============================================================================
# Support functions
# =============================================================================


def apply_sub_group(X, subFunc, bunn, maks, sattrin, trin, grov,
                    unitincrease=False
                    ):
    """
    Wrapper function to derive subsides in an group->apply() statement
    """
    
    pool = Pool()
    t0 = X.name.year
    t1 = t0+1
    print('Func ',subFunc, 'year ',t0)
    # Create Subsidy scheme
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
    
    if unitincrease is False:
        # Call the method of deriveSub speficied as string in subFunc
        sub = getattr(deriveSub, subFunc)(X)
    else:
        # Call the method of deriveSub speficied as string in subFunc
        sub = getattr(deriveSub, subFunc)(X, unitincrease)
   
    pool.close() 
    pool.join()
    return sub


# %%
# =============================================================================
# Luigi classes
# =============================================================================


class DeriveSubs(luigi.Task):

    # date_interval = luigi.DateIntervalParameter()
    # targetFileName = "../../data/processed/subs.csv"
    targetFileName = data_path+"data/processed/subs.pkl"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return BlancePanel()

    def run(self):

        #  Import file and set intext to year
        df = pd.read_csv(BlancePanel().targetFileName)

        df['year'] = pd.to_datetime(df['year'])  # transform to datatime
        df.set_index(['year', 'KGB'], inplace=True)  # set index to year
        
        # Select all years except last
        df.reset_index(inplace=True)
        df.set_index('year',inplace=True)
        df_exLast = df.loc[:str((df.index.unique().max()+ pd.DateOffset(days=-1)))].copy()
        df_exLast.reset_index(inplace=True)
        df_exLast.set_index(['year', 'KGB'], inplace=True)  # set index to year
        
        # Sorting the index is important otherwise slicing e.g. by year
        # is not possible
        df_exLast.sort_index(inplace=True)

        # Load subsidy scheme data
        bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()

        # Create group variable based on year
        grouped = df_exLast.groupby(level='year')

        # Calculate subsidies
        sub = grouped.apply(lambda x: apply_sub_group(x, 'get_sub',
                                                      bunn, maks, sattrin,
                                                      trin, grov))

        #  Save dataframe to file
        # sub.to_csv(self.targetFileName)
        save_obj(sub, self.targetFileName)

# %%


class DeriveSubsNext(luigi.Task):
    """
    Derive subsidies bases on the activities in t but the rates in t+1
    """

    # date_interval = luigi.DateIntervalParameter()
    # targetFileName = "../../data/processed/subs.csv"
    targetFileName = data_path+"data/processed/subsNext.pkl"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return BlancePanel()

    def run(self):

        #  Import file and set intext to year
        df = pd.read_csv(BlancePanel().targetFileName)

        df['year'] = pd.to_datetime(df['year'])  # transform to datatime
        df.set_index(['year', 'KGB'], inplace=True)  # set index to year

        # Select all years except last
        df.reset_index(inplace=True)
        df.set_index('year',inplace=True)
        df_exLast = df.loc[:str((df.index.unique().max()+ pd.DateOffset(days=-1)))].copy()
        df_exLast.reset_index(inplace=True)
        df_exLast.set_index(['year', 'KGB'], inplace=True)  # set index to year
        #
        # Sorting the index is important otherwise slicing e.g. by year
        # is not possible
        df_exLast.sort_index(inplace=True)

        # Load subsidy scheme data
        bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()

        # Create group variable based on year
        grouped = df_exLast.groupby(level='year')

        # Calculate subsidies
        subNext = grouped.apply(lambda x: apply_sub_group(x, 'get_subNext',
                                                      bunn, maks, sattrin,
                                                      trin, grov))

        subNext.columns = ['subNext_' + str(col) for col in subNext.columns]

        #  Save dataframe to file
        # sub.to_csv(self.targetFileName)
        save_obj(subNext, self.targetFileName)


class DeriveSubD1(luigi.Task):
    """
    Calculate delta subsidies for one additional unit
    """
    # date_interval = luigi.DateIntervalParameter()
    # targetFileName = "../../data/processed/subs.csv"
    targetFileName = data_path+"data/processed/subD1.pkl"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return BlancePanel()

    def run(self):

        #  Import file and set intext to year
        df = pd.read_csv(BlancePanel().targetFileName)

        df['year'] = pd.to_datetime(df['year'])  # transform to datatime
        df.set_index(['year', 'KGB'], inplace=True)  # set index to year
        
        # Select all years except last
        df.reset_index(inplace=True)
        df.set_index('year',inplace=True)
        df_exLast = df.loc[:str((df.index.unique().max()+ pd.DateOffset(days=-1)))].copy()
        df_exLast.reset_index(inplace=True)
        df_exLast.set_index(['year', 'KGB'], inplace=True)  # set index to year
        #

        # Sorting the index is important otherwise slicing e.g. by year
        # is not possible
        df_exLast.sort_index(inplace=True)

        # Load subsidy scheme data
        bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()

        # Create group variable based on year
        grouped = df_exLast.groupby(level='year')

        # Calculate subsidies
        subD1 = grouped.apply(lambda x: apply_sub_group(x,
                                                        'get_subDUnit',
                                                        bunn, maks, sattrin,
                                                        trin, grov,
                                                        1))
        subD1.columns = ['subD1_' + str(col) for col in subD1.columns]

        #  Save dataframe to file
        # subD1.to_csv(self.targetFileName)
        save_obj(subD1, self.targetFileName)


class DeriveCsubD1(luigi.Task):
    """
    Calculate change from t0 to t1 in delta subidies for 1 additional unit
    """
    # date_interval = luigi.DateIntervalParameter()
    # targetFileName = "../../data/processed/subs.csv"
    targetFileName = data_path+"data/processed/CsubD1.pkl"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return BlancePanel()

    def run(self):

        
        # Import file and set intext to year
        df = pd.read_csv(BlancePanel().targetFileName)

        df['year'] = pd.to_datetime(df['year'])  # transform to datatime
        df.set_index(['year', 'KGB'], inplace=True)  # set index to year

        # Select all years except last
        df.reset_index(inplace=True)
        df.set_index('year',inplace=True)
        df_exLast = df.loc[:str((df.index.unique().max()+ pd.DateOffset(days=-1)))].copy()
        df_exLast.reset_index(inplace=True)
        df_exLast.set_index(['year', 'KGB'], inplace=True)  # set index to year
        
        # Sorting the index is important otherwise slicing e.g. by year
        # is not possible
        df_exLast.sort_index(inplace=True)
        
        
        # Load subsidy scheme data
        bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()

        # Create group variable based on year
        grouped = df_exLast.groupby(level='year')

        # Calculate change from t0 to t1 in delta subidies for
        # 10 additional unit
        CsubD1 = grouped.apply(lambda x: apply_sub_group(x,
                                                         'get_CsubDUnit',
                                                         bunn, maks, sattrin,
                                                         trin, grov,
                                                         1))
        CsubD1.columns = ['CsubD1_' + str(col) for col in CsubD1.columns]

        #  Save dataframe to file
        # CsubD1.to_csv(self.targetFileName)
        save_obj(CsubD1, self.targetFileName)


class DeriveCsubD10(luigi.Task):
    """
    Calculate change from t0 to t1 in delta subidies for 10 additional unit
    """
    # date_interval = luigi.DateIntervalParameter()
    # targetFileName = "../../data/processed/subs.csv"
    targetFileName = data_path+"data/processed/CsubD10.pkl"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return BlancePanel()

    def run(self):

        #  Import file and set intext to year
        df = pd.read_csv(BlancePanel().targetFileName)

        df['year'] = pd.to_datetime(df['year'])  # transform to datatime
        df.set_index(['year', 'KGB'], inplace=True)  # set index to year

        # Select all years except last
        df.reset_index(inplace=True)
        df.set_index('year',inplace=True)
        df_exLast = df.loc[:str((df.index.unique().max()+ pd.DateOffset(days=-1)))].copy()
        df_exLast.reset_index(inplace=True)
        df_exLast.set_index(['year', 'KGB'], inplace=True)  # set index to year

        # Sorting the index is important otherwise slicing e.g. by year
        # is not possible
        df_exLast.sort_index(inplace=True)

        # Load subsidy scheme data
        bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()

        # Create group variable based on year
        grouped = df_exLast.groupby(level='year')

        # Calculate change from t0 to t1 in delta subidies for 10
        # additional unit
        CsubD10 = grouped.apply(lambda x: apply_sub_group(x,
                                                          'get_CsubDUnit',
                                                          bunn, maks, sattrin,
                                                          trin, grov,
                                                          10))
        CsubD10.columns = ['CsubD10_' + str(col) for col in CsubD10.columns]

        #  Save dataframe to file
        CsubD10.to_csv(self.targetFileName)
        save_obj(CsubD10, self.targetFileName)


class DeriveCsubD50(luigi.Task):
    """
    Calculate change from t0 to t1 in delta subidies for 50 additional unit
    """
    # date_interval = luigi.DateIntervalParameter()
    # targetFileName = "../../data/processed/subs.csv"
    # targetFileName = data_path+"data/processed/CsubD50.csv"
    targetFileName = data_path+"data/processed/CsubD50.pkl"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return BlancePanel()

    def run(self):

        #%  Import file and set intext to year
        df = pd.read_csv(BlancePanel().targetFileName)

        df['year'] = pd.to_datetime(df['year'])  # transform to datatime
        df.set_index(['year', 'KGB'], inplace=True)  # set index to year
        
        # Select all years except last
        df.reset_index(inplace=True)
        df.set_index('year',inplace=True)
        df_exLast = df.loc[:str((df.index.unique().max()+ pd.DateOffset(days=-1)))].copy()
        df_exLast.reset_index(inplace=True)
        df_exLast.set_index(['year', 'KGB'], inplace=True)  # set index to year
        
        # Sorting the index is important otherwise slicing e.g. by year
        # is not possible
        df_exLast.sort_index(inplace=True)      
        
        # Load subsidy scheme data
        bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()

        # Create group variable based on year
        grouped = df_exLast.groupby(level='year')

        # Calculate change from t0 to t1 in delta subidies for
        # 10 additional unit
        CsubD50 = grouped.apply(lambda x: apply_sub_group(x,
                                                          'get_CsubDUnit',
                                                          bunn, maks, sattrin,
                                                          trin, grov,
                                                          50))
        CsubD50.columns = ['CsubD50_' + str(col) for col in CsubD50.columns]

        #%  Save dataframe to file
        # CsubD50.to_csv(self.targetFileName)
        save_obj(CsubD50, self.targetFileName)

class DeriveDAvgSub(luigi.Task):
    """
    Calculate change in average subsidy per activity
    """
    # date_interval = luigi.DateIntervalParameter()
    # targetFileName = "../../data/processed/subs.csv"
    # targetFileName = data_path+"data/processed/CsubD50.csv"
    targetFileName = data_path+"data/processed/dAvgSub.pkl"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return BlancePanel()

    def run(self):

        #  Import file and set intext to year
        df = pd.read_csv(BlancePanel().targetFileName)

        df['year'] = pd.to_datetime(df['year'])  # transform to datatime
        df.set_index(['year', 'KGB'], inplace=True)  # set index to year
        
        # Select all years except last
        df.reset_index(inplace=True)
        df.set_index('year',inplace=True)
        df_exLast = df.loc[:str((df.index.unique().max()+ pd.DateOffset(days=-1)))].copy()
        df_exLast.reset_index(inplace=True)
        df_exLast.set_index(['year', 'KGB'], inplace=True)  # set index to year
        
        # Sorting the index is important otherwise slicing e.g. by year
        # is not possible
        df_exLast.sort_index(inplace=True)

        # Load subsidy scheme data
        bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()

        # Create group variable based on year
        grouped = df_exLast.groupby(level='year')

        # Calculate change from t0 to t1 in delta subidies for
        # 10 additional unit
        DAvgSub = grouped.apply(lambda x: apply_sub_group(x,
                                                          'get_dAvgSub',
                                                          bunn, maks, sattrin,
                                                          trin, grov))
        DAvgSub.columns = ['DAvgSub_' + str(col) for col in DAvgSub.columns]

        #  Save dataframe to file
        save_obj(DAvgSub, self.targetFileName)


class DeriveDSub(luigi.Task):
    """
    Calculate change in total subsides between t0 and t1 with the acticties
    from t0
    """
    # date_interval = luigi.DateIntervalParameter()
    # targetFileName = "../../data/processed/subs.csv"
    # targetFileName = data_path+"data/processed/CsubD50.csv"
    targetFileName = data_path+"data/processed/dSub.pkl"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return BlancePanel()

    def run(self):

        #  Import file and set intext to year
        df = pd.read_csv(BlancePanel().targetFileName)

        df['year'] = pd.to_datetime(df['year'])  # transform to datatime
        df.set_index(['year', 'KGB'], inplace=True)  # set index to year
        
        
        # Select all years except last
        df.reset_index(inplace=True)
        df.set_index('year',inplace=True)
        df_exLast = df.loc[:str((df.index.unique().max()+ pd.DateOffset(days=-1)))].copy()
        df_exLast.reset_index(inplace=True)
        df_exLast.set_index(['year', 'KGB'], inplace=True)  # set index to yea
        

        # Sorting the index is important otherwise slicing e.g. by year
        # is not possible
        df_exLast.sort_index(inplace=True)

        # Load subsidy scheme data
        bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()

        # Create group variable based on year
        grouped = df_exLast.groupby(level='year')

        # Calculate change from t0 to t1 in delta subidies for
        # 10 additional unit
        dSub = grouped.apply(lambda x: apply_sub_group(x,
                                                          'get_dSub',
                                                          bunn, maks, sattrin,
                                                          trin, grov))
        dSub.columns = ['dSub' + str(col) for col in dSub.columns]

        #  Save dataframe to file
        save_obj(dSub, self.targetFileName)

#%%
class MergeFeatures(luigi.Task):
    """
    Merge all individual feature files
    """
    # date_interval = luigi.DateIntervalParameter()
    # targetFileName = "../../data/processed/merged_features.csv"
    # targetFileName = data_path+"data/processed/merged_features.csv"
    targetFileName = data_path+"data/processed/merged_features.pkl"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return [BlancePanel(),
                DeriveSubs(),
                DeriveSubsNext(),
                DeriveSubD1(),
                DeriveCsubD1(),
                DeriveCsubD10(),
                DeriveCsubD50(),
                DeriveDAvgSub(),
                DeriveDSub()
                ]

    def run(self):
        #%%  Import individual dataframes
        panel = pd.read_csv(BlancePanel().targetFileName)
        panel['year'] = pd.to_datetime(panel['year'])
        panel.set_index(['year', 'KGB'], inplace=True)
        
        subs = load_obj(DeriveSubs().targetFileName)
        subs.reset_index(inplace=True)
        subs['year'] = pd.to_datetime(subs['year'])
        subs.set_index(['year', 'KGB'], inplace=True)

        subsNext = load_obj(DeriveSubsNext().targetFileName)
        
        subsNext.reset_index(inplace=True)
        subsNext['year'] = pd.to_datetime(subsNext['year'])
        subsNext.set_index(['year', 'KGB'], inplace=True)
        
        subD1 = load_obj(DeriveSubD1().targetFileName)
        subD1.reset_index(inplace=True)
        subD1['year'] = pd.to_datetime(subD1['year'])
        subD1.set_index(['year', 'KGB'], inplace=True)

        CsubD1 = load_obj(DeriveCsubD1().targetFileName)
        CsubD1.reset_index(inplace=True)
        CsubD1['year'] = pd.to_datetime(CsubD1['year'])
        CsubD1.set_index(['year', 'KGB'], inplace=True)

        CsubD10 = load_obj(DeriveCsubD10().targetFileName)
        CsubD10.reset_index(inplace=True)
        CsubD10['year'] = pd.to_datetime(CsubD10['year'])
        CsubD10.set_index(['year', 'KGB'], inplace=True)

        CsubD50 = load_obj(DeriveCsubD50().targetFileName)
        CsubD50.reset_index(inplace=True)
        CsubD50['year'] = pd.to_datetime(CsubD50['year'])
        CsubD50.set_index(['year', 'KGB'], inplace=True)

        DAvgSub = load_obj(DeriveDAvgSub().targetFileName)
        DAvgSub .reset_index(inplace=True)
        DAvgSub ['year'] = pd.to_datetime(DAvgSub ['year'])
        DAvgSub .set_index(['year', 'KGB'], inplace=True)

        DSub = load_obj(DeriveDSub().targetFileName)
        DSub.reset_index(inplace=True)
        DSub['year'] = pd.to_datetime(DSub ['year'])
        DSub.set_index(['year', 'KGB'], inplace=True)
        
        #%% Load EAA price and merge with panel
        df_price = pd.read_csv('data/raw/importEAAprices.csv',sep=';')
        # df_price = pd.read_csv('Z:/ptFarmDataNorway/EAAPrices/importEAAprices.csv',sep=';')
        df_price['year'] = pd.to_datetime(df_price['year'].astype(str)+'-06-15')
        df_price.set_index(['year'], inplace=True)
        
        # Backfill missing values        
        df_price.fillna(method='backfill',inplace=True)
        #%%
        panel_price = pd.merge(panel, df_price, left_index=True, right_index=True, how='inner')
        
        panel_price.sort_index(inplace=True)
        #%%
        merged_df = pd.concat([panel_price,
                               subs,
                               subsNext,
                               CsubD1,
                               CsubD10,
                               CsubD50,
                               DAvgSub,
                               DSub], axis=1
                              )

        #%% Save dataframe to file
        # merged_df.to_csv(self.targetFileName)
        save_obj(merged_df, self.targetFileName)


#%%
class TestTrainSplit(luigi.Task):
    """
    Spilt data in train and test
    """
    # date_interval = luigi.DateIntervalParameter()
    # targetFileName = "../../data/processed/merged_features.csv"
    targetFileName = data_path+"data/processed/train_test_split.pkl"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return MergeFeatures()

    def run(self):
        #%% Import individual dataframes
        panel = load_obj(MergeFeatures().targetFileName)

        #%% Split sample in test and training_set
        #X_train, X_test = train_test_split(panel,
        #                                   test_size=0.01,
        #                                   random_state=44)
        
        X_train, X_test = train_test_split_panel(panel,
                                                 vId='KGB',
                                                 test_size=0.01,
                                                 random_state=100)
        #
        train_test_tuple = (X_train, X_test)
        
        save_obj(train_test_tuple, self.targetFileName)
        

class TestTrainSplit_unitTest(luigi.Task):
    """
    Save a small data file for unit testing, same structure as TestTrainSplit
    """
    # date_interval = luigi.DateIntervalParameter()
    # targetFileName = "../../data/processed/merged_features.csv"
    targetFileName = data_path+"data/processed/train_test_split_unit_test.pkl"
    # targetFileName = "processed/train_test_split_unit_test.pkl"

    def output(self):
        return luigi.LocalTarget(self.targetFileName)

    def requires(self):
        return TestTrainSplit()

    def run(self):
        # Import individual dataframes
        panel = load_obj(MergeFeatures().targetFileName)
        
        
        # get a sample from all kgbs        
        kgb_all = pd.DataFrame(panel.index.get_level_values('KGB').unique())
        kgb = kgb_all.sample(10)
        
        # extract sample from full panel
        panel_sample = panel.loc[(slice(None),list(kgb['KGB'].values)),:]
        
        
        # Split sample in test and training_set
        X_train, X_test = train_test_split_panel(panel_sample,
                                                 vId='KGB',
                                                 test_size=0.5,
                                                 random_state=100)
        #
        train_test_tuple = (X_train, X_test)

        save_obj(train_test_tuple, self.targetFileName)


# %%
"""
class AllReports(luigi.WrapperTask):
    date = luigi.DateParameter(default=datetime.date.today())
    def requires(self):
        yield SomeReport(self.date)
        yield SomeOtherReport(self.date)
        yield CropReport(self.date)
        yield TPSReport(self.date)
        yield FooBarBazReport(self.date)
"""

# %%
# Note: To run this open in anaconda promt "luigid"
# Then open in browser http://agpcom3:8082
if __name__ == '__main__':
    luigi.run(main_task_cls=TestTrainSplit_unitTest)
