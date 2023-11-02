#%%
import pandas as pd
import numpy as np
import time

import sys
#from multiprocessing import Pool#, Array, Process
#from multiprocessing.dummy import Pool as ThreadPool

from itertools import repeat

#%%
class SubsidyScheme():
    """
    Class to calculate subsidies for PT data
    """


    def __init__(self,
                 t,
                 satTrin,bunn,maks,grov
                 ):
        # Year t0
        self.t = t
        # Payment Rates and  bounderies for each Payment type
        self.satTrin = satTrin
        # Amount of BUNNFRADRAG deduction for TAKTL payment
        self.bunn = bunn
        # MAKSSATS, cap on total abount for different payments
        self.maks = maks
        # Grovfact, factors to determin eligable FOOD area
        self.grov = grov

        self.vFODD = ['x210','x211','x212','x213'];
        self.zNam = ['zoneTAKTL', 'zoneTDISE', 'zoneTDISG',
                     'zoneTDISK','zoneTDISM','zoneTDMLK',
                     'zoneTPROD']

        # Derive names of subsidies
        #self.vSub = self.satTrin['cat_sub'].unique()
        #self.vSub = self.satTrin.index.get_level_values('cat_sub').unique().tolist()
        self.vSub = self.satTrin['vSub']

        # Derive vAct
        #vActRaw = self.satTrin['cat_act'].unique()
        #vActRaw = self.satTrin.index.get_level_values('cat_act').unique().tolist()
        #vActRaw = self.satTrin['rate'].columns.unique().tolist()
        vActRaw = self.satTrin['vActRaw']
        # Append vAct with a x for all numberic variable names
        self.vAct =[]

        for s in vActRaw :
            try:  # try if it is a number with potentially leading zeros
               # Remove leading zeros by transforming to int and back to str
               self.vAct.append('x'+str(int(s)))
            except ValueError:
               self.vAct.append(s)

        # Define all variable names required to calcualte subs
        self.vVar = self.vAct+ self.vFODD+self.zNam
        


    def eligableFood(self,zSubX,zGrov):
        """
        Calcualte eligable food based on grovfo and immark factors
        """
        # Reshape grov dataframe
        zGrov.index = [zGrov.cat_sub, zGrov.cat_act]
        zGrov = zGrov.drop(['cat_act', 'year', 'cat_sub'], axis=1)

        # Calcualte product of activities and inmark rates
        vInmark = zGrov['rate']['INMARK'].index
        ratesInmark = (zGrov['rate']['INMARK']
                        .values.reshape(1, vInmark.shape[0])
                      )
        inmark = zSubX[vInmark].values*ratesInmark
        sumInm = np.sum(inmark, axis=1)
        # Calcualte product of activities and grovfo rates
        vGrovfo = zGrov['rate']['GROVFO'].index
        ratesGrovfo = (zGrov['rate']['GROVFO']
                        .values
                        .reshape(1, vGrovfo.shape[0])
                      )
        grovfo = zSubX[vGrovfo].values*ratesGrovfo
        sumGro = np.sum(grovfo, axis=1)

        # Step 1: get min of INMARK and x212
        assert(zSubX.x212.values.shape ==sumInm.shape)
        areaStep1 = np.minimum(zSubX.x212.values, sumInm)

        # Step 2 a):  get sum of 210,211,213 and 60% of eligible area
        #               calcualted in Step 1
        areaStep2  = zSubX.x210.values*1.
        areaStep2 += zSubX.x211.values*1.
        areaStep2 += zSubX.x213.values*1.
        areaStep2 += 0.6* areaStep1*1.

        # Step 2 b): get minimum of area calcualted in a) and GROVFO
        assert(areaStep2.shape ==sumGro.shape)
        eligibleFODD =  np.minimum(areaStep2, sumGro);

        # replace FOOD with eligibleFODD
        zSubX.FODD = eligibleFODD

        return zSubX

    def correctAmmekyr(self,X):
        """
        Correction to TDMLK ammekyr from 2002 to 2008
        From 01.07.02 a special subside (driftstilskudd) was added
        for milking cows in meat production ( TDMLK.121 ammekyr).
        This subside is not giving to companies that already get
        subsides for their milk production (so you can either get
        TDMLK for code 120 or for 121???). In 2008 this was changed
        and companies already receiving other subsides for milk
        production could also receive this special subside.
        Note: The TDMLK subsidy for ammekyr (121) is also only
        for 6-40 cows. Farms with less then6  cows does not
        receive anything.
        """
        #  Check if year is between 2002-2007
        #    In these years you do not get anything for 121 if you
        #    already get something for 120
        if self.t>=2002 and self.t<2008:
            X.loc[(X["x120"]>0),"x121"] = 0
            #X.x121[X.x120>0] = 0 # Set x121 if you have x120

        # In all years payment for 121 is only for farms with
        #   more then 5 Suckler cows. Set number of suckler cows to zero for
        #  all farms with less then 6 ammekyr (121)
        X.loc[(X.x121<6),"x121"] = 0

        return X

    def applyMaks(self,pay):
       """
       Apply MAKSSATS, cap on total payments
       Argument:
           pay -- dataframe of payments with observations in rows and types
                  for subsidies in columns
           maks -- makssats table for specific year one number for each subsidy

       Returns:
           pay -- Same as input but with makssats applied
       """
       #
       # Loop over all entries in maks and apply clip values from top if
       # bound if not inf and if subsidy is in pay (subsidy might not be in pay
       # if self.vSub does not include all subsidies i.e. if only a specific 
       # one should be calcualted)
       for i in range(0,self.maks.shape[0]):
           if (np.isinf(self.maks['rate'].iloc[i])==False) & (self.maks['cat_sub'].iloc[i] in pay.columns):
               pay[self.maks['cat_sub'].iloc[i]] = (pay[self.maks['cat_sub'].iloc[i]]
                                           .clip_upper(self.maks['rate'].iloc[i]))
       #        
       return pay

    def applyBunn(self, pay):
       """
       Substract BUNNFRADRAG
       """
       # Loop over all entries in maks and apply clip values from top if
       # bound if not inf
       for i in range(0, self.bunn.shape[0]):
           if self.bunn.iloc[i,1] in pay.columns:
               pay[self.bunn.iloc[i,1]] -= self.bunn.iloc[i,2]
               pay[self.bunn.iloc[i,1]] = pay[self.bunn.iloc[i,1]].clip_lower(0)

       return pay

    def subZone(self,zSubX,subName,zSubSatTrin,zGrov):
        """
        Function to caclualte payments in one zone
        """
   
        rate, lowcut, upcut, ordCol = zSubSatTrin

        # Calcualte eligable food if subsidy is TAKTL
        if subName=='TAKTL':
             zSubX = self.eligableFood(zSubX, zGrov)

        # check if columns need to be selected
        #   Performance wise extracting columns is expensive
        if np.array_equal(zSubX.columns, ordCol):
            XX = zSubX.values  # transform zSubX to numpy
        else:
            XX = zSubX[ordCol].values  # transform zSubX to numpy
 
        XLow = XX-lowcut  # Substract lower bound from activtiy

        XLow[XLow<0] = 0  # set differnce to zero if negative

        XStep = np.minimum(XLow, upcut-lowcut)  # find minimum of
                                               # upper-lower bound and Xlow
        XPay = rate*XStep  #multiply rates with XStep
        XsumStep = np.sum(XPay,axis=0)  # sum over steps
        Xsum = np.sum(XsumStep,axis=1, keepdims=True)
        #%
        return Xsum

    def subCore(self,subX,subName,subSatTrin):
        """
        Function to calcualte payment for one particlar subsidy

        """
        
        # Get unique list of zones
        zones = self.satTrin[subName]['zones']

        if len(zones)>1:
            
            vZone = 'zone'+subName
            
            subX.reset_index(inplace=True)
            subX.set_index([vZone,'year','KGB'],inplace=True)
            subX.sort_index(inplace=True)
            
        sub =  pd.DataFrame(0.0, index=subX.index, columns=['sub'])

        # Make corrections for Ammekry if subsidy is TDMLK
        if subName=='TDMLK' and self.t >=2002:
            subX = self.correctAmmekyr(subX)
        
        if len(zones)>1:
            
            for z in zones:
                # Check if zone is actuall in subX
                if (z in subX.index.get_level_values(vZone).unique().tolist()):
                        
                    zSubSatTrin = self.satTrin[subName]['rlu'][z]
                    
                    zSubX = subX.loc[(z),:]
                    
                    idxGrovZone = (self.grov['zone'] ==z)
                    zGrov = self.grov[idxGrovZone]
                    zGrov = zGrov.drop('zone', axis=1)
    
                    # Calculate sub for zone
                    sub.loc[(z),:] = self.subZone(zSubX, subName, zSubSatTrin, zGrov)
                    
            sub.index = sub.index.droplevel(vZone)
            
        else:
            zSubSatTrin = self.satTrin[subName]['rlu']
            sub = self.subZone(subX,subName,zSubSatTrin,self.grov)
            
        return sub

    def oneSub(self,subName,X):
        """
        Wrapper function to calculate one single subsidy 
        """
        #%
        #print(subName)
        subSatTrin = self.satTrin[subName]
        
        subVar = self.satTrin[subName]['subVar']
        subX = X[subVar].copy()
        #
        sub = self.subCore(subX,subName,subSatTrin)
        #%
        return sub
        
    
    def sub(self,X,onlyDpay=True):
        """
        Calculate Subsidies for all observations in X for one particular year.

        Argument:
         X -- (NxK)Input data with all acitivities and zones

        Returns:
         pay -- (Nxp) Matrix with different payements (columns) for each
                 observation (rows)
        """
        #%
        X.reset_index(inplace=True)
        X.set_index(['year','KGB'],inplace=True)
        X.sort_index(inplace=True)
        
        #
        pay = pd.DataFrame(0.0, index=X.index, columns=self.vSub)

        # Fill nans in X with 0
        X.fillna(0,inplace=True)
        
        # Loop over subsidies
        for subName in self.vSub:
            
             pay[subName] =  self.oneSub(subName,X)
        
        # Apply MAKSSATS, cap on total payments for some payments types
        pay = self.applyMaks(pay)
     
        # Substract BUNNFRADRA
        pay = self.applyBunn(pay)
       
        # Transform payments in 1000 NOK
        pay = pay/1000

        if onlyDpay:
            pay = pay.sum(axis=1)
        else:
            pay['DPAY'] = pay.sum(axis=1)
            
        #elapsed = time.time() - t
        #print(elapsed)
        #%
        return pay


#%%


class DeriveSubsidies():
    """
    Class to derive subsidies, delta subsidies and changes in detla subsidies
    """
    def __init__(self,
                 t0,
                 sattrint0,
                 bunnt0,
                 makst0,
                 grovt0,
                 t1 = [],
                 sattrint1 = [],
                 bunnt1 = [],
                 makst1 = [],
                 grovt1 = [],
                 vDiffAct = ['CERE','GRON','SAU','STOR','HEST','VSAU','GEIT',
                             'USAU','FRUK','BAER',
                             'x410','x420','x440',
                             'x521','x522',
                             'x210','x211','x212','x230',
                             'x120','x121','x136','x140', 'x142',
                             'x155', 'x157','x160','x180']
                 ):
      

        # List of variable names for which diff in Subs should be derived
        self.vDiffAct = vDiffAct

        self.t0 = t0
        self.sattrint0= sattrint0
        self.bunnt0 = bunnt0
        self.makst0 = makst0
        self.grovt0 = grovt0
        self.t1 = t1
        self.sattrint1= sattrint1
        self.bunnt1 = bunnt1
        self.makst1 = makst1
        self.grovt1 = grovt1
        
        #self.pool = Pool(processes=1)

    def subDUnit(self, X, t,
                 sattrin, bunn, maks, grov,
                 unitIncrease=1,
                 setZero = False,
                 currentSub=pd.DataFrame(),
                 usePool=True):
        

        """
        Difference in Subsidies for a unitIncrease in activities

        Argument:
         X -- (NxK)Input data with all acitivities and zones
         setZero -- Flag to set activity to zero
                    (used to derive the contribution of that activity to
                    overall subsidies)
         unitIncrease -- Amount by which activity sould be increased

        Returns:
         subD -- Difference in Subsidies
        """

        print("Beginn SubDUnit")
        #%%
        tic = time.time()    
        # Creat subscheme
        subScheme = SubsidyScheme(t,
                                  sattrin,
                                  bunn,
                                  maks,
                                  grov
                                  )
        print("inSubDUnit after SubsidyScheme")
        
        #%% Check if current sub is supplied 
        if currentSub.empty:
            # if not, calcualte current sub
            currentDpay = subScheme.sub(X.copy(),onlyDpay=True)
        else:
            # Check if a Dataframe with all columns or a series of DPAY
            # Most efficintly a series with only DPAY is supplied
            # This can be derived using "subScheme.sub(X,onlyDpay=True)"
            if isinstance(currentSub, pd.DataFrame):
                currentDpay = currentSub['DPAY']      
            else:
                currentDpay = currentSub
            
        if usePool:
            
            subDArray  = np.zeros((currentDpay.shape[0],len(self.vDiffAct)))
        else:
            #Create dataframe to hold results
            subD = pd.DataFrame(index=X.index, columns=[self.vDiffAct])
        #%%

        print("inSubDUnit after 1stIf")
        Xpool = []
        # Loop over avtivities
        for vAct in self.vDiffAct:
            Xincrease = X.copy()  # Copy X

            # Check if setZero is false, if yes increase activity oterwise set
            # to zero
            if setZero == False:
                Xincrease[vAct] +=unitIncrease  # increase selected column by one
            else:
                Xincrease[vAct] = 0.0

            # Check if activtiy is either x210 or x212 if yes also increase FODD
            if (vAct == 'x210') | (vAct == 'x212'):
                if setZero == False:
                    # Increase fodder by one
                    Xincrease['FODD']+=unitIncrease
                else: # If setZero Ture substract x210 or x212 from FODD
                    if (vAct == 'x210'):
                        Xincrease['FODD']-=X['x210']
                    elif (vAct == 'x212'):
                        Xincrease['FODD']-=X['x212']

            if usePool:
                Xpool.append(Xincrease)
            else:
                increasesSub = subScheme.sub(Xincrease,onlyDpay=True)  # Calc new subsidies
                subD[vAct] = increasesSub-currentDpay  # calc diff
        #%% Version to use a Pool

        print("inSubDUnit after forloop")
        if usePool:
            
            # for i in range(len(Xpool)):
            #     print(Xpool[i].shape)


            #sys.exit()


            #pool = Pool(processes=1)
            #payMap = pool.map(subScheme.sub, Xpool)
            print("before dpayMap")
            # print("Function called : ", subScheme.sub)
            #dpayMap = pool.map(subScheme.sub, Xpool)
            dpayMap = list(map(subScheme.sub, Xpool))
            # print(type(dpayMap))
            #pool.close() 
            #pool.join()
            
            for i in range(0,len(dpayMap)):
                subDArray[:,i] = dpayMap[i]-currentDpay
            subD = pd.DataFrame(subDArray,index=X.index, columns=self.vDiffAct) 
        
        elapsed = time.time() - tic
        #print('Elapsed Time in subDUnit ', elapsed)
        #%%    
        print("inSubDUnit after usePool")
        return subD

    def avgSub(self, X,t,sattrin,bunn,maks,grov,currentSub=pd.DataFrame()):
        """
        Calculate average subsidy per unit
        1) Calculate actual subsidy
        2) Calculate subsidy without specific actifity
        3) Calculate difference between 1) and 2)
        4) Divide difference by number of unities in activity to get the
           average subsidy

        Argument:
         X -- (NxK)Input data with all acitivities and zones
         t -- year
         sattrin,bunn,maks,grov from the particular year to be considered

         unitIncrease -- Amount by which activity sould be increased

        Returns:
         CsubD -- Change in difference in Subsidies
        """
       
        #%% Step 1-3) Calculate difference between actuall subsidy and
        # subsidies without specific activity
        subWithout = self.subDUnit(X,
                               t,
                               sattrin,
                               bunn,
                               maks,
                               grov,
                               unitIncrease = 0,
                               setZero = True,
                               currentSub=currentSub
                               )
        # Step 4 devide difference by number of units
        # Multiply by -1 to make aveaged per unit positve
        #%% Fill nan's
        avgSub = subWithout.copy()
        #%%
        for vAct in self.vDiffAct:
            avgSub[vAct] = subWithout[[vAct]]/X[[vAct]]*(-1)
        avgSub.fillna(0,inplace=True)
        #%%
        return avgSub

    def get_avgSub(self, X,currentSub=pd.DataFrame()):
        """
        Method to get average subsidy for one year
        """
        #%% Get average subsidies for t0
        avgSubt0 = self.avgSub(X,
                               self.t0,
                               self.sattrint0,
                               self.bunnt0,
                               self.makst0,
                               self.grovt0,
                               currentSub=currentSub
                               )
        #%%
        return avgSubt0

    def get_dAvgSub(self, X,currentSub=pd.DataFrame()):
        """
        Method to get change in average subsidy from t0 to t1 with the
        production activities from t0
        """
        #%% Get average subsidies for t0
        avgSubt0 = self.get_avgSub(X,currentSub=currentSub)
        #%%
        avgSubt1 = self.avgSub(X,
                               self.t1,
                               self.sattrint1,
                               self.bunnt1,
                               self.makst1,
                               self.grovt1
                               )

        # Calculate difference in avgSubt0 and avgSubt1
        dAvgSub = avgSubt1-avgSubt0

        #%%
        return dAvgSub


    def get_sub(self, X):
        """
        Method to get subsidies for one year
        """
        # Creat subscheme
        subScheme = SubsidyScheme(self.t0,
                                  self.sattrint0,
                                  self.bunnt0,
                                  self.makst0,
                                  self.grovt0
                                  )

        # Calcualte current sub
        currentSub = subScheme.sub(X,onlyDpay=False)
        return currentSub

    def get_subNext(self, X):
        """
        Method to get subsidies for one year
        """
        # Creat subscheme
        subScheme = SubsidyScheme(self.t1,
                                  self.sattrint1,
                                  self.bunnt1,
                                  self.makst1,
                                  self.grovt1
                                  )

        # Calcualte with current X and next year sub
        subNext = subScheme.sub(X,onlyDpay=False)
        return subNext


    def get_dSub(self, X,currentSub=pd.DataFrame()):
        """
        Method to get difference in subsidies between t0 and t1
        """
        if currentSub.empty:
            # Get sub for t0
            subt0 = self.get_sub(X)
        else:
            subt0 = currentSub

        # Get sub for t1
        subt1 = self.get_subNext(X)

        # Calcualte change in sub between t0 and t1
        dSub = subt1-subt0
        return dSub

    def get_subDUnit(self, X, unitIncrease=1,setZero = False,currentSub=pd.DataFrame()):
        """
        Provide difference in Subsidies for a unitIncrease in activities for t0

        Argument:
            ---

        Returns:
            subD -- Difference in Subsidies
        """

        subDt0 = self.subDUnit(X,
                               self.t0,
                               self.sattrint0,
                               self.bunnt0,
                               self.makst0,
                               self.grovt0,
                               unitIncrease,
                               setZero,
                               currentSub=currentSub
                               )
        return subDt0

    def get_CsubDUnit(self, X, unitIncrease=1, setZero = False, currentSub=pd.DataFrame()):
        """
        Calculate change in difference in Subsidies between scheme in t0 and t1

        Argument:
         X -- (NxK)Input data with all acitivities and zones

         unitIncrease -- Amount by which activity sould be increased

        Returns:
         CsubD -- Change in difference in Subsidies
        """
        #%% Calculate subsidy for t0
        subDt0 = self.subDUnit(X,
                               self.t0,
                               self.sattrint0,
                               self.bunnt0,
                               self.makst0,
                               self.grovt0,
                               unitIncrease,
                               setZero = False,
                               currentSub=currentSub
                               )
        # Calculate subsidy for t1
        subDt1 = self.subDUnit(X,
                               self.t1,
                               self.sattrint1,
                               self.bunnt1,
                               self.makst1,
                               self.grovt1,
                               unitIncrease,
                               setZero = False
                               )

        #%% Calculate difference in subt0 and subt1
        CSubD = subDt1-subDt0
        #%%
        return CSubD

#%% Append numeric variable names with an leading "x"
def xAppendCode(codes):
    vActRaw = codes.unique()
    #Append vAct with a x for all numberic variable names
    vAct =[]
    vActSel = []
    for s in vActRaw:
        try: # try if it is a number with potentially leading zeros
           # Remove leading zeros by transforming to int and back to str
           vAct.append('x'+str(int(s)))
           vActSel.append(s)
        except:
            pass
    # Create dict
    dictAct = dict(zip(vActSel,vAct))
    return dictAct

#%%
def sattrinToDict(dsat,grov):
    """
    Prepear a dictionary with the for sattrin in the correct format
    required to calculate subsidies efficiently
    Doing this here once is more efficient because reashping the pandas
    Dataframe is computationally expensive
    """
    #%%
    dsat.reset_index(inplace=True)
    dsat.set_index(['year','cat_sub','zone','step'],inplace=True)
    dsat.sort_index(inplace=True,sort_remaining=True)
    
    #
    dsat.fillna(0,inplace=True)
    vSub = dsat.index.get_level_values('cat_sub').unique().tolist()
    years = dsat.index.get_level_values('year').unique().tolist()
    
    zNam = ['zoneTAKTL', 'zoneTDISE', 'zoneTDISG',
            'zoneTDISK','zoneTDISM','zoneTDMLK',
            'zoneTPROD']
    #%%
    satDict = {}
    allvAct = []
    for year in years:
        ysat =  dsat.loc[(year),:]
        satDict[year] = {}
    
        satDict[year]['vSub'] = ysat.index.get_level_values('cat_sub').unique().tolist()
    
        satDict[year]['vActRaw'] = ysat['rate'].columns.unique().tolist()

        # Loop over subsidies
        for subName in satDict[year]['vSub']:
        #for subName in vSub:
            #
            satDict[year][subName] = {}
            subSatTrin = ysat.loc[(subName),:]
            
            subVar = subSatTrin['rate'].columns.unique().tolist()
            
            zones = subSatTrin.index.get_level_values('zone').unique().tolist()
            
            satDict[year][subName]['zones'] = zones
            satDict[year][subName]['rlu'] = {}

            #First get names of all variables used in sattrin in all zones               
            subVar = []            
            if len(zones)>1:
                for z in zones:
                    zSubSatTrin = subSatTrin.loc[(z),:]
                    
                    selCol = zSubSatTrin['rate'].columns[zSubSatTrin['rate'].sum(axis=0)>0]
                    
                    subVar +=list(selCol)
            else:
                selCol =  subSatTrin['rate'].columns[subSatTrin['rate'].sum(axis=0)>0]
                subVar +=list(selCol)

            selCol = list(set(subVar))
            selCol.sort()
            allvAct +=selCol
            if len(zones)>1:
                
                for z in zones:
                    
                    # Find sattrin corresponding to zone
                    zSubSatTrin = subSatTrin.loc[(z),:]
                    
                    zSubSatTrin = zSubSatTrin.loc[:,(slice(None),selCol)]
                                        
                    rate =  zSubSatTrin['rate'].values
                    lowcut =  zSubSatTrin['lowcut'].values
                    upcut =  zSubSatTrin['upcut'].values
                    #
                    rate = rate.reshape(rate.shape[0], 1, rate.shape[1])
                    lowcut = lowcut.reshape(lowcut.shape[0], 1, lowcut.shape[1])
                    upcut = upcut.reshape(upcut.shape[0], 1, upcut.shape[1])
                    #
                    ordCol =  zSubSatTrin['rate'].columns  
                    satDict[year][subName]['rlu'][z] = (rate,lowcut,upcut,ordCol)
                    
            else:
                subSatTrin = subSatTrin.loc[:,(slice(None),selCol)]
                
                rate =  subSatTrin['rate'].values
                lowcut =  subSatTrin['lowcut'].values
                upcut =  subSatTrin['upcut'].values
                #
                rate = rate.reshape(rate.shape[0], 1, rate.shape[1])
                lowcut = lowcut.reshape(lowcut.shape[0], 1, lowcut.shape[1])
                upcut = upcut.reshape(upcut.shape[0], 1, upcut.shape[1])
                #
                ordCol =  subSatTrin['rate'].columns
                satDict[year][subName]['rlu'] = (rate,lowcut,upcut,ordCol)
                 
            if ('zone'+subName) in zNam: # Check if zone in required
                selCol.append('zone'+subName) # Add zone
    
            if subName=='TAKTL':
                vTAKTL = (grov.cat_act.unique().tolist()
                            + ['x210','x211','x212','x213'])
                selCol += vTAKTL
                allvAct += vTAKTL
                
            satDict[year][subName]['subVar'] = list(set(selCol))
            satDict[year][subName]['subVar'].sort() 
    
    allvAct = list(set(allvAct))
    allvAct.sort()
    satDict['allvAct'] = allvAct+zNam
    #%%
    return satDict

#%%
def loadSubsidySchemeFiles():
    """
    Load all file required for the subsidy scheme
    """
    # %% 
    # import src.data.make_dataset as make_dataset
    import make_dataset

    # Load files for subsidy scheme
    bunn = pd.read_csv(make_dataset.deriveBunn().targetFileName)
    maks = pd.read_csv(make_dataset.deriveMaks().targetFileName)
    satser = pd.read_csv(make_dataset.deriveSatser().targetFileName)
    trin = pd.read_csv(make_dataset.deriveTrin().targetFileName)
    grov = pd.read_csv(make_dataset.deriveGrovfact().targetFileName)

    # Create sattrin Table (i.e merge satser and trin table)
    lowTrin = trin.copy()
    lowTrin['step'] +=1   # Increase step by one, such that on merge cut from
                          # the previous step is merged as an lower bound
    lowTrin.rename(columns={'cut':'lowcut'}, inplace=True)
    sattrin = pd.merge(satser,
                       lowTrin,
                       how='left',
                       on=['year', 'cat_sub', 'cat_act', 'step']
                       )

    sattrin = pd.merge(sattrin,
                       trin,
                       how='left',
                       on=['year', 'cat_sub', 'cat_act','step']
                       )
    sattrin.rename(columns={'cut':'upcut'}, inplace=True)

    # set lower cut to zero if nan
    sattrin['lowcut'] = sattrin['lowcut'].fillna(0)
    sattrin['upcut'] = sattrin['upcut'].fillna(np.inf)

    #------ Append numeric variable names with an leading "x"
    dictAct = xAppendCode(sattrin.cat_act)
    sattrin = sattrin.replace({"cat_act": dictAct})
    #
    sattrin.reset_index(inplace=True)
    #sattrin.set_index(['year', 'cat_sub','cat_act','step','zone'], inplace=True)  # set index to year
    #sattrin.sort_index(inplace=True)
    # %%
    # -----------------------------------------------------------
    # Adjust year 2. Von Klaus
    # Beihilfeninformation (Zonen, Sätze, trinns, max und bunnfradrag) bis zu 
    # 2018 in «tilskudd_2018.gms». Für folgende Beihilfen musst Du ein 
    # Jahr zurückgehen, d.h. PT-Daten in 2017 mit Beihilfen aus 2018 
    # verknüpfen: TDMLK, TAKTL, TPROD, TVELF, TBEIT und TUTMK. Für 
    # alle anderen Beihilfen ist die Jahr gleich.

    for k in ['TDMLK', 'TAKTL', 'TPROD', 'TVELF', 'TBEIT', 'TUTMK']:
        sattrin.loc[sattrin['cat_sub']==k,'year'] += -1
        maks.loc[maks['cat_sub']==k,'year'] += -1
        bunn.loc[bunn['cat_sub']==k,'year'] += -1


    
    sattrin = pd.pivot_table(sattrin,index=['year','cat_sub','zone','step'],columns='cat_act',values=['rate','lowcut','upcut'])
    sattrin.sort_index(inplace=True)

    dictGrov = xAppendCode(grov.cat_act)
    grov = grov.replace({"cat_act": dictGrov})
    #%%
    satDict = sattrinToDict(sattrin.copy(),grov)
    
    # %%    
    return bunn, maks, satDict, trin, grov, sattrin 


# %%

def compareToMatlab():
    """
    Compare caclcualted subsidies to the once calculated in Matlab
    This is not longer correct because there was in error in Matlab
    with respect to the zone of TDMLK which change over the years.
    The error is resoved in Matlab but the subsidies are not calcualted 
    anew and the saved csv file is not correct. Before this comparison 
    is used the csv file would need to be derived again
    
    """

    # Import of activties
    df = pd.read_csv('Z:/5_NeuralNetworkPolicy/Model/nn_norway/data/raw/featureTargetWithSubs.csv',delim_whitespace=True)

    # Delete missing observations base on year
    isNanIdx = df[np.isnan(df['year'])].index # get index of nan observations
    df = df.drop(isNanIdx) # get rid of all nan's

    df.fillna(0,inplace=True)

    # load Subsidi data
    bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()

    # Compute Subsidies= bb and CsubDUnit==aa
    t0 = 2010
    t1 = 2011


    XAll = df
    X = XAll[df.year == t0]
    Xyear = X

    sattrint0 = sattrin.loc[(t0,slice(None),slice(None),slice(None),slice(None)),:]
    bunnt0 = bunn[bunn.year == t0]
    makst0 = maks[maks.year == t0]
    grovt0 = grov[grov.year == t0]

    sattrint1 = sattrin.loc[(t1,slice(None),slice(None),slice(None),slice(None)),:]
    bunnt1 = bunn[bunn.year == t1]
    makst1= maks[maks.year == t1]
    grovt1 = grov[grov.year == t1]

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

    tic = time.time()

    bb = deriveSub.get_sub(X)
    # aa = deriveSub.get_subDUnit(X)
    aa = deriveSub.get_CsubDUnit(X, 1)

    elapsed = time.time() - tic
    print('Time:', elapsed)

    # 'TAKTL', 'TDISE', 'TDISG', 'TDISK', ...
    # 'TDISM', 'TDMLK', 'TGRUN', ...
    # 'TPROD', 'TUTMK', 'TVELF','DPAY'
    vSub = "DPAY"
    diffVSub = "dpayDeltaDiff_" + vSub

    # print(
    #         ("Difference in sum:" + vSub + "={}")
    #         .format(np.sum(Xyear[vSub] - bb[vSub]))
    #      )
    dbb = pd.concat([Xyear[vSub], bb[vSub]], axis=1)
    # print(dbb.head(5))

    vSub = "x120"
    diffVSub = "dpayDeltaDiff_"+vSub

    # print(
    #         ("Difference in dpayDeltaDiff_"+vSub+"={}")
    #         .format(np.sum(Xyear[diffVSub] - aa[vSub]))
    #      )
    daa = pd.concat([Xyear[diffVSub], aa[vSub]], axis=1)
    # print(daa.head(5))

    return dbb, daa

#%%
def playground():
    # %%
    
    aa = sattrin.loc[(slice(None),'TPROD',1),(slice(None),'SAU')].droplevel([1,2]).droplevel(1,axis=1)
    # aa = sattrin.loc[(slice(None),'TPROD',1),(slice(None),'x136')].droplevel([1,2]).droplevel(1,axis=1)

    fig,ax = plt.subplots()
    for year in range(2012,2015):

        ai = pd.concat([aa.loc[year,['lowcut','rate']].rename(columns={'lowcut':'cut'}),
                        aa.loc[year,['upcut','rate']].rename(columns={'upcut':'cut'})]).sort_values(
                            ['cut','rate'], ascending=[True, False])
        ai
        ax.plot(ai['cut'],ai['rate'], label=year)
    ax.legend(frameon=False)
    # %%
    aa = sattrin.loc[(slice(2007,2015),'TPROD',1),(slice(None),'SAU')]
    aa
    # %%
    maks.loc[maks['cat_sub']=='TPROD',:]
    # %%
    sattrin.columns
    
    # %%
    sattrin.loc[(slice(2013,2014),'TPROD',slice(None)),:].index
    # %%
    aa.loc[slice(2011,2014),:]
#%%
#-----------------------------------------------------------------------------
#                           Code for testing
#-----------------------------------------------------------------------------
"""
import luigi
import pandas as pd
import pickle

import sys
sys.path.append("src/data")
sys.path.append("src/features")

# from luigi import configuration
from make_dataset import BlancePanel
from calc_dpay import loadSubsidySchemeFiles
from calc_dpay import DeriveSubsidies
from sklearn.model_selection import train_test_split
data_path = 'C:/temp/storm/nn_norway/'  # define path to data


#%%
# %%
wd = "/nn_norway"#
os.chdir(wd)
# %%
sys.path.append(os.path.join("src","features"))
sys.path.append(os.path.join("src","models"))
sys.path.append(os.path.join("src","mod_lib"))
sys.path.append(os.path.join("src","lib"))
sys.path.append(os.path.join("src","data"))

from make_dataset import BlancePanel


#  Import file and set intext to year
df = pd.read_csv(BlancePanel().targetFileName)

df['year'] = pd.to_datetime(df['year'])  # transform to datatime
df.set_index(['year', 'KGB'], inplace=True)  # set index to year

# Sorting the index is important otherwise slicing e.g. by year
# is not possible
df.sort_index(inplace=True)

# Load subsidy scheme data
bunn, maks, sattrin, trin, grov, _ = loadSubsidySchemeFiles()

#%%
t0 = 2014
t1 = 2015
df.reset_index(inplace=True)
df['year'] = pd.to_datetime(df['year'])  # transform to datatime
df.set_index(['year'], inplace=True)  # set index to year

#%%
X = df.loc["2014"]

#%%
# Create Subsidy scheme
deriveSub = DeriveSubsidies(t0,
                            sattrin.loc[(t0,slice(None),slice(None),slice(None),slice(None)),:],
                            bunn[bunn.year == t0],
                            maks[maks.year == t0],
                            grov[grov.year == t0],
                            t1,
                            sattrin.loc[(t1,slice(None),slice(None),slice(None),slice(None)),:],
                            bunn[bunn.year == t1],
                            maks[maks.year == t1],
                            grov[grov.year == t1]
                            )
self = deriveSub

#%%
deriveSub.get_CsubDUnit(X)
"""