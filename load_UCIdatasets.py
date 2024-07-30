from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import warnings
from sklearn.datasets import  load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
import tools as tl
warnings.filterwarnings("ignore")

#=========================================================================================================
url_obesity = 'data/UCIdataset/ObesityDataSet_raw_and_data_sinthetic.csv'
url_wine = 'data/UCIdataset/winequality-white.csv'
#=========================================================================================================



#======================================
#======================================
#           UCI DATASETS
#======================================
#======================================


#=============================================================================================
#===================================    BREAST CANCER   ======================================
#=============================================================================================   

def breast_cancer(c = False):
    bc = load_breast_cancer()
    df_bc = pd.DataFrame(data = bc.data, columns = bc.feature_names)
    
    df_bc['output'] = bc.target
    #df_bc['output'][df_bc['output'] == 0] = -1
    
    if c == True:
        cor_bc = np.abs(df_bc.corr()['output'][:-1]).sort_values(ascending = False)
        names = cor_bc.index
        return cor_bc, names
    
    
    X_bc = df_bc.drop('output', axis = 1)
    y_bc = df_bc['output']
    y_bc = 2*y_bc-1
    return X_bc, y_bc


#=============================================================================================
#======================================    OBESITY   =========================================
#=============================================================================================

def obesity(c = False):
    
    df = pd.read_csv(url_obesity, sep = ',')
    
    df['Gender'][df['Gender'] == 'Female'] = 0
    df['Gender'][df['Gender'] == 'Male'] = 1
    
    for i in ['FAVC', 'SMOKE', 'SCC', 'family_history_with_overweight']:
        df[i][df[i] == 'no'] = 0
        df[i][df[i] == 'yes'] = 1
        
    for i in ['CAEC', 'CALC']:
        df[i][df[i] == 'no'] = 0
        df[i][df[i] == 'Sometimes'] = 1
        df[i][df[i] == 'Frequently'] = 2
        df[i][df[i] == 'Always'] = 3
        
    for i in ['CAEC', 'CALC']:
        df[i][df[i] == 'no'] = 0
        df[i][df[i] == 'Sometimes'] = 1
        df[i][df[i] == 'Frequently'] = 2
        df[i][df[i] == 'Always'] = 3
        
    for count, j in enumerate(df['MTRANS'].unique()):
        df['MTRANS'][df['MTRANS'] == j] = count
    
    #OUTPUT
    
    for i in ['NObeyesdad']:
        df[i][df[i] == 'Insufficient_Weight'] = 1
        df[i][df[i] == 'Normal_Weight'] = 1
        df[i][df[i] == 'Overweight_Level_I'] = 1
        df[i][df[i] == 'Overweight_Level_II'] = 0
        df[i][df[i] == 'Obesity_Type_I'] = 0
        df[i][df[i] == 'Obesity_Type_II'] = 0
        df[i][df[i] == 'Obesity_Type_III'] = 0
        
    df_cat = df.select_dtypes(include=['object','category'])    
    for i in df_cat.columns:
        df[i] = pd.to_numeric(df[i])
        

    if c == True:
        cor = np.abs(df.corr()['NObeyesdad'][:-1]).sort_values(ascending = False)
        names = cor.index
        return cor, names
        
    X = df.drop('NObeyesdad', axis = 1)
    y = df['NObeyesdad']    
    y = 2*y-1
    
    return X, y
    

#=============================================================================================
#=========================================    WINE   =========================================
#=============================================================================================

def wine(c = False):
    df = pd.read_csv(url_wine, sep = ';')
    df.quality[df.quality<=5] = 0
    df.quality[df.quality>5] = 1
    
    if c == True:
        cor = np.abs(df.corr()['quality'][:-1]).sort_values(ascending = False)
        names = cor.index
        return cor, names
    
    X = df.drop('quality', axis = 1)
    y = df['quality'] 
    y = 2*y-1
    
    return X, y





#DROGAS
def drugs(c = False):
    
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data', header = None)
    df = df.drop([0,14,  15, 16, 17, 19, 21 , 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], axis = 1)
    df = df.rename(columns = {1: 'age',  2: 'gender',  3: 'education',  4: 'country',  5: 'ethnicity',  6: 'nscore',  7: 'escore',  
                         8: 'oscore',  9: 'ascore', 10: 'cscore', 11: 'impulsive', 12: 'ss', 13: 'alcohol', 
                         18: 'cannabis', 20: 'coke'})
    
    for i in ['alcohol', 'cannabis']:
        df[i][df[i] == 'CL0'] = 0
        df[i][df[i] == 'CL1'] = 1
        df[i][df[i] == 'CL2'] = 2
        df[i][df[i] == 'CL3'] = 3
        df[i][df[i] == 'CL4'] = 4
        df[i][df[i] == 'CL5'] = 5
        df[i][df[i] == 'CL6'] = 6
        
    for i in ['coke']:
        df[i][df[i] == 'CL0'] = 0
        df[i][df[i] == 'CL1'] = 0
        df[i][df[i] == 'CL2'] = 1
        df[i][df[i] == 'CL3'] = 1
        df[i][df[i] == 'CL4'] = 1
        df[i][df[i] == 'CL5'] = 1
        df[i][df[i] == 'CL6'] = 1
        
    df_cat = df.select_dtypes(include=['object','category'])    
    for i in df_cat.columns:
        df[i] = pd.to_numeric(df[i])
        
    if c == True:
        cor = np.abs(df.corr()['coke'][:-1]).sort_values(ascending = False)
        names = cor.index
        return cor, names
    
    
    X = df.drop('coke', axis = 1)
    y = df['coke']   
    y = 2*y-1

    return X, y





#Kc2
def kc2():
    dfs = pd.read_csv('data/kc2.csv')
    dfs['problems'] = (dfs['problems'] == 'yes')*1
    X = dfs.drop('problems', axis = 1)
    y = dfs['problems']
    pi_features= list(X.columns[14:])
    y = 2*y-1
    return X, y, pi_features


#Parkinsons
def parkinsons():
    df = pd.read_csv('data/parkinsons.data')
    X = df.drop(['name', 'status'], axis = 1)
    y = df['status']
    y = 2*y-1
    return X, y






