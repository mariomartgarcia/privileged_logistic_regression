from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import warnings
from sklearn.datasets import  load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
import tools as tl
warnings.filterwarnings("ignore")

#=========================================================================================================
url_obesity = r'/Users/mmartinez/Desktop/Code/Data/Random_BBDD/ObesityDataSet_raw_and_data_sinthetic.csv'
url_wine = r'/Users/mmartinez/Desktop/Code/Data/Random_BBDD/winequality-white.csv'
#=========================================================================================================

def data(frame):
    X = frame.drop('output', axis = 1)
    y = frame['output']
    
    col = X.columns
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    Xnorm = scaler.fit_transform(X)
    Xn = pd.DataFrame(Xnorm, columns = col)
    return Xn,y

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
    
    
    X_bc, y_bc = data(df_bc)
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
    
    col = X.columns
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    Xnorm = scaler.fit_transform(X)
    Xn = pd.DataFrame(Xnorm, columns = col)
    return Xn, y
    

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
    
    col = X.columns
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    Xnorm = scaler.fit_transform(X)
    Xn = pd.DataFrame(Xnorm, columns = col)
    return Xn, y




