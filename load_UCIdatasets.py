from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import warnings
from sklearn.datasets import  load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
import tools as tl
from ucimlrepo import fetch_ucirepo 
warnings.filterwarnings("ignore")

#=========================================================================================================
url_obesity = r'/Users/mmartinez/Desktop/Code/Python/LRPI/Data/UCIdataset/ObesityDataSet_raw_and_data_sinthetic.csv'
url_wine = r'/Users/mmartinez/Desktop/Code/Python/LRPI/Data/UCIdataset/winequality-white.csv'
url_heart = r'/Users/mmartinez/Desktop/Code/Python/LRPI/Data/Discussion_dataset/framingham.csv'
url_wm = r'/Users/mmartinez/Desktop/Code/Python/LRPI/Data/Discussion_dataset/WM_data.csv'
url_heart2 = r'/Users/mmartinez/Desktop/Code/Python/LRPI/Data/Discussion_dataset/heart.csv'
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

def wine_reg(c = False):
    df = pd.read_csv(url_wine, sep = ';')
    
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


#SPAM
def spam(c = False):
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data', header = None)
    df = df.rename(columns = {57: 'output'})
    #df.output[df.output == 0] = -1

    if c == True:
        cor = np.abs(df.corr()['output'][:-1]).sort_values(ascending = False)
        names = cor.index
        return cor, names

    X = df.drop('output', axis = 1)
    y = df['output']
    y = 2*y-1    

    return X, y




#Heart
def heart():
    df = pd.read_csv(url_heart, sep = ',')
    imp = df.drop('TenYearCHD', axis = 1)
    index = []
    for i in range(imp.shape[1]):
        if len(imp.iloc[:,i].unique()) < 6:
            if imp.iloc[:,i].isnull().sum() != 0:
                index.append(i)

    imputer = KNNImputer(n_neighbors=5)
    i = imputer.fit_transform(imp)
    imp_correct = pd.DataFrame(i, columns = imp.columns)

    for i in index:
        for j in range(imp_correct.shape[0]):
            imp_correct.iloc[j, i] = round(imp_correct.iloc[j, i])
            
    X = imp_correct
    y = df['TenYearCHD']  
    y = 2*y-1
    
    return X, y


#WM

def wm():
    df = pd.read_csv(url_wm, sep = ',')
    
    df.age5[df.age5 == '20-24'] = 1
    df.age5[df.age5 == '25-29'] = 2
    df.age5[df.age5 == '30-34'] = 3
    df.age5[df.age5 == '35-39'] = 4
    df.age5[df.age5 == '40-44'] = 5
    df.age5[df.age5 == '45-49'] = 6
    df.age5[df.age5 == '50-54'] = 7
    df.age5[df.age5 == '55-59'] = 8
    df.age5[df.age5 == '60-64'] = 9
    df.age5[df.age5 == '65-69'] = 10
    df.age5[df.age5 == '70-74'] = 11
    df.age5[df.age5 == '75-79'] = 12
    df.age5[df.age5 == '80-84'] = 13

    df.age5.astype(int)
    
    imp = df.drop('wm', axis = 1)
    index = []
    for i in range(imp.shape[1]):
        if len(imp.iloc[:,i].unique()) < 6:
            if imp.iloc[:,i].isnull().sum() != 0:
                index.append(i)

    imputer = KNNImputer(n_neighbors=5)
    i = imputer.fit_transform(imp)
    imp_correct = pd.DataFrame(i, columns = imp.columns)

    for i in index:
        for j in range(imp_correct.shape[0]):
            imp_correct.iloc[j, i] = round(imp_correct.iloc[j, i])
            
    X = imp_correct
    y = df.wm
    y = 2*y-1
    return X, y





#Heart

def heart2():
    df = pd.read_csv(url_heart2, sep = ',')
    X = df.drop('target', axis = 1)
    y = df.target
    
    return X, y


#Abalone
def abalone():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', header = None)
    df = df.rename(columns = {0: 'sex', 1: 'length',  2: 'diameter',  3: 'height',  4: 'whole_weight',  5: 'shucked_weight',  6: 'viscera_weight',  7: 'shell_weight', 8: 'rings'})
    df = df[df['sex'] != 'I'].reset_index(drop = True )
    df['sex'][df['sex'] == 'M'] = 1
    df['sex'][df['sex'] == 'F'] = 0
    
    X = df.drop('sex', axis = 1)
    y = pd.to_numeric(df.sex)
    y = 2*y-1

    return X, y


def car():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', header = None)
    df = df.rename(columns = {0: 'buying', 1: 'maint',  2: 'doors',  3: 'persons',  4: 'lug_boot',  5: 'safety',  6: 'output'})     
    df.buying = pd.Categorical(df.buying, ordered = True, categories=['low', 'med', 'high', 'vhigh']).codes
    df.maint = pd.Categorical(df.maint, ordered = True, categories=['low', 'med', 'high', 'vhigh']).codes 
    df.doors = pd.Categorical(df.doors, ordered = True, categories=['2', '3', '4', '5more']).codes   
    df.persons = pd.Categorical(df.persons, ordered = True, categories=['2', '4', 'more']).codes   
    df.lug_boot = pd.Categorical(df.lug_boot, ordered = True, categories=['small', 'med', 'big']).codes   
    df.safety = pd.Categorical(df.safety, ordered = True, categories=['low', 'med', 'high']).codes       
    df.output = pd.Categorical(df.output, ordered = True, categories=['unacc', 'acc', 'good', 'vgood']).codes   
    
    df['output'][df['output'] == 1] = 1
    df['output'][df['output'] == 2] = 1
    df['output'][df['output'] == 3] = 1
    
    X = df.drop('output', axis = 1)
    y = pd.to_numeric(df.output)
    y = 2*y-1

    return X, y    
    
    



#Kc2
def kc2():
    dfs = pd.read_csv(r'/Users/mmartinez/Desktop/KnowledgeDistillation/data/PI/kc2.csv')
    dfs['problems'] = (dfs['problems'] == 'yes')*1
    X = dfs.drop('problems', axis = 1)
    y = dfs['problems']
    pi_features= list(X.columns[14:])
    y = 2*y-1
    return X, y, pi_features


#Parkinsons
def parkinsons():
    df = pd.read_csv(r'/Users/mmartinez/Desktop/KnowledgeDistillation/data/PI/parkinsons/parkinsons.data')
    X = df.drop(['name', 'status'], axis = 1)
    y = df['status']
    y = 2*y-1
    return X, y


#--------------------------------------------------------
#1. PHISHING | (1250x9) | Binario

def phishing(from_csv = True):
    if from_csv:
        df = pd.read_csv('data/phishing.csv', index_col = False)
        y = df['0']
        X = df.drop(['Unnamed: 0', '0'], axis = 1)
        return X,y
    else:
        dataset = Phishing()
        X = pd.DataFrame()
        y = []
        for xx, yy in dataset.take(5000):
            X = pd.concat([X, pd.DataFrame([xx])], ignore_index=True)
            y.append(yy)
        y = pd.Series(y)*1
        return X, y



#--------------------------------------------------------
#2. DIABETES | (768x8) | Binario

def diabetes():
    df = pd.read_csv('data/diabetes.csv')
    X = df.drop('Outcome', axis = 1)
    y = df['Outcome']
    return X, y


#--------------------------------------------------------
#12. WHITE MATTER | (1904 x 24)  | Binario 


def wm():
    df = pd.read_csv('data/WM_data.csv', sep = ',')
    
    df.age5[df.age5 == '20-24'] = 1
    df.age5[df.age5 == '25-29'] = 2
    df.age5[df.age5 == '30-34'] = 3
    df.age5[df.age5 == '35-39'] = 4
    df.age5[df.age5 == '40-44'] = 5
    df.age5[df.age5 == '45-49'] = 6
    df.age5[df.age5 == '50-54'] = 7
    df.age5[df.age5 == '55-59'] = 8
    df.age5[df.age5 == '60-64'] = 9
    df.age5[df.age5 == '65-69'] = 10
    df.age5[df.age5 == '70-74'] = 11
    df.age5[df.age5 == '75-79'] = 12
    df.age5[df.age5 == '80-84'] = 13

    df.age5.astype(int)
    
    imp = df.drop('wm', axis = 1)
    index = []
    for i in range(imp.shape[1]):
        if len(imp.iloc[:,i].unique()) < 6:
            if imp.iloc[:,i].isnull().sum() != 0:
                index.append(i)

    imputer = KNNImputer(n_neighbors=5)
    i = imputer.fit_transform(imp)
    imp_correct = pd.DataFrame(i, columns = imp.columns)

    for i in index:
        for j in range(imp_correct.shape[0]):
            imp_correct.iloc[j, i] = round(imp_correct.iloc[j, i])
            
    X = imp_correct
    y = df.wm
    
    return X, y

