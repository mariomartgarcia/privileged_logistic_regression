
import pandas as pd
from sklearn.model_selection import StratifiedKFold



#=========================================================================================================
#---------------------------------------------------------------------------------------------------------
#=========================================================================================================
#Stratified folds


def skfold(X, y, n, r = 0):
    skf = StratifiedKFold(n_splits = n, shuffle = True, random_state = r)
    d= {}
    j = 0
    for train_index, test_index in skf.split(X, y): 
            d['X_train' + str(j)] = X.loc[train_index]
            d['X_test' + str(j)] = X.loc[test_index]
            d['y_train' + str(j)] = y.loc[train_index]
            d['y_test' + str(j)] = y.loc[test_index]

            d['X_train' + str(j)].reset_index(drop = True, inplace = True)
            d['X_test' + str(j)].reset_index(drop = True, inplace = True)
            d['y_train' + str(j)].reset_index(drop = True, inplace = True)
            d['y_test' + str(j)].reset_index(drop = True, inplace = True)
            j+=1
    return d