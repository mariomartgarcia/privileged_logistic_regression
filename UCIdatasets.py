# %%

import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import  accuracy_score, brier_score_loss
from sklearn.linear_model import LogisticRegression, LinearRegression
import lrplus as lr
import load_UCIdatasets as bs
import random
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn import svm
import tools as tl
import time

warnings.filterwarnings("ignore")


def priv_gain(x, lb, ub):
    pg = ( (np.array(x) - np.array(lb)) / (np.array(ub) - np.array(lb)) )*100
    return pg


# %%

text = ['parkinsons', 'kc2', 'breast_cancer', 'obesity', 'wine']
dc = [ bs.parkinsons(), bs.kc2(), bs.breast_cancer(), bs.obesity(), bs.wine()]


datasets_dict = dict(zip(text, dc))

repetitions = 30  #Number of repetitions for each PI feature
cv = 5
r = random.sample(range(500), repetitions)  #Seed number without replacement

dataLR = pd.DataFrame([])
dataSVM = pd.DataFrame([])


for te  in text:

    if te == 'kc2':
        X, y, pi_features = datasets_dict[te]
        y = pd.Series(y)  
        pi_var = pi_features
    
    if te == 'parkinsons':
        X, y = datasets_dict[te]  

        col = X.columns
        scaler = MinMaxScaler()
        Xnorm = scaler.fit_transform(X)
        Xn = pd.DataFrame(Xnorm, columns = col)

        mi = mutual_info_classif(Xn, y)
        mi_df = pd.DataFrame({'name': list(Xn.columns), 'mi': mi })
        mi_sort = mi_df.sort_values(by='mi', ascending=False)
        pi_var = list(mi_sort['name'][0:10])

    if te in ['breast_cancer', 'obesity', 'wine']:
        X, y = datasets_dict[te]
        X.rename(columns = {'family_history_with_overweight': 'fam. his.'}, inplace = True)
        X.rename(columns = {'fixed acidity': 'fix. acid.', 'volatile acidity': 'vol. acid.', 
                    'citric acid': 'cit. acid.', 'residual sugar': 'res. sug.', 
                    'free sulfur dioxide': 'free sulf diox.', 
                    'total sulfur dioxide': 'tot. sulf diox'}, inplace = True)
        X.rename(columns = {'mean radius': 'avg. rad.', 'mean texture': 'avg. tex.', 'mean perimeter': 'avg. per.', 'mean area': 'avg. ar.',
                    'mean smoothness': 'avg. smo.', 'mean compactness': 'avg. comp.', 'mean concavity': 'avg. conc.',
                    'mean concave points': 'avg. conc. p.', 'mean symmetry': 'avg. sym.', 'mean fractal dimension': 'avg. frac. dim.',
                    'radius error': 'rad. er.', 'texture error': 'tex. er.', 'perimeter error': 'per. er.', 'area error': 'ar. er.',
                    'smoothness error': 'smo. er.', 'compactness error': 'comp. er.', 'concavity error': 'conc. er.',
                    'concave points error': 'conc. p. er.', 'symmetry error': 'sym. er.', 'fractal dimension error': 'frac. dim. er.',
                    'worst radius': 'wor. rad.', 'worst texture': 'wor. tex.', 'worst perimeter': 'wor. per.', 'worst area': 'wor. ar.',
                    'worst smoothness': 'wor. smo.', 'worst compactness': 'wor. comp.', 'worst concavity': 'wor. conc.',
                    'worst concave points': 'wor. conc. p.', 'worst symmetry': 'wor. sym.', 'worst fractal dimension': 'wor. frac. dim.'}, inplace = True)

        #=============================================================
        # UNREAL PRIVILEGED CLASSIFIER. LIST OF PI CANDIDATES
        #=============================================================
        col = X.columns
        scaler = MinMaxScaler()
        Xnorm = scaler.fit_transform(X)
        Xn = pd.DataFrame(Xnorm, columns = col)

        lg = LogisticRegression()    
        lg.fit(Xn, y)
        coef = pd.DataFrame({'names': X.columns, 'coef': lg.coef_[0]})
        values = np.abs(coef.coef).sort_values(ascending = False)
        names = list(coef.names[values.index])
        log_coef = pd.DataFrame({'names': names , 'value': values })
        pi_var =  [names[0]]

            


    #===============================================================
    # N TIME. 5-CV. INCREASING THE NUMBER OF PI FEATURES. MAIN CODE
    #===============================================================
    # DEVELOPMENT




    print(te)




    acc_lb, mae_lb = [], []
    acc_ub, mae_ub = [], []
    acc_realit, mae_realit  = [], []
    acc_realp, mae_realp = [], []
    acc_kl_st = []
    acc_kl_ts = []
    acc_kt = []
    train_lrit, train_lrp, train_kl_st, train_kl_ts  = [], [], [], []
    #svmup, svmplus, svmb = [], [], []

    for k in r:
        
        dr = tl.skfold(X, y, cv, r = k)

        for h in range(cv):
            X_train = dr['X_train' + str(h)]
            y_train = dr['y_train' + str(h)]
            X_test = dr['X_test' + str(h)]
            y_test = dr['y_test' + str(h)]
            
            X_train = X_train.reset_index(drop = True)
            y_train = y_train.reset_index(drop = True)
            
            SS = MinMaxScaler()
            X_train = pd.DataFrame(SS.fit_transform(X_train), columns = X_train.columns)
            X_test = pd.DataFrame(SS.transform(X_test), columns = X_train.columns)
            

            #-------------------------------------------
            #Define the regular space
            X_trainr = X_train.drop(pi_var, axis = 1)
            X_testr = X_test.drop(pi_var, axis = 1)
            

            #-------------------------------------------
            #Unreal Privileged model
            lg = LogisticRegression()    
            lg.fit(X_train, y_train)
            omega = lg.coef_[0]
            beta = lg.intercept_
            pre = lg.predict(X_test)
            proba = lg.predict_proba(X_test)
            
            acc_ub.append(accuracy_score(y_test, pre))
            
            #-------------------------------------------
            #Base model
            lg = LogisticRegression()    
            lg.fit(X_trainr, y_train)
            prep = lg.predict(X_testr)
    
            acc_lb.append(accuracy_score(y_test, prep))

            #-------------------------------------------
            #LRIT+ model
            al = lr.LRIT_plus()

            start_time = time.time()
            al.fit(X_train, X_trainr,  omega, beta)
            end_time = time.time()
            train_lrit.append(end_time - start_time)

            test = al.predict(X_testr)

            
            acc_realit.append(accuracy_score(y_test, test))
            
            #-------------------------------------------
            #LR+ model
            alp = lr.LR_plus()

            start_time = time.time()
            alp.fit(X_train, X_trainr,  omega, beta)
            end_time = time.time()
            train_lrp.append(end_time - start_time)

            testp = alp.predict(X_testr)
            
            acc_realp.append(accuracy_score(y_test, testp))


            #-------------------------------------------
            #LR_KL model ST
            kl_st = lr.LR_plusKL(loss = 'st')
            start_time = time.time()
            kl_st.fit(X_train, X_trainr,  omega, beta)
            end_time = time.time()
            train_kl_st.append(end_time - start_time)

            test_st = kl_st.predict(X_testr)
            
            acc_kl_st.append(accuracy_score(y_test, test_st))


            #-------------------------------------------
            #LR_KL model ST

            kl_ts = lr.LR_plusKL(loss = 'ts')
            start_time = time.time()
            kl_ts.fit(X_train, X_trainr,  omega, beta)
            end_time = time.time()
            train_kl_ts.append(end_time - start_time)



            test_ts = kl_ts.predict(X_testr)
            
            acc_kl_ts.append(accuracy_score(y_test, test_ts))

            #-------------------------------------
            #KNOWLEDGE TRANSFER
            lin = {}
            for i in pi_var:
                lin[i] = LinearRegression()
                lin[i].fit(X_trainr, X_train[i])

            kt_train = pd.DataFrame([])
            kt_test = pd.DataFrame([])
            for i in pi_var:
                privil = lin[i].predict(X_trainr)
                kt_train[i] = privil
                privil_t = lin[i].predict(X_testr)
                kt_test[i] = privil_t
            X_trainr = pd.concat([X_trainr, kt_train], axis = 1)
            X_testr = pd.concat([X_testr, kt_test], axis = 1)

            kt = LogisticRegression()
            kt.fit(X_trainr, y_train)
            pre_kt = kt.predict(X_testr)
            acc_kt.append(accuracy_score(y_test, pre_kt))


            
    gan_it = priv_gain(np.mean(acc_realit), np.mean(acc_lb), np.mean(acc_ub))
    gan_p = priv_gain(np.mean(acc_realp), np.mean(acc_lb), np.mean(acc_ub))
    gan_kl_st = priv_gain(np.mean(acc_kl_st), np.mean(acc_lb), np.mean(acc_ub))
    gan_kl_ts = priv_gain(np.mean(acc_kl_ts), np.mean(acc_lb), np.mean(acc_ub))
    gan_kt = priv_gain(np.mean(acc_kt), np.mean(acc_lb), np.mean(acc_ub))


    data_lr = pd.DataFrame({#'nPI': range(1, number_pi),
                        'dataset': te,
                        'ACClb':       np.round(np.mean(acc_lb), 3),
                        'ACCreal_it':  np.round(np.mean(acc_realit), 3),
                        'ACCreal_p':   np.round(np.mean(acc_realp), 3),
                        'ACC_kl_st':  np.round(np.mean(acc_kl_st), 3),
                        'ACC_kl_ts':   np.round(np.mean(acc_kl_ts), 3),
                        'ACC_kt':      np.round(np.mean(acc_kt), 3),
                        'ACCub':       np.round(np.mean(acc_ub), 3),
                        'stdlb':       np.round(np.std(acc_lb), 3),
                        'stdreal_it':  np.round(np.std(acc_realit), 3),
                        'stdreal_p':   np.round(np.std(acc_realp), 3),
                        'std_kl_st':   np.round(np.std(acc_kl_st), 3),
                        'std_kl_ts':   np.round(np.std(acc_kl_ts), 3),
                        'std_KT':      np.round(np.std(acc_kt), 3),
                        'stdub':       np.round(np.std(acc_ub), 3),                    
                        'gain_it':      np.round(gan_it, 3),
                        'gain_p':       np.round(gan_p, 3),
                        'gain_kl_st':      np.round(gan_kl_st, 3),
                        'gain_kl_ts':       np.round(gan_kl_ts, 3),
                        'gain_kt':       np.round(gan_kt, 3),
                        'time_lrit':      np.round(np.mean(train_lrit),3),
                        'time_lrp':    np.round(np.mean(train_lrp), 3),
                        'time_kl_st':    np.round(np.mean(train_kl_st),3),
                        'time_kl_ts':    np.round(np.mean(train_kl_ts), 3)
                        }, index = [0])

    dataLR = pd.concat([dataLR, data_lr]).reset_index(drop = True)



dataLR.to_csv('dataLR_kl.csv')


