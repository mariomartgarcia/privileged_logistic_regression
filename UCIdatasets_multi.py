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
from privileged_lr import PrivilegedLogisticRegression

warnings.filterwarnings("ignore")


def priv_gain(x, lb, ub):
    pg = ( (np.array(x) - np.array(lb)) / (np.array(ub) - np.array(lb)) )*100
    return pg


# %%

text = ['kc2', 'parkinsons', 'obesity', 'wine', 'breast_cancer', 'phishing', 'diabetes', 'wm', 'drugs', 'spam', 'heart', 'heart2', 'abalone', 'car']
dc = [bs.kc2(), bs.parkinsons(), bs.obesity(), bs.wine(), bs.breast_cancer(), bs.phishing(), bs.diabetes(), bs.wm(), bs.drugs(), bs.spam(), bs.heart(), bs.heart2(), bs.abalone(), bs.car()]




datasets_dict = dict(zip(text, dc))

repetitions = 30  #Number of repetitions for each PI feature
cv = 5
r = random.sample(range(500), repetitions)  #Seed number without replacement

dataLR = pd.DataFrame([])
dataSVM = pd.DataFrame([])
dataOTHER = pd.DataFrame([])


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

    if te in ['breast_cancer', 'obesity', 'wine', 'phishing', 'diabetes', 'wm', 'drugs', 'spam', 'heart', 'heart2', 'abalone', 'car']:
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



    print('XXXXXXX')
    print(te)

    for p in [1]:
        print('----------')
        print(p)
        print('---------')

        Tacc_lb, Tacc_ub, Tacc_oub, Tacc_realit, Tacc_realp = [ [] for i in range(5)]
        Tsvmup, Tsvmplus, Tsvmb = [], [], []
        Tplr, Tktsvme, Tgd_e, Tpfd_e = [[] for i in range(4)]


        for k in r:
            
            dr = tl.skfold(X, y, cv, r = k)

            acc_lb, acc_ub, acc_oub, acc_realit, acc_realp = [ [] for i in range(5)]
            svmup, svmplus, svmb = [], [], []
            plr_m, ktsvme, gd_e, pfd_e = [[] for i in range(4)]

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
                
                if p<1:
                    X_train = X_train.sample(frac = p, random_state = 2)
                    y_train = y_train[X_train.index]

                    X_train = X_train.reset_index(drop = True)
                    y_train = y_train.reset_index(drop = True)

                #-------------------------------------------
                #Define the regular space
                X_trainr = X_train.drop(pi_var, axis = 1)
                X_testr = X_test.drop(pi_var, axis = 1)
                X_trainp = X_train[pi_var]
                X_testp = X_test[pi_var]

                #-------------------------------------------
                #Only Privileged model
                lg = LogisticRegression()    
                lg.fit(X_trainp, y_train)
                omega_o = lg.coef_[0]
                beta_o = lg.intercept_
                pre = lg.predict(X_testp)
                proba = lg.predict_proba(X_testp)
                
                acc_oub.append(accuracy_score(y_test, pre))

                #------------------------------------------

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
                al.fit(X_train, X_trainr,  omega, beta)
                test = al.predict(X_testr)
                
                acc_realit.append(accuracy_score(y_test, test))
                
                #-------------------------------------------
                #LR+ model
                alp = lr.LR_plus()
                alp.fit(X_train, X_trainr,  omega, beta)
                testp = alp.predict(X_testr)
                
                acc_realp.append(accuracy_score(y_test, testp))



                #-------------------------------------------
                #-------------------------------------------
                #-------------------------------------------
                #SVMUP
                sv = svm.SVC(kernel = 'linear')
                sv.fit(X_train, y_train)
                test_sup = sv.predict(X_test)
                
                svmup.append(accuracy_score(y_test, test_sup))
                

                #SVM+ model
                svmp = tl.svmplus_CVX()
                
		

                svmp.fit(X_trainr, X_trainp, y_train)
                tests= svmp.predict(X_testr)

                svmplus.append(accuracy_score(y_test, tests))
                #print('prediction', tests)

                print(accuracy_score(y_test, tests))
                
                #SVMB
                svb = svm.SVC(kernel = 'linear' )
                svb.fit(X_trainr, y_train)
                test_sb = svb.predict(X_testr)
                
                svmb.append(accuracy_score(y_test, test_sb))

                #-------------------------------------------
                #-------------------------------------------
                #-------------------------------------------


                #PLR
                plr = PrivilegedLogisticRegression()
                plr.fit(X_trainr, y_train, X_star=X_trainp, y_star=y_train)
                pre = plr.predict(X_testr)

                plr_m.append(accuracy_score(y_test, pre))

                #KTSVM
                ktsvm = tl.KT_svm()
                ktsvm.fit(X_train, y_train, pi_var)
                pre = ktsvm.predict(X_testr)

                ktsvme.append(accuracy_score(y_test, pre))

                y_train01 = (y_train + 1)/2
                y_test01 = (y_test + 1)/2

                #GD
                gd = tl.GD()
                gd.fit(X_trainp, X_trainr, y_train01, omega_o, beta_o)
                pre = gd.predict(X_testr)

                gd_e.append(accuracy_score(y_test01, pre))


                #PFD
                pfd = tl.GD()
                pfd.fit(X_train, X_trainr, y_train01, omega, beta)
                pre = pfd.predict(X_testr)

                pfd_e.append(accuracy_score(y_test01, pre))






            Tacc_lb.append(np.mean(acc_lb))
            Tacc_realit.append(np.mean(acc_realit))
            Tacc_realp.append(np.mean(acc_realp))
            Tacc_ub.append(np.mean(acc_ub))
            Tacc_oub.append(np.mean(acc_oub))
            Tsvmb.append(np.mean(svmb))
            Tsvmplus.append(np.mean(svmplus))
            Tsvmup.append(np.mean(svmup))

            Tplr.append(np.mean(plr_m))
            Tktsvme.append(np.mean(ktsvme))
            Tgd_e.append(np.mean(gd_e))
            Tpfd_e.append(np.mean(pfd_e))

                
        gan_it = priv_gain(np.mean(Tacc_realit), np.mean(Tacc_lb), np.mean(Tacc_ub))
        gan_p = priv_gain(np.mean(Tacc_realp), np.mean(Tacc_lb), np.mean(Tacc_ub))
        gan_svm = priv_gain(np.mean(Tsvmplus), np.mean(Tsvmb), np.mean(Tsvmup))

        data_lr = pd.DataFrame({#'nPI': range(1, number_pi),
                            'dataset': te,
                            'per_train': p,
                            'ACClb':       np.round(np.mean(Tacc_lb), 3),
                            'ACCreal_it':  np.round(np.mean(Tacc_realit), 3),
                            'ACCreal_p':   np.round(np.mean(Tacc_realp), 3),
                            'ACCub':       np.round(np.mean(Tacc_ub), 3),
                            'ACCoub':       np.round(np.mean(Tacc_oub), 3),
                            'stdlb':       np.round(np.std(Tacc_lb), 3),
                            'stdreal_it':  np.round(np.std(Tacc_realit), 3),
                            'stdreal_p':   np.round(np.std(Tacc_realp), 3),
                            'stdub':       np.round(np.std(Tacc_ub), 3),          
                            'stdoub':       np.round(np.std(Tacc_oub), 3),                              
                            'gain_it':      np.round(gan_it, 3),
                            'gain_p':       np.round(gan_p, 3),

                            }, index = [0])

        data_svm = pd.DataFrame({#'nPI': range(1, number_pi),
                            'dataset': te,
                            'per_train': p,
                            'ACCsvmb':    np.round(np.mean(Tsvmb), 3),
                            'ACCsvmplus': np.round(np.mean(Tsvmplus), 3),
                            'ACCsvmup':   np.round(np.mean(Tsvmup), 3),
                            'stdsvmb':    np.round(np.std(Tsvmb), 3),
                            'stdsvmplus': np.round(np.std(Tsvmplus), 3),
                            'stdsvmup':   np.round(np.std(Tsvmup), 3),                      
                            'gain_svm':   np.round(gan_svm, 3),

                            }, index = [0])
        
        data_other = pd.DataFrame({#'nPI': range(1, number_pi),
                            'dataset': te,
                            'ACCplr':    np.round(np.mean(Tplr), 3),
                            'std_plr':    np.round(np.std(Tplr), 3),  
                            'ACCktsvm':    np.round(np.mean(Tktsvme), 3),
                            'std_ktsvm':    np.round(np.std(Tktsvme), 3), 
                             'ACCgd':    np.round(np.mean(Tgd_e), 3),
                            'std_gd':    np.round(np.std(Tgd_e), 3), 
                             'ACCpfd':    np.round(np.mean(Tpfd_e), 3),
                            'std_pfd':    np.round(np.std(Tpfd_e), 3),                    
                            }, index = [0])

        dataLR = pd.concat([dataLR, data_lr]).reset_index(drop = True)
        dataSVM = pd.concat([dataSVM, data_svm]).reset_index(drop = True)
        dataOTHER = pd.concat([dataOTHER, data_other]).reset_index(drop = True)



dataLR.to_csv('dataLR.csv')
dataSVM.to_csv('dataSVM.csv')
dataOTHER.to_csv('dataOTHER.csv')


# %%
