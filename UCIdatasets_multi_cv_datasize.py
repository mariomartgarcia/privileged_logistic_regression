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
import itertools

warnings.filterwarnings("ignore")


def priv_gain(x, lb, ub):
    pg = ( (np.array(x) - np.array(lb)) / (np.array(ub) - np.array(lb)) )*100
    return pg


# %%

def regularization_LR(X, y, pi_features, param):
 
    cv = 5
    dr = tl.skfold(X, y, cv, r = 0)
    Tacc_ulr, Tacc_blr, Tacc_svm_up, Tacc_svm_b, C_value  = [], [], [], [], []
    for i  in param:
            acc_ulr, acc_blr, svm_up, svm_b  = [], [], [], []
            for h in range(cv):
                X_trainv = dr['X_train' + str(h)]
                y_trainv = dr['y_train' + str(h)]
                X_val= dr['X_test' + str(h)]
                y_val = dr['y_test' + str(h)]
                
                X_trainv = X_trainv.reset_index(drop = True)
                y_trainv = y_trainv.reset_index(drop = True)

                X_trainv_reg = X_trainv.drop(pi_features, axis = 1)
                X_val_reg = X_val.drop(pi_features, axis = 1)
                
                #Upper
                lr = LogisticRegression(C = i)
                lr.fit(X_trainv, y_trainv)
                pre_upper = lr.predict(X_val)
                acc_ulr.append(accuracy_score(y_val, pre_upper))

                
                X_trainv_reg = X_trainv.drop(pi_features, axis = 1)
                X_val_reg = X_val.drop(pi_features, axis = 1)
                
                #-------------------------------------------
                #Base
                lr = LogisticRegression(C = i)
                lr.fit(X_trainv_reg, y_trainv)
                pre_base = lr.predict(X_val_reg)
                acc_blr.append(accuracy_score(y_val, pre_base))


                #SVMUP
                sv = svm.SVC(kernel = 'linear', C  = i)
                sv.fit(X_trainv, y_trainv)
                test_sup = sv.predict(X_val)
                
                svm_up.append(accuracy_score(y_val, test_sup))
                
    
                #SVMB
                svb = svm.SVC(kernel = 'linear', C = i)
                svb.fit(X_trainv_reg, y_trainv)
                test_sb = svb.predict(X_val_reg)
                svm_b.append(accuracy_score(y_val, test_sb))
                
            C_value.append(i)

            Tacc_ulr.append(np.mean(acc_ulr)) 
            Tacc_blr.append(np.mean(acc_blr))
            Tacc_svm_up.append(np.mean(svm_up))
            Tacc_svm_b.append(np.mean(svm_b))

        #-------------------------------------------
    cv_upper_lr   = pd.DataFrame({'C': C_value, 'ACC': Tacc_ulr})
    cv_base_lr  = pd.DataFrame({'C': C_value, 'ACC': Tacc_blr})
    cv_upper_svm   = pd.DataFrame({'C': C_value, 'ACC': Tacc_svm_up})
    cv_base_svm  = pd.DataFrame({'C': C_value, 'ACC': Tacc_svm_b})


     
    C_uplr= cv_upper_lr.sort_values('ACC', ascending = False).reset_index(drop = True)['C'][0]  
    C_blr = cv_base_lr.sort_values('ACC', ascending = False).reset_index(drop = True)['C'][0]  
    C_upsvm= cv_upper_svm.sort_values('ACC', ascending = False).reset_index(drop = True)['C'][0]  
    C_bsvm = cv_base_svm.sort_values('ACC', ascending = False).reset_index(drop = True)['C'][0]  

    return C_uplr, C_blr, C_upsvm, C_bsvm
     

def regularization_lrplus(X, y, pi_features, C_upper, param):
 
    cv = 5
    dr = tl.skfold(X, y, cv, r = 0)
    Taccrealp, Taccrealit, C_value = [], [], []
    for i  in param:
            accrealp, accrealit = [], []
            for h in range(cv):
                X_trainv = dr['X_train' + str(h)]
                y_trainv = dr['y_train' + str(h)]
                X_val= dr['X_test' + str(h)]
                y_val = dr['y_test' + str(h)]
                
                X_trainv = X_trainv.reset_index(drop = True)
                y_trainv = y_trainv.reset_index(drop = True)

                X_trainv_reg = X_trainv.drop(pi_features, axis = 1)
                X_val_reg = X_val.drop(pi_features, axis = 1)
                
                #Upper
                lg = LogisticRegression(C = C_upper)
                lg.fit(X_trainv, y_trainv)
                pre_upper = lg.predict(X_val)
                omega = lg.coef_[0]
                beta = lg.intercept_

                
                X_trainv_reg = X_trainv.drop(pi_features, axis = 1)
                X_val_reg = X_val.drop(pi_features, axis = 1)
                
                #-------------------------------------------
                #LRIT+ model
                al = lr.LRIT_plus(l = i)
                al.fit(X_trainv, X_trainv_reg,  omega, beta)
                test = al.predict(X_val_reg)
                accrealit.append(accuracy_score(y_val, test))
                
                
                #LR+ model
                alp = lr.LR_plus(l = i)
                alp.fit(X_trainv, X_trainv_reg,  omega, beta)
                testp = alp.predict(X_val_reg)
                accrealp.append(accuracy_score(y_val, testp))

                
            C_value.append(i)

            Taccrealit.append(np.mean(accrealit)) 
            Taccrealp.append(np.mean(accrealp))


        #-------------------------------------------
    cv_it   = pd.DataFrame({'C': C_value, 'ACC': Taccrealit})
    cv_p = pd.DataFrame({'C': C_value, 'ACC': Taccrealp})



     
    C_it= cv_it.sort_values('ACC', ascending = False).reset_index(drop = True)['C'][0]  
    C_p = cv_p.sort_values('ACC', ascending = False).reset_index(drop = True)['C'][0]  
 
    return C_it, C_p
     

def regularization_svmplus(X, y, pi_features, param_grid):
 
    cv = 5
    dr = tl.skfold(X, y, cv, r = 0)
    all_hyperparam_combinations = list(itertools.product(*map(param_grid.get, list(param_grid))))

    Tacc_svm = []
    for h in range(cv):
        X_trainv = dr['X_train' + str(h)]
        y_trainv = dr['y_train' + str(h)]
        X_val= dr['X_test' + str(h)]
        y_val = dr['y_test' + str(h)]
        
        X_trainv = X_trainv.reset_index(drop = True)
        y_trainv = y_trainv.reset_index(drop = True)
        
        accsvmp = []

        X_trainv_reg = X_trainv.drop(pi_features, axis = 1)
        X_val_reg = X_val.drop(pi_features, axis = 1)
        X_trainv_p = X_trainv[pi_features]

        for i, hyper_param_values in enumerate(all_hyperparam_combinations):
            kwarg = dict(zip(list(param_grid.keys()), hyper_param_values))
            svmp = tl.svmplus_CVX(**kwarg)

            svmp.fit(X_trainv_reg, X_trainv_p, y_trainv)
            tests= svmp.predict(X_val_reg)
            accsvmp.append(accuracy_score(y_val, tests))

        Tacc_svm.append(accsvmp)

    res = np.mean(Tacc_svm, axis = 0)


    values = {'C': np.array(all_hyperparam_combinations)[:,0],
              'gamma': np.array(all_hyperparam_combinations)[:,1],
              'ACC': res}

    cv_svm = pd.DataFrame(values)


     
    C = cv_svm.sort_values('ACC', ascending = False).reset_index(drop = True)['C'][0]  
    gamma = cv_svm.sort_values('ACC', ascending = False).reset_index(drop = True)['gamma'][0]  
 
    return C, gamma


def regularization_plr(X, y, pi_features, param_grid):
 
    cv = 5
    dr = tl.skfold(X, y, cv, r = 0)
    all_hyperparam_combinations = list(itertools.product(*map(param_grid.get, list(param_grid))))

    Tplrm = []
    for h in range(cv):
        X_trainv = dr['X_train' + str(h)]
        y_trainv = dr['y_train' + str(h)]
        X_val= dr['X_test' + str(h)]
        y_val = dr['y_test' + str(h)]
        
        X_trainv = X_trainv.reset_index(drop = True)
        y_trainv = y_trainv.reset_index(drop = True)
        
        plrm = []

        X_trainv_reg = X_trainv.drop(pi_features, axis = 1)
        X_val_reg = X_val.drop(pi_features, axis = 1)
        X_trainv_p = X_trainv[pi_features]

        for i, hyper_param_values in enumerate(all_hyperparam_combinations):
            kwarg = dict(zip(list(param_grid.keys()), hyper_param_values))
            plr = PrivilegedLogisticRegression(**kwarg)
            plr.fit(X_trainv_reg, y_trainv, X_star=X_trainv_p, y_star=y_trainv)
            pre = plr.predict(X_val_reg)

            plrm.append(accuracy_score(y_val, pre))

        Tplrm.append(plrm)

    res = np.mean(Tplrm, axis = 0)


    values = {'lambda_base': np.array(all_hyperparam_combinations)[:,0],
              'lambda_star': np.array(all_hyperparam_combinations)[:,1],
              'alpha': np.array(all_hyperparam_combinations)[:,2],
              'xi_link': np.array(all_hyperparam_combinations)[:,3],
              'penalty':  np.array(all_hyperparam_combinations)[:,4],
              'ACC': res}

    cv_svm = pd.DataFrame(values)


     
    lambda_base = cv_svm.sort_values('ACC', ascending = False).reset_index(drop = True)['lambda_base'][0]  
    lambda_star = cv_svm.sort_values('ACC', ascending = False).reset_index(drop = True)['lambda_star'][0]
    alpha = cv_svm.sort_values('ACC', ascending = False).reset_index(drop = True)['alpha'][0]  
    xi_link = cv_svm.sort_values('ACC', ascending = False).reset_index(drop = True)['xi_link'][0]  
    penalty = cv_svm.sort_values('ACC', ascending = False).reset_index(drop = True)['penalty'][0]  

    return lambda_base, lambda_star, alpha, xi_link, penalty

# %%

text = ['kc2', 'parkinsons']
dc = [bs.kc2(), bs.parkinsons()]


datasets_dict = dict(zip(text, dc))

repetitions = 10  #Number of repetitions for each PI feature
cv = 5
r = random.sample(range(500), repetitions)  #Seed number without replacement

dataLR = pd.DataFrame([])
dataSVM = pd.DataFrame([])
dataOTHER = pd.DataFrame([])

param_grid_plr = {
    'lambda_base': [0.01, 0.1, 1, 10],
    'lambda_star': [0.01, 0.1, 1, 10],
    'alpha': [0.01, 0.1, 1, 10],
    'xi_link': [0.01, 0.1, 1, 10],
    'penalty' : ['l1']}


param_C = [0.01, 0.1, 1, 10]

param_grid_svmp = {
    'C': [0.01, 0.1, 1, 10],
    'gamma': [0.01, 0.1, 1, 10],
    }


'''
param_grid_plr = {
    'lambda_base': [0.01, 0.1],
    'lambda_star': [0.01, 1],
    'alpha': [0.01, 1],
    'xi_link': [0.01, 1],
    'penalty' : ['l2']}


param_C = [0.01, 1]

param_grid_svmp = {
    'C': [0.01, 1],
    'gamma': [0.01, 1],
    }

'''

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

    for p in [0.25, 0.5, 0.75, 1]:
        print('----------')
        print(p)
        print('---------')

        acc_lb, acc_ub, acc_oub, acc_realit, acc_realp = [ [] for i in range(5)]
        svmup, svmplus, svmb = [], [], []
        plr_m, ktsvme, gd_e, pfd_e = [[] for i in range(4)]



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
                

                C_uplr, C_blr, C_upsvm, C_bsvm = regularization_LR(X_train, y_train, pi_var, param_C)
                C_it, C_p = regularization_lrplus(X_train, y_train, pi_var, C_uplr, param_C)
                C_s, gamma_s =  regularization_svmplus(X_train, y_train, pi_var,  param_grid_svmp)
                lamb_base, lamb_star, alph, xi_lk, pen = regularization_plr(X_train, y_train, pi_var,  param_grid_plr)
                
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
            
                #------------------------------------------

                #Unreal Privileged model
                lg = LogisticRegression(C = C_uplr)    
                lg.fit(X_train, y_train)
                omega = lg.coef_[0]
                beta = lg.intercept_
                pre = lg.predict(X_test)
                proba = lg.predict_proba(X_test)
                
                acc_ub.append(accuracy_score(y_test, pre))
                
                #-------------------------------------------
                #Base model
                lg = LogisticRegression(C = C_blr)    
                lg.fit(X_trainr, y_train)
                prep = lg.predict(X_testr)
        
                acc_lb.append(accuracy_score(y_test, prep))

                #-------------------------------------------
                #LRIT+ model
                al = lr.LRIT_plus(l = C_it)
                al.fit(X_train, X_trainr,  omega, beta)
                test = al.predict(X_testr)
                
                acc_realit.append(accuracy_score(y_test, test))
                
                #-------------------------------------------
                #LR+ model
                alp = lr.LR_plus(l = C_p)
                alp.fit(X_train, X_trainr,  omega, beta)
                testp = alp.predict(X_testr)
                
                acc_realp.append(accuracy_score(y_test, testp))



                #-------------------------------------------
                #-------------------------------------------
                #-------------------------------------------
                #SVMUP
                sv = svm.SVC(kernel = 'linear', C = C_upsvm)
                sv.fit(X_train, y_train)
                test_sup = sv.predict(X_test)
                
                svmup.append(accuracy_score(y_test, test_sup))
                

                #SVM+ model
                svmp = tl.svmplus_CVX(C = C_s, gamma = gamma_s)
                
                svmp.fit(X_trainr, X_trainp, y_train)
                tests= svmp.predict(X_testr)

                svmplus.append(accuracy_score(y_test, tests))
                #print('prediction', tests)

                print(accuracy_score(y_test, tests))
                
                #SVMB
                svb = svm.SVC(kernel = 'linear', C = C_bsvm )
                svb.fit(X_trainr, y_train)
                test_sb = svb.predict(X_testr)
                
                svmb.append(accuracy_score(y_test, test_sb))

                #-------------------------------------------
                #-------------------------------------------
                #-------------------------------------------


                #PLR
                plr = PrivilegedLogisticRegression(penalty=pen , lambda_base=lamb_base , lambda_star=lamb_star, xi_link=xi_lk, alpha=alph)
                plr.fit(X_trainr, y_train, X_star=X_trainp, y_star=y_train)
                pre = plr.predict(X_testr)

                plr_m.append(accuracy_score(y_test, pre))


                
        gan_it = priv_gain(np.mean(acc_realit), np.mean(acc_lb), np.mean(acc_ub))
        gan_p = priv_gain(np.mean(acc_realp), np.mean(acc_lb), np.mean(acc_ub))
        gan_svm = priv_gain(np.mean(svmplus), np.mean(svmb), np.mean(svmup))
        gan_plr= priv_gain(np.mean(plr_m), np.mean(acc_lb), np.mean(acc_ub))


        data_lr = pd.DataFrame({#'nPI': range(1, number_pi),
                            'dataset': te,
                            'per_train': p,
                            'ACClb':       np.round(np.mean(acc_lb), 3),
                            'ACCreal_it':  np.round(np.mean(acc_realit), 3),
                            'ACCreal_p':   np.round(np.mean(acc_realp), 3),
                            'ACCplr':    np.round(np.mean(plr_m), 3), 
                            'ACCub':       np.round(np.mean(acc_ub), 3),
                            'stdlb':       np.round(np.std(acc_lb), 3),
                            'stdreal_it':  np.round(np.std(acc_realit), 3),
                            'stdreal_p':   np.round(np.std(acc_realp), 3),
                            'std_plr':    np.round(np.std(plr_m), 3), 
                            'stdub':       np.round(np.std(acc_ub), 3),                                       
                            'gain_it':      np.round(gan_it, 3),
                            'gain_p':       np.round(gan_p, 3),
                            'gain_plr':       np.round(gan_plr, 3),

                            }, index = [0])

        data_svm = pd.DataFrame({#'nPI': range(1, number_pi),
                            'dataset': te,
                            'per_train': p,
                            'ACCsvmb':    np.round(np.mean(svmb), 3),
                            'ACCsvmplus': np.round(np.mean(svmplus), 3),
                            'ACCsvmup':   np.round(np.mean(svmup), 3),
                            'stdsvmb':    np.round(np.std(svmb), 3),
                            'stdsvmplus': np.round(np.std(svmplus), 3),
                            'stdsvmup':   np.round(np.std(svmup), 3),                      
                            'gain_svm':   np.round(gan_svm, 3),

                            }, index = [0])
        

        dataLR = pd.concat([dataLR, data_lr]).reset_index(drop = True)
        dataSVM = pd.concat([dataSVM, data_svm]).reset_index(drop = True)



dataLR.to_csv('dataLRdatasize.csv')
dataSVM.to_csv('dataSVMdatasize.csv')


# %%
