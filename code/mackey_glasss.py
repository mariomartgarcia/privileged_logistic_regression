import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  accuracy_score
from sklearn.linear_model import LogisticRegression
import lrplus as lr
import tools as tl
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler


def priv_gain(x, lb, ub):
    pg = ( (np.array(x) - np.array(lb)) / (np.array(ub) - np.array(lb)) )*100
    return pg
# %%
def mackey_glass(n, T, tau = 17):
    
    #======== TIME SERIES MG ==========================================================
    def time_serie_mackey_glass(T, a = 0.1, b = 0.2, tau = tau, initial = 0.9, N = 1000):
        x = list(np.ones(tau +1)*initial) 
        dim = N + T + 6
        for i in range(tau, dim + tau):
            #print(i-tau)
            x.append(x[i] - a*x[i] + b*x[i-tau]/(1+x[i-tau]**10))
        return x
    #==================================================================================
    
    
    #   Regular features    |      Privileged features    |    Output
        # xt3 = x(t-3)      |        xp1 = x(t+T-2)       |   xT = x(t+T)
        # xt2 = x(t-2)      |        xp2 = x(t+T-1) 
        # xt1 = x(t-1)      |        xp3 = x(t+T+1) 
        # xt = x(t)         |        xp4 = x(t+T+2) 
    
    x = time_serie_mackey_glass(T, N = n)
    dim = n + T + 6
    xt3, xt2, xt1, xt = [], [], [], []
    xp1, xp2, xp3, xp4 = [], [], [], []
    xT = []
    
 
    for t in range(tau+4, dim+tau-(T+2)):
        
        #Regular features
        xt3.append( x[t-3] )
        xt2.append( x[t-2] )
        xt1.append( x[t-1] )
        xt.append( x[t] )
        
        #Privileged Features
        xp1.append( x[t+T-2] )
        xp2.append( x[t+T-1] )
        xp3.append( x[t+T+1] )
        xp4.append( x[t+T+2] )
        
        #Output
        xT.append( x[t+T] )
    
    output = [1 if xT[i]>xt[i] else -1  for i in range(len(xT)) ]
    
    mg = pd.DataFrame({'xt3': xt3,
                       'xt2': xt2,
                       'xt1': xt1,
                       'xt': xt,
                       'xp1': xp1,
                       'xp2': xp2,
                       'xp3': xp3,
                       'xp4': xp4,
                       'output': output})
    X = mg.drop('output', axis = 1)
    y = mg.output
    return X, y, x


# %%
#=====================================================
# N TIME. 5-CV. For different T and N
#=====================================================
# DEVELOPMENT

#Change the number of T.
#3, 5, 8

for j in [3, 5, 8]:
    T = j
    repetitions = 2  #Number of repetitions for each PI feature
    r = random.sample(range(500), repetitions)  #Seed number without replacement
    
    #Results boxes for every seed
    TACClb, TACCub, TACCreal_it, TACCreal_p = [], [], [], []
    Tstdlb, Tstdub, Tstdreal_it, Tstdreal_p  = [], [], [], []
    TACCsvmup, TACCsvmplus, TACCsvmplusq, TACCsvmb = [], [], [], []
    Tstdsvmup, Tstdsvmplus, Tstdsvmplusq, Tstdsvmb = [], [], [], []
    
    for i in [500, 1000, 1500]:
        
        mg = mackey_glass(i, T)
        X = mg[0]
        y = mg[1]
        
        
        #Results boxes for every seed
        ACClb, ACCub, ACCreal_it, ACCreal_p = [], [], [], []
        ACCsvmup, ACCsvmplus, ACCsvmplusq, ACCsvmb = [], [], [], []
        print(i)
        for k in r:
            cv = 5
            dr = tl.skfold(X, y, cv, r = k)
        
            acc_lb, mae_lb = [], []
            acc_ub, mae_ub = [], []
            acc_realit, acc_realp  = [], []
            svmup, svmplus, svmb = [], [], []
            
            
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
            
                #----------------------
                pi_var =  ['xp1', 'xp2', 'xp3', 'xp4']
                #----------------------
                
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
                X_trainp = X_train[pi_var]
                svmp.fit(X_trainr, X_trainp, y_train)
                tests= svmp.predict(X_testr)
                svmplus.append(accuracy_score(y_test, tests))
                
                
                #SVMB
                svb = svm.SVC(kernel = 'linear')
                svb.fit(X_trainr, y_train)
                test_sb = svb.predict(X_testr)
                
                svmb.append(accuracy_score(y_test, test_sb))
       
                
                
            
            #Computation od the mean for the 5 folds
            
            #Base and Unreal Privileged model
            ACClb.append(np.mean(acc_lb))
            ACCub.append(np.mean(acc_ub))
            
    
            #LRIT+
            ACCreal_it.append(np.mean(acc_realit))
    
            #LR+
            ACCreal_p.append(np.mean(acc_realp))
            
            #SVM
            ACCsvmup.append(np.mean(svmup))
            ACCsvmplus.append(np.mean(svmplus))
            ACCsvmb.append(np.mean(svmb))
    
         
        #Base and Unreal Privileged model
        TACClb.append(np.mean(ACClb))
        Tstdlb.append(np.std(ACClb))
        TACCub.append(np.mean(ACCub))
        Tstdub.append(np.std(ACCub))
        
    
    
        #LRIT+
        TACCreal_it.append(np.mean(ACCreal_it))
        Tstdreal_it.append(np.std(ACCreal_it))
        
        #LR+
        TACCreal_p.append(np.mean(ACCreal_p))
        Tstdreal_p.append(np.std(ACCreal_p))
        
        
        #SVM
        TACCsvmup.append(np.mean(ACCsvmup))
        TACCsvmplus.append(np.mean(ACCsvmplus))
        TACCsvmb.append(np.mean(ACCsvmb))
        
        Tstdsvmup.append(np.std(ACCsvmup))
        Tstdsvmplus.append(np.std(ACCsvmplus))
        Tstdsvmb.append(np.std(ACCsvmb))


    
    gan_it = priv_gain(TACCreal_it, TACClb, TACCub)
    gan_p = priv_gain(TACCreal_p, TACClb, TACCub)
    gan_svm = priv_gain(TACCsvmplus, TACCsvmb, TACCsvmup)
    
    
    
    data_lr = pd.DataFrame({'size': [500, 1000, 1500],
                           'TACCub': TACCub,
                           'TACClb': TACClb,
                           'TACCreal_it':TACCreal_it,
                           'TACCreal_p': TACCreal_p,
                           'Tstdlb': Tstdlb,
                           'Tstdub': Tstdub,
                           'Tstdreal_p':Tstdreal_p,
                           'Tstdreal_it': Tstdreal_it,
                           'gain_it': gan_it,
                           'gain_p': gan_p,
    
                           })
    
    data_svm = pd.DataFrame({ 'size': [500, 1000, 1500],
                            'TACCsvmb': TACCsvmb,
                            'TACCsvmplus': TACCsvmplus,
                            'TACCsvmup': TACCsvmup,
                            'Tstdsvmb': Tstdsvmb,
                            'Tstdsvmplus': Tstdsvmplus,
                            'Tstdsvmup': Tstdsvmup,
                           'gain_svm': gan_svm,
    
                           })
    
    print('LRIT', gan_it)
    print('LR', gan_p)
    print('SVM', gan_svm)

    texto = 'T'+str(j)
    #data_lr.to_csv(r'/Users/mmartinez/Desktop/LR+ Paper/privileged_logistic_regression/code/results/mackeyglass/' + texto + '_LR.csv')
    #data_svm.to_csv(r'/Users/mmartinez/Desktop/LR+ Paper/privileged_logistic_regression/code/results/mackeyglass/' + texto + '_SVM.csv')


