import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import  accuracy_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import lrplus as lr
import load_UCIdatasets as bs
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats
import tools as tl
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# %%

#===================================
# DATASETS (select what you want)
#===================================

#X, y = bs.breast_cancer()
#X, y = bs.obesity()
#X, y = bs.wine()



#X, y = bs.drugs()
X, y = bs.spam()
#X, y = bs.heart()
#X, y = bs.wm()
#X, y = bs.heart()
#X, y = bs.abalone()
#X, y = bs.car()

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

# %%

#===============================================================
# N TIME. 5-CV. INCREASING THE NUMBER OF PI FEATURES. MAIN CODE
#===============================================================
# DEVELOPMENT


number_pi = 2    #Up to 4 privileged features
repetitions = 30  #Number of repetitions for each PI feature


r = random.sample(range(500), repetitions)  #Seed number without replacement
q = {}

#Results boxes for every seed
TACClb, TACCub, TACCreal_it, TACCreal_p = [], [], [], []
Tstdlb, Tstdub, Tstdreal_it, Tstdreal_p  = [], [], [], []
Tmedias = []

for i in range(1, number_pi):
    
    #Results boxes for every seed
    ACClb, ACCub, ACCreal_it, ACCreal_p = [], [], [], []
    medias_p = []
    
    print(i)
    print('-----')
    for k in r:
        print(k)
        cv = 5
        dr = tl.skfold(X, y, cv, r = k)
    
        acc_lb, mae_lb = [], []
        acc_ub, mae_ub = [], []
        acc_realit, mae_realit  = [], []
        acc_realp, mae_realp = [], []
        medias = []
        
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
            pi_var =  names[0:i]
            #pi_var = ['dias_hospi']
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
            proba = lg.predict_proba(X_test)
            
            
            medias.append(np.mean([proba[i][1] if pre[i] == 1 else proba[i][0] for i in range(len(pre))]))
            
            
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

        
        #Computation od the mean for the 5 folds
        
        #Base and Unreal Privileged model
        ACClb.append(np.mean(acc_lb))
        ACCub.append(np.mean(acc_ub))
        
        medias_p.append(np.mean(medias))

        #LRIT+
        ACCreal_it.append(np.mean(acc_realit))

        #LR+
        ACCreal_p.append(np.mean(acc_realp))

    #Computation od the mean for the N iterations
    
    q['lower' + str(i)] = ACClb
    q['upper' + str(i)] = ACCub
    q['real_it' + str(i)] = ACCreal_it
    q['real_p' + str(i)] = ACCreal_p
     
    #Base and Unreal Privileged model
    TACClb.append(np.mean(ACClb))
    Tstdlb.append(np.std(ACClb))
    TACCub.append(np.mean(ACCub))
    Tstdub.append(np.std(ACCub))
    
    Tmedias.append(np.mean(medias_p))


    #LRIT+
    TACCreal_it.append(np.mean(ACCreal_it))
    Tstdreal_it.append(np.std(ACCreal_it))
    
    #LR+
    TACCreal_p.append(np.mean(ACCreal_p))
    Tstdreal_p.append(np.std(ACCreal_p))
# %%
print(Tmedias)


def priv_gain(x, lb, ub):
    pg = ( (np.array(x) - np.array(lb)) / (np.array(ub) - np.array(lb)) )*100
    return pg

gan_it = priv_gain(TACCreal_it[0], TACClb[0], TACCub[0])
gan_p = priv_gain(TACCreal_p[0], TACClb[0], TACCub[0])

print('LRUP', np.round(TACCub[0],3), '+-', np.round(Tstdub[0],3))
print('LRIT+', np.round(TACCreal_it[0],4), '+-', np.round(Tstdreal_it[0],3))
print('LR+', np.round(TACCreal_p[0],4), '+-', np.round(Tstdreal_p[0],3))
print('LRB', np.round(TACClb[0],3), '+-', np.round(Tstdlb[0],3))

print('Ganancia LR+', np.round(gan_p,1))#, '+-', np.round(std_p,1))
print('Ganancia LRIT+', np.round(gan_it,1))#, '+-', np.round(std_it,1))