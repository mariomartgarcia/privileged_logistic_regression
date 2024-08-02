# %%
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import  accuracy_score 
from sklearn.linear_model import LogisticRegression
import lrplus as lr
import load_UCIdatasets as bs
import random
from scipy import stats
import tools as tl
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

warnings.filterwarnings("ignore")

def priv_gain(x, lb, ub):
    pg = ( (np.array(x) - np.array(lb)) / (np.array(ub) - np.array(lb)) )*100
    return pg

# %%
text = ['breast_cancer', 'obesity', 'wine', 'drugs', 'spam', 'heart_f', 'heart', 'wm', 'abalone', 'kc2', 'parkinsons']
dc = [ bs.breast_cancer(), bs.obesity(), bs.wine(), bs.drugs(), bs.spam(), bs.heart_f(), bs.heart(), bs.wm(), bs.abalone(), bs.kc2(), bs.parkinsons()]

datasets_dict = dict(zip(text, dc))

repetitions = 30  #Number of repetitions for each PI feature
cv = 5
r = random.sample(range(500), repetitions)  #Seed number without replacement

dataLR = pd.DataFrame([])


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

    if te in ['breast_cancer', 'obesity', 'wine', 'drugs', 'spam', 'heart_f', 'heart', 'wm', 'abalone']:
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

 
    Tacc_lb, Tacc_ub, Tacc_realp = [], [], []

    for k in r:
        
        dr = tl.skfold(X, y, cv, r = k)

        acc_lb, acc_ub, acc_realp = [], [], []
        medias, medias_p, medias_n= [], [], []

        
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
            
            
            medias.append(np.mean([proba[i][1] if pre[i] == 1 else proba[i][0] for i in range(len(pre))]))
            

            delta = (pre == y_test)
            medias_p.append(np.mean([np.max(proba[i]) for i in range(len(proba)) if delta[i] == True ]))
            medias_n.append(np.mean([np.max(proba[i]) for i in range(len(proba)) if delta[i] == False ]))



            acc_ub.append(accuracy_score(y_test, pre))
            
            #-------------------------------------------
            #Base model
            lg = LogisticRegression()    
            lg.fit(X_trainr, y_train)
            prep = lg.predict(X_testr)
    
            acc_lb.append(accuracy_score(y_test, prep))
            
            #-------------------------------------------
            #LR+ model
            alp = lr.LR_plus()
            alp.fit(X_train, X_trainr,  omega, beta)
            testp = alp.predict(X_testr)
            
            acc_realp.append(accuracy_score(y_test, testp))

        
        Tacc_lb.append(np.mean(acc_lb))
        Tacc_realp.append(np.mean(acc_realp))
        Tacc_ub.append(np.mean(acc_ub))

        gan_p = priv_gain(np.mean(Tacc_realp), np.mean(Tacc_lb), np.mean(Tacc_ub))



    data_lr = pd.DataFrame({#'nPI': range(1, number_pi),
                    'dataset': te,
                    'ACClb':       np.round(np.mean(Tacc_lb), 3),
                    'ACCreal_p':   np.round(np.mean(Tacc_realp), 3),
                    'ACCub':       np.round(np.mean(Tacc_ub), 3),
                    'MPP':       np.round(np.mean(medias), 3),
                    'MPP_correct':       np.round(np.mean(medias_p), 3),
                    'MPP_errors':       np.round(np.mean(medias_n), 3),
                    'gain_p':       np.round(gan_p, 3),
                    }, index = [0])
    dataLR = pd.concat([dataLR, data_lr]).reset_index(drop = True)

dataLR.to_csv('discussion.csv')


