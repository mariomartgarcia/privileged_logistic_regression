import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import  accuracy_score, brier_score_loss
from sklearn.linear_model import LogisticRegression, LinearRegression
import lrplus as lr
import load_UCIdatasets as bs
import matplotlib.pyplot as pl
import seaborn as sns
import random
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn import svm
import tools as tl

warnings.filterwarnings("ignore")


def priv_gain(x, lb, ub):
    pg = ( (np.array(x) - np.array(lb)) / (np.array(ub) - np.array(lb)) )*100
    return pg

# %%

#===================================
# DATASETS (select what you want)
#===================================

X, y = bs.breast_cancer()
X, y = bs.obesity()
X, y = bs.wine()






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

X, y, pi_features = bs.kc2()
y = pd.Series(y)  
text = 'kc2'


# %%
X, y = bs.parkinsons()    
text = 'parkinsons'


col = X.columns
scaler = MinMaxScaler()
Xnorm = scaler.fit_transform(X)
Xn = pd.DataFrame(Xnorm, columns = col)

mi = mutual_info_classif(Xn, y)
mi_df = pd.DataFrame({'name': list(Xn.columns), 'mi': mi })
mi_sort = mi_df.sort_values(by='mi', ascending=False)
#Select 10 privileged features
# %%
#===============================================================
# N TIME. 5-CV. INCREASING THE NUMBER OF PI FEATURES. MAIN CODE
#===============================================================
# DEVELOPMENT


number_pi = 5    #Up to 4 privileged features
repetitions = 30  #Number of repetitions for each PI feature


  
r = random.sample(range(500), repetitions)  #Seed number without replacement
q = {}

#Results boxes for every seed
TACClb, TACCub, TACCreal_it, TACCreal_p = [], [], [], []
Tstdlb, Tstdub, Tstdreal_it, Tstdreal_p  = [], [], [], []
TACCsvmup, TACCsvmplus, TACCsvmb = [], [], []
Tstdsvmup, Tstdsvmplus, Tstdsvmb = [], [], [] 
Tmedias, Tbrier = [], []

for i in range(1, number_pi):
    
    #Results boxes for every seed
    ACClb, ACCub, ACCreal_it, ACCreal_p = [], [], [], []
    ACCsvmup, ACCsvmplus, ACCsvmplusq, ACCsvmb = [], [], [], []
    medias_p, brier_p = [], []
    
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
            pi_var =  names[0:i]
            #pi_var = pi_features
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
            svb = svm.SVC(kernel = 'linear' )
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

data_lr = pd.DataFrame({#'nPI': range(1, number_pi),

                       'TACClb': TACClb,
                       'TACCreal_it':TACCreal_it,
                       'TACCreal_p': TACCreal_p,
                       'TACCub': TACCub,
                       'Tstdlb': Tstdlb,
                       'Tstdreal_it': Tstdreal_it,
                       'Tstdreal_p':Tstdreal_p,
                       'Tstdub': Tstdub,                      
                       'gain_it': gan_it,
                       'gain_p': gan_p,

                       })

data_svm = pd.DataFrame({#'nPI': range(1, number_pi),

                       'TACCsvmb': TACCsvmb,
                       'TACCsvmplus': TACCsvmplus,
                       'TACCsvmup': TACCsvmup,
                       'Tstdsvmb': Tstdsvmb,
                       'Tstdsvmplus': Tstdsvmplus,
                       'Tstdsvmup': Tstdsvmup,                       
                       'gain_svm': gan_svm,

                       })


#data_lr.to_csv(r'/Users/mmartinez/Desktop/LR+ Paper/privileged_logistic_regression/code/results/'+ text +'/d_LR.csv')
#data_svm.to_csv(r'/Users/mmartinez/Desktop/LR+ Paper/privileged_logistic_regression/code/results/'+ text +'/d_SVM.csv')





# %%

#=================================================================
# COMPARISON OF PARAMETERS. TRAINING WITH THE FULL DATA COHORT
#=================================================================
#WITH PI REPRESENTATION
#======================
texto = 'breastcancer/bc'
#----------------------
number_pi = 2
pi_var =  names[0:number_pi]

col = X.columns
scaler = MinMaxScaler()
Xnorm = scaler.fit_transform(X)
X = pd.DataFrame(Xnorm, columns = col)
#-------------------------------------------
#Define the regular space
Xr = X.drop(pi_var, axis = 1)

#-------------------------------------------
#Unreal Privileged model
lg = LogisticRegression()    
lg.fit(X, y)
omega = lg.coef_[0]
beta = lg.intercept_


mask = [w  for w, j in enumerate(list(X.columns)) if j in pi_var]
omega_up = [j for w, j in enumerate(omega) if w not in mask]
w_priv = [j for w, j in enumerate(omega) if w in mask]
omega_up = omega_up + w_priv

#-------------------------------------------
#Base model
lgb = LogisticRegression()    
lgb.fit(Xr, y)
omega_b = list(lgb.coef_[0])


#-------------------------------------------
#LRIT+ model
al = lr.LRIT_plus()
al.fit(X, Xr,  omega, beta)
w_it = list(al.coef_())
#-------------------------------------------
#LR+ model
alp = lr.LR_plus()
alp.fit(X, Xr,  omega, beta)
w_p = list(alp.coef_())

#-------------------------------------------
#Add zero for parameters of the privileged features
for i in range(0, number_pi):
    omega_b.append(None)
    w_it.append(None)
    w_p.append(None)
    
#-------------------------------------------
#Graphic representation
    
features = list(Xr.columns) + pi_var

#Colors
color_lb = 'firebrick'
color_ub = 'forestgreen'
color_realit = 'darkorange'
color_realp = 'royalblue'

# %%
#GENERAL
sns.set_theme(style="whitegrid")
plt.figure(figsize=(17,4))
plt.grid()
plt.title('Breast Cancer' + ' | Parallel Comparison of Parameters | #PI = 2', fontweight="bold", fontsize = 24)
plt.plot(features, omega_up, 'o--', c = color_ub , label = r'LR$_{UP}$')
plt.plot(features, omega_b, 'o--', c = color_lb, label = r'LR$_{B}$')

plt.plot(features, w_it, 'o--', c = color_realit, label = 'LRIT+', alpha = 0.9)
plt.plot(features, w_p, 'o--', c = color_realp, label = 'LR+', alpha = 0.9) #para el vino 0.6
plt.xlabel('Features', fontweight="bold", fontsize = 22)
plt.ylabel('Value', fontweight="bold", fontsize = 22)
#plt.axvspan(xmin = 8.2, xmax = 8.22, color = 'black')
plt.axvline(x = X.shape[1]-number_pi-0.75, linewidth = 3,  color = 'black')
plt.grid()
plt.legend(fontsize = 16)
plt.xticks(rotation=90, fontsize = 20)
plt.yticks(fontsize = 20)
plt.savefig(r'/Users/mmartinez/Desktop/LR+ Paper/privileged_logistic_regression/code/results/' + texto + 'parametersPI.pdf', format='pdf', transparent = True, dpi = 300,  bbox_inches='tight')
plt.show()
