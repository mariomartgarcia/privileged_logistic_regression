import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import  accuracy_score
from sklearn.linear_model import LogisticRegression
import lrplus as lr
import load_UCIdatasets as bs
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats

warnings.filterwarnings("ignore")


# %%

#===================================
# DATASETS (select what you want)
#===================================

X, y = bs.breast_cancer()
#X, y = bs.drugs()
#X, y = bs.wine()


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

lg = LogisticRegression()    
lg.fit(X, y)
coef = pd.DataFrame({'names': X.columns, 'coef': lg.coef_[0]})
values = np.abs(coef.coef).sort_values(ascending = False)
names = list(coef.names[values.index])
log_coef = pd.DataFrame({'names': names , 'value': values })

# %%

#===============================================================
# N TIME. 5-CV. INCREASING THE NUMBER OF PI FEATURES. MAIN CODE
#===============================================================
# DEVELOPMENT


number_pi = 5     #Up to 4 privileged features
repetitions = 1  #Number of repetitions for each PI feature


r = random.sample(range(500), repetitions)  #Seed number without replacement
q = {}

#Results boxes for every seed
TACClb, TACCub, TACCreal_it, TACCreal_p = [], [], [], []
Tstdlb, Tstdub, Tstdreal_it, Tstdreal_p  = [], [], [], []

for i in range(1, number_pi):
    
    #Results boxes for every seed
    ACClb, ACCub, ACCreal_it, ACCreal_p = [], [], [], []

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

        
        for h in range(cv):
            X_train = dr['X_train' + str(h)]
            y_train = dr['y_train' + str(h)]
            X_test = dr['X_test' + str(h)]
            y_test = dr['y_test' + str(h)]
            
            X_train = X_train.reset_index(drop = True)
            y_train = y_train.reset_index(drop = True)
        
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


    #LRIT+
    TACCreal_it.append(np.mean(ACCreal_it))
    Tstdreal_it.append(np.std(ACCreal_it))
    
    #LR+
    TACCreal_p.append(np.mean(ACCreal_p))
    Tstdreal_p.append(np.std(ACCreal_p))



# %%
#=====================================================
# N TIME. 5-CV. INCREASING THE NUMBER OF PI FEATURES
#=====================================================
#text = 'spam/s'
name = 'Breast Cancer'


# GRAPHICAL REPRESENTATION
number_pi = 5
pilist = np.arange(1, number_pi)


sns.set_theme(style="whitegrid")

#Colors
color_lb = 'firebrick'
color_ub = 'forestgreen'
color_realit = 'darkorange'
color_realp = 'royalblue'

#Lines
plt.figure(figsize=(7.5,4))
plt.plot(pilist, TACClb, 'D-.', c = color_lb, label = r'$\hat{\Omega}_{B}$')
plt.plot(pilist, TACCub, c = color_ub, label = r'$\hat{\Omega}_{UP}$', marker = 'o')
plt.plot(pilist, TACCreal_it, 'o--', c = color_realit, label = 'LRIT+')
plt.plot(pilist, TACCreal_p, 'o--', c = color_realp, label = 'LR+')

#Error
plt.fill_between(pilist, np.array(TACClb) - np.array(Tstdlb),  np.array(TACClb) + np.array(Tstdlb), alpha = 0.1, color = color_lb, edgecolor = 'black', linestyle = '-.')
plt.fill_between(pilist, np.array(TACCub) - np.array(Tstdub),  np.array(TACCub) + np.array(Tstdub), alpha = 0.07, color = color_ub, edgecolor = 'black', linestyle = '--')
plt.fill_between(pilist, np.array(TACCreal_it) - np.array(Tstdreal_it),  np.array(TACCreal_it) + np.array(Tstdreal_it), alpha = 0.1, color = color_realit, edgecolor = 'black', facecolor = color_realit, linestyle = '--')
plt.fill_between(pilist, np.array(TACCreal_p) - np.array(Tstdreal_p),  np.array(TACCreal_p) + np.array(Tstdreal_p), alpha = 0.1, color = color_realp, edgecolor = 'black', facecolor = color_realp, linestyle = '--')

plt.title(name, fontweight="bold", fontsize = 14)
plt.ylabel('Accuracy', fontweight="bold", fontsize = 14)
plt.xlabel('Number of privileged features', fontweight="bold", fontsize = 14)
plt.xticks(pilist, fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(fontsize = 13)
#plt.savefig('Photos/standard_datasets/'+ texto +'_n_privileged.pdf', format='pdf', transparent = True, dpi = 300,  bbox_inches='tight')
plt.show()
#plt.show(sns)

# PRIVILEGED GAIN
plt.figure(figsize=(7.5,4))


def priv_gain(x, lb, ub):
    pg = ( (np.array(x) - np.array(lb)) / (np.array(ub) - np.array(lb)) )*100
    return pg

gan_it = priv_gain(TACCreal_it, TACClb, TACCub)
gan_p = priv_gain(TACCreal_p, TACClb, TACCub)

std_it = priv_gain(Tstdreal_it, Tstdlb, Tstdub)
std_p = priv_gain(Tstdreal_p, Tstdlb, Tstdub)




plt.plot(pilist, gan_it, 'o--', c = color_realit, label = 'LRIT+')
plt.plot(pilist, gan_p, 'o--', c = color_realp, label = 'LR+')
plt.fill_between(pilist, gan_it - std_it,  gan_it + std_it, alpha = 0.1, color = color_realit, edgecolor = 'black', facecolor = color_realit, linestyle = '--')
plt.fill_between(pilist, gan_p - std_p,  gan_p - std_p,  alpha = 0.1, color = color_realp, edgecolor = 'black', facecolor = color_realp, linestyle = '--')
plt.axhline(y = 0, c = 'black')
plt.title(name, fontweight="bold", fontsize = 14)
plt.ylabel('Privileged Gain (%)', fontweight="bold", fontsize = 14)
plt.xlabel('Number of privileged features', fontweight="bold", fontsize = 14)
plt.xticks(pilist, fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(fontsize = 13)
#plt.savefig('Photos/standard_datasets/' + texto +'_gain.pdf', format='pdf', transparent = True, dpi = 300,  bbox_inches='tight')
plt.show()


# %%

#=================================================================
# COMPARISON OF PARAMETERS. TRAINING WITH THE FULL DATA COHORT
#=================================================================
#WITH PI REPRESENTATION
#======================

#----------------------
number_pi = 2
pi_var =  names[0:number_pi]


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
omega_up = omega_up + [0] + w_priv

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
for i in range(0, number_pi+1):
    omega_b.append(0)
    w_it.append(0)
    w_p.append(0)
    
#-------------------------------------------
#Graphic representation
    
features = list(Xr.columns) + ['----------'] + pi_var

#Colors
color_lb = 'firebrick'
color_ub = 'forestgreen'
color_realit = 'darkorange'
color_realp = 'royalblue'

# %%
#GENERAL
plt.figure(figsize=(17,4))
plt.grid()
plt.title(name + ' | Parallel Comparison of Parameters | #PI = 2', fontweight="bold", fontsize = 14)
plt.plot(features, omega_up, 'o--', c = color_ub , label = r'$\hat{\Omega}_{UP}$')
plt.plot(features, omega_b, 'o--', c = color_lb, label = r'$\hat{\Omega}_{B}$')
plt.plot(features, w_it, 'o--', c = color_realit, label = 'LRIT+', alpha = 0.8)
plt.plot(features, w_p, 'o--', c = color_realp, label = 'LR+', alpha = 0.8) #para el vino 0.6
plt.xlabel('Features', fontweight="bold", fontsize = 14)
plt.ylabel('Value', fontweight="bold", fontsize = 14)
plt.axvline(x = Xr.shape[1], color = 'black')
plt.grid()
plt.legend(fontsize = 13)
plt.xticks(rotation=90, fontsize = 13)
plt.yticks(fontsize = 13)
#plt.savefig('Photos/standard_datasets/'+ text +'parametersPI.pdf', format='pdf', transparent = True, dpi = 300,  bbox_inches='tight')
plt.show()
