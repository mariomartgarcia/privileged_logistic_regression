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
    
    output = [1 if xT[i]>xt[i] else 0  for i in range(len(xT)) ]
    
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
#3, 5, 8, 12


T = 5
repetitions = 30  #Number of repetitions for each PI feature
r = random.sample(range(500), repetitions)  #Seed number without replacement

#Results boxes for every seed
TACClb, TACCub, TACCreal_it, TACCreal_p = [], [], [], []
Tstdlb, Tstdub, Tstdreal_it, Tstdreal_p  = [], [], [], []


for i in [500, 1000, 1500, 2000]:
    
    mg = mackey_glass(i, T)
    X = mg[0]
    y = mg[1]
    
    
    #Results boxes for every seed
    ACClb, ACCub, ACCreal_it, ACCreal_p = [], [], [], []

    print(i)
    for k in r:
        cv = 5
        dr = tl.skfold(X, y, cv, r = k)
    
        acc_lb, mae_lb = [], []
        acc_ub, mae_ub = [], []
        acc_realit, acc_realp  = [], []
        
        
        for h in range(cv):
            X_train = dr['X_train' + str(h)]
            y_train = dr['y_train' + str(h)]
            X_test = dr['X_test' + str(h)]
            y_test = dr['y_test' + str(h)]
            
            X_train = X_train.reset_index(drop = True)
            y_train = y_train.reset_index(drop = True)
        
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
            #al = lr.LogisticRegressionPI_logit()
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
texto = 't12/t12'
s = pd.read_csv( r'/Users/mmartinez/Desktop/Code/Python/LRPI/photos/mackeyglass/' + texto + 'dataMG.csv')
#s = pd.read_csv('Photos/standard_datasets/' + texto + '/5_30bc.csv')

dim = 4
TACClb = s['ACClb'][0:dim]
TACCub = s['ACCub'][0:dim]
TACCreal_it = s['ACCreal_it'][0:dim]
TACCreal_p = s['ACCreal_p'][0:dim]

Tstdlb = s['stdlb'][0:dim]
Tstdub = s['stdub'][0:dim]
Tstdreal_it = s['stdreal_it'][0:dim]
Tstdreal_p = s['stdreal_p'][0:dim]



# %%


#=====================================================
# N TIME. 5-CV. REPRESENTATION
#=====================================================
#texto = 't3/t3'
# GRAPHICAL REPRESENTATION

pilist = [500, 1000, 1500, 2000]


sns.set_theme(style="whitegrid")

#Colors
color_lb = 'firebrick'
color_ub = 'forestgreen'
color_realit = 'darkorange'
color_realp = 'royalblue'

#Lines
plt.figure(figsize=(7.5,4))
plt.plot(pilist, TACClb, 'D-.', c = color_lb, label = r'LR$_{B}$')
plt.plot(pilist, TACCub, c = color_ub, label = r'LR$_{UP}$', marker = 'o')
plt.plot(pilist, TACCreal_it, 'o--', c = color_realit, label = 'LRIT+')
plt.plot(pilist, TACCreal_p, 'o--', c = color_realp, label = 'LR+')

#Error
plt.fill_between(pilist, np.array(TACClb) - np.array(Tstdlb),  np.array(TACClb) + np.array(Tstdlb), alpha = 0.1, color = color_lb, edgecolor = 'black', linestyle = '-.')
plt.fill_between(pilist, np.array(TACCub) - np.array(Tstdub),  np.array(TACCub) + np.array(Tstdub), alpha = 0.07, color = color_ub, edgecolor = 'black', linestyle = '--')
plt.fill_between(pilist, np.array(TACCreal_it) - np.array(Tstdreal_it),  np.array(TACCreal_it) + np.array(Tstdreal_it), alpha = 0.1, color = color_realit, edgecolor = 'black', facecolor = color_realit, linestyle = '--')
plt.fill_between(pilist, np.array(TACCreal_p) - np.array(Tstdreal_p),  np.array(TACCreal_p) + np.array(Tstdreal_p), alpha = 0.1, color = color_realp, edgecolor = 'black', facecolor = color_realp, linestyle = '--')

plt.title('T = 12', fontweight="bold", fontsize = 14)
plt.ylabel('Accuracy', fontweight="bold", fontsize = 14)
plt.xlabel('Data Size (N)', fontweight="bold", fontsize = 14)
plt.xticks(pilist, fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(fontsize = 13)
plt.savefig(r'/Users/mmartinez/Desktop/Code/Python/LRPI/photos/mackeyglass/' + texto +'trainMG.pdf', format='pdf', transparent = True, dpi = 300,  bbox_inches='tight')
plt.show()


# PRIVILEGED GAIN
plt.figure(figsize=(7.5,4))


def priv_gain(x, lb, ub):
    pg = ( (np.array(x) - np.array(lb)) / (np.array(ub) - np.array(lb)) )*100
    return pg

gan_it = priv_gain(TACCreal_it, TACClb, TACCub)
gan_p = priv_gain(TACCreal_p, TACClb, TACCub)




plt.plot(pilist, gan_it, 'o--', c = color_realit, label = 'LRIT+')
plt.plot(pilist, gan_p, 'o--', c = color_realp, label = 'LR+')
plt.axhline(y = 0, c = 'black')
plt.title('T = 12', fontweight="bold", fontsize = 14)
plt.ylabel('Privileged Gain (%)', fontweight="bold", fontsize = 14)
plt.xlabel('Data Size (N)', fontweight="bold", fontsize = 14)
plt.xticks(pilist, fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(fontsize = 13)
plt.savefig(r'/Users/mmartinez/Desktop/Code/Python/LRPI/photos/mackeyglass/' + texto +'privgainMG.pdf', format='pdf', transparent = True, dpi = 300,  bbox_inches='tight')
plt.show()

