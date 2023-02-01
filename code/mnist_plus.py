import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import lrplus as lr
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")



url_val_features = r'/Users/mmartinez/Desktop/Code/Data/mnist+/val_features.txt'
url_val_labels = r'/Users/mmartinez/Desktop/Code/Data/mnist+/val_labels.txt'
url_train_features = r'/Users/mmartinez/Desktop/Code/Data/mnist+/train_features.txt'
url_train_labels = r'/Users/mmartinez/Desktop/Code/Data/mnist+/train_labels.txt'
url_train_PFfeatures = r'/Users/mmartinez/Desktop/Code/Data/mnist+/train_PFfeatures.txt'
url_train_YYfeatures = r'/Users/mmartinez/Desktop/Code/Data/mnist+/train_YYfeatures.txt'
url_test_features = r'/Users/mmartinez/Desktop/Code/Data/mnist+/test_features.txt'
url_test_labels = r'/Users/mmartinez/Desktop/Code/Data/mnist+/test_labels.txt'

# %%

#======================
# VALIDATION | n = 4002
#======================

val_features = pd.read_csv(url_val_features, header = None)
val_labels = pd.read_csv(url_val_labels, header = None)

col = [str(i)+'r' for i in range(val_features.shape[1])]
scaler = MinMaxScaler()
fited = scaler.fit(val_features)
val_features_N = scaler.transform(val_features)
val_features = pd.DataFrame(val_features_N , columns = col)

val_labels[val_labels == 5] = 0
val_labels[val_labels == 8] = 1


#=================
# TRAIN | n = 100
#=================
train_features = pd.read_csv(url_train_features, header = None)
train_labels = pd.read_csv(url_train_labels, header = None)
train_PFfeatures = pd.read_csv(url_train_PFfeatures, header = None)
train_YYfeatures = pd.read_csv(url_train_YYfeatures, header = None)

#Normalization of regular features
col = [str(i)+'r' for i in range(train_features.shape[1])]
scaler_r = MinMaxScaler()
fited = scaler_r.fit(train_features)
train_features_N = scaler_r.transform(train_features)
train_features = pd.DataFrame(train_features_N, columns = col)

#Normalization of privileged features
col = [str(i)+'p' for i in range(train_PFfeatures.shape[1])]
scaler_p = MinMaxScaler()
fited = scaler_p.fit(train_PFfeatures)
train_PFfeatures_N = scaler_p.transform(train_PFfeatures)
train_PFfeatures = pd.DataFrame(train_PFfeatures_N, columns = col) #.iloc[:,0:3]

train = pd.concat([train_features, train_PFfeatures], axis = 1)



train_labels[train_labels == 5] = 0
train_labels[train_labels == 8] = 1

#=================
# TEST | n = 1866
#=================
test_features = pd.read_csv(url_test_features , header = None)
test_labels = pd.read_csv(url_test_labels , header = None)

col = [str(i)+'r' for i in range(test_features.shape[1])]
test_features_N = scaler_r.transform(test_features)
test_features = pd.DataFrame(test_features_N, columns = col)
test_labels[test_labels == 5] = 0
test_labels[test_labels == 8] = 1


# %%

#=========================================================================================
        #HYPERPARAMETER SELECTION FOR BASE AND UNREAL PRIVILEGED MODELS
                  ##C, L2 REGULARIZATION PARAMETER, LR (SCIKIT)##
#=========================================================================================
#C definition in SCIKIT: Inverse of regularization strength; must be a positive float. 
#Like in support vector machines, smaller values specify stronger regularization.

lgp = LogisticRegression() 
parameters = {'C': np.arange(0.05, 1.05, 0.05)}
clf = GridSearchCV(lgp, parameters, cv = 5)
clf.fit(val_features, val_labels)

print('Best Param for Base Model:', clf.best_params_ )
print('Best Score:', clf.best_score_)

lgp = LogisticRegression() 
parameters = {'C': np.arange(0.05, 1.05, 0.05)}
clf = GridSearchCV(lgp, parameters, cv = 5)
clf.fit(train, train_labels)

print('Best Param for Unreal Privileged:', clf.best_params_ )
print('Best Score:', clf.best_score_)

# %%

#========================================================================
                #HYPERPARAMETER SELECTION FOR LR+ AND LRIT+
                  ##lambda, REGULARIZATION PARAMETER##
#========================================================================
#lambda definition: Regularization strength; must be a positive float. 
#smaller values specify smaller regularization. lambda = 1/C (SCIKIT)

def regularization(data, data_labels, dataPF, val_feat, val_labels):

    lista = [0, 0.1,  0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 5, 6, 7, 8]
    accr_logit, accr_proba = [], []
    
    for k in lista:
       
        #-------------------------------------------
        #Define the regular space
        pi_var =  dataPF.columns
        datar = data.drop(pi_var, axis = 1)
        
        #-------------------------------------------
        #Unreal Privileged model
        lg = LogisticRegression(C  = 0.2)    
        lg.fit(data, data_labels)
        omega = lg.coef_[0]
        beta = lg.intercept_

        #-------------------------------------------
        #LRIT+ model
        al = lr.LRIT_plus(l = k, optimizer = 'scipy')
        al.fit(data, datar, omega, beta)
        test = al.predict(val_feat)
        
        accr_logit.append(accuracy_score(val_labels, test))
      
        #-------------------------------------------
        #LR+ model
        alp = lr.LR_plus(l = k)
        alp.fit(data, datar, omega, beta)
        testp = alp.predict(val_feat)
        
        accr_proba.append(accuracy_score(val_labels, testp))
        
        #-------------------------------------------

    cv_logit = pd.DataFrame({'values': lista, 'ACC': accr_logit})
    cv_proba = pd.DataFrame({'values': lista, 'ACC': accr_proba})
    
    l_logit = cv_logit.sort_values('ACC', ascending = False).reset_index(drop = True)['values'][0]
    l_proba = cv_proba.sort_values('ACC', ascending = False).reset_index(drop = True)['values'][0]
    return l_logit, l_proba

# %%

#==============================================
      #REPLICATION OF VAPNIK EXPERIMENT
#==============================================

n_iterations = 2                      #number of repetition foe each training data size
tds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   #training data sizes
train['output'] = train_labels[0]


ACC_lb = []
ACC_logit = []
ACC_proba = []


for i in tds:
    acc_lb = []
    acc_logit = []
    acc_proba = []
    
    n_er_lb = []
    n_er_logit = []
    n_er_proba = []

    print('--------------')
    print(i)
    for j in range(0, n_iterations):
        train_fullset = train.sample(frac = i, replace = False).reset_index(drop = True)
        train_set = train_fullset.drop('output', axis = 1)
        train_rset = train_set.iloc[:,0:100]
        train_pset = train_set.iloc[:,100:]
        train_labels_set = train_fullset['output']

        if j%5 == 0:
            print(j, 'computed of ', i)
        
        #-------------------------------------------
        #LRIT+ and LR+ Regularization
        l_logit, l_proba = regularization(train_set, train_labels_set, train_pset, val_features, val_labels)
        #-------------------------------------------
        
        #-------------------------------------------
        #Unreal Privileged model
        
        lgp = LogisticRegression(C = 0.2)  
        lgp.fit(train_set, train_labels_set)
        omega = lgp.coef_[0]
        beta = lgp.intercept_

            
        #-------------------------------------------
        #Base model
        lgb = LogisticRegression(C = 0.25)    
        lgb.fit(train_rset, train_labels_set)
        pre = lgb.predict(test_features)
        acc_lb.append(accuracy_score(test_labels, pre))
        
        #-------------------------------------------
        #LRIT+ model
        lit = lr.LRIT_plus(l = l_logit, optimizer = 'scipy')
        lit.fit(train_set, train_rset, omega, beta)
        plit= lit.predict(test_features)
        acc_logit.append(accuracy_score(test_labels, plit))
        
        
        #-------------------------------------------
        #LR+ model
        lp = lr.LR_plus(l = l_proba)
        lp.fit(train_set, train_rset, omega, beta)
        pl = lp.predict(test_features)
        acc_proba.append(accuracy_score(test_labels, pl))
        
    ACC_lb.append(np.mean(acc_lb))
    ACC_logit.append(np.mean(acc_logit))
    ACC_proba.append(np.mean(acc_proba))

    print('===================')
    print('Lower:', ACC_lb)
    print('Logit:', ACC_logit)
    print('Proba:', ACC_proba)

    print('===================')
        
# %%
#pd.DataFrame({'ACC_logit': ACC_logit, 'ACC_proba': ACC_proba, 'ACC_lb': ACC_lb}).to_csv('rep_Vapnik30.csv')
#s = pd.read_csv(r'/Users/mmartinez/Desktop/Code/Python/LRPI/rep_Vapnik.csv')

# %%
s = pd.read_csv(r'/Users/mmartinez/Desktop/Code/Python/LRPI/rep_Vapnik30.csv')
ACC_logit = s['ACC_logit']
ACC_proba = s['ACC_proba']
ACC_lb = s['ACC_lb']
BS = s['BS']

# %%
#==============================================
      #GRAPHIC REPRESENTATION
#==============================================
color_lb = 'firebrick'
color_realit = 'darkorange'
color_realp = 'royalblue'

tds = [40, 50, 60, 70, 80, 90]
plt.plot(tds, np.array(ACC_lb), 'o--',  c = color_lb, label = r'$\hat{\Omega}_{B}$', alpha = 0.8)
plt.plot(tds, np.array(ACC_logit), 'o--', c = color_realit, label = 'LRIT+', alpha = 0.8)
plt.plot(tds, np.array(ACC_proba), 'o--', c = color_realp, label = 'LR+', alpha = 0.8)
plt.grid(True)
plt.ylabel('Accuracy', fontweight="bold")
plt.xlabel('Training data size', fontweight="bold")
plt.legend()
#plt.savefig('Photos/mnist/mnist30.pdf', format='pdf', transparent = True, dpi = 300,  bbox_inches='tight')
plt.show()

        