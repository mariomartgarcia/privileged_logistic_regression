
from sklearn.model_selection import StratifiedKFold
import numpy as np
import cvxpy as cp
from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
import scipy.optimize as so
import pandas as pd

#=========================================================================================================
#---------------------------------------------------------------------------------------------------------
#=========================================================================================================
#Stratified folds


def skfold(X, y, n, r = 0):
    skf = StratifiedKFold(n_splits = n, shuffle = True, random_state = r)
    d= {}
    j = 0
    for train_index, test_index in skf.split(X, y): 
            d['X_train' + str(j)] = X.loc[train_index]
            d['X_test' + str(j)] = X.loc[test_index]
            d['y_train' + str(j)] = y.loc[train_index]
            d['y_test' + str(j)] = y.loc[test_index]

            d['X_train' + str(j)].reset_index(drop = True, inplace = True)
            d['X_test' + str(j)].reset_index(drop = True, inplace = True)
            d['y_train' + str(j)].reset_index(drop = True, inplace = True)
            d['y_test' + str(j)].reset_index(drop = True, inplace = True)
            j+=1
    return d

#=============================================================================
                        #SVM+ (kernel lineal) | CVX 
#=============================================================================

class svmplus_CVX():
    #l número de instancias
    #n número de variables involucradas
    def __init__(self, C = 1, gamma = 1):
        #Real Space
        self.C = C
        
        #Privileged Space
        self.gamma = gamma

    def fit(self, X, Xp, y):
        #Real Space
        self.w = cp.Variable((1, X.shape[1]))
        self.b = cp.Variable()
        
        #Privileged Space
        self.wp = cp.Variable((1, Xp.shape[1]))
        self.bp = cp.Variable()

        x_dot_weights  = cp.matmul(self.w, X.transpose()) + self.b
        x_dot_weights_p  = cp.matmul(self.wp, Xp.transpose()) + self.bp
        y_reshaped = y.values.reshape((1,-1))
        
        constraints = [cp.multiply(y_reshaped, x_dot_weights) >= 1 - x_dot_weights_p,
                       x_dot_weights_p >= 0]

        loss = self.loss(Xp, self.C, self.gamma, self.w, self.wp, self.bp)
        obj = cp.Minimize(loss)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        self.w = self.w.value
        self.b = self.b.value
        
        
    def loss(self, Xp, C, gamma, w, wp, bp):
        
        margin =  cp.norm(w)**2
        margin_p = gamma*( cp.norm(wp)**2)
        x_dot_weights_p  = cp.matmul(wp, Xp.transpose()) + bp
        slack = C*cp.sum(x_dot_weights_p)
        return 0.5*(margin + margin_p) +slack
    
    
    
    def predict(self, x):
        pred = np.matmul(self.w, x.transpose()) + self.b
        self.values = list(pred.iloc[0])
        pre = np.sign(self.values)
        return pre
    
    def parameters(self):
        return  self.w, self.b
    
#=============================================================================
                        #KNOWLEDGE TRANSFER SVM
#=============================================================================

class KT_svm():
    def __init__(self, kernel_ridge = 'linear', kernel_svm = 'linear'):
        self.kernel_svm = kernel_svm
        self.kernel_ridge = kernel_ridge
        
    def fit(self, X, y, pi_features):  
        #pi_features = [pi_features]
        Xp = X[pi_features]
        Xr = X.drop(pi_features, axis = 1)
        self.m = {}
        kt = pd.DataFrame([])
        for i, j in enumerate(pi_features):
            self.m[j] = KernelRidge(kernel = self.kernel_ridge)
            self.m[j].fit(Xr, Xp.iloc[:,i])
            xkt = self.m[j].predict(Xr)
            kt[j] = xkt
            
        Xkt = pd.concat([Xr, kt], axis = 1)

        
        self.pi_columns = Xp.columns
        self.model = svm.SVC(kernel =  self.kernel_svm)
        self.model.fit(Xkt, y)
        
        
        
    def predict(self, Xr):
        kt = pd.DataFrame([])
        for i in range(len(self.m)):
            col = self.pi_columns[i]
            xkt = self.m[col].predict(Xr)
            kt[col] = xkt
                    
            
        Xkt = pd.concat([Xr, kt], axis = 1)
        pre = self.model.predict(Xkt)
        return pre
    
#=============================================================================
                        #LRGD | SCIPY | SOLVER: 'LBFGS'#
#=============================================================================


class GD():
    #l número de instancias
    #n número de variables involucradas
    def __init__(self, l = 1, T = 1):
        if l < 0:
            raise Exception("l hyperparameter must be 0 or a real positive number")
        self.l = l
        self.T = T

    
    def fit(self, Xc, Xr, y, omega, beta ):
        
        zp = np.matmul(Xc, omega.T) + beta
        self.y = np.array([[np.abs(i-1), i] for i in y])
        self.zp = np.array(zp).reshape((1,-1))
        self.Xr = Xr
        
        #SCIPY
        #ini = np.ones(self.Xr.shape[1] + 1 )
        ini = np.random.randn(self.Xr.shape[1] + 1 )*0.01

        result = so.minimize(self.loss, ini, method='L-BFGS-B')
        
        self.w = result['x'][0:-1].reshape(1,-1)
        self.b = result['x'][-1]
        
    
    def loss(self, w):
        
       self.z = np.array(list(np.matmul(w[0:-1], self.Xr.transpose()) + w[-1])).reshape((1,-1))
       y_pred = np.array([[1-i,i] for i in np.ravel(self.sigmoid(self.z))])
       y_upper = np.array([[1-i,i] for i in np.ravel(self.sigmoid(self.zp/self.T)) ])
        
       d1 =  np.mean(-np.sum(self.y*np.log(y_pred + 1e-15), axis = 1))
       d2 =  np.mean(-np.sum(y_upper*np.log(y_pred + 1e-15), axis = 1))
        
       return  (1-self.l)*d1 + self.l * d2
   
    def sigmoid(self, x):
        z = np.exp(-x)
        return 1 / (1 + z)
    
    def predict(self, x):
        self.x_dot_weights = np.matmul(self.w, x.transpose()) + self.b
        
        probabilities = self.sigmoid(self.x_dot_weights.iloc[0])
        pre = [1 if p > 0.5 else 0 for p in probabilities]
        return  pre
    
    def predict_proba(self, x):
        x_dot_weights = np.matmul(self.w, x.transpose()) + self.b
        probabilities = self.sigmoid(x_dot_weights.iloc[0])
        return  probabilities
    
    def coef_(self):
        return self.w[0]
        
    def intercept_(self):
        return self.b 
    