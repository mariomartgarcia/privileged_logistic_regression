
from sklearn.model_selection import StratifiedKFold
import numpy as np
import cvxpy as cp



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
    