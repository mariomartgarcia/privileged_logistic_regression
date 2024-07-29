import scipy.optimize as so
import numpy as np
import cvxpy as cp

#=============================================================================
                                #LRIT+ | CVX#
#=============================================================================

class LRIT_plus():
    #l número de instancias
    #n número de variables involucradas
    def __init__(self, l = 0, optimizer = "cvx"):
        if l < 0:
            raise Exception("l hyperparameter must be 0 or a real positive number")
        self.l = l
        
        if optimizer not in  ['cvx', 'scipy']:
            raise Exception("optimizer must be 'cvx' or 'scipy' ")
        self.optimizer = optimizer
    
    def fit(self, Xc, Xr, omega, beta):
        
        zp = np.matmul(Xc, omega.T) + beta
        self.zp = np.array(zp).reshape((1,-1))
        
        
        if self.optimizer == 'cvx':
               
            #Initialization of privileged model parameters
            self.w = cp.Variable((1, Xr.shape[1]))
            self.b = cp.Variable()
    
            
            self.z = (cp.matmul(self.w, Xr.transpose()) + self.b)
          
            loss = self.loss_cvx()
            obj = cp.Minimize(loss)
            prob = cp.Problem(obj) #, constraints)
            prob.solve()#verbose = True)
      
            self.w = self.w.value
            self.b = self.b.value
            
        if self.optimizer == 'scipy':
            
            self.Xr = Xr
            ini = np.ones(self.Xr.shape[1] + 1 )
            
            result = so.minimize(self.loss_scipy, ini, method='L-BFGS-B')
            
            self.w = result['x'][0:-1].reshape(1,-1)
            self.b = result['x'][-1]
    
        
    def loss_cvx(self):
        d = cp.sum((self.z-self.zp)**2) + self.l * cp.sum(self.w**2)
        return d 
    
    def loss_scipy(self, w):
       self.z = np.array(list(np.matmul(w[0:-1], self.Xr.transpose()) + w[-1])).reshape((1,-1))
       d = np.sum((self.z-self.zp)**2)  + self.l * np.sum(w[0:-1]**2)
       return d

    def sigmoid(self, x):
        z = np.exp(-x)
        return 1 / (1 + z)
    
    def predict(self, x):
        x_dot_weights = np.matmul(self.w, x.transpose()) + self.b
        probabilities = self.sigmoid(x_dot_weights.iloc[0])
        pre = [1 if p > 0.5 else -1 for p in probabilities]
        return  pre
    
    def predict_proba(self, x):
        x_dot_weights = np.matmul(self.w, x.transpose()) + self.b
        probabilities = self.sigmoid(x_dot_weights.iloc[0])
        return  probabilities
    
    def coef_(self):
        return self.w[0]
        
    def intercept_(self):
        return self.b 
        


    
#=============================================================================
                        #LR+ | SCIPY | SOLVER: 'LBFGS'#
#=============================================================================

class LR_plus():
    #l número de instancias
    #n número de variables involucradas
    def __init__(self, l = 0):
        if l < 0:
            raise Exception("l hyperparameter must be 0 or a real positive number")
        self.l = l
    
    def fit(self, Xc, Xr, omega, beta ):
        
        zp = np.matmul(Xc, omega.T) + beta
        
        self.zp = np.array(zp).reshape((1,-1))
        self.Xr = Xr
        
        
        ini = np.ones(self.Xr.shape[1] + 1 )
        
        result = so.minimize(self.loss, ini, method='L-BFGS-B')

        self.w = result['x'][0:-1].reshape(1,-1)
        self.b = result['x'][-1]
        
    def loss(self, w):
       self.z = np.array(list(np.matmul(w[0:-1], self.Xr.transpose()) + w[-1])).reshape((1,-1))
       d = np.sum((self.sigmoid(self.z)-self.sigmoid(self.zp))**2) + self.l * np.sum(w[0:-1]**2)
       return d
    
    
    def sigmoid(self, x):
        z = np.exp(-x)
        return 1 / (1 + z)
    
    def predict(self, x):
        x_dot_weights = np.matmul(self.w, x.transpose()) + self.b
        probabilities = self.sigmoid(x_dot_weights.iloc[0])
        pre = [1 if p > 0.5 else -1 for p in probabilities]
        return  pre
    
    def predict_proba(self, x):
        x_dot_weights = np.matmul(self.w, x.transpose()) + self.b
        probabilities = self.sigmoid(x_dot_weights.iloc[0])
        return  probabilities
    
    def coef_(self):
        return self.w[0]
        
    def intercept_(self):
        return self.b 
    
    
    

    
    
    


