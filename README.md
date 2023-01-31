# Extending the use of privileged information paradigm to logistic regression
> LRIT+ and LR+ methods

[python-img]: https://img.shields.io/badge/Made%20with-Python-blue
[ama-img]: https://img.shields.io/badge/Ask%20me-anything-yellowgreen

![Made with Python][python-img]
![Ask me anything][ama-img]

This repository contains the code for the paper "Extending the use of privileged information paradigm to logistic regression"

<img width="944" alt="Screenshot 2023-01-31 at 09 53 26" src="https://user-images.githubusercontent.com/63496191/215713753-297dd1c8-5147-4357-903a-2465ff702941.png">


## Abstract

Learning Using Privileged Information (LUPI) paradigm [^1] exploits privileged features, not available at the testing stage, as additional information for training models. In this paper, the privileged information paradigm is addressed from the perspective of logistic regression. A reduced set of methods have addressed the handling of privileged information, but its implementation on logistic regression had not yet been formalised. Hence, two algorithms, LRIT+ and LR+, learned using the privileged information paradigm and preserving the interpretability of conventional logistic regression, are proposed. For its development, a traditional logistic regression trained with all available features, privileged and regular, is projected onto the solution space that exclusively contains regular features. The projection to obtain the model parameters is performed by the minimization of two different squared losses functions: for LRIT+ classifier the function is governed by logit terms and for LR+  by posterior probabilities. Experimental results on datasets report improvements of our proposals over the performance of traditional logistic regression learned without privileged information.


## Content

- **code:**
  - _lrplus.py_. Main file with LRIT+ and LR+ algorithms.
  - _load_UCIdatasets.py_. Load UCI datasets examples.
  - _UCIdatasets.py_. Implementation of LRIT+ and LR+ on UCI datasets.
  - _mnistplus.py_. Implementation of LRIT+ and LR+ on MNIST+ dataset.
  - _mackey-glass.py_. Implementation of LRIT+ and LR+ on Mackey-Glass time series datasets.
- **data:**
  - _mnistplus_. MNIST+ dataset. Validation, train and test cohorts.
  - _mackey-glass_.   Mackey-Glass dataset for different data sizes ( N = 500, 1000, 1500, 2000)



## How to Install LRIT+ and LR+






## Description of the classifiers

> #### LRIT+. Logit model.

**LRIT_plus**(_l = 0, optimizer = 'cvx'_)

  - **_l: float, default = 0_**
  
    "_l_" is directly proportional to the regularization term: bigger values of "_l_" imply stronger regularization.
    
  - **_optimizer: {'cvx', 'scipy'}, default = 'cvx'_**
  
    Package used to obtain the optimal parameters of the loss function.
    - 'cvx'. [CVXPY](https://www.cvxpy.org/tutorial/intro/index.html) library is implemented. This method is usually faster.
    - 'scipy'. [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) from [SCIPY](https://docs.scipy.org/doc/scipy/index.html) library is used.




> #### LR+. Posterior probabilities model.

**LR_plus**(_l = 0_)

  - **_l: float, default = 0_**
  
    "_l_" is directly proportional to the regularization term: bigger values of "_l_" imply stronger regularization.
 
---

| Method | Description | 
| -----: | :--- | 
|   **fit(X', X, $\omega$', $\beta$')**     |    Fit the model according to regular and privileged features X', regular features X and  ( $\omega$ ', $\beta$') parameters of the unreal privileged model  |  
|   **predict(X)**      |    Predict class lables for samples in X   | 
|    **predict_proba(X)**     |   Probability estimates    | 

| Attribute | Description | 
| -----: | :--- | 
|      **coef_()**    | ndarray of shape (1, n_features).  Coefficient of the features in the decision function.    | 
|    **intercept_()**      |  ndarray of shape (1,).   Intercept (a.k.a. bias) added to the decision function.   | 
    
## Example of use

## References

[^1]: Vapnik, V., Vashist, A.: A new learning paradigm: Learning using privileged information. Neural Networks 22(5), 544–557 (2009)
  
  
  

