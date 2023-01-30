# Extending the use of privileged information paradigm to logistic regression
> LRIT+ and LR+ methods

[python-img]: https://img.shields.io/badge/Made%20with-Python-blue
[ama-img]: https://img.shields.io/badge/Ask%20me-anything-yellowgreen

![Made with Python][python-img]
![Ask me anything][ama-img]


#
### Description of the classifiers

> #### LRIT+. Logit model.

**LRIT_plus**(_l = 0, optimizer = 'cvx'_)

  - **_l: float, default = 0_**
  
    "_l_" is directly proportional to the regularization term: bigger values of "_l_" imply stronger regularization.
    
  - **_optimizer: {'cvx', 'scipy'}, default = 'cvx'_**
  
    Package used to minimize the loss function.
    - 'cvx'. [CVXPY](https://www.cvxpy.org/tutorial/intro/index.html) library is implemented. This method is usually faster.
    - 'scipy'. [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) from [SCIPY](https://docs.scipy.org/doc/scipy/index.html) library is used.

> #### LR+. Posterior probabilities model.

**LR_plus**(_l = 0_)

  - **_l: float, default = 0_**
  
    "_l_" is directly proportional to the regularization term: bigger values of "_l_" imply stronger regularization.
    

  
  
  
  

