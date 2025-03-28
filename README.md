
# Extending the learning using privileged information paradigm to logistic regression


> LRIT+ and LR+ classifiers

[python-img]: https://img.shields.io/badge/Made%20with-Python-blue
[ama-img]: https://img.shields.io/badge/Ask%20me-anything-yellowgreen

![Made with Python][python-img]
![Ask me anything][ama-img]

This repository contains the code for the paper _"Extending the learning using privileged information paradigm to logistic regression"_. Learning using privileged information paradigm [^1] is a learning scenario that exploits privileged features, available at training time, but not at prediction, as additional information for training models. Specifically, learning using privileged information paradigm is addressed from the logistic regression perspective. Two new approaches, LRIT+ and LR+, learned using the privileged information paradigm and preserving the interpretability of conventional logistic regression, are proposed. For its development, the parameters of a traditional logistic regression trained with all available features, privileged and regular, are projected onto the parameter space associated to regular features. The projection to obtain the model parameters is performed by the minimization of two different convex loss functions: for LRIT+ classifier the function is governed by logit terms, and for LR+  by posterior probabilities. Experimental results report improvements of our proposals over the performance of traditional logistic regression learned without privileged information.


<img width="512" alt="Screenshot 2023-01-31 at 13 29 33" src="https://user-images.githubusercontent.com/63496191/215760145-fa62ca77-d303-4ddd-a563-033d44a7fbfa.png">


## Content

- **code:**
  - `lrplus.py`. Main file with LRIT+ and LR+ algorithms.
  - `load_UCIdatasets.py`. Load UCI datasets examples.
  - `UCIdatasets.py`. Implementation of LRIT+ and LR+ on UCI datasets.
  - `mnist_plus.py`. Implementation of LRIT+ and LR+ on MNIST+ dataset.
  - `mackey_glass.py`. Implementation of LRIT+ and LR+ on Mackey-Glass time series datasets.
  - `tools.py`. Used functions.
  - `discussion.py`. Used functions.
- **data:**
  - `mnistplus`. MNIST+ dataset. Validation, train and test cohorts.
  - `UCI_dataset`. Obesity and Wine UCI datasets.
 


## How to Install LRIT+ and LR+

Step 1: Download the repository.

```git
git clone https://github.com/mariomartgarcia/privileged_logistic_regression 
cd privileged_logistic_regression/code
```

Step 2: Load the library `lrplus.py`.

```python
import lrplus as lrp
lrpy = lrp.LR_plus()
```

Alternative: Place `lrplus.py` file in the folder where you are working and continue with step 2.

## Description of the classifiers

#### LRIT+. Logit model.

>**LRIT_plus**(_l = 0, optimizer = 'cvx'_)
>
>  - **_l: float, default = 0_**
>  
>    "_l_" is directly proportional to the regularization term: bigger values of "_l_" imply stronger regularization.
>    
>  - **_optimizer: {'cvx', 'scipy'}, default = 'cvx'_**
>  
>    Package used to obtain the optimal parameters of the loss function.
>    - 'cvx'. [CVXPY](https://www.cvxpy.org/tutorial/intro/index.html) library is implemented. This method is usually faster.
>    - 'scipy'. [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) from [SCIPY](https://docs.scipy.org/doc/scipy/index.html) library is used.




#### LR+. Posterior probabilities model.

>**LR_plus**(_l = 0_)
>
>  - **_l: float, default = 0_**
>  
>    "_l_" is directly proportional to the regularization term: bigger values of "_l_" imply stronger regularization.
> 


| Method | Description | 
| -----: | :--- | 
|   **fit(X', X, $\omega$', $\beta$')**     |    Fit the model according to regular and privileged features X', regular features X and  ( $\omega$ ', $\beta$') parameters of the unreal privileged model  |  
|   **predict(X)**      |    Predict class lables for samples in X   | 
|    **predict_proba(X)**     |   Probability estimates    | 

| Attribute | Description | 
| -----: | :--- | 
|      **coef_()**    | ndarray of shape (1, n_features).  Coefficient of the features in the decision function.    | 
|    **intercept_()**      |  ndarray of shape (1,).   Intercept (a.k.a. bias) added to the decision function.   | 
    
## Example 

Formulation of the method:

- _X_train_. Samples with regular features for the training set. 
- _X_train_priv_. Samples with regular and privileged features for the training set. 
- _y_train_. Output of the training samples.
- _X_test_. Samples with regular features for the test set


1. Train a standard logistic regression (sklearn package is recommended) with _X_train_priv_ (regular and privileged features) to obtain the parameters ( $\omega$ ', $\beta$'). Note that hyperparameter tuning can be done to obtain a better performance.


```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train_priv, y_train)
wp = lr.coef_
bp = lr.intercept_
```

2. LR+ classifier training and predictions.

```python
import lrplus as lrp

clas = lrp.LR_plus()
clas.fit(X_train_priv, X_train, wp, bp)
pre = clas.predict(X_test_regular)
```

## Contact

Mario Martínez García - mmartinez@bcamath.org


## Citation

The corresponding BiBTeX citation is given below:

```bib
@article{cite-key, 
	author = {Martínez-García, Mario and García-Gutierrez, Susana and Barreñada, Lasai and Inza, Iñaki and Lozano, Jose A.}, 
	journal = {Neurocomputing}, 
	pages = {128869},
	title = {Extending the learning using privileged information paradigm to logistic regression}, 
	year = {2024}, 
 	doi = {https://doi.org/10.1016/j.neucom.2024.128869}}
```



## References

[^1]: Vapnik, V., Vashist, A.: A new learning paradigm: Learning using privileged information. Neural Networks 22(5), 544–557 (2009)
  
  
  

