# Income Classifier using Bagging and Boosting with Bayesian Optimization

This project is an implementation of bagging and boosting techniques with decision tree as base classifier to classify income data. It also uses bayes optimization technique to find the best hyperparameter for tree depth size and number of estimates for bagging and boosting. 
The following libraries are used in this project:
* [Bagging](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) 
* [Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) 
* [Bayes Optimization](https://github.com/fmfn/BayesianOptimization) 

The dataset is obtained from UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income).

To run the program make sure you have [pip](https://pip.pypa.io/en/stable/installing/) installed. Then open a terminal and run the following command: 

For linux and mac users:
```
sh run_code.sh
```
For windows users, you need to install [Cygwin](http://www.cygwin.com/) or any other linux command line utility and run the above mentioned command.



