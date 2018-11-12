from __future__ import print_function
from __future__ import division
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from bayes_opt import BayesianOptimization
warnings.filterwarnings('ignore')

global X_train, y_train, X_dev, y_dev, X_test, y_test

def read_data(train_data_path, dev_data_path, test_data_path):
    global X_train, y_train, X_dev, y_dev, X_test, y_test
    train_dataset = pd.read_csv(train_data_path, header = None)
    dev_dataset = pd.read_csv(dev_data_path, header = None)
    test_dataset = pd.read_csv(test_data_path, header = None)
    
    concat_set = pd.concat([train_dataset, dev_dataset, test_dataset], keys=[0,1,2])
    
    temp = pd.get_dummies(concat_set, columns = [1,2,3,4,5,6,8])

    train, dev, test = temp.xs(0), temp.xs(1), temp.xs(2)

    y_train = train[[9]]
    y_dev = dev[[9]]
    y_test = test[[9]]

    y_train = y_train.values.reshape((y_train.shape[0]))
    y_dev = y_dev.values.reshape((y_dev.shape[0]))
    y_test = y_test.values.reshape((y_test.shape[0]))

    train = train.drop([9], axis = 1)
    dev = dev.drop([9], axis = 1)
    test = test.drop([9], axis = 1)
    
    X_train = train.iloc[:, :].values

    X_dev = dev.iloc[:, :].values

    X_test = test.iloc[:, :].values

    # Encode Y
    labelencoder_y = LabelEncoder()
    y_train = labelencoder_y.fit_transform(y_train)
    y_dev = labelencoder_y.fit_transform(y_dev)
    y_test = labelencoder_y.fit_transform(y_test)

    # Convert 0 to -1
    y_train[y_train == 0] = -1
    y_dev[y_dev == 0] = -1
    y_test[y_test == 0] = -1

def optimize_bagging(max_depth, n_estimators):
    predictor = BaggingClassifier(DecisionTreeClassifier(max_depth = int(max_depth)), n_estimators = int(n_estimators))
    predictor.fit(X_train, y_train)
    val = predictor.score(X_dev, y_dev)
    return val

def optimize_boosting(max_depth, n_estimators):
    predictor = AdaBoostClassifier(DecisionTreeClassifier(max_depth = int(max_depth)), n_estimators = int(n_estimators))
    predictor.fit(X_train, y_train)
    val = predictor.score(X_dev, y_dev)
    return val

def get_accuracies(bagging, boosting):
    bag_depth = bagging.res['max']['max_params']['max_depth']
    bag_n = bagging.res['max']['max_params']['n_estimators']
    boost_depth = boosting.res['max']['max_params']['max_depth']
    boost_n = boosting.res['max']['max_params']['n_estimators']
    bag = BaggingClassifier(DecisionTreeClassifier(max_depth = int(bag_depth)), n_estimators = int(bag_n))
    bag.fit(X_train, y_train)
    boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth = int(boost_depth)), n_estimators = int(boost_n))
    boost.fit(X_train, y_train)
    return (round(bag.score(X_test, y_test) * 100, 2), round(boost.score(X_test, y_test) * 100, 2))
    
def generate_results(bagging, boosting):
    
    max_depth_bag = int(bagging.res['max']['max_params']['max_depth'])
    n_estimate_bag = int(bagging.res['max']['max_params']['n_estimators'])
    
    max_depth_boost = int(boosting.res['max']['max_params']['max_depth'])
    n_estimate_boost = int(boosting.res['max']['max_params']['n_estimators'])
    
    train_score_bag = round(bagging.res['max']['max_val'] * 100, 2)
    train_score_boost = round(boosting.res['max']['max_val'] * 100, 2)
    (test_score_bag, test_score_boost) = get_accuracies(bagging, boosting)
    
    file = open('Output.txt', 'a+')
    file.write('\n\n')
    file.write('##### Bayesian Optimization for Bagging #####')
    file.write('\n\n')
    file.write('Best Validation accuracy = ' + str(train_score_bag) + ' %\n')
    file.write('Best Test accuracy = ' + str(test_score_bag) + ' %\n')
    file.write('Best Hyperparameters: max_depth = ' + str(max_depth_bag) + ' n_estimator = ' + str(n_estimate_bag) + '\n\n')
    file.write('Candidate hyperparameter at each iteration:\n\n')
    for i in range(50):
        file.write('For iteration ' + str(i+1) + ' max_depth = ' + str(int(bagging.X[i+10][0])) + ' n_estimator = ' + str(int(bagging.X[i+10][1])) + '\n')
    file.write('\n\n')
    file.write('##### Bayesian Optimization for Boosting #####')
    file.write('\n\n')
    file.write('Best Validation accuracy = ' + str(train_score_boost) + ' %\n')
    file.write('Best Test accuracy = ' + str(test_score_boost) + ' %\n')
    file.write('Best Hyperparameters: max_depth = ' + str(max_depth_boost) + ' n_estimator = ' + str(n_estimate_boost) + '\n\n')
    file.write('Candidate hyperparameter at each iteration:\n\n')
    for i in range(50):
        file.write('For iteration ' + str(i+1) + ' max_depth = ' + str(int(boosting.X[i+10][0])) + ' n_estimator = ' + str(int(boosting.X[i+10][1])) + '\n')
    file.write('\n\n')
    file.close()

def plot_curve(bagging, boosting):
    iterations = np.arange(1, 51)
    bag_performance = bagging.Y[10:]
    boost_performance = boosting.Y[10:]
    bag_performance = [x * 100 for x in bag_performance]
    boost_performance = [x * 100 for x in boost_performance]
    plt.xlim(0, 51)
    plt.plot(iterations, bag_performance, 'r-', label='Bagging')
    plt.plot(iterations, boost_performance, 'b-', label='Boosting')
    plt.legend()
    plt.title('Bayesian Optimization performance vs Iterations')
    plt.xlabel('Number of iterations')
    plt.ylabel('Validation Set Accuracy')
    plt.savefig('bayes_opt.png')
    plt.close(1)

train_data_path = 'income-data/income.train.txt' 
dev_data_path = 'income-data/income.dev.txt'
test_data_path = 'income-data/income.test.txt'
read_data(train_data_path, dev_data_path, test_data_path)
    
# Perform Bayesian Optimization for Bagging
bagging = BayesianOptimization(optimize_bagging, {'max_depth': [1, 100], 'n_estimators': [1, 100]})
    
bagging.explore({'max_depth': [1, 2, 3, 5, 10], 'n_estimators': [1, 2, 5, 10, 20]})

# Run for 50 iterations
bagging.maximize(n_iter = 50)
    
# Perform Bayesian Optimization for Boosting
boosting = BayesianOptimization(optimize_boosting, {'max_depth': [1, 100], 'n_estimators': [1, 100]})
    
boosting.explore({'max_depth': [1, 2, 3, 5, 10], 'n_estimators': [1, 2, 5, 10, 20]})

# Run for 50 iterations
boosting.maximize(n_iter = 50)

# Write output to file
generate_results(bagging, boosting)

# Plot curve
plot_curve(bagging, boosting)