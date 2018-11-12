import warnings
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
warnings.filterwarnings('ignore')

def read_data(train_data_path, dev_data_path, test_data_path):
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

    return (X_train, y_train, X_dev, y_dev, X_test, y_test)

def perform_bagging(tree_depths, num_estimators, train_x, train_y, dev_x, dev_y, test_x, test_y):
    
    train_accuracies = []
    dev_accuracies = []
    test_accuracies = []
    x_axis = []
    
    for depth in tree_depths:
        train_depth = []
        dev_depth = []
        test_depth = []
        x = []
        
        for estimate in num_estimators:
            
            tree = DecisionTreeClassifier(max_depth = depth)
            bagging = BaggingClassifier(tree, n_estimators=estimate)
            
            bagging.fit(train_x, train_y)
            
            train_acc = bagging.score(train_x, train_y) * 100
            dev_acc = bagging.score(dev_x, dev_y) * 100
            test_acc = bagging.score(test_x, test_y) * 100
            
            train_depth.append(train_acc)
            dev_depth.append(dev_acc)
            test_depth.append(test_acc)
            x.append(estimate)
    
        train_accuracies.append(train_depth)
        dev_accuracies.append(dev_depth)
        test_accuracies.append(test_depth)
        x_axis.append(x)
      
    return(train_accuracies, dev_accuracies, test_accuracies, x_axis)

def perform_boosting(tree_depths, num_estimators, train_x, train_y, dev_x, dev_y, test_x, test_y):
    
    train_accuracies = []
    dev_accuracies = []
    test_accuracies = []
    x_axis = []
    
    for depth in tree_depths:
        train_depth = []
        dev_depth = []
        test_depth = []
        x = []
        for estimate in num_estimators:
            
            tree = DecisionTreeClassifier(max_depth = depth)
            boosting = AdaBoostClassifier(tree, n_estimators=estimate)
            
            boosting.fit(train_x, train_y)
            
            train_acc = boosting.score(train_x, train_y) * 100
            dev_acc = boosting.score(dev_x, dev_y) * 100
            test_acc = boosting.score(test_x, test_y) * 100
            
            train_depth.append(train_acc)
            dev_depth.append(dev_acc)
            test_depth.append(test_acc)
            x.append(estimate)
    
        train_accuracies.append(train_depth)
        dev_accuracies.append(dev_depth)
        test_accuracies.append(test_depth)
        x_axis.append(x)
      
    return(train_accuracies, dev_accuracies, test_accuracies, x_axis)

def plot_curves(train_accuracies, dev_accuracies, test_accuracies, x_axis, depths, ensemble):
    
    for i in range(len(x_axis)):
        plt.xlim(0, 101)
        plt.plot(x_axis[i], train_accuracies[i], 'r-', label='Train accuracy')
        plt.plot(x_axis[i], dev_accuracies[i], 'b-', label='Dev accuracy')
        plt.plot(x_axis[i], test_accuracies[i], 'g-', label='Test accuracy')
        plt.xlabel('Number of estimators')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title("(" + ensemble + ") Accuracies vs n_estimators for depth " + str(depths[i]), loc = 'center')
        plt.savefig(ensemble+"_Depth_" + str(depths[i]) + ".png")
        plt.close(1)
    
def write_to_file(train_accuracies_bag, dev_accuracies_bag, test_accuracies_bag, x_axis_bag, tree_depths_bag, train_accuracies_boost, dev_accuracies_boost, test_accuracies_boost, x_axis_boost, tree_depths_boost):
    
    file = open('Output.txt', 'w+')
    file.write('##### Bagging #####\n\n')
    
    for i in range(len(tree_depths_bag)):
        for j in range(len(x_axis_bag[0])):
            file.write('Tree depth = ' + str(tree_depths_bag[i]) + ' Bag Size = ' + str(x_axis_bag[i][j]) + ' Train Accuracy = ' + str(round(train_accuracies_bag[i][j], 2)) + ' %\n')
            file.write('Tree depth = ' + str(tree_depths_bag[i]) + ' Bag Size = ' + str(x_axis_bag[i][j]) + ' Dev Accuracy = ' + str(round(dev_accuracies_bag[i][j], 2)) + ' %\n')
            file.write('Tree depth = ' + str(tree_depths_bag[i]) + ' Bag Size = ' + str(x_axis_bag[i][j]) + ' Test Accuracy = ' + str(round(test_accuracies_bag[i][j], 2)) + ' %\n')
        file.write('\n')
    
    file.write('\n\n##### Boosting #####\n\n')
    
    for i in range(len(tree_depths_boost)):
        for j in range(len(x_axis_boost[0])):
            file.write('Tree depth = ' + str(tree_depths_boost[i]) + ' Bag Size = ' + str(x_axis_boost[i][j]) + ' Train Accuracy = ' + str(round(train_accuracies_boost[i][j], 2)) + ' %\n')
            file.write('Tree depth = ' + str(tree_depths_boost[i]) + ' Bag Size = ' + str(x_axis_boost[i][j]) + ' Dev Accuracy = ' + str(round(dev_accuracies_boost[i][j], 2)) + ' %\n')
            file.write('Tree depth = ' + str(tree_depths_boost[i]) + ' Bag Size = ' + str(x_axis_boost[i][j]) + ' Test Accuracy = ' + str(round(test_accuracies_boost[i][j], 2)) + ' %\n')
        file.write('\n')
        
    file.close()
        
def main():
    train_data_path = 'income-data/income.train.txt' 
    dev_data_path = 'income-data/income.dev.txt'
    test_data_path = 'income-data/income.test.txt'
    
    tree_depths_bag = [1, 2, 3, 5, 10]
    tree_depths_boost = [1, 2, 3]
    num_estimators = [10, 20, 40, 60, 80, 100]
    
    # Read and preprocess dataset
    (train_x, train_y, dev_x, dev_y, test_x, test_y) = read_data(train_data_path, dev_data_path, test_data_path)
    
    # Bagging Operation
    (train_accuracies_bag, dev_accuracies_bag, test_accuracies_bag, x_axis_bag) = perform_bagging(tree_depths_bag, num_estimators, train_x, train_y, dev_x, dev_y, test_x, test_y)
    
    # Plot curves for bagging iterations
    plot_curves(train_accuracies_bag, dev_accuracies_bag, test_accuracies_bag, x_axis_bag, tree_depths_bag, 'Bagging')
    
    # Boosting operation
    (train_accuracies_boost, dev_accuracies_boost, test_accuracies_boost, x_axis_boost) = perform_boosting(tree_depths_boost, num_estimators, train_x, train_y, dev_x, dev_y, test_x, test_y)
    
    # Plot curves for boosting iterations
    plot_curves(train_accuracies_boost, dev_accuracies_boost, test_accuracies_boost, x_axis_boost, tree_depths_boost, 'Boosting')
    
    # Write all the output to file
    write_to_file(train_accuracies_bag, dev_accuracies_bag, test_accuracies_bag, x_axis_bag, tree_depths_bag, train_accuracies_boost, dev_accuracies_boost, test_accuracies_boost, x_axis_boost, tree_depths_boost)

main()