from sklearn.model_selection import KFold
from sklearn import svm
import pandas as pd
import numpy as np
import csv

from . import utils
from .constants import XMLMIDI_FEATURE_KEYS, FEATURE_KEYS

class Runner():
    def __init__(self, total_data, train_data, test_data):
        feature_key = FEATURE_KEYS
        self.total_data = total_data
        self.train_data = train_data
        self.test_data = test_data

        self.total_X, self.total_Y = utils.make_X_Y_not_splitted(self.total_data, feature_key)
        self.train_X, self.train_Y = utils.make_X_Y_not_splitted(self.train_data, feature_key)
        self.test_X, self.test_Y = utils.make_X_Y_not_splitted(self.test_data, feature_key)

    def run_svm(self):
        kf = KFold(n_splits=5)
        svm_options = {'C': 10, 'kernel': 'linear',
                       'decision_function_shape': 'ovr', 'gamma': 'scale'}
        clf = svm.SVC(C=svm_options['C'], kernel=svm_options['kernel'], 
                      decision_function_shape=svm_options['decision_function_shape'], 
                      gamma=svm_options['gamma'])
        
        self._run_cross_validation(kf, clf)
        self._run_train_test(clf)
    
    def _run_cross_validation(self, kf, classifier):
        fold_total_accuracy = []
        fold_total_result = []
        for train_index, test_index in kf.split(self.total_X):
            # split
            train_X_for_cv, test_X_for_cv = self.total_X[train_index], self.total_X[test_index]
            train_Y_for_cv, test_Y_for_cv = self.total_Y[train_index], self.total_Y[test_index]

            # training
            classifier.fit(train_X_for_cv, train_Y_for_cv)

            # get prediction
            predicted = classifier.predict(test_X_for_cv)

            # check accuracy
            total_accuracy = classifier.score(test_X_for_cv, test_Y_for_cv)
            total_result = utils.get_total_result(predicted, test_Y_for_cv)

            fold_total_accuracy.append(total_accuracy)
            fold_total_result.append(total_result)
        
        # check stats of k-fold cross validation accuracy
        acc_list = np.array(fold_total_accuracy)
        print('total_accuracy', ": %0.3f (+/- %0.3f)" % (acc_list.mean(), acc_list.std()*2))

        final_total_result_mean = dict()
        final_total_result_std = dict()
        for key in fold_total_result[0].keys():
            (row, col) = np.array(fold_total_result[0][key]).shape
            tmp = []
            for r in range(0, row):
                tmp.append([])
                for c in range(0, col):
                    tmp[r].append([])
            for fold_result in fold_total_result:
                cur_result = fold_result[key]
                for i, cur_result_row in enumerate(cur_result):
                    for j, item in enumerate(cur_result_row):
                        tmp[i][j].append(item)

            mean = np.zeros((row, col))
            std = np.zeros((row, col))
            for i, final_total_result_row in enumerate(tmp):
                for j, item in enumerate(final_total_result_row):
                    acc_list = np.array(tmp[i][j])
                    mean[i][j] = acc_list.mean()
                    std[i][j] = acc_list.std()

            final_total_result_mean[key] = mean
            final_total_result_std[key] = std * 2

        utils.print_total_result(final_total_result_mean)
        utils.print_total_result(final_total_result_std)
    

    def _run_train_test(self, classifier):
        classifier.fit(self.train_X, self.train_Y)

        predicted = classifier.predict(self.test_X)
        total_accuracy = classifier.score(self.test_X, self.test_Y)
        
        total_result = utils.get_total_result(predicted, self.test_Y)
        
        print("total_accuracy : %0.3f" % total_accuracy)
        utils.print_total_result(total_result)
