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

        self.svm_options = {'C': 10, 'kernel': 'linear',
                            'decision_function_shape': 'ovr', 'gamma': 'scale'}

    def run_svm(self):
        kf = KFold(n_splits=5)
        self._run_cross_validation_svm(kf)
        self._run_train_test_svm()
    
    def _run_cross_validation_svm(self, kf):
        fold_total_accuracy = []
        fold_total_result = []
        for train_index, test_index in kf.split(self.total_X):
            # split
            train_X_for_cv, test_X_for_cv = self.total_X[train_index], self.total_X[test_index]
            train_Y_for_cv, test_Y_for_cv = self.total_Y[train_index], self.total_Y[test_index]

            # training
            clf = svm.SVC(C=self.svm_options['C'], kernel=self.svm_options['kernel'],
                          decision_function_shape=self.svm_options['decision_function_shape'], gamma=self.svm_options['gamma'])
            clf.fit(train_X_for_cv, train_Y_for_cv)

            # get prediction
            predicted = clf.predict(test_X_for_cv)

            # check accuracy
            total_accuracy = clf.score(test_X_for_cv, test_Y_for_cv)
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
    

    def _run_train_test_svm(self):
        clf = svm.SVC(C=self.svm_options['C'], kernel=self.svm_options['kernel'], decision_function_shape=self.svm_options['decision_function_shape'], gamma=self.svm_options['gamma'])
        clf.fit(self.train_X, self.train_Y)

        predicted = clf.predict(self.test_X)
        total_accuracy = clf.score(self.test_X, self.test_Y)
        
        total_result = utils.get_total_result(predicted, self.test_Y)
        
        print("total_accuracy : %0.3f" % total_accuracy)
        utils.print_total_result(total_result)

    '''
    def _print_total_result(self, total_result):
        for key in total_result.keys():
            print(key)
            df = pd.DataFrame(total_result[key])
            print(df)


    def _get_total_result(self, predicted, answer):
        result = self._get_result_num(predicted, answer)

        total_result = dict()
        # 1. Single emotion -> Single emotion accuracy
        ratio_result = self._get_ratio_result(result)
        total_result['single_to_single'] = ratio_result

        # 2. Single emotion -> Arousal accuracy
        e2_to_HA = ratio_result[1][3] + ratio_result[1][4]
        e2_to_LA = ratio_result[1][1] + ratio_result[1][2]
        e3_to_HA = ratio_result[2][3] + ratio_result[2][4]
        e3_to_LA = ratio_result[2][2] + ratio_result[2][1]
        e4_to_HA = ratio_result[3][3] + ratio_result[3][4]
        e4_to_LA = ratio_result[3][1] + ratio_result[3][2]
        e5_to_HA = ratio_result[4][4] + ratio_result[4][3]
        e5_to_LA = ratio_result[4][1] + ratio_result[4][2]
        total_result['single_to_arousal'] = [[e2_to_HA, e2_to_LA],
                                            [e3_to_HA, e3_to_LA],
                                            [e4_to_HA, e4_to_LA],
                                            [e5_to_HA, e5_to_LA]]

        # 3. Single emotion -> Valence accuracy
        e2_to_PV = ratio_result[1][1] + ratio_result[1][3]
        e2_to_NV = ratio_result[1][2] + ratio_result[1][4]
        e3_to_PV = ratio_result[2][1] + ratio_result[2][3]
        e3_to_NV = ratio_result[2][2] + ratio_result[2][4]
        e4_to_PV = ratio_result[3][3] + ratio_result[3][1]
        e4_to_NV = ratio_result[3][2] + ratio_result[3][4]
        e5_to_PV = ratio_result[4][1] + ratio_result[4][3]
        e5_to_NV = ratio_result[4][4] + ratio_result[4][2]
        total_result['single_to_valence'] = [[e2_to_PV, e2_to_NV],
                                            [e3_to_PV, e3_to_NV],
                                            [e4_to_PV, e4_to_NV],
                                            [e5_to_PV, e5_to_NV]]

        # 4. Arousal -> Arousal accuracy
        HA_to_HA = (result[3][3]+result[3][4]+result[4][3] +
                    result[4][4]) / (sum(result[3]) + sum(result[4]))
        HA_to_LA = (result[3][1]+result[3][2]+result[4][1] +
                    result[4][2]) / (sum(result[3]) + sum(result[4]))
        LA_to_LA = (result[1][1]+result[1][2]+result[2][1] +
                    result[2][2]) / (sum(result[1])+sum(result[2]))
        LA_to_HA = (result[1][3]+result[1][4]+result[2][3] +
                    result[2][4]) / (sum(result[1])+sum(result[2]))
        total_result['arousal_to_arousal'] = [[HA_to_HA, HA_to_LA],
                                            [LA_to_HA, LA_to_LA]]

        # 5. Valence -> Valence accuracy
        PV_to_PV = (result[1][1]+result[1][3]+result[3][1] +
                    result[3][3]) / (sum(result[1]) + sum(result[3]))
        PV_to_NV = (result[1][2]+result[1][4]+result[3][2] +
                    result[3][4]) / (sum(result[1]) + sum(result[3]))
        NV_to_NV = (result[2][2]+result[2][4]+result[4][2] +
                    result[4][4]) / (sum(result[2]) + sum(result[4]))
        NV_to_PV = (result[2][1]+result[2][3]+result[4][1] +
                    result[4][3]) / (sum(result[2]) + sum(result[4]))
        total_result['valence_to_valence'] = [[PV_to_PV, PV_to_NV],
                                            [NV_to_PV, NV_to_NV]]

        return total_result

    def _get_result_num(self, predicted, answer):
        result = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
        for pred, ans in zip(predicted, answer):
            result[ans-1][pred-1] += 1
        return result
    
    def _get_ratio_result(self, result):
        ratio_result = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
        for i, emotion in enumerate(result):
            for j, pred_num in enumerate(emotion):
                ratio_result[i][j] = pred_num/sum(emotion)
            
        return ratio_result
    '''
