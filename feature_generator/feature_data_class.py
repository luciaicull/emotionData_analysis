from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import skew, kurtosis
import math

from . import utils

class RawFeatureData:
    def __init__(self, set_name, performance_data):
        self.set_name = set_name
        self.emotion_number = performance_data.emotion_number
        self.performance_data = performance_data
        self.feature_data = dict()

        self.statistics = dict()
        self.scaled_statistics = dict()

    def convert_to_dict(self): 
        data_dict = {'set_name':self.set_name,
                     'emotion_number':self.emotion_number,
                     'feature_data':self.feature_data,
                     'statistics':self.statistics,
                     'scaled_statistics':self.scaled_statistics}
        return data_dict

class RawFeatureDataset:
    def __init__(self, set_list):
        '''
        set_list => list of set_dict
            set_dict => {name:set_name, data_list:performance data list}
                set_dict['data_list'] => list of RawFeatureData class
        '''
        self.set_list = self._init_set_list(set_list)

    def _init_set_list(self, raw_set_list):
        set_list = []
        for set_dict in raw_set_list:
            set_name = set_dict['name']
            performance_data_list = set_dict['list']

            new_dict = {'name': set_name, 'data_list': []}
            for performance_data in performance_data_list:
                new_dict['data_list'].append(RawFeatureData(set_name, performance_data))

            set_list.append(new_dict)
        return set_list
    
    def get_statistics(self):
        for set_dict in tqdm(self.set_list):
            data_list = set_dict['data_list']
            feature_keys = list(data_list[0].feature_data.keys())

            # calculate stats
            for eN_class in data_list:
                for key in feature_keys:
                    if '_diff' in key:
                        feat_data = [feat for feat in eN_class.feature_data[key] if (feat != None) and (feat != 0)]
                    else:
                        feat_data = [feat for feat in eN_class.feature_data[key] if feat != None]
                    eN_class.statistics[key+'_mean'] = np.mean(feat_data)
                    eN_class.statistics[key+'_std'] = np.std(feat_data)
                    eN_class.statistics[key+'_skew'] = skew(feat_data)
                    eN_class.statistics[key+'_kurt'] = kurtosis(feat_data)
            
            # normalize stats
            stat_keys = list(data_list[0].statistics.keys())
            total_stats = []
            for eN_class in data_list:
                eN_data = [eN_class.statistics[key] for key in stat_keys]
                total_stats.append(eN_data)
            scaler = StandardScaler()
            scaler.fit(total_stats)
            transformed_stats = scaler.transform(total_stats)
            for eN_class, eN_transformed_stats in zip(data_list, transformed_stats):
                for i, key in enumerate(stat_keys):
                    eN_class.scaled_statistics[key] = eN_transformed_stats[i]

    def save_into_dict(self, save_path):
        raw_feature_data_dicts = []
        for set_dict in self.set_list:
            data_list = set_dict['data_list']
            for eN_class in data_list:
                data_dict = eN_class.convert_to_dict()
                raw_feature_data_dicts.append(data_dict)
        
        utils.save_datafile(save_path, 'raw_feature_dataset_dict.dat', raw_feature_data_dicts)


class SplittedFeatureData:
    def __init__(self, set_name, emotion_number, start, end):
        self.set_name = set_name
        self.emotion_number = emotion_number
        self.start = start
        self.end = end
        self.feature_data = dict()
        self.statistics = dict() 
        self.scaled_statistics = dict()

    
    def convert_to_dict(self):
        data_dict = {'set_name':self.set_name,
                     'emotion_number':self.emotion_number,
                     'start_measure':self.start,
                     'end_measure':self.end,
                     'statistics':self.statistics,
                     'scaled_statistics':self.scaled_statistics}
        return data_dict


class SplittedFeatureDataset:
    def __init__(self, raw_feature_dataset, hop, split):
        self.hop = hop
        self.split = split
        self.set_list = self._split_dataset(raw_feature_dataset.set_list)
    
    def _split_dataset(self, raw_set_list):
        set_list = []

        for set_dict in tqdm(raw_set_list):
            data_list = set_dict['data_list']

            # get measure ranges
            e1_class = data_list[0]
            feature_keys = list(e1_class.feature_data.keys())
            max_measure = max([note.measure_number for note in e1_class.performance_data.xml_notes])
            if max_measure < self.split:
                measure_ranges = [(1, max_measure+1)]
            else:
                split_num = math.ceil((max_measure - self.split) / self.hop) + 1
                measure_ranges = []
                for i in range(split_num+1):
                    measure_ranges.append((i*self.hop+1, i*self.hop+self.split+1))
            
            # split
            splitted_data_list = []
            for eN_raw_feature_class in set_dict['data_list']:
                for i, (start, end) in enumerate(measure_ranges):
                    if i == len(measure_ranges)-1:
                        data = SplittedFeatureData(eN_raw_feature_class.set_name, eN_raw_feature_class.emotion_number, start, 'last')
                    else:
                        data = SplittedFeatureData(eN_raw_feature_class.set_name, eN_raw_feature_class.emotion_number, start, end)
                    
                    for key in feature_keys:
                        data.feature_data[key] = []
                    for i, note in enumerate(eN_raw_feature_class.performance_data.xml_notes):
                        if note.measure_number in range(start, end):
                            for key in feature_keys:
                                if i < len(eN_raw_feature_class.feature_data[key]):
                                    data.feature_data[key].append(eN_raw_feature_class.feature_data[key][i])
                    
                    # calculate statistics
                    for key in feature_keys:
                        if '_diff' in key:
                            feat_list = [feat for feat in data.feature_data[key] if (feat!=None) and (feat!=0)]
                        else:
                            feat_list = [feat for feat in data.feature_data[key] if (feat!=None)]
                        data.statistics[key+'_mean'] = np.mean(feat_list)
                        data.statistics[key+'_std'] = np.std(feat_list)
                        data.statistics[key+'_skew'] = skew(feat_list)
                        data.statistics[key+'_kurt'] = kurtosis(feat_list)
                    
                    splitted_data_list.append(data)
            
            stat_keys = list(splitted_data_list[0].statistics.keys())
            total_stats = []
            for splitted_data in splitted_data_list:
                stat_list = [splitted_data.statistics[key] for key in stat_keys]
                total_stats.append(stat_list)
            scaler = StandardScaler()
            scaler.fit(total_stats)
            transformed_stats = scaler.transform(total_stats)
            for splitted_data, eN_transformed_stats in zip(splitted_data_list, transformed_stats):
                for i, key in enumerate(stat_keys):
                    splitted_data.scaled_statistics[key] = eN_transformed_stats[i]
            
            set_list += splitted_data_list
        
        return set_list
                    
        
    def save_into_dict(self, save_path):
        splitted_feature_data_dicts = []

        for splitted_feature_data in self.set_list:
            data_dict = splitted_feature_data.convert_to_dict()
            splitted_feature_data_dicts.append(data_dict)
        save_name = 'splitted_hop_{}_split_{}.dat'.format(self.hop, self.split)
        utils.save_datafile(save_path, save_name, splitted_feature_data_dicts)
