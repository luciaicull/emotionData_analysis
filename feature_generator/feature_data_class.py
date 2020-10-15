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

        self.scaled_feature_data = dict()
        self.total_scaled_statistics = dict()

    def convert_to_dict(self): 
        data_dict = {'set_name':self.set_name,
                     'emotion_number':self.emotion_number,
                     'feature_data':self.feature_data,
                     'scaled_feature_data':self.scaled_feature_data,
                     'total_scaled_statistics':self.total_scaled_statistics}
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
    
    def scale_dataset(self):
        for set_dict in tqdm(self.set_list):
            data_list = set_dict['data_list']
            # 1. get set-level normalized features
            self._normalize_set_level(data_list)

            # 2. get total scaled features' statistics
            self._get_total_statistics(data_list)
    
    def _normalize_set_level(self, data_list):
        feature_keys = data_list[0].feature_data.keys()
        for key in feature_keys:
            indices = [0]
            total_feat_list = []
            for eN_class in data_list:
                feature_list = [[feat] for feat in eN_class.feature_data[key]]
                indices.append(indices[-1]+len(feature_list))
                total_feat_list += feature_list

            scaler = StandardScaler()
            scaler.fit(total_feat_list)
            transformed = scaler.transform(total_feat_list)
            for i, eN_class in enumerate(data_list):
                feature_list = [feat[0] for feat in transformed[indices[i]:indices[i+1]]]
                eN_class.scaled_feature_data[key] = feature_list
    
    def _get_total_statistics(self, data_list):
        feature_keys = list(data_list[0].feature_data.keys())

        for eN_class in data_list:
            feature_data_list = eN_class.scaled_feature_data
            stat_dict = eN_class.total_scaled_statistics

            for key in feature_keys:
                feat_data = [feat for feat in feature_data_list[key] if not math.isnan(feat)]
                stat_dict[key+'_mean'] = np.mean(feat_data)
                stat_dict[key+'_std'] = np.std(feat_data)
                stat_dict[key+'_skew'] = skew(feat_data)
                stat_dict[key+'_kurt'] = kurtosis(feat_data)
    
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
        self.scaled_feature_data = dict()
        self.statistics = dict() 

    
    def convert_to_dict(self):
        data_dict = {'set_name':self.set_name,
                     'emotion_number':self.emotion_number,
                     'start_measure':self.start,
                     'end_measure':self.end,
                     'scaled_feature_data':self.scaled_feature_data,
                     'statistics':self.statistics}
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
            feature_keys = list(e1_class.scaled_feature_data.keys())
            max_measure = max([note.measure_number for note in e1_class.performance_data.xml_notes])
            split_num = math.ceil((max_measure - self.split) / self.hop) + 1
            measure_ranges = []
            for i in range(split_num+1):
                measure_ranges.append((i*self.hop+1, i*self.hop+self.split+1))
            
            # split
            for raw_feature_data in set_dict['data_list']:
                for i, (start, end) in enumerate(measure_ranges):
                    if i == len(measure_ranges)-1:
                        data = SplittedFeatureData(raw_feature_data.set_name, raw_feature_data.emotion_number, start, 'last')
                    else:
                        data = SplittedFeatureData(raw_feature_data.set_name, raw_feature_data.emotion_number, start, end)
                    
                    for key in feature_keys:
                        data.scaled_feature_data[key] = []
                    for i, note in enumerate(raw_feature_data.performance_data.xml_notes):
                        if note.measure_number in range(start, end):
                            for key in feature_keys:
                                if i < len(raw_feature_data.scaled_feature_data[key]):
                                    data.scaled_feature_data[key].append(raw_feature_data.scaled_feature_data[key][i])
                    
                    # calculate statistics
                    for key in feature_keys:
                        feat_list = [feat for feat in data.scaled_feature_data[key] if not math.isnan(feat)]
                        data.statistics[key+'_mean'] = np.mean(feat_list)
                        data.statistics[key+'_std'] = np.std(feat_list)
                        data.statistics[key+'_skew'] = skew(feat_list)
                        data.statistics[key+'_kurt'] = kurtosis(feat_list)
                    set_list.append(data)
        
        return set_list
                    
        
    def save_into_dict(self, save_path):
        splitted_feature_data_dicts = []

        for splitted_feature_data in self.set_list:
            data_dict = splitted_feature_data.convert_to_dict()
            splitted_feature_data_dicts.append(data_dict)
        save_name = 'splitted_hop_{}_split_{}.dat'.format(self.hop, self.split)
        utils.save_datafile(save_path, save_name, splitted_feature_data_dicts)
