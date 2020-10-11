from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from matplotlib import pyplot as plt
from pathlib import Path


class FeatureData:
    def __init__(self, set_name, performance_data):
        self.set_name = set_name
        self.emotion_number = performance_data.emotion_number
        self.performance_data = performance_data
        self.feature_data = dict()
        self.scaled_feature_data = dict()

    def make_features_into_dict(self):
        pass


class FeatureDataset:
    def __init__(self, set_list):
        '''
        set_list => list of set_dict
            set_dict => {name:set_name, data_list:performance data list}
                set_dict['data_list'] => list of FeatureData class
        '''
        self.set_list = self._init_set_list(set_list)

    def _init_set_list(self, raw_set_list):
        set_list = []
        for set_dict in raw_set_list:
            set_name = set_dict['name']
            performance_data_list = set_dict['list']

            new_dict = {'name': set_name, 'data_list': []}
            for performance_data in performance_data_list:
                new_dict['data_list'].append(FeatureData(set_name, performance_data))

            set_list.append(new_dict)
        return set_list

    def get_final_data(self, split, hop):
        for set_dict in self.set_list:
            set_name = set_dict['name']
            data_list = set_dict['data_list']
            # 1. get set-level normalized features
            self._normalize_set_level(data_list)

            # 2. split

            # 3. get statistics

        # 4. return in dict format
        return self._convert_to_dict()

    def _normalize_set_level(self, data_list):
        feature_keys = data_list[0].feature_data.keys()
        for key in tqdm(feature_keys):
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
            
    def _convert_to_dict(self):
        # TODO
        return
    
    