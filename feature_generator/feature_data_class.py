from tqdm import tqdm

class FeatureData:
    def __init__(self, set_name, performance_data):
        self.set_name = set_name
        self.emotion_number = performance_data.emotion_number
        self.performance_data = performance_data
        self.feature_data = dict()

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
        for set_dict in tqdm(raw_set_list):
            set_name = set_dict['name']
            performance_data_list = set_dict['list']

            new_dict = {'name': set_name, 'data_list':[]}
            for performance_data in performance_data_list:
                new_dict['data_list'].append(FeatureData(set_name, performance_data))
                # set_list.append(FeatureData(set_name, performance_data))

            set_list.append(new_dict)
        return set_list
    
    def get_final_data(self, split):
        for set_dict in self.set_list:
            set_name = set_dict['name']
            data_list = set_dict['data_list']
            for feature_data in data_list:
                feature_data.split_into_measures(split)
            
        self._split_into_measures(split)
        self._get_statistics()
        return self._convert_to_dict()


    def _split_into_measures(self, split):
        pass

    def _get_statistics(self):
        pass

    def _convert_to_dict(self):
        pass
    
