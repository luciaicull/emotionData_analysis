import _pickle as cPickle
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .constants import DATA_PATH, FILE_NAME, VALID_LIST, TEST_LIST, STAT_TYPE, FEATURE_KEYS, BATCH_SIZE

class RawDataLoader(object):
    def __init__(self, data_path, file_name):
        self.path = data_path.joinpath(file_name)
        self.dict_data = self._load_dict_data()
    
    def _load_dict_data(self):
        with open(self.path, 'rb') as f:
            u = cPickle.Unpickler(f)
            dict_data = u.load()
        return dict_data

    def load_dataset(self, mode, stat, x_keys):
        if mode == 'valid':
            list_name = VALID_LIST
        elif mode == 'test':
            list_name = TEST_LIST
        else:
            list_name = None

        data_list = []
        for dicts in self.dict_data:
            name = dicts[0]['name'] + '.' + dicts[0]['performer']
            if mode == 'train':
                if (name not in VALID_LIST) and (name not in TEST_LIST):
                    data_list.append(dicts)
            else:
                if name in list_name :
                    data_list.append(dicts)
        
        x, y = self.make_X_and_Y(data_list, stat, x_keys)
        return x, y

    '''
    def split_data(self, test_list, valid_list):
        train_data = []
        valid_data = []
        test_data = []
        
        for dicts in self.dict_data:
            name = dicts[0]['name'] + '.' + dicts[0]['performer']
            if name in test_list:
                test_data.append(dicts)
            elif name in valid_list:
                valid_data.append(dicts)
            else:
                train_data.append(dicts)

        return train_data, valid_data, test_data

    '''
    def make_X_and_Y(self, data, stat, x_keys):
        X = []
        Y = []
        
        for dicts in data:
            for dic in dicts:
                data = []
                for key in x_keys:
                    if key in dic[stat].keys():
                        data.append(dic[stat][key])
                    elif 'cross' in key:
                        feature_name = key.split('_cross')[0]
                        #print(feature_name)
                        data.append(dic[stat][feature_name+'_mean']
                                    * dic[stat][feature_name+'_std'])
                    else:
                        print("ERROR : No key named " + key)

                X.append(data)
                Y.append(dic['emotion_number'])

        return np.array(X), np.array(Y)

class EmotionDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        #y = np.eye(5)[self.y[index] - 1]
        #return self.x[index], y
        return self.x[index], self.y[index] - 1
    
    def __len__(self):
        return self.x.shape[0]
            

def get_dataloader():
    DL = RawDataLoader(DATA_PATH, FILE_NAME)
    x_train, y_train = DL.load_dataset('train', STAT_TYPE, FEATURE_KEYS)
    x_valid, y_valid = DL.load_dataset('valid', STAT_TYPE, FEATURE_KEYS)
    x_test, y_test = DL.load_dataset('test', STAT_TYPE, FEATURE_KEYS)

    train_set = EmotionDataset(x_train, y_train)
    valid_set = EmotionDataset(x_valid, y_valid)
    test_set = EmotionDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,  shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE,  shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,  shuffle=True, drop_last=False)

    return train_loader, valid_loader, test_loader
