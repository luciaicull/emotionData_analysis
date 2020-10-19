import _pickle as cPickle
from pathlib import Path
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader

from .constants import VALID_LIST, TEST_LIST, BATCH_SIZE

class RawDataLoader(object):
    def __init__(self, data_path, frag_data_name, total_data_name, test_with_split):
        self.path = data_path

        if not test_with_split:
            self.total_dataset = self._load_file(total_data_name)

        self.frag_dataset = self._load_file(frag_data_name)
    
    def _load_file(self, file_name):
        with open(self.path.joinpath(file_name), 'rb') as f:
            u = cPickle.Unpickler(f)
            dataset = u.load()
        return dataset


    def load_dataset(self, mode, x_keys, load_fragment):
        if mode == 'valid':
            list_name = VALID_LIST
        elif mode == 'test':
            list_name = TEST_LIST
        else:
            list_name = None
        
        dataset_list = []
        found_nan = False
        if load_fragment:
            for eN_dataset in self.frag_dataset:
                set_name = eN_dataset[0]['set_name']
                if mode == 'train':
                    if (set_name not in VALID_LIST) and (set_name not in TEST_LIST):
                        dataset_list.append(eN_dataset)
                else:
                    if set_name in list_name:
                        dataset_list.append(eN_dataset)
            
            X, Y = [], []
            for eN_dataset in dataset_list:
                eN_list = []
                for dataset in eN_dataset:
                    data = []
                    for key in x_keys:
                        if key in dataset['scaled_statistics'].keys():
                            if math.isnan(dataset['scaled_statistics'][key]):
                                found_nan = True
                                break
                            data.append(dataset['scaled_statistics'][key])
                        else:
                            print("ERROR : No key named " + key)
                    if found_nan:
                        found_nan = False
                        continue
                    if mode == 'train':
                        X.append(data)
                        Y.append(dataset['emotion_number'])
                    else:
                        eN_list.append(data)
                if mode != 'train':
                    X.append(eN_list)
                    Y.append(dataset['emotion_number'])
        # load total dataset
        else:
            for dataset in self.total_dataset:
                set_name = dataset['set_name']
                if mode == 'train':
                    if (set_name not in VALID_LIST) and (set_name not in TEST_LIST):
                        dataset_list.append(dataset)
                else:
                    if set_name in list_name:
                        dataset_list.append(dataset)
            
            X, Y = [], []
            for dataset in dataset_list:
                data = []
                for key in x_keys:
                    if key in dataset['scaled_statistics'].keys():
                        if math.isnan(dataset['scaled_statistics'][key]):
                            found_nan = True
                            break
                        data.append(dataset['scaled_statistics'][key])
                    else:
                        print("ERROR : No key named " + key)
                if found_nan:
                    found_nan = False
                    continue
                X.append(data)
                Y.append(dataset['emotion_number'])
        '''    
        for dataset in self.total_dataset:
            set_name = dataset['set_name']
            if mode == 'train':
                if (set_name not in VALID_LIST) and (set_name not in TEST_LIST):
                    dataset_list.append(dataset)
            else:
                if set_name in list_name:
                    dataset_list.append(dataset)
        '''
        '''
        X, Y = [], []
        if load_fragment:
            for dataset in dataset_list:
                for dataset_piece in dataset['splitted_scaled_feature_data']:
                    data = []
                    for key in x_keys:
                        if key in dataset_piece['statistics'].keys():
                            data.append(dataset_piece['statistics'][key])
                        else:
                            print("ERROR : No key named " + key)
                    X.append(data)
                    Y.append(dataset['emotion_number'])
        else:
            for dataset in dataset_list:
                data = []
                for key in x_keys:
                    if key in dataset['total_scaled_statistics']:
                        data.append(dataset['total_scaled_statistics'][key])
                    else:
                        print("ERROR : No key named " + key)
                X.append(data)
                Y.append(dataset['emotion_number'])
        '''
        return np.array(X), np.array(Y)


class EmotionDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        return self.x[index], self.y[index] - 1
    
    def __len__(self):
        return self.x.shape[0]
            

def get_dataloader(data_path, frag_data_name, total_data_name, feature_keys, test_with_split):
    DL = RawDataLoader(data_path, frag_data_name, total_data_name, test_with_split)
    x_train, y_train = DL.load_dataset('train', feature_keys, load_fragment=True)
    x_valid, y_valid = DL.load_dataset('valid', feature_keys, load_fragment=test_with_split)
    x_test, y_test = DL.load_dataset('test', feature_keys, load_fragment=test_with_split)

    train_set = EmotionDataset(x_train, y_train)
    valid_set = EmotionDataset(x_valid, y_valid)
    test_set = EmotionDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,  shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE,  shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,  shuffle=True, drop_last=False)

    return train_loader, valid_loader, test_loader
