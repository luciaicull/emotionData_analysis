import _pickle as cPickle
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .constants import VALID_LIST, TEST_LIST, STAT_TYPE, BATCH_SIZE

class RawDataLoader(object):
    def __init__(self, data_path, file_name):
        self.path = data_path.joinpath(file_name)
        self.total_dataset = self._load_dict_data()
    
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
        for dataset in self.total_dataset:
            #name = dicts[0]['name'] + '.' + dicts[0]['performer']
            set_name = dataset['set_name']
            if mode == 'train':
                #if (name not in VALID_LIST) and (name not in TEST_LIST):
                if (set_name not in VALID_LIST) and (set_name not in TEST_LIST):
                    data_list.append(dataset)
            else:
                if set_name in list_name :
                    data_list.append(dataset)
        
        x, y = self.make_X_and_Y(data_list, stat, x_keys)
        return x, y

    def make_X_and_Y(self, dataset_list, stat, x_keys):
        X = []
        Y = []
        
        for dataset in dataset_list:
            data = []
            for key in x_keys:
                if key in dataset[stat].keys():
                    data.append(dataset[stat][key])
                else:
                    print("ERROR : No key named " + key)
            X.append(data)
            Y.append(dataset['emotion_number'])

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
            

def get_dataloader(data_path, file_name, feature_keys):
    DL = RawDataLoader(data_path, file_name)
    x_train, y_train = DL.load_dataset('train', STAT_TYPE, feature_keys)
    x_valid, y_valid = DL.load_dataset('valid', STAT_TYPE, feature_keys)
    x_test, y_test = DL.load_dataset('test', STAT_TYPE, feature_keys)

    train_set = EmotionDataset(x_train, y_train)
    valid_set = EmotionDataset(x_valid, y_valid)
    test_set = EmotionDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,  shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE,  shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,  shuffle=True, drop_last=False)

    return train_loader, valid_loader, test_loader
