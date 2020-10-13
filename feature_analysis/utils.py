import _pickle as cPickle
import pickle
import numpy as np

def save_datafile(path, name, data):
    with open(path.joinpath(name), "wb") as f:
        pickle.dump(data, f, protocol=2)


def load_datafile(path, name):
    with open(path.joinpath(name), 'rb') as f:
        u = cPickle.Unpickler(f)
        data = u.load()
    return data

def make_X_Y(total_dataset):
    x = []
    y = []
    feature_key_list = [key for key in list(total_dataset[0]['total_scaled_statistics'].keys())
                            if ('_diff' in key) or ('_ratio' in key) or ('relative_' in key)]

    for feature_data in total_dataset:
        emotion_number = feature_data['emotion_number']
        total_scaled_statistics = feature_data['total_scaled_statistics']

        data = []
        for key in feature_key_list:
            data.append(total_scaled_statistics[key])
        x.append(data)
        y.append(emotion_number)
    return np.array(x), np.array(y), feature_key_list
