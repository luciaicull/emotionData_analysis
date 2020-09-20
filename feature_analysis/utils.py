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

def make_X_Y(feature_data):
    x = []
    y = []
    exist_keys = list(feature_data[0]['set'][0]['scaled_stats'].keys())
    feature_key_list = [key for key in exist_keys if ('_diff' in key) or ('_ratio' in key) or ('relative_' in key)]

    for emotion_set in feature_data:
        set_name = emotion_set['name']
        feature_set_list = emotion_set['set']
        for feature_set in feature_set_list:
            emotion_number = feature_set['emotion_number']
            scaled_stats = feature_set['scaled_stats']
            
            data = []
            for key in feature_key_list:
                data.append(scaled_stats[key])
            x.append(data)
            y.append(emotion_number)

    return np.array(x), np.array(y), feature_key_list
