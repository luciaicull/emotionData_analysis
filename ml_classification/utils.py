import _pickle as cPickle
import pickle
import numpy as np

from .constants import TEST_LIST, MIDIMIDI_FEATURE_KEYS

def save_datafile(path, name, data):
    with open(path.joinpath(name), "wb") as f:
        pickle.dump(data, f, protocol=2)


def load_datafile(path, name):
    with open(path.joinpath(name), 'rb') as f:
        u = cPickle.Unpickler(f)
        data = u.load()
    return data


def split_train_test(feature_data):
    train_data = []
    test_data = []

    for emotion_set in feature_data:
        set_name = emotion_set['name']
        eN_feature_set_list = emotion_set['set']
        if set_name in TEST_LIST:
            test_data.append(emotion_set)
        else:
            train_data.append(emotion_set)

    total_data = train_data + test_data

    return total_data, train_data, test_data


def make_X_Y(feature_data):
    x = []
    y = []

    for emotion_set in feature_data:
        set_name = emotion_set['name']
        eN_feature_set_list = emotion_set['set']
        for eN_feature_set in eN_feature_set_list:
            emotion_number = eN_feature_set['emotion_number']
            scaled_stats = eN_feature_set['scaled_stats']
            fragment_num = len(list(scaled_stats.values())[0])

            for i in range(0, fragment_num):    # in no_split, fragment_num=1
                data = []
                for key in MIDIMIDI_FEATURE_KEYS:
                    data.append(scaled_stats[key][i])
                x.append(data)
                y.append(emotion_number)

    return np.array(x), np.array(y)
