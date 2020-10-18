import _pickle as cPickle
import pickle
import numpy as np
import math

from .constants import TEST_LIST

def save_datafile(path, name, data):
    with open(path.joinpath(name), "wb") as f:
        pickle.dump(data, f, protocol=2)


def load_datafile(path, name):
    with open(path.joinpath(name), 'rb') as f:
        u = cPickle.Unpickler(f)
        data = u.load()
    return data


def split_train_test(total_feature_data):
    train_data = []
    test_data = []

    #for emotion_set in feature_data:
    for feature_data in total_feature_data:
        #set_name = emotion_set['name']
        set_name = feature_data['set_name']
        if set_name in TEST_LIST:
            #test_data.append(emotion_set)
            test_data.append(feature_data)
        else:
            #train_data.append(emotion_set)
            train_data.append(feature_data)

    total_data = train_data + test_data

    return total_data, train_data, test_data

def make_X_Y_not_splitted(total_dataset, feature_keys):
    x = []
    y = []

    for dataset in total_dataset:
        emotion_number = dataset['emotion_number']

        data = []
        for key in feature_keys:
            data.append(dataset['scaled_statistics'][key])
        x.append(data)
        y.append(emotion_number)

    return np.array(x), np.array(y)
'''
def make_X_Y(feature_data):
    x = []
    y = []
    feature_key_list = XMLMIDI_FEATURE_KEYS 
    #feature_key_list = MIDIMIDI_FEATURE_KEYS
    
    for emotion_set in feature_data:
        set_name = emotion_set['name']
        # eN_feature_set_list = emotion_set['set']
        eN_feature_set_list = emotion_set['splitted_set']
        for eN_feature_set in eN_feature_set_list:
            emotion_number = eN_feature_set['emotion_number']
            scaled_stats = eN_feature_set['scaled_stats']
            #fragment_num = len(list(scaled_stats.values())[0])

            #for i in range(0, fragment_num):    # in no_split, fragment_num=1
            data = []
            for key in feature_key_list:
                #data.append(scaled_stats[key][i])
                data.append(scaled_stats[key])
            x.append(data)
            y.append(emotion_number)

    return np.array(x), np.array(y)

def make_X_Y_for_xmlmidi(feature_data):
    x = []
    y = []
    info = []
    feature_key_list = XMLMIDI_FEATURE_KEYS

    for emotion_set in feature_data:
        set_name = emotion_set['name']
        feature_set_list = emotion_set['splitted_set']
        for feature_set in feature_set_list:
            emotion_number = feature_set['emotion_number']
            scaled_stats = feature_set['scaled_stats']

            detect_NaN = False
            data = []
            for key in feature_key_list:
                if math.isnan(scaled_stats[key]):
                    detect_NaN = True
                    break
                else:
                    data.append(scaled_stats[key])
            if detect_NaN:
                continue
            else:
                x.append(data)
                y.append(emotion_number)
                if 'bucket_index' in feature_set.keys():
                    info.append((set_name, feature_set['bucket_index'], feature_set['total_bucket']))
    return np.array(x), np.array(y), info
'''