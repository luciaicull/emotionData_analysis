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


def print_total_result(total_result):
    for key in total_result.keys():
        print(key)
        df = pd.DataFrame(total_result[key])
        print(df)


def get_total_result(predicted, answer):
    result = get_result_num(predicted, answer)

    total_result = dict()
    # 1. Single emotion -> Single emotion accuracy
    ratio_result = get_ratio_result(result)
    total_result['single_to_single'] = ratio_result

    # 2. Single emotion -> Arousal accuracy
    e2_to_HA = ratio_result[1][3] + ratio_result[1][4]
    e2_to_LA = ratio_result[1][1] + ratio_result[1][2]
    e3_to_HA = ratio_result[2][3] + ratio_result[2][4]
    e3_to_LA = ratio_result[2][2] + ratio_result[2][1]
    e4_to_HA = ratio_result[3][3] + ratio_result[3][4]
    e4_to_LA = ratio_result[3][1] + ratio_result[3][2]
    e5_to_HA = ratio_result[4][4] + ratio_result[4][3]
    e5_to_LA = ratio_result[4][1] + ratio_result[4][2]
    total_result['single_to_arousal'] = [[e2_to_HA, e2_to_LA],
                                        [e3_to_HA, e3_to_LA],
                                        [e4_to_HA, e4_to_LA],
                                        [e5_to_HA, e5_to_LA]]

    # 3. Single emotion -> Valence accuracy
    e2_to_PV = ratio_result[1][1] + ratio_result[1][3]
    e2_to_NV = ratio_result[1][2] + ratio_result[1][4]
    e3_to_PV = ratio_result[2][1] + ratio_result[2][3]
    e3_to_NV = ratio_result[2][2] + ratio_result[2][4]
    e4_to_PV = ratio_result[3][3] + ratio_result[3][1]
    e4_to_NV = ratio_result[3][2] + ratio_result[3][4]
    e5_to_PV = ratio_result[4][1] + ratio_result[4][3]
    e5_to_NV = ratio_result[4][4] + ratio_result[4][2]
    total_result['single_to_valence'] = [[e2_to_PV, e2_to_NV],
                                        [e3_to_PV, e3_to_NV],
                                        [e4_to_PV, e4_to_NV],
                                        [e5_to_PV, e5_to_NV]]

    # 4. Arousal -> Arousal accuracy
    HA_to_HA = (result[3][3]+result[3][4]+result[4][3] +
                result[4][4]) / (sum(result[3]) + sum(result[4]))
    HA_to_LA = (result[3][1]+result[3][2]+result[4][1] +
                result[4][2]) / (sum(result[3]) + sum(result[4]))
    LA_to_LA = (result[1][1]+result[1][2]+result[2][1] +
                result[2][2]) / (sum(result[1])+sum(result[2]))
    LA_to_HA = (result[1][3]+result[1][4]+result[2][3] +
                result[2][4]) / (sum(result[1])+sum(result[2]))
    total_result['arousal_to_arousal'] = [[HA_to_HA, HA_to_LA],
                                        [LA_to_HA, LA_to_LA]]

    # 5. Valence -> Valence accuracy
    PV_to_PV = (result[1][1]+result[1][3]+result[3][1] +
                result[3][3]) / (sum(result[1]) + sum(result[3]))
    PV_to_NV = (result[1][2]+result[1][4]+result[3][2] +
                result[3][4]) / (sum(result[1]) + sum(result[3]))
    NV_to_NV = (result[2][2]+result[2][4]+result[4][2] +
                result[4][4]) / (sum(result[2]) + sum(result[4]))
    NV_to_PV = (result[2][1]+result[2][3]+result[4][1] +
                result[4][3]) / (sum(result[2]) + sum(result[4]))
    total_result['valence_to_valence'] = [[PV_to_PV, PV_to_NV],
                                        [NV_to_PV, NV_to_NV]]

    return total_result

def get_result_num(predicted, answer):
    result = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
    for pred, ans in zip(predicted, answer):
        result[ans-1][pred-1] += 1
    return result

def get_ratio_result(result):
    ratio_result = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
    for i, emotion in enumerate(result):
        for j, pred_num in enumerate(emotion):
            ratio_result[i][j] = pred_num/sum(emotion)
        
    return ratio_result
