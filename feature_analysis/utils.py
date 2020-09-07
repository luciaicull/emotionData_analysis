import _pickle as cPickle
import pickle

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
    feature_key_list = feature_data[0]['set'][0]['scaled_stats'].keys()

    for emotion_set in feature_data:
        set_name = emotion_set['name']
        eN_feature_set_list = emotion_set['set']
        for eN_feature_set in eN_feature_set_list:
            emotion_number = eN_feature_set['emotion_number']
            scaled_stats = eN_feature_set['scaled_stats']
            fragment_num = len(list(scaled_stats.values())[0])
            
            for i in range(0, fragment_num):
                data = []
                for key in feature_key_list:
                    data.append(scaled_stats[key][i])
                x.append(data)
                y.append(emotion_number)

    return x, y, feature_key_list
