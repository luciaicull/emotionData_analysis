from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import math
from matplotlib import pyplot as plt

from . import feature_utils
from .feature_data_class import RawFeatureDataset

class XmlMidiFeatureExtractor:
    def __init__(self, set_list, feature_list):
        self.set_list = set_list
        self.feature_key_list = feature_list

    def extract_features(self):
        dataset = RawFeatureDataset(self.set_list)

        for set_dict in tqdm(dataset.set_list):
            data_list = set_dict['data_list']

            # get basic features
            for data_class in data_list:
                for feature_key in self.feature_key_list:
                    feat_list = getattr(self, 'extract_'+feature_key)(data_class.performance_data)
                    data_class.feature_data[feature_key] = feat_list
                if data_class.emotion_number == 1:
                    e1_class = data_class

            # get e1-relative, e1-ratio, self-diff features
            for data_class in data_list:
                for feature_key in self.feature_key_list:
                    relative_feature = self._get_relative_feature(e1_class.feature_data[feature_key], data_class.feature_data[feature_key])
                    ratio_feature = self._get_ratio_feature(e1_class.feature_data[feature_key], data_class.feature_data[feature_key])
                    diff_feature = self._get_diff_feature(data_class.feature_data[feature_key])

                    data_class.feature_data['relative_'+feature_key] = relative_feature
                    data_class.feature_data[feature_key+'_ratio'] = ratio_feature
                    data_class.feature_data[feature_key+'_diff'] = diff_feature
        
        #scale feature_data and return
        dataset.scale_dataset()
        return dataset
        
    def _get_relative_feature(self, e1_list, eN_list):
        feature_list = []
        for ref, infer in zip(e1_list, eN_list):
            feature_list.append(infer-ref)
        return feature_list
    
    def _get_ratio_feature(self, e1_list, eN_list):
        feature_list = []
        for ref, infer in zip(e1_list, eN_list):
            if ref != 0:
                feature_list.append(infer / ref)
            else:
                feature_list.append(None)
        return feature_list

    def _get_diff_feature(self, eN_list):
        feature_list = []
        for i, _ in enumerate(eN_list):
            if i == len(eN_list)-1:
                break
            feature_list.append(eN_list[i+1] - eN_list[i])
        return feature_list


    def extract_beat_tempo(self, performance_data):
        beat_positions = performance_data.xml_obj.get_beat_positions()
        tempos = feature_utils._cal_tempo_by_positions(beat_positions, performance_data.valid_position_pairs)
    
        return [feature_utils.get_item_by_xml_position(tempos, note).qpm for note in performance_data.xml_notes]


    def extract_measure_tempo(self, performance_data):
        measure_positions = performance_data.xml_obj.get_measure_positions()
        tempos = feature_utils._cal_tempo_by_positions(measure_positions, performance_data.valid_position_pairs)
        
        return [feature_utils.get_item_by_xml_position(tempos, note).qpm for note in performance_data.xml_notes]


    def extract_velocity(self, performance_data):
        features = []
        prev_velocity = 64
        for pair in performance_data.pairs:
            if pair == []:
                velocity = prev_velocity
            else:
                velocity = pair['midi'].velocity
                prev_velocity = velocity
            features.append(velocity)
        return features

    def extract_original_duration(self, performance_data):
        features = []
        for pair in performance_data.pairs:
            if pair == []:
                duration = 0
            else:
                midi = pair['midi']
                duration = midi.end - midi.start
            features.append(duration)

        return features

    def extract_elongated_duration(self, performance_data):
        features = []
        for pair in performance_data.pairs:
            if pair == []:
                duration = 0
            else:
                midi = pair['midi']
                if midi.elongated_offset_time > midi.end:
                    duration = midi.elongated_offset_time - midi.start
                else:
                    duration = midi.end - midi.start
            features.append(duration)

        return features
    
    def extract_onset_deviation(self, performance_data):
        beat_positions = performance_data.xml_obj.get_beat_positions()
        tempos = feature_utils._cal_tempo_by_positions(beat_positions, performance_data.valid_position_pairs)

        features = []
        for pair in performance_data.pairs:
            if pair == []:
                deviation = 0
            else:
                note = pair['xml']
                midi=  pair['midi']
                tempo = feature_utils.get_item_by_xml_position(tempos, note)
                tempo_start = tempo.time_position

                passed_duration = note.note_duration.xml_position - tempo.xml_position
                actual_passed_second = midi.start - tempo_start
                actual_passed_duration = actual_passed_second / 60 * tempo.qpm * note.state_fixed.divisions

                xml_pos_diff = actual_passed_duration - passed_duration
                pos_diff_in_quarter_note = xml_pos_diff / note.state_fixed.divisions
                deviation = pos_diff_in_quarter_note
            features.append(deviation)
        return features

    
    def extract_onset_timing(self, performance_data):
        features = []
        '''
        for note in performance_data.midi_notes:
            features.append(note.start)
        '''
        for pair in performance_data.pairs:
            if pair == []:
                timing = -1
            else:
                timing = pair['midi'].start
            features.append(timing)
        return features
    
    def extract_IOI(self, performance_data):
        features = []
        onset_timings = self.extract_onset_timing(performance_data)
        '''
        for i, onset in enumerate(onset_timings):
            if i < len(onset_timings)-1:
                features.append(onset_timings[i+1] - onset_timings[i])
        '''
        for i, _ in enumerate(onset_timings):
            if i < len(onset_timings) - 1:
                if onset_timings[i+1] == -1 or onset_timings[i] == -1:
                    ioi = 0
                else:
                    ioi = onset_timings[i+1] - onset_timings[i]
            features.append(ioi)
        return features

        

class MidiMidiFeatureExtractor:
    def __init__(self, set_list, feature_list):
        self.set_list = set_list
        self.feature_key_list = feature_list
    
    def extract_features(self):
        feature_data = []

        for set_dict in tqdm(self.set_list):
            set_name = set_dict['name']
            set_list = set_dict['list']

            feature_set_dict = {'name':set_name, 'set':[]}
            for performance_set in set_list:
                emotion_number = performance_set['emotion_number']
                midi_note_pair_list = performance_set['pairs']

                feature_dict = self._init_feature_dict()
                for pairs in midi_note_pair_list:
                    for feature_key in self.feature_key_list:
                        try:
                            feat_list, relative_feat_list, ratio_feat_list = getattr(self, 'extract_'+feature_key)(pairs)
                            feature_dict[feature_key].append(feat_list)
                            if relative_feat_list is not None:
                                feature_dict['relative_'+feature_key].append(relative_feat_list)
                            if ratio_feat_list is not None:
                                feature_dict[feature_key+'_ratio'].append(ratio_feat_list)
                        except:
                            print(set_name, emotion_number, feature_key)
                
                stats = self._get_stats(feature_dict)
                dic = {'emotion_number':emotion_number, 'feature_dict':feature_dict, 'stats':stats}
                feature_set_dict['set'].append(dic)
            
            feature_set_dict['set'] = self._add_normalized_stats(feature_set_dict['set'])
            feature_data.append(feature_set_dict)
        
        return feature_data
    

    def _get_stats(self, feature_dict):
        # feature dict = {'key1':feat_list, 'key2':feat_list, ...}
        stats = dict()
        for key in feature_dict.keys():
            stats[key+'_mean'] = [np.mean(feat_list) for feat_list in feature_dict[key]]
            stats[key+'_std'] = [np.std(feat_list) for feat_list in feature_dict[key]]
            stats[key+'_skew'] = [skew(feat_list) for feat_list in feature_dict[key]]
            stats[key+'_kurt'] = [kurtosis(feat_list) for feat_list in feature_dict[key]]
        return stats


    def _add_normalized_stats(self, set_dict_list):
        # set_dict_list = [e1 dict, e2 dict, e3 dict, e4 dict, e5 dict]
        # eN dict = {'emotion_number':emotion_number, 'feature_dict':feature_dict, 'stats':stats}
        # stats = {'key1_mean':[fragment1 value, fragment2 value, ..], 'key1_std':[fragment1 value, fragment2 value, ..], ...}
        stat_keys = list(set_dict_list[0]['stats'].keys())
        num_fragment = len(set_dict_list[0]['stats'][stat_keys[0]])

        total_stat_list = []  # (emotion_num x num_fragment, num_feature)
        for eN_dict in set_dict_list:
            stat_list = [eN_dict['stats'][k] for k in stat_keys]
            stat_list = np.array(stat_list).T
            total_stat_list += list(stat_list)
        
        scaler = StandardScaler()
        scaler.fit(total_stat_list)

        scaled_total_stat_list = scaler.transform(total_stat_list)  # (emotion_num x num_fragment, num_feature)
        for i, eN_dict in enumerate(set_dict_list):
            scaled_stat_list = scaled_total_stat_list[i*num_fragment:(i+1)*num_fragment]
            scaled_stat_list = list(np.array(scaled_stat_list).T) # (num_feature, num_fragment)

            eN_dict['scaled_stats'] = dict()
            for i, k in enumerate(stat_keys):
                eN_dict['scaled_stats'][k] = scaled_stat_list[i]
        
        return set_dict_list

                        
    def _init_feature_dict(self):
        feature_dict = dict()
        for feature_key in self.feature_key_list:
            feature_dict[feature_key] = []
            if feature_key is not 'interval':
                feature_dict['relative_'+feature_key] = []
                feature_dict[feature_key+'_ratio'] = []
        return feature_dict

        
    def _get_relative_feature(self, e1_list, eN_list):
        feature_list = []
        for ref, infer in zip(e1_list, eN_list):
            feature_list.append(infer-ref)
        return feature_list

    
    def _get_ratio_feature(self, e1_list, eN_list):
        feature_list = []
        for ref, infer in zip(e1_list, eN_list):
            if ref != 0:
                feature_list.append(infer / ref)
        return feature_list


    def extract_velocity(self, midi_note_pair_list):
        eN_velocity = self._get_velocity(midi_note_pair_list, 'perf')
        e1_velocity = self._get_velocity(midi_note_pair_list, 'ref')

        relative_velocity = self._get_relative_feature(e1_velocity, eN_velocity)
        ratio_velocity = self._get_ratio_feature(e1_velocity, eN_velocity)

        return eN_velocity, relative_velocity, ratio_velocity


    def _get_velocity(self, midi_note_pair_list, key):
        feature_list = []

        prev_velocity = 64
        for pair in midi_note_pair_list:
            note = pair[key]
            if note is None:
                velocity = prev_velocity
            else:
                velocity = note.velocity
                prev_velocity = velocity
            feature_list.append(velocity)
        return feature_list


    def extract_original_duration(self, midi_note_pair_list):
        eN_original_duration = self._get_original_duration(midi_note_pair_list, 'perf')
        e1_original_duration = self._get_original_duration(midi_note_pair_list, 'ref')

        relative_original_duration = self._get_relative_feature(e1_original_duration, eN_original_duration)
        ratio_original_duration = self._get_ratio_feature(e1_original_duration, eN_original_duration)

        return eN_original_duration, relative_original_duration, ratio_original_duration


    def _get_original_duration(self, midi_note_pair_list, key):
        feature_list = []
        for pair in midi_note_pair_list:
            note = pair[key]
            if note is None:
                duration = 0
            else:
                duration = note.end - note.start
            feature_list.append(duration)
        return feature_list

            
    def extract_elongated_duration(self, midi_note_pair_list):
        eN_elongated_duration = self._get_elongated_duration(midi_note_pair_list, 'perf')
        e1_elongated_duration = self._get_elongated_duration(midi_note_pair_list, 'ref')

        relative_elongated_duration = self._get_relative_feature(e1_elongated_duration, eN_elongated_duration)
        ratio_elongated_duration = self._get_ratio_feature(e1_elongated_duration, eN_elongated_duration)

        return eN_elongated_duration, relative_elongated_duration, ratio_elongated_duration


    def _get_elongated_duration(self, midi_note_pair_list, key):
        feature_list = []
        for pair in midi_note_pair_list:
            note = pair[key]
            if note is None:
                duration = 0
            else:
                if note.elongated_offset_time > note.end:
                    duration = note.elongated_offset_time - note.start
                else:
                    duration = note.end - note.start
            feature_list.append(duration)
        return feature_list


    def extract_interval(self, midi_note_pair_list):
        # interval between two note in eN note which is 1 second between aligned e1 notes
        interval_sec = 1
        interval_list = []
        cur_index = 0
        #cur_pair = midi_note_pair_list[cur_index]
        while True:
            while (midi_note_pair_list[cur_index]['perf'] is None) and (cur_index < len(midi_note_pair_list)-1):
                cur_index += 1

            if cur_index >= len(midi_note_pair_list)-1:
                break
            
            cur_pair = midi_note_pair_list[cur_index]
            cur_sec = cur_pair['ref'].start
            next_sec = cur_sec + interval_sec
            for i in range(cur_index, len(midi_note_pair_list)):
                pair = midi_note_pair_list[i]

                if pair['ref'].start < next_sec:
                    continue
                else:
                    if pair['perf'] is None:
                        if (i == len(midi_note_pair_list) - 1) or (midi_note_pair_list[i+1]['perf'] is None):
                            pair = midi_note_pair_list[i-1]
                        else:
                            pair = midi_note_pair_list[i+1]
                    
                    if pair['perf'] is None:
                        break
                        
                    interval = pair['perf'].start - cur_pair['perf'].start
                    interval_list.append(interval)
                    break

            cur_index = i
            cur_pair = pair
        
        return interval_list, None, None


