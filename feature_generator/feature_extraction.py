
class Extractor:
    def __init__(self, set_list, feature_list):
        self.set_list = set_list
        self.feature_list = feature_list
    
    def extract_features(self):
        for set_dict in self.set_list:
            set_name = set_dict['name']
            set_list = set_dict['list']

            for performance_set in set_list:
                emotion_number = performance_set['emotion_number']
                midi_note_pair_list = performance_set['pairs']

                for feature_key in self.feature_list:
                    feat_list, relative_feat_list, ratio_feat_list = getattr(self, 'extract_'+feature_key)(midi_note_pair_list)
    
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
        pairs = sorted(midi_note_pair_list, key=lambda dic: dic['ref'].start)
        for pair in pairs:
            e1_note = pair['ref']
            eN_note = pair['perf']
            

