
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
                    feature_list = getattr(self, 'get_'+feature_key)(midi_note_pair_list)
    
    def get_velocity(self, midi_note_pair_list):
        feature_list = []

        prev_velocity = 64
        for pair in midi_note_pair_list:
            ref_note = pair['ref']
            perf_note = pair['perf']
            if perf_note is None:
                velocity = prev_velocity
            else:
                velocity = perf_note.velocity
                prev_velocity = velocity
            feature_list.append(velocity)
        return feature_list

    def get_original_duration(self, midi_note_pair_list):
        features = []
        for pair in midi_note_pair_list:
            
    
    def get_elongated_duration(self, midi_note_pair_list):
        pass

