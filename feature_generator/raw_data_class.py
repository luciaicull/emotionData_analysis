from alignment import MidiMidiAlignmentTool
import matching
from midi_utils import midi_utils


class MidiMidiDataset:
    def __init__(self, path):
        self.path = path        # type=Path()

        '''
        Main Variable
        # set_list : 
        '''
        self.set_list = self.load_dataset()

    def load_dataset(self):
        set_list = []

        midi_file_path_list = sorted(self.path.glob('*.E[1-5].mid'))
        file_set_dict = dict()
        
        for midi_file_path in midi_file_path_list:
            file_name = midi_file_path.name[:-len('.E0.mid')]
            if file_name not in file_set_dict.keys():
                file_set_dict[file_name] = []
            file_set_dict[file_name].append(midi_file_path)
        
        for set_name in file_set_dict.keys():
            performance_set = PerformanceSet(file_set_dict[set_name])
            
            set_dict = {'name':set_name, 'list':performance_set.performance_set_list}
            set_list.append(set_dict)

        return set_list



class PerformanceSet:
    def __init__(self, performance_set_path_list):
        self.ref_path = None  # original emotion performance
        self.infer_path_list = []  # e1 ~ e5 emotion performance list

        self.path_dict_list = self._check_alignment(performance_set_path_list)

        '''
        Main Variable
        # performance_set_list
             : list of performance data dictionary from PerformanceData class
               performance data -> {emotion number, note pair}
        '''
        self.performance_set_list = self._make_performance_set()
    
    def _check_alignment(self, performance_set_path_list):
        
        for perf_path in performance_set_path_list:
            if '.E1.mid' in perf_path.name:
                self.ref_path = perf_path
            
            self.infer_path_list.append(perf_path)

        tool = MidiMidiAlignmentTool(self.ref_path, self.infer_path_list)
        corresp_file_path_list = tool.align()

        self.infer_path_list = sorted(self.infer_path_list)
        corresp_file_path_list = sorted(corresp_file_path_list)

        path_dict_list = []

        for midi, corresp in zip(self.infer_path_list, corresp_file_path_list):
            path_dict = {'midi_path':midi, 'corresp_path': corresp}
            path_dict_list.append(path_dict)

        return path_dict_list
    
    def _make_performance_set(self):
        performance_set = []

        for path_dict in self.path_dict_list:
            data = PerformanceData(self.ref_path, path_dict['midi_path'], path_dict['corresp_path'])
            
            emotion_number = data.emotion_number
            pairs = data.pairs
            
            performance_data = {'emotion_number':emotion_number, 'pairs':pairs}
            performance_set.append(performance_data)

        return performance_set
            
'''
class for E1 note - E[1-5] note pairs
'''
class PerformanceData:
    def __init__(self, ref_path, midi_path, corresp_path):
        self.ref_path = ref_path
        self.midi_path = midi_path

        self.ref_notes = self._get_midi(str(ref_path))
        self.perf_notes = self._get_midi(str(midi_path))
        
        self.corresp = matching.read_corresp(corresp_path)

        '''
        Main Variable
        # pairs : list of pair
                  pair -> {'ref': E1 note, 'perf': corresponding E[1-5] note}
        # emotion number : e1(original), e2(sad), e3(relaxed), e4(happy), e5(anger)
        '''
        self.pairs = matching.match_midis(self.ref_notes, self.perf_notes, self.corresp)
        self.emotion_number = self._get_emotion_number(midi_path.name)
        
    def _get_midi(self, path):
        midi = midi_utils.to_midi_zero(path, save_name='tmp.mid')
        midi = midi_utils.add_pedal_inf_to_notes(midi)
        midi_notes = midi.instruments[0].notes

        return midi_notes

    def _get_emotion_number(self, name):
        emotion_name = name.split('.')[-2] # '~~~.E1.mid'
        return int(emotion_name[1])
    
