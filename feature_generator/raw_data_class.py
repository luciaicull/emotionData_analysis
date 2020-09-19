
from .alignment import MidiMidiAlignmentTool, XmlMidiAlignmentTool
from . import matching, xml_utils
from .midi_utils import midi_utils
from .musicxml_parser import MusicXMLDocument

from tqdm import tqdm
from abc import abstractmethod

class Dataset:
    def __init__(self, path, split=0):
        self.path = path        # type=Path()
        self.split = split
        '''
        Main Variable
        # set_list : list of set_dict
                     set_dict -> {performance set name, performance set list}
        '''
        self.set_list = self.load_dataset()

    @classmethod
    @abstractmethod
    def load_dataset(self):
        raise NotImplementedError

    

class XmlMidiDataset(Dataset):
    def __init__(self, path, split=0):
        super().__init__(path, split)
    
    def load_dataset(self):
        set_list = []

        xml_file_path_list = sorted(self.path.glob('*.xml'))
        midi_file_path_list = sorted(self.path.glob('*.E[1-5].mid'))
        file_set_dict = dict()

        for midi_file_path in midi_file_path_list:
            file_name = midi_file_path.name[:-len('.E0.mid')]
            if file_name not in file_set_dict.keys():
                file_set_dict[file_name] = {'score': None, 
                                            'emotion_midi_list':[]}
            file_set_dict[file_name]['emotion_midi_list'].append(midi_file_path)
        
        for xml_file_path in xml_file_path_list:
            score_name = xml_file_path.name[:-len('.xml')]
            for file_name in file_set_dict.keys():
                if score_name in file_name:
                    file_set_dict[file_name]['score'] = xml_file_path

        for set_name in tqdm(file_set_dict.keys()):
            #print(set_name)
            performance_set = XmlMidiPerformanceSet(file_set_dict[set_name], self.split)
            set_dict = {'name': set_name,
                        'list': performance_set.performance_set_list}
            set_list.append(set_dict)
        
        return set_list


class MidiMidiDataset(Dataset):
    def __init__(self, path, split=0):
        super().__init__(path, split)

    def load_dataset(self):
        set_list = []

        midi_file_path_list = sorted(self.path.glob('*.E[1-5].mid'))
        file_set_dict = dict()
        
        for midi_file_path in midi_file_path_list:
            file_name = midi_file_path.name[:-len('.E0.mid')]
            if file_name not in file_set_dict.keys():
                file_set_dict[file_name] = []
            file_set_dict[file_name].append(midi_file_path)
        
        for set_name in tqdm(file_set_dict.keys()):
            performance_set = MidiMidiPerformanceSet(file_set_dict[set_name], self.split)
            
            set_dict = {'name':set_name, 
                        'list':performance_set.performance_set_list}
            set_list.append(set_dict)

        return set_list

class PerformanceSet:
    def __init__(self, performance_set_paths, split=0):
        # determine whether split performance into {split} seconds or not
        self.split = split

        self.ref_path = None  # original emotion performance
        self.infer_path_list = []  # e1 ~ e5 emotion performance list
    
    @classmethod
    @abstractmethod
    def _check_alignment(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _make_performance_set(self):
        raise NotImplementedError


class MidiMidiPerformanceSet(PerformanceSet):
    def __init__(self, performance_set_paths, split=0):
        super().__init__(performance_set_paths, split)
        self.path_dict_list = self._check_alignment(performance_set_paths)

        '''
        Main Variable
        # performance_set_list
             : list of performance data dictionary from PerformanceData class
               performance data -> {emotion number, note pair}
        '''
        self.performance_set_list = self._make_performance_set()


    def _check_alignment(self, performance_set_paths):
        
        for perf_path in performance_set_paths:
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
        performance_set_list = []

        for path_dict in self.path_dict_list:
            data = MidiMidiPerformanceData(self.ref_path, path_dict['midi_path'], path_dict['corresp_path'])
            
            emotion_number = data.emotion_number
            pairs = self._split_pairs(data.pairs)

            performance_data = {'emotion_number':emotion_number, 'pairs':pairs}
            performance_set_list.append(performance_data)

        return performance_set_list
    

    def _split_pairs(self, pairs):
        # split data with {self.split} second
        
        if self.split == 0:
            return [pairs]
        
        splitted_pairs = []
        cur_index = 0
        cur_time = pairs[cur_index]['ref'].start
        while True:
            if cur_index >= len(pairs)-1:
                break

            pair_fragment = []
            next_frag_time = cur_time + self.split
            for i in range(cur_index, len(pairs)):
                dic = pairs[i]
                note = dic['ref']
                if cur_time <= note.start < next_frag_time:
                    pair_fragment.append(dic)
                else:
                    break
            splitted_pairs.append(pair_fragment)
            cur_time = note.start
            cur_index = i

        min_fragment_length = len(sorted(splitted_pairs[:-1], key=len)[0])
        if len(splitted_pairs[-1]) < min_fragment_length/2:
            splitted_pairs[-2] += splitted_pairs[-1]
            del splitted_pairs[-1]

        return splitted_pairs


class XmlMidiPerformanceSet(PerformanceSet):
    def __init__(self, performance_set_paths, split=0):
        super().__init__(performance_set_paths, split)

        # score xml path
        self.ref_path = performance_set_paths['score']
        # e1 ~ e5 emotion performance list
        self.infer_path_list = performance_set_paths['emotion_midi_list']
        
        self.path_dict_list = self._check_alignment()

        '''
        Main Variable
        # performance_set_list
             : list of performance data dictionary from PerformanceData class
               performance data -> {emotion number, note pair}
        '''
        self.performance_set_list = self._make_performance_set()


    def _check_alignment(self):
        tool = XmlMidiAlignmentTool(self.ref_path, self.infer_path_list)
        match_file_path_list = tool.align()

        self.infer_path_list = sorted(self.infer_path_list)
        match_file_path_list = sorted(match_file_path_list)

        path_dict_list = []

        for midi, match in zip(self.infer_path_list, match_file_path_list):
            path_dict = {'midi_path': midi, 'match_path': match}
            path_dict_list.append(path_dict)

        return path_dict_list

    def _make_performance_set(self):
        performance_set_list = []
        #print(self.infer_path_list[0].name[:-len('.E0.mid')])
        for path_dict in self.path_dict_list:
            data = XmlMidiPerformanceData(self.ref_path, path_dict['midi_path'], path_dict['match_path'])

            performance_set_list.append(data)

        return performance_set_list
    


'''
class for ref note - E[1-5] note pairs
'''
class PerformanceData:
    def __init__(self, ref_path, midi_path, txt_path):
        self.ref_path = ref_path
        self.midi_path = midi_path
        self.txt_path = txt_path

        
    def _get_midi(self, path):
        midi = midi_utils.to_midi_zero(path, save_name='tmp.mid')
        midi = midi_utils.add_pedal_inf_to_notes(midi)
        midi_notes = midi.instruments[0].notes

        return midi_notes

    def _get_emotion_number(self, name):
        emotion_name = name.split('.')[-2] # '~~~.E1.mid'
        return int(emotion_name[1])
    
    @classmethod
    @abstractmethod
    def _get_pairs(self):
        raise NotImplementedError
    

class XmlMidiPerformanceData(PerformanceData):
    def __init__(self, ref_path, midi_path, txt_path):
        super().__init__(ref_path, midi_path, txt_path)
        
        # need for feature extraction
        self.xml_obj = MusicXMLDocument(str(self.ref_path))
        self.xml_notes = self._get_xml_notes()
        '''
        Main Variable
        # pairs : list of pair
                  pair -> {'ref': xml note, 'perf': corresponding E[1-5] note}
        # emotion number : e1(original), e2(sad), e3(relaxed), e4(happy), e5(anger)
        '''
        self.pairs, self.valid_position_pairs, self.missing_pair_num = self._get_pairs()
        self.emotion_number = self._get_emotion_number(midi_path.name)
    
    def _get_xml_notes(self):
        notes, _ = self.xml_obj.get_notes()
        directions = self.xml_obj.get_directions()
        time_signatures = self.xml_obj.get_time_signatures()

        xml_notes = xml_utils.apply_directions_to_notes(notes, directions, time_signatures)

        return xml_notes

    def _get_pairs(self):
        # TODO
        # need xml_notes, xml_beat_poisitions, midi_notes
        # or just xml-midi note pair and xml object?
        midi_notes = self._get_midi(str(self.midi_path))
        match, missing = matching.read_match_file(self.txt_path)
        
        pairs = matching.match_xml_midi(self.xml_notes, midi_notes, match, missing)
        pairs, valid_position_pairs = matching.make_available_xml_midi_positions(pairs)
        # split => after feature extraction..

        count = 0
        for pair in pairs:
            if pair == []:
                count += 1
        #print(str(count), ' / ', str(len(self.xml_notes)))

        return pairs, valid_position_pairs, count


class MidiMidiPerformanceData(PerformanceData):
    def __init__(self, ref_path, midi_path, txt_path):
        super().__init__(ref_path, midi_path, txt_path)

        '''
        Main Variable
        # pairs : list of pair
                  pair -> {'ref': E1 note, 'perf': corresponding E[1-5] note}
        # emotion number : e1(original), e2(sad), e3(relaxed), e4(happy), e5(anger)
        '''
        self.pairs = self._get_pairs()
        self.emotion_number = self._get_emotion_number(midi_path.name)
    
    def _get_pairs(self):
        ref_notes = self._get_midi(str(self.ref_path))
        perf_notes = self._get_midi(str(self.midi_path))
        corresp = matching.read_corresp(self.txt_path)
        return matching.match_midis(ref_notes, perf_notes, corresp)
