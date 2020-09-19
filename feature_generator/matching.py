""" 
Utilities for matching process after applying Nakamura Algorithm
"""
import pretty_midi
import csv
import copy
'''
modules for xml note - eN note matching
'''

def read_match_file(match_path):
    f = open(match_path, 'r')
    reader = csv.reader(f, dialect='excel', delimiter='\t')
    match_txt = {'match_list': [], 'missing': []}
    for row in reader:
        if len(row) == 1:
            continue
        elif len(row) == 2:
            dic = {'scoreTime': int(row[0].split(' ')[-1]), 'xmlNoteID': row[1]}
            match_txt['missing'].append(dic)
        else:
            dic = {'midiID': int(row[0]), 'midiStartTime': float(row[1]), 'midiEndTime': float(row[2]), 'pitch': row[3], 'midiOnsetVel': int(row[4]), 'midiOffsetVel': int(
                row[5]), 'channel': int(row[6]), 'matchStatus': int(row[7]), 'scoreTime': int(row[8]), 'xmlNoteID': row[9], 'errorIndex': int(row[10]), 'skipIndex': row[11], 'used': False}
            match_txt['match_list'].append(dic)
    
    return match_txt['match_list'], match_txt['missing']


def match_xml_midi(xml_notes, midi_notes, match_list, missing_xml_list):
    """
    Main method for score xml - performance midi direct matching process

    Parameters
    -----------
    xml_notes : list of xml note object
    midi_notes : list of midi note object
    match_list : lines of match result in _match.txt, in dictionary(dict) format
    missing_xml_list : lines of missing result in _match.txt, in dictionary(dict) format

    Returns
    -----------
    pairs : list of match pair in dictionary format - {'xml': xml_index, 'midi': midi_index}

    """
    index_dict_list = []
    for xml_index, xml_note in enumerate(xml_notes):
        dic = find_matching_midi_note(xml_index, xml_note, match_list, midi_notes)
        index_dict_list.append(dic)
        #print(dic)

    pairs = pair_transformation(xml_notes, midi_notes, index_dict_list)
    return pairs


def find_matching_midi_note(xml_index, xml_note, match_list, midi_notes):
    """
    Match method for one xml note

    Parameters
    -----------
    xml_index : index of xml_note in xml_notes list
    xml_note : xml note object
    midi_notes : list of midi note object
    match_list : lines of match result in _match.txt, in dictionary(dict) format

    Returns
    -----------
    dic : result of match result in dictionary format
    """
    dic = {'match_index': [], 'xml_index': {'idx': xml_index, 'pitch': xml_note.pitch[0]}, 'midi_index': [], 
           'is_trill': False, 'is_ornament': False, 'is_overlapped': xml_note.is_overlapped, 'overlap_xml_index': [], 
           'unmatched': False, 'fixed_trill_idx': []}

    # find correspond match and midi
    score_time = xml_note.note_duration.xml_position
    score_pitch = xml_note.pitch[0]
    for match_index, match in enumerate(match_list):
        if match['xmlNoteID'] != '*':
            if score_time == match['scoreTime'] and score_pitch == match['pitch']:
                dic['match_index'].append(match_index)
                midi_index = find_midi_note_index(midi_notes, match['midiStartTime'], match['midiEndTime'], match['pitch'])

                if midi_index != -1:
                    dic['midi_index'].append(midi_index)
                    match['used'] = True
    
    # find trill midis
    dic = find_trill_midis(dic, match_list, midi_notes)

    # find ornament midis
    dic = find_ornament_midis(dic, score_time, match_list, midi_notes)

    if len(dic['midi_index']) == 0:
        dic['unmatched'] = True
    else:
        for idx in dic['match_index']:
            match_list[idx]['used'] = True

    return dic


def find_trill_midis(dic, match_list, midi_notes):
    """
    find possible trill midi note indices

    Parameters
    -----------
    dic : result of match result in dictionary format
    match_list : lines of match result in _match.txt, in dictionary(dict) format
    midi_notes : list of midi note object
    """
    if len(dic['match_index']) > 1:
        # 미디 여러 개 - xml 하나라서 match가 여러 개 뜰 경우
        dic['is_trill'] = True

        start_idx = dic['match_index'][0]
        end_idx = dic['match_index'][-1]
        match_id = match_list[start_idx]['xmlNoteID']
        pitch = match_list[start_idx]['pitch']

        new_match_idx = []

        # find trill
        trill_pitch = None
        for idx in range(start_idx, end_idx + 1):
            if idx in dic['match_index']:
                continue
            else:
                if (match_list[idx]['xmlNoteID'] == match_id) or (match_list[idx]['errorIndex'] == 3):
                    midi_idx = find_midi_note_index(
                        midi_notes, match_list[idx]['midiStartTime'], match_list[idx]['midiEndTime'], match_list[idx]['pitch'])
                    if midi_idx != -1:
                        dic['midi_index'].append(midi_idx)
                        new_match_idx.append(idx)
                        trill_pitch = match_list[idx]['pitch']
                        if match_list[idx]['xmlNoteID'] != match_id:
                            dic['fixed_trill_idx'].append(midi_idx)
                        match_list[idx]['used'] = True

        # find one prev trill
        prev_idx = start_idx - 1
        prev = match_list[prev_idx]
        if prev['pitch'] == trill_pitch:
            if (prev['xmlNoteID'] == match_id) or (prev['errorIndex'] == 3):
                midi_idx = find_midi_note_index(
                    midi_notes, prev['midiStartTime'], prev['midiEndTime'], prev['pitch'])
                if midi_idx != -1:
                    dic['midi_index'].append(midi_idx)
                    new_match_idx.append(prev_idx)
                    if prev['xmlNoteID'] != match_id:
                        dic['fixed_trill_idx'].append(midi_idx)
                    prev['used'] = True

        dic['match_index'] += new_match_idx
        dic['match_index'].sort()
        prev_midi_index = dic['midi_index']
        dic['midi_index'] = sorted(
            prev_midi_index, key=lambda prev_midi_index: prev_midi_index['idx'])
    
    return dic


def find_ornament_midis(dic, score_time, match_list, midi_notes):
    """
    find possible ornament midi note indices

    Parameters
    -----------
    dic : result of match result in dictionary format
    score_time : score time of xml note
    match_list : lines of match result in _match.txt, in dictionary(dict) format
    midi_notes : list of midi note object
    
    """
    if len(dic['match_index']) > 0:
        match = match_list[dic['match_index'][0]]
        cand_match_idx = [idx for idx, match in enumerate(
            match_list) if match['scoreTime'] == score_time]
        new_match_idx = []
        for cand in cand_match_idx:
            cand_match = match_list[cand]
            if not cand_match['used']:
                if (cand_match['xmlNoteID'] == match['xmlNoteID']):
                    midi_idx = find_midi_note_index(
                        midi_notes, cand_match['midiStartTime'], cand_match['midiEndTime'], match['pitch'], ornament=True)
                    if midi_idx != -1:
                        dic['midi_index'].append(midi_idx)
                        new_match_idx.append(cand)
                        if cand_match['xmlNoteID'] != match['xmlNoteID']:
                            dic['fixed_trill_idx'].append(midi_idx)
                        cand_match['used'] = True
                        dic['is_ornament'] = True
        dic['match_index'] += new_match_idx
        new_match_idx = []
        if len(dic['match_index']) >= 2:
            for cand in cand_match_idx:
                cand_match = match_list[cand]
                if not cand_match['used']:
                    if (cand_match['errorIndex'] == 3):
                        midi_idx = find_midi_note_index(
                            midi_notes, cand_match['midiStartTime'], cand_match['midiEndTime'], match['pitch'], ornament=True)
                        if midi_idx != -1:
                            dic['midi_index'].append(midi_idx)
                            new_match_idx.append(cand)
                            if cand_match['xmlNoteID'] != match['xmlNoteID']:
                                dic['fixed_trill_idx'].append(midi_idx)
                            cand_match['used'] = True
                            dic['is_ornament'] = True

        dic['match_index'] += new_match_idx
        dic['match_index'].sort()
        prev_midi_index = dic['midi_index']
        dic['midi_index'] = sorted(prev_midi_index, key=lambda prev_midi_index: prev_midi_index['idx'])
    
    return dic


def find_midi_note_index(midi_notes, start, end, pitch, ornament=False):
    """
    find corresponding midi note index for one xml note

    Parameters
    -----------
    midi_notes : list of midi note object
    start: midi start time in match_list
    end : midi end time in match_list
    pitch : midi pitch in match_list (in string)
    ornament : whether it's ornament

    Returns
    -----------
    dictionary of midi index and pitch
    """
    pitch = check_pitch(pitch)
    if not ornament:
        for i, note in enumerate(midi_notes):
            if (abs(note.start - start) < 0.001) and (abs(note.end - end) < 0.001) and (note.pitch == pretty_midi.note_name_to_number(pitch)):
                return {'idx': i, 'pitch': pretty_midi.note_number_to_name(note.pitch)}
    else:
        for i, note in enumerate(midi_notes):
            if (abs(note.start - start) < 0.001) and (abs(note.end - end) < 0.001) and (abs(note.pitch - pretty_midi.note_name_to_number(pitch)) <= 2):
                return {'idx': i, 'pitch': pretty_midi.note_number_to_name(note.pitch)}
    return -1


def check_pitch(pitch):
    """
    check string pitch format and fix it

    Parameters
    -----------
    pitch : midi string pitch

    Returns
    -----------
    pitch : midi string pitch
    """
    if len(pitch) == 4:
        base_pitch_num = pretty_midi.note_name_to_number(pitch[0]+pitch[-1])
        if pitch[1:3] == 'bb':
            pitch = pretty_midi.note_number_to_name(base_pitch_num-2)
        if pitch[1:3] == '##':
            pitch = pretty_midi.note_number_to_name(base_pitch_num+2)
    return pitch


def pair_transformation(xml_notes, midi_notes, index_dict_list):
    """
    Transform pair format from index_dict_list to original pair

    Parameters
    -----------
    xml_notes : list of xml note object
    midi_notes : list of midi note object
    index_dict_list 
            : list of dictionary
            {'match_index': [], 'xml_index': {xml_index, xml_note.pitch[0]}, 'midi_index': [], 
             'is_trill': False, 'is_ornament': False, 'is_overlapped': xml_note.is_overlapped, 
             'overlap_xml_index': [], 'unmatched': False, 'fixed_trill_idx': []}
    
    Returns
    -----------
    pairs : list of dictionary
            {'xml': xml_notes[i], 'midi': midi_notes[match_list[i]]}
    """
    pairs = []
    for dic in index_dict_list:
        xml_idx = dic['xml_index']['idx']
        midi_idx_list = dic['midi_index']
        if dic['unmatched']:
            pair = []
        else:
            midi_idx = midi_idx_list[0]['idx']
            pair = {'xml': xml_notes[xml_idx], 'midi': midi_notes[midi_idx]}

        pairs.append(pair)

    return pairs


def make_available_xml_midi_positions(pairs):
    available_pairs = []
    num_pairs = len(pairs)
    for i in range(num_pairs):
        pair = pairs[i]
        if not pair == []:
            xml_note = pair['xml']
            midi_note = pair['midi']
            xml_pos = xml_note.note_duration.xml_position
            time = midi_note.start
            divisions = xml_note.state_fixed.divisions
            if not xml_note.note_duration.is_grace_note:
                pos_pair = {'xml_position': xml_pos, 'time_position': time, 'pitch': xml_note.pitch[1], 'index':i, 'divisions':divisions}
                if xml_note.note_notations.is_arpeggiate:
                    pos_pair['is_arpeggiate'] = True
                else:
                    pos_pair['is_arpeggiate'] = False
                available_pairs.append(pos_pair)

    available_pairs, mismatched_indexes = make_average_onset_cleaned_pair(available_pairs)
    #print('Number of mismatched notes: ', len(mismatched_indexes))
    for index in mismatched_indexes:
        pairs[index] = []

    return pairs, available_pairs

def make_average_onset_cleaned_pair(position_pairs, maximum_qpm=600):
    length = len(position_pairs)
    previous_position = -float("Inf")
    previous_time = -float("Inf")
    previous_index = 0
    # position_pairs.sort(key=lambda x: (x.xml_position, x.pitch))
    cleaned_list = list()
    notes_in_chord = list()
    mismatched_indexes = list()
    for i in range(length):
        pos_pair = position_pairs[i]
        current_position = pos_pair['xml_position']
        current_time = pos_pair['time_position']
        if current_position > previous_position >= 0:
            minimum_time_interval = (current_position - previous_position) / pos_pair['divisions'] / maximum_qpm * 60 + 0.001
        else:
            minimum_time_interval = 0
        if current_position > previous_position and current_time > previous_time + minimum_time_interval:
            if len(notes_in_chord) > 0:
                average_pos_pair = copy.copy(notes_in_chord[0])
                notes_in_chord_cleaned, average_pos_pair['time_position'] = get_average_onset_time(notes_in_chord)
                if len(cleaned_list) == 0 or average_pos_pair['time_position'] > cleaned_list[-1]['time_position'] + (
                        (average_pos_pair['xml_position'] - cleaned_list[-1]['xml_position']) /
                        average_pos_pair['divisions'] / maximum_qpm * 60 + 0.01):
                    cleaned_list.append(average_pos_pair)
                    for note in notes_in_chord:
                        if note not in notes_in_chord_cleaned:
                            # print('the note is far from other notes in the chord')
                            mismatched_indexes.append(note['index'])
                else:
                    # print('the onset is too close to the previous onset', average_pos_pair.xml_position, cleaned_list[-1].xml_position, average_pos_pair.time_position, cleaned_list[-1].time_position)
                    for note in notes_in_chord:
                        mismatched_indexes.append(note['index'])
            notes_in_chord = list()
            notes_in_chord.append(pos_pair)
            previous_position = current_position
            previous_time = current_time
            previous_index = i
        elif current_position == previous_position:
            notes_in_chord.append(pos_pair)
        else:
            # print('the note is too close to the previous note', current_position - previous_position, current_time - previous_time)
            # print(previous_position, current_position, previous_time, current_time)
            mismatched_indexes.append(position_pairs[previous_index]['index'])
            mismatched_indexes.append(pos_pair['index'])

    return cleaned_list, mismatched_indexes


def get_average_onset_time(notes_in_chord_saved, threshold=0.2):
    # notes_in_chord: list of PosTempoPair Dictionary, len > 0
    notes_in_chord = copy.copy(notes_in_chord_saved)
    average_onset_time = 0
    for pos_pair in notes_in_chord:
        average_onset_time += pos_pair['time_position']
        if pos_pair['is_arpeggiate']:
            threshold = 1
    average_onset_time /= len(notes_in_chord)

    # check whether there is mis-matched note
    deviations = list()
    for pos_pair in notes_in_chord:
        dev = abs(pos_pair['time_position'] - average_onset_time)
        deviations.append(dev)
    if max(deviations) > threshold:
        # print(deviations)
        if len(notes_in_chord) == 2:
            del notes_in_chord[0:2]
        else:
            index = deviations.index(max(deviations))
            del notes_in_chord[index]
            notes_in_chord, average_onset_time = get_average_onset_time(notes_in_chord, threshold)

    return notes_in_chord, average_onset_time

'''
modules for e1 midi - eN midi matching
'''
def read_corresp(txtpath):
    file = open(txtpath, 'r')
    reader = csv.reader(file, dialect='excel', delimiter='\t')
    corresp_list = []
    for row in reader:
        if len(row) == 1:
            continue
        temp_dic = {'alignID': row[0], 'alignOntime': row[1], 'alignSitch': row[2], 'alignPitch': row[3], 'alignOnvel': row[4],
                    'refID': row[5], 'refOntime': row[6], 'refSitch': row[7], 'refPitch': row[8], 'refOnvel': row[9]}
        corresp_list.append(temp_dic)

    return corresp_list



def match_midis(ref_notes, perf_notes, corresp_list):
    pairs = []

    for ref_note in ref_notes:
        corresp_dict = find_corresp_dict(corresp_list, 'refOntime', ref_note.start, 'refPitch', ref_note.pitch)
        if corresp_dict == None:
            print("Missing ref")
        perf_note = find_perf_note(perf_notes, corresp_dict)

        pair = {'ref':ref_note, 'perf':perf_note}
        pairs.append(pair)
    
    pairs = sorted(pairs, key=lambda dic: dic['ref'].start)
    return pairs


def find_corresp_dict(corresp_list, onset_key, onset_value, pitch_key, pitch_value):
    for dic in corresp_list:
        if abs(float(dic[onset_key]) - onset_value) < 0.02 and int(dic[pitch_key]) == pitch_value:
            return dic
    return None

def find_perf_note(perf_note_list, dic):
    onset_time = float(dic['alignOntime'])
    pitch = int(dic['alignPitch'])
    for note in perf_note_list:
        if abs(note.start - onset_time) < 0.02 and note.pitch == pitch:
            return note
    return None

