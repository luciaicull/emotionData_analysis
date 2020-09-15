""" 
Utilities for matching process after applying Nakamura Algorithm
"""

import csv

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

    # find ornament midis

    if len(dic['midi_index']) == 0:
        dic['unmatched'] = True
    else:
        for idx in dic['match_index']:
            match_list[idx]['used'] = True

    return dic

def find_midi_note_index(midi_notes, start, end, pitch, ornament=False):
    pass

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

