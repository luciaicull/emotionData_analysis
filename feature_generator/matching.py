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

def match_midis(ref_notes, perf_notes, corresp_list):
    pairs = []

    for ref_note in ref_notes:
        corresp_dict = find_corresp_dict(corresp_list, 'refOntime', ref_note.start, 'refPitch', ref_note.pitch)
        if corresp_dict == None:
            print("Missing ref")
        perf_note = find_perf_note(perf_notes, corresp_dict)

        pair = {'ref':ref_note, 'perf':perf_note}
        pairs.append(pair)
    
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

