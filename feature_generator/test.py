from pathlib import Path
from .raw_data_class import MidiMidiDataset
from . import utils

import csv

emotion_path = Path('/home/yoojin/data/emotionDataset/final/total/')
emotion_save_path = Path('/home/yoojin/data/emotionDataset/final/save/')

dataset = utils.load_datafile(emotion_save_path, 'raw_dataset.dat')


pitch_f = open('/home/yoojin/pitch_1228.csv', 'w', encoding='utf-8', newline='')
duration_f = open('/home/yoojin/duration_1228.csv', 'w', encoding='utf-8', newline='')
pitch_wr = csv.writer(pitch_f)
duration_wr = csv.writer(duration_f)

for set_l in dataset.set_list:
    name = set_l['name']
    xml_notes = set_l['list'][0].xml_notes
    pitches = [note.pitch[1] for note in xml_notes]
    durations = [note.note_duration.type for note in xml_notes]
    pitch_wr.writerow([name] + pitches)
    duration_wr.writerow([name] + durations)

pitch_f.close()
duration_f.close()
