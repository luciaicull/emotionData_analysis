from pathlib import Path
import csv

from . import utils

if __name__=="__main__":
    path = Path("/home/yoojin/data/emotionDataset/final/save")
    name = "no_split_dataset.dat"

    f = open(path.joinpath('alignment_result.csv'), 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(['name', 'num_matched_notes', 'num_unmatched_notes'])

    dataset = utils.load_datafile(path, name)

    for emotionset in dataset.set_list:
        for eN_set in emotionset['list']:
            for pair_list in eN_set['pairs']:
                num_matched_notes = 0
                num_unmatched_notes = 0
                for pair in pair_list:
                    if pair['perf'] is None:
                        num_unmatched_notes += 1
                    else:
                        num_matched_notes += 1
                name = emotionset['name'] + '.E' + str(eN_set['emotion_number'])

                wr.writerow([name, num_matched_notes, num_unmatched_notes])
            

    f.close()
