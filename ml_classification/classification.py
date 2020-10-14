from .arg_parser import get_parser
from . import utils
from .runner import Runner

from pathlib import Path

def main():
    p = get_parser()
    args = p.parse_args()

    total_feature_data = utils.load_datafile(args.path, args.name)

    total_data, train_data, test_data = utils.split_train_test(total_feature_data)

    r = Runner(total_data, train_data, test_data)
    r.run_svm()

    print('')
'''
def fragment_main():
    dir_path = Path('/home/yoojin/data/emotionDataset/final/save')

    total_data = utils.load_datafile(dir_path, 'xml_midi_no_split_feature_data.dat')
    fragment_data = utils.load_datafile(dir_path, 'xml_midi_8measure_split_feature_data.dat')

    _, train_data, test_data = utils.split_train_test(total_data)

    r = Runner(None, train_data, test_data)
    r._run_train_test_svm()

    r = Runner(None, total_data, fragment_data)
    r.test_fragment()

    print('')
'''
if __name__ == "__main__":
    main()