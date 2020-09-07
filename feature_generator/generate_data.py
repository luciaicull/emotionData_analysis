
from .arg_parser import get_parser
from .raw_data_class import MidiMidiDataset
from .feature_extraction import Extractor
from .constant import FEATURE_LIST
from . import utils
'''
# for debugging
from arg_parser import get_parser
from raw_data_class import MidiMidiDataset
from feature_extraction import Extractor
from constant import FEATURE_LIST
import utils
'''
def generate():
    p = get_parser()
    args = p.parse_args()

    emotion_path = args.path.joinpath("total")
    emotion_save_path = args.path.joinpath("save")
    
    # make dataset
    if args.direct:
        # midi - midi direct matching
        dataset = MidiMidiDataset(emotion_path, split=5)
    else:
        # score - midi matching
        pass
    
    utils.save_datafile(emotion_save_path, 'dataset.dat', dataset)
    
    dataset = utils.load_datafile(emotion_save_path, 'dataset.dat')
    # extract features
    extractor = Extractor(dataset.set_list, FEATURE_LIST)
    feature_data = extractor.extract_features()
    
    utils.save_datafile(emotion_save_path, args.save_name, feature_data)



if __name__ == "__main__":
    generate()
