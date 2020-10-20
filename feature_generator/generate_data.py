
from .arg_parser import get_parser
from .raw_data_class import MidiMidiDataset, XmlMidiDataset
from .feature_extraction import MidiMidiFeatureExtractor, XmlMidiFeatureExtractor
from .constant import MIDI_MIDI_FEATURE_LIST, XML_MIDI_FEATURE_LIST, FEATURE_LIST_TMP
from .feature_data_class import SplittedFeatureDataset
from . import utils

def generate():
    p = get_parser()
    args = p.parse_args()

    emotion_path = args.path.joinpath("total")
    emotion_save_path = args.path.joinpath("save")
    
    # make dataset
    if args.direct:
        # midi - midi direct matching
        dataset = MidiMidiDataset(emotion_path, split=5)
        utils.save_datafile(emotion_save_path,'5sec_split_dataset.dat', dataset)
        dataset = utils.load_datafile(emotion_save_path, '5sec_split_dataset.dat')
        # extract features
        extractor = MidiMidiFeatureExtractor(dataset.set_list, MIDI_MIDI_FEATURE_LIST)
    else:
        # score - midi matching
        dataset = XmlMidiDataset(emotion_path)
        utils.save_datafile(emotion_save_path, 'raw_dataset.dat', dataset)
        dataset = utils.load_datafile(emotion_save_path, 'raw_dataset.dat')
        # extract features
        extractor = XmlMidiFeatureExtractor(dataset.set_list, XML_MIDI_FEATURE_LIST)
    
    raw_feature_dataset = extractor.extract_features()
    raw_feature_dataset.save_into_dict(emotion_save_path)
    utils.save_datafile(emotion_save_path, 'raw_feature_dataset.dat', raw_feature_dataset)
    
    raw_feature_dataset = utils.load_datafile(emotion_save_path, 'raw_feature_dataset.dat')
    
    r_list = [(1,8), (2,8), (4,8), (2,16), (4,16), (8,16)]
    for (hop, split) in r_list:
        splitted_feature_dataset = SplittedFeatureDataset(raw_feature_dataset, hop=hop, split=split)
        splitted_feature_dataset.save_into_dict(emotion_save_path)


if __name__ == "__main__":
    generate()
