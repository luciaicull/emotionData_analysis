
from .arg_parser import get_parser
from .raw_data_class import MidiMidiDataset, XmlMidiDataset
from .feature_extraction import MidiMidiFeatureExtractor, XmlMidiFeatureExtractor
from .constant import MIDI_MIDI_FEATURE_LIST, XML_MIDI_FEATURE_LIST, FEATURE_LIST_TMP
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
    
    feature_dataset = extractor.extract_features()
    utils.save_datafile(emotion_save_path, 'feature_dataset.dat', feature_dataset)
    #feature_dataset = utils.load_datafile(emotion_save_path, 'feature_dataset.dat')
    final_feature_data = feature_dataset.get_final_data(split=8, hop=1)
    utils.save_datafile(emotion_save_path, args.save_name, final_feature_data)



if __name__ == "__main__":
    generate()
