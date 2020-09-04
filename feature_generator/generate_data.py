from parser import get_parser
from raw_data_class import MidiMidiDataset
from feature_extraction import Extractor
from constants import FEATURE_LIST

def main():
    parser = get_parser()
    args = parser.parse_args()

    emotion_path = args.path.joinpath("total")
    emotion_save_path = args.path.joinpath("save")

    # make dataset
    if args.direct:
        # midi - midi direct matching
        dataset = MidiMidiDataset(emotion_path)
    else:
        # score - midi matching
        pass
    
    # extract features
    extractor = Extractor(dataset.set_list, FEATURE_LIST)
    ex.extract_features()



if __name__ == "__main__":
    main()
