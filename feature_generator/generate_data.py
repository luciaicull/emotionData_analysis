from parser import get_parser
from raw_data_class import MidiMidiDataset

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
    



if __name__ == "__main__":
    main()
