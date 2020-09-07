from .arg_parser import get_parser
from . import utils
from .analyser_class import Analyser


def main():
    p = get_parser()
    args = p.parse_args()

    feature_data = utils.load_datafile(args.path, args.name)
    
    X, Y, feature_key_list = utils.make_X_Y(feature_data)

    analyser = Analyser(X, Y, feature_key_list)
    analyser.run_feature_selection()


if __name__ == "__main__":
    main()
