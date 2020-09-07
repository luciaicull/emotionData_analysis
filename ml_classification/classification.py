from .arg_parser import get_parser
from . import utils
from .runner import Runner


def main():
    p = get_parser()
    args = p.parse_args()

    feature_data = utils.load_datafile(args.path, args.name)

    total_data, train_data, test_data = utils.split_train_test(feature_data)

    r = Runner(total_data, train_data, test_data)



if __name__ == "__main__":
    main()
