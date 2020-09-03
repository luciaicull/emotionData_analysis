import argparse
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser("feature_generator")

    parser.add_argument("--path", type=Path, 
                        default="/home/yoojin/data/emotionDataset/final/", help="emotion data folder name")

    parser.add_argument("--direct", type=bool,
                        default=False, help="midi-midi direct or not(Default)")
    
    parser.add_argument("--save_name", type=str,
                        default="test", help=".dat file name")
    
    return parser