import argparse
from pathlib import Path


def get_parser():
    p = argparse.ArgumentParser("feature_generator")

    p.add_argument("--path", type=Path, 
                        default="/home/yoojin/data/emotionDataset/final/", help="emotion data folder name")

    p.add_argument("--direct", type=bool,
                        default=False, help="midi-midi direct or not(Default)")
    
    p.add_argument("--save_name", type=str,
                        default="test", help=".dat file name")
    
    return p