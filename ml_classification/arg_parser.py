import argparse
from pathlib import Path


def get_parser():
    p = argparse.ArgumentParser("feature_generator")

    p.add_argument("--path", type=Path,
                   default="/home/yoojin/data/emotionDataset/final/tmp_test/save/", help="emotion data folder name")
    p.add_argument("--name", type=str,
                   default="no_split.dat")

    return p
