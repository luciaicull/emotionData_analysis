import argparse
from pathlib import Path


def get_parser():
    p = argparse.ArgumentParser("classification")

    p.add_argument("--path", type=Path,
                   default="/home/yoojin/data/emotionDataset/final/save/", help="emotion data folder name")
    p.add_argument("--name", type=str,
                   default="final_feature_data.dat")

    return p
