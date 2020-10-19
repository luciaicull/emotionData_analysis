import argparse
from pathlib import Path


def get_parser():
    p = argparse.ArgumentParser("classification")

    p.add_argument("--path", type=Path,
                   default="/home/yoojin/data/emotionDataset/final/save/", help="emotion data folder name")
    p.add_argument("--frag_data_name", type=str,
                   default="splitted_hop_8_split_16.dat")
    p.add_argument("--total_data_name", type=str,
                   default="raw_feature_dataset_dict.dat")

    return p
