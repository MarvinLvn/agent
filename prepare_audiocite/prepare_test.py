import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm




def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/home/engaclew/agent/datasets/audiocite_prepared',
                        help='Path to LibriSpeech')
    args = parser.parse_args(argv)
    data_path = Path(args.data_path)


if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)