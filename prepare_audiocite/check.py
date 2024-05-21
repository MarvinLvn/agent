import argparse
import json
from pathlib import Path
import sys
import numpy as np
import random
import bisect
import librosa
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
AUTHORIZED = ['M_Daniel Luttringer', 'M_Raminagrobis',
              'M_Jean-Paul Alexis', 'M_Jean-François Ricou',
              'F_Ar Men', 'F_Christiane-Jehanne',
              'F_Sabine', 'F_Juliette']

def get_metadata(metadata_path):
    with open(metadata_path, 'r') as fin:
        original_metadata = json.load(fin)

    metadata = {}
    for book_id in original_metadata.keys():
        path = Path(metadata_path.parent.parent / original_metadata[book_id]['path'].replace('../', ''))
        duration = original_metadata[book_id]['duration']
        spk_id = original_metadata[book_id]['spk_id']
        spk_gender = original_metadata[book_id]['spk_gender']
        spk_id = f'{spk_gender}_{spk_id}'
        if spk_gender in ['M', 'F'] and path.is_file():
            entry = {'book_id': book_id, 'path': path, 'duration': duration}
            if spk_id in metadata:
                metadata[spk_id].append(entry)
            else:
                metadata[spk_id] = [entry]
    return metadata


def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/home/engaclew/agent/datasets/audiocite',
                        help='Path to the data folder.')
    args = parser.parse_args(argv)
    data_path = Path(args.data_path)
    metadata_path = data_path / 'metadata' / 'train_files.json'
    metadata = get_metadata(metadata_path)
    for spk_id, book_rows in metadata.items():
        for book_row in book_rows:
            file_path = book_row['path']
            if not Path(file_path).exists():
                print(f'Cannot find {file_path} from {spk_id}')
            else:
                print("Found")



if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)