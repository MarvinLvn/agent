import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

DUR_FILE = 10

def get_files_per_speaker(data_path):
    speaker_folders = data_path.glob('*')
    out = {}
    for speaker_folder in speaker_folders:
        out[speaker_folder.name] = list(speaker_folder.glob('*.wav'))
    return out

def increasing_size(data_path, files_per_speaker, durations):
    for tgt_dur in tqdm(durations):
        nb_chunks = int(tgt_dur*60/DUR_FILE)
        files_to_copy = files_per_speaker['M0'][:nb_chunks]
        output_folder = data_path / f'M0_{tgt_dur}_mn'
        output_folder.mkdir(parents=True, exist_ok=True)
        for file in tqdm(files_to_copy, leave=False):
            os.symlink(file, output_folder / file.name)
        print(f"Done creating training set of size {tgt_dur} mn with speaker M0.")

def varying_speakers(data_path, files_per_speaker, duration):
    for nb_speakers in tqdm([2, 4, 8]):
        nb_files_per_speaker = int(duration*60 / (DUR_FILE*nb_speakers))
        for idx_speaker in tqdm(range(nb_speakers), leave=False):
            if idx_speaker % 2 == 0:
                speaker_id = f'M{idx_speaker//2}'
            else:
                speaker_id = f'F{idx_speaker//2}'
            files_to_copy = files_per_speaker[speaker_id][:nb_files_per_speaker]
            output_folder = data_path / f'{nb_speakers}_speakers_{duration}_mn'
            output_folder.mkdir(parents=True, exist_ok=True)
            for file in tqdm(files_to_copy, leave=False):
                os.symlink(file, output_folder / file.name)
        print(f"Done creating training set of size {duration} mn with {nb_speakers} speakers.")




def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/home/engaclew/agent/datasets/audiocite_prepared',
                        help='Path to LibriSpeech')
    parser.add_argument('--mode', type=str, choices=['increasing', 'speaker'],
                        help='If increasing mode is activated, will create training sets of varying duration.'
                             'If speaker mode is activated, will create training sets of size --dur with varying number of speakers.')
    parser.add_argument('--durations', nargs='+', default=[10, 60, 600, 6000],
                        help='Durations (in mn) of the subsets to generate')
    args = parser.parse_args(argv)
    data_path = Path(args.data_path)

    files_per_speaker = get_files_per_speaker(data_path)
    if args.mode == 'increasing':
        increasing_size(data_path, files_per_speaker, args.durations)
    elif args.mode == 'speaker':
        varying_speakers(data_path, files_per_speaker, args.durations[-1])
    print('Done.')

if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)