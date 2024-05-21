import argparse
import json
from pathlib import Path
import sys
import numpy as np
import random
import bisect
import librosa
from tqdm import tqdm
import soundfile as sf

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
        if spk_gender in ['M', 'F'] and spk_id in AUTHORIZED and path.is_file():
            entry = {'book_id': book_id, 'path': path, 'duration': duration}
            if spk_id in metadata:
                metadata[spk_id].append(entry)
            else:
                metadata[spk_id] = [entry]
    return metadata

def extract_chunks(metadata, seq_dur, tgt_dur, output_path):
    spk_ids = list(metadata.keys())
    cnt_m = 0
    cnt_f = 0
    # 1) For each speaker
    for idx_spk in tqdm(range(len(spk_ids))):
        spk_id = spk_ids[idx_spk]
        spk_gender = spk_id.split('_')[0]
        if spk_gender == 'M':
            new_spk_id = f'{spk_gender}{cnt_m}'
        elif spk_gender == 'F':
            new_spk_id = f'{spk_gender}{cnt_f}'
        path_list = [e['path'] for e in metadata[spk_id]]
        duration_list = [e['duration'] for e in metadata[spk_id]]
        spkr_folder = output_path / new_spk_id
        spkr_folder.mkdir(parents=True, exist_ok=True)

        if new_spk_id == 'M0':
            actual_tgt_dur = tgt_dur
        elif new_spk_id == 'M1':
            actual_tgt_dur = tgt_dur/4
        elif new_spk_id == 'M2':
            actual_tgt_dur = tgt_dur/8
        elif new_spk_id == 'M3':
            actual_tgt_dur = tgt_dur/8
        elif new_spk_id == 'F0':
            actual_tgt_dur = tgt_dur/2
        elif new_spk_id == 'F1':
            actual_tgt_dur = tgt_dur/4
        elif new_spk_id == 'F2':
            actual_tgt_dur = tgt_dur/8
        elif new_spk_id == 'F3':
            actual_tgt_dur = tgt_dur/8

        tot_chunks = int(actual_tgt_dur * 3600 / seq_dur)
        curr_nb_chunk = len(list(spkr_folder.glob('*.wav')))
        if curr_nb_chunk < tot_chunks:
            idx_chunk = 0
            # 2) Extract N chunks of size --seq_dur
            for idx_file, file_path in tqdm(enumerate(path_list), leave=False):
                audio = librosa.load(file_path, sr=16000, mono=True)[0]
                for onset in range(0, int(duration_list[idx_file]-seq_dur), int(seq_dur)):
                    if idx_chunk >= curr_nb_chunk:
                        beg = onset*16000
                        end = (onset+seq_dur)*16000
                        chunk = audio[beg:end]
                        chunk_path = spkr_folder / f'{new_spk_id}_{idx_file}_{idx_chunk}.wav'
                        sf.write(chunk_path, chunk, 16000, subtype='PCM_16')
                    idx_chunk += 1
                    if idx_chunk == tot_chunks:
                         break
                if idx_chunk == tot_chunks:
                    break
        if spk_gender == 'M':
            cnt_m += 1
        elif spk_gender == 'F':
            cnt_f += 1


def get_cum_dur_per_spkr(metadata):
    cum_dur_per_spkr = [(spk_id, np.sum([file['duration'] for file in metadata[spk_id]])/60) for spk_id in metadata.keys()]
    cum_dur_per_spkr = sorted(cum_dur_per_spkr, key=lambda x: x[1], reverse=True)
    return cum_dur_per_spkr

def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/home/engaclew/agent/datasets/audiocite',
                        help='Path to the data folder.')
    parser.add_argument('--output_path', type=str, default='/home/engaclew/agent/datasets/audiocite_prepared',
                        help='Path to the metadata file.')
    parser.add_argument('--dur', type=int, default=100,
                        help='Select --dur hours of speech per spkr.')
    parser.add_argument('--seq_dur', type=int, default=10,
                        help='Extract chunks of --seg_dur seconds of speech.')
    args = parser.parse_args(argv)
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    metadata_path = data_path / 'metadata' / 'train_files.json'
    metadata = get_metadata(metadata_path)

    extract_chunks(metadata, args.seq_dur, args.dur, output_path)



if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)