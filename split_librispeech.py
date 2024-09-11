import argparse
from pathlib import Path
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from synthesizer.synthesizer import Synthesizer
from external import lpcynet
from tqdm import tqdm
from lib import utils
from scipy.io import wavfile
import shutil
import librosa
import random
import os
SEED=0
random.seed(SEED)
# First convert flac to wav files using the bash command:
# for flac in $(find . -name "*.flac"); do sox $flac ${flac/.flac/.wav}; done
# And then delete unused flac files using:
# find . -name "*.flac" -exec rm {} \;

def get_durations(data_path):
    files = data_path.glob('**/*.wav')
    duration_files = [(file.stem, librosa.get_duration(filename=file)) for file in files]
    print("Found %.2f minutes of audio files" % (np.sum([e[1] for e in duration_files]) / 60))
    random.shuffle(duration_files)
    return duration_files

def sample_files(duration_files, durations_to_sample):
    sampled_files = {dur_sample: [] for dur_sample in durations_to_sample}
    cum_sum = np.cumsum([e[1] for e in duration_files])
    for dur_sample in durations_to_sample:
        # Find index of the cumulated duration that is closest to our target duration
        idx = min(range(len(cum_sum)), key=lambda i: abs(cum_sum[i] - dur_sample))
        # Keep as many files as we need to reach the target duration
        sampled_files[dur_sample] = [e[0] for e in duration_files[:idx]]
    return sampled_files

def splice_and_copy(original_path, dest_path, curr_dur, max_dur, symlink=True):
    end_points = np.arange(0, max_dur*curr_dur//max_dur, max_dur)
    if curr_dur != end_points[-1] and curr_dur - end_points[-1] > .3:
        end_points = np.append(end_points, curr_dur)
    for i in range(len(end_points)-1):
        onset, offset = end_points[i], end_points[i+1]
        new_dest_path = dest_path.parent / (dest_path.stem + '_from_%.2f_to_%.2f.wav' % (onset, offset))
        if not symlink:
            # Load wav between onset and offset
            input_wav = librosa.load(original_path, sr=16000, mono=True,
                                     offset=onset, duration=offset-onset)[0]
            librosa.output.write_wav(new_dest_path, input_wav, 16000)
        else:
            new_original_path = original_path.parent / (original_path.stem  + '_from_%.2f_to_%.2f.wav' % (onset, offset))
            os.symlink(new_original_path, new_dest_path)
def copy_files(sampled_files, duration_files, data_path, out_path, max_dur=10):
    target_durations = sorted(sampled_files.keys(), reverse=True)
    max_target = target_durations[0]
    for i, target_dur in tqdm(enumerate(target_durations)):
        curr_folder = out_path / ('librispeech_%d_mn' % (target_dur/60))
        curr_folder.mkdir(parents=True, exist_ok=True)
        for stem in sampled_files[target_dur]:
            dest_path = curr_folder / (stem + '.wav')
            if i == 0:
                splitted_stem = stem.split('-')
                original_path = data_path / splitted_stem[0] / splitted_stem[1] / (stem + '.wav')
                splice_and_copy(original_path, dest_path, duration_files[stem], max_dur, symlink=False)
                #shutil.copy(original_path, dest_path)
            else:
                original_path = out_path / ('librispeech_%d_mn' % (max_target/60)) / (stem + '.wav')
                splice_and_copy(original_path, dest_path, duration_files[stem], max_dur, symlink=True)
                #os.symlink(original_path, dest_path)

def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/home/engaclew/agent/datasets/train-clean-100/LibriSpeech/train-clean-100',
                        help='Path to LibriSpeech')
    parser.add_argument('--out_path', type=str,
                        default='/home/engaclew/agent/datasets',
                        help='Output path')
    parser.add_argument('--duration', nargs='+', default=[10, 60, 600, 6000],
                        help='Durations (in mn) of the subsets to generate')
    parser.add_argument('--max_dur', type=int, default=6,
                        help='Maximum duration (in s) above which a segment will be spliced')
    args = parser.parse_args(argv)
    data_path = Path(args.data_path)
    out_path = Path(args.out_path)
    durations_to_sample = np.array(args.duration)*60
    duration_files = get_durations(data_path)
    sampled_files = sample_files(duration_files, durations_to_sample)
    copy_files(sampled_files, dict(duration_files), data_path, out_path, max_dur=args.max_dur)
    print('Done.')

if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)