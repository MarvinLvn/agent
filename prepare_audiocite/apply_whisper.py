import argparse
import os
import sys
import whisperx
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import pandas as pd

# See installation instructions at: https://github.com/m-bain/whisperX
# conda create --name whisperx python=3.10
# conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# pip install git+https://github.com/m-bain/whisperx.git
# Notes:
# On oberon I had to run:
# export LD_LIBRARY_PATH="/home/mlavechin/.conda/envs/whisperx/lib:$LD_LIBRARY_PATH"

# Usage example:
# python apply_whisper.py --data_path /scratch1/projects/MarvinTmp/audiocite_prepared/test --device cuda --save_dir $SCRATCH/whisper_models
def save_audio(audio, onset, offset, output_path):
    onset = int(onset*16000)
    offset = int(offset*16000)
    audio = audio[onset:offset]
    sf.write(output_path, audio, 16000, subtype='PCM_16')

def save_sentence_level_transcript(transcript, output_path):
    with open(output_path, 'w') as fout:
        fout.write("word\tstart\tend\n%s\t%.2f\t%.2f" % (transcript['text'], 0.0, (transcript['end']-transcript['start'])))

def save_word_level_transcript(transcript, output_path):
    # In some rare cases alignment can fail, we'll catch these cases and simply skip them
    try:
        data = pd.DataFrame(transcript['words'])
        data['end'] = data['end'] - data['start'].iloc[0]
        data['start'] = data['start'] - data['start'].iloc[0]
        data.to_csv(output_path, index=False, float_format='%.2f', sep='\t')
        return True
    except KeyError:
        return False

def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/home/engaclew/agent/datasets/audiocite_prepared/test',
                        help='Path to the audio files that need to be transcribed.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use.')
    parser.add_argument( '--compute_type', type=str, default='float16', choices=['float16', 'int8'],
                        help='Switch to int8 if low on GPU mem (may reduce accuracy).')
    parser.add_argument('--batch_size', type=int, default='16',
                        help='Batch size.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Where to save the model.')
    args = parser.parse_args(argv)
    data_path = Path(args.data_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = whisperx.load_model("large-v2", device=args.device, compute_type=args.compute_type,
                                language='fr', download_root=save_dir)
    # Default alignment model is "VOXPOPULI_ASR_BASE_10K_FR"
    # But I noticed it wasn't so good.
    model_a, metadata = whisperx.load_align_model(language_code='fr', device=args.device)

    audio_files = list(data_path.glob('*_raw/*.wav'))
    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {data_path}")

    for audio_file in tqdm(audio_files):
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=args.batch_size)
        result = whisperx.align(result['segments'], model_a, metadata, audio, args.device, return_char_alignments=False)

        folder_name = audio_file.parent.name.replace('_raw', '')
        output_folder = audio_file.parent.parent / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)

        # Skip audio files that have already been fully processed
        nb_utt = len(result['segments'])
        nb_already_processed = len(list(output_folder.glob(f'{audio_file.stem}_s_*_word.csv')))
        if nb_utt == nb_already_processed:
            continue

        for utt_idx, utterance in enumerate(result['segments']):
            output_utt_level = output_folder / f'{audio_file.stem}_s_{utt_idx}_word.csv'
            has_succeed = save_word_level_transcript(utterance, output_utt_level)
            if has_succeed:
                output_utt_level = output_folder / f'{audio_file.stem}_s_{utt_idx}_utt.csv'
                save_sentence_level_transcript(utterance, output_utt_level)
                output_audio_path = output_folder / f'{audio_file.stem}_s_{utt_idx}.wav'
                save_audio(audio, float(utterance['start']), float(utterance['end']), output_audio_path)

    print("Done.")

if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)