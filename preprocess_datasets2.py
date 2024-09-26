import math
import os
import pickle
from glob import glob

import librosa
import numpy as np
from meldataset import mel_spectrogram
from scipy.io import wavfile
from tqdm import tqdm
from external import lpcynet

import torch
from lib import art_model
from lib import utils

MAX_WAV_VALUE = 32768.0
INT16_MAX_VALUE = 32767
H_N_FFT = 1024
H_NUM_MELS = 80
H_SAMPLING_RATE = 16000
H_HOP_SIZE = 320
H_WIN_SIZE = 640
H_FMIN = 0
H_FMAX = 8000

def rms(y):
    return np.sqrt(np.mean((y * 1.0) ** 2))


def compute_wav_rms(wav_pathname):
    wavfiles_path = list(glob(wav_pathname))

    # No need to compute the rms on every single file
    # + this does not fit in memory when processing a large number of files
    if len(wavfiles_path) > 1000:
        wavfiles_path = wavfiles_path[:1000]

    dataset_pcm = []
    for wavfile_path in tqdm(wavfiles_path):
        pcm, _ = librosa.load(wavfile_path, sr=None)
        dataset_pcm.append(pcm)

    dataset_pcm = np.concatenate(dataset_pcm, axis=0)
    dataset_wav_rms = rms(dataset_pcm)
    return dataset_wav_rms


def preprocess_wav(
    dataset_name, wav_pathname, target_sampling_rate, dataset_wav_rms, target_wav_rms
):
    export_dir = "datasets/%s/wav" % dataset_name
    utils.mkdir(export_dir)

    wav_scaling_factor = 1
    if target_wav_rms is not None:
        wav_scaling_factor = np.sqrt(target_wav_rms ** 2 / dataset_wav_rms ** 2)

    wavfiles_path = glob(wav_pathname)
    for wavfile_path in tqdm(wavfiles_path):
        pcm, wavfile_sampling_rate = librosa.load(wavfile_path, sr=None)
        pcm = pcm * wav_scaling_factor
        if wavfile_sampling_rate != target_sampling_rate:
            pcm = librosa.resample(pcm, wavfile_sampling_rate, target_sampling_rate)

        pcm = pcm * INT16_MAX_VALUE
        assert np.abs(pcm).max() <= INT16_MAX_VALUE
        pcm = pcm.astype("int16")
        item_name = utils.parse_item_name(wavfile_path) if not dataset_name.startswith('librispeech') \
            else os.path.basename(wavfile_path)
        wavfile.write("%s/%s.wav" % (export_dir, item_name), target_sampling_rate, pcm)

def get_mel(x):
    return mel_spectrogram(x, H_N_FFT, H_NUM_MELS, H_SAMPLING_RATE, H_HOP_SIZE, H_WIN_SIZE, H_FMIN, H_FMAX)

def get_source(pcm):
    # to avoid buffer source array is read-only
    pcm = np.copy(pcm)
    lpcnet_features = lpcynet.analyze_frames(pcm)
    source = lpcnet_features[:, 18:]
    return source

def extract_source_and_mel(dataset_name, format='.bin'):
    mel_export_dir = "datasets/%s/mel" % dataset_name
    utils.mkdir(mel_export_dir)
    source_export_dir = "datasets/%s/source" % dataset_name
    utils.mkdir(source_export_dir)

    wavfiles_dir = "datasets/%s/wav" % dataset_name
    wavfiles_path = glob("%s/*.wav" % wavfiles_dir)
    lengths = {}
    for wavfile_path in tqdm(wavfiles_path):
        item_name = utils.parse_item_name(wavfile_path)

        sr, pcm = wavfile.read(wavfile_path)
        assert sr == 16000

        audio = pcm / MAX_WAV_VALUE
        audio = torch.FloatTensor(audio).unsqueeze(0)
        item_mel = get_mel(audio).squeeze(0).numpy()
        lengths[item_name] = item_mel.shape[1]
        item_source = get_source(pcm)
        current_length = item_source.shape[0]
        tgt_length = item_mel.shape[1]
        if current_length != tgt_length:
            current_indices = np.arange(current_length)
            target_indices = np.linspace(0, current_length - 1, tgt_length)
            item_source = np.column_stack([
                np.interp(target_indices, current_indices, item_source[:, i])
                for i in range(item_source.shape[1])
            ])


        if format == '.bin':
            item_mel.tofile("%s/%s.bin" % (mel_export_dir, item_name))
            item_source.tofile("%s/%s.bin" % (source_export_dir, item_name))
        elif format == '.npy':
            np.save("%s/%s.npy" % (mel_export_dir, item_name), item_mel)
            np.save("%s/%s.npy" % (source_export_dir, item_name), item_source)
        else:
            raise ValueError(f"Unknown output format for mel-spec {format}")
    return lengths

def preprocess_ema(
    dataset_name,
    ema_pathname,
    ema_format,
    ema_sampling_rate,
    ema_scaling_factor,
    ema_coils_order,
    ema_needs_lowpass,
    target_sampling_rate,
):
    export_dir = "datasets/%s/ema" % dataset_name
    utils.mkdir(export_dir)

    items_ema = {}
    ema_limits = {
        "xmin": math.nan,
        "xmax": math.nan,
        "ymin": math.nan,
        "ymax": math.nan,
    }

    if ema_needs_lowpass:
        lowpass_filter = utils.create_lowpass_filter(ema_sampling_rate, 50)

    emafiles_path = glob(ema_pathname)
    for emafile_path in tqdm(emafiles_path):
        item_ema = utils.read_ema_file(emafile_path, ema_format)

        # lowpass filtering
        if ema_needs_lowpass:
            item_ema = lowpass_filter(item_ema)

        # resampling
        if ema_sampling_rate != target_sampling_rate:
            item_ema = utils.interp_2d(
                item_ema, ema_sampling_rate, target_sampling_rate
            )

        # reordering, target coils order:
        #   lower incisor, tongue tip, tongue middle, tongue back, lower lip, upper lip and velum

        item_ema = item_ema[:, ema_coils_order]
        # scaling to mm
        item_ema = item_ema / ema_scaling_factor

        item_name = utils.parse_item_name(emafile_path)
        item_ema.astype("float32").tofile("%s/%s.bin" % (export_dir, item_name))
        items_ema[item_name] = item_ema

        ema_limits["xmin"] = min(item_ema[:, 0::2].min(), ema_limits["xmin"])
        ema_limits["xmax"] = max(item_ema[:, 0::2].max(), ema_limits["xmax"])
        ema_limits["ymin"] = min(item_ema[:, 1::2].min(), ema_limits["ymin"])
        ema_limits["ymax"] = max(item_ema[:, 1::2].max(), ema_limits["ymax"])

    with open("datasets/%s/ema_limits.pickle" % dataset_name, "wb") as file:
        pickle.dump(ema_limits, file)

    return items_ema


def extract_art_parameters(dataset_name, items_ema, tgt_lengths, format='.bin'):
    dataset_dir = "datasets/%s" % dataset_name

    all_ema_frames = np.concatenate(list(items_ema.values()), axis=0)
    art_model_params = art_model.build_art_model(all_ema_frames)
    with open("%s/art_model.pickle" % dataset_dir, "wb") as file:
        pickle.dump(art_model_params, file)

    export_dir = "datasets/%s/art_params" % dataset_name
    utils.mkdir(export_dir)
    for i, (item_name, item_ema) in tqdm(enumerate(items_ema.items())):
        item_art = art_model.ema_to_art(art_model_params, item_ema)
        current_length = item_art.shape[0]
        tgt_length = tgt_lengths[item_name]
        if current_length != tgt_length:
            current_indices = np.arange(current_length)
            target_indices = np.linspace(0, current_length - 1, tgt_length)
            item_art = np.column_stack([
                np.interp(target_indices, current_indices, item_art[:, i])
                for i in range(item_art.shape[1])
            ])
        if format == '.bin':
            item_art.astype("float32").tofile("%s/%s.bin" % (export_dir, item_name))
        elif format == '.npy':
            np.save("%s/%s.npy" % (export_dir, item_name), item_art.astype("float32"))
        else:
            raise ValueError(f"Unknown output format for art_params {format}")


def preprocess_lab(dataset_name, lab_pathname, dataset_resolution, target_resolution):
    export_dir = "datasets/%s/lab" % dataset_name
    utils.mkdir(export_dir)

    resolution_ratio = target_resolution / dataset_resolution

    labfiles_path = glob(lab_pathname)
    for labfile_path in tqdm(labfiles_path):
        item_lab = utils.read_lab_file(labfile_path, resolution_ratio)
        item_name = utils.parse_item_name(labfile_path)
        utils.save_lab_file("%s/%s.lab" % (export_dir, item_name), item_lab)


def main():
    features_config = utils.read_yaml_file("./features_config.yaml")
    datasets_infos = utils.read_yaml_file("./datasets_infos.yaml")
    format = '.npy'
    datasets_wav_rms = {}

    for dataset_name, dataset_infos in datasets_infos.items():
        #if dataset_name == 'pb2009': #dataset_name.startswith('M0'):
        #if dataset_name in ['2_speakers_6000_mn', '4_speakers_6000_mn', '8_speakers_6000_mn',
        #                    'M0_10_mn', 'M0_60_mn', 'M0_600_mn', 'M0_6000_mn', 'heldout', 'test']:

        if dataset_name in ['2_speakers_6000_mn', '4_speakers_6000_mn', '8_speakers_6000_mn',
                            'M0_6000_mn']:
            print("Preprocessing %s..." % dataset_name)

            wavfiles_path = glob(dataset_infos["wav_pathname"])

            if len(wavfiles_path) == 0:
                print("Dataset %s not found" % dataset_name)
                print("")
                continue

            print("Computing RMS...")
            dataset_wav_rms = compute_wav_rms(dataset_infos["wav_pathname"])
            datasets_wav_rms[dataset_name] = dataset_wav_rms
            print("Computing RMS done")

            print("Resampling WAV files...")
            target_wav_rms = (
                datasets_wav_rms[dataset_infos["wav_rms_reference"]]
                if "wav_rms_reference" in dataset_infos
                else None
            )

            preprocess_wav(
                dataset_name,
                dataset_infos["wav_pathname"],
                features_config["wav_sampling_rate"],
                dataset_wav_rms,
                target_wav_rms,
            )
            print("Resampling WAV files done")

            print("Extracting source & mel-spectrograms...")
            tgt_lengths = extract_source_and_mel(dataset_name, format=format)
            print("Extracting source & mel-spectrograms done")

            if "ema_pathname" in dataset_infos:
                frames_sampling_rate = features_config["ema_sampling_rate"]

                print("Preprocessing EMA...")
                items_ema = preprocess_ema(
                    dataset_name,
                    dataset_infos["ema_pathname"],
                    dataset_infos["ema_format"],
                    dataset_infos["ema_sampling_rate"],
                    dataset_infos["ema_scaling_factor"],
                    dataset_infos["ema_coils_order"],
                    dataset_infos["ema_needs_lowpass"],
                    frames_sampling_rate,
                )
                print("Preprocessing EMA done")

                print("Extracting articulatory model and parameters...")
                extract_art_parameters(dataset_name, items_ema, tgt_lengths, format='.npy')
                print("Extracting articulatory model and parameters done")
            if "lab_pathname" in dataset_infos:
                print("Resampling LAB files...")
                preprocess_lab(
                    dataset_name,
                    dataset_infos["lab_pathname"],
                    dataset_infos["lab_resolution"],
                    frames_sampling_rate,
                )
                print("Resampling LAB files done")

            print("Preprocessing %s done" % dataset_name)
            print("")


if __name__ == "__main__":
    main()
