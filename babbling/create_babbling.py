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
import random
SEED=4
random.seed(SEED)

def load_art_params(data_path, item):
    param_file = data_path / 'art_params' / f'{item}.bin'
    param_dim = 12 // 2
    return np.fromfile(param_file, dtype="float32").reshape((-1, param_dim))

def load_lab(data_path, item):
    lab_file = data_path / 'lab' / f'{item}.lab'
    data = pd.read_csv(lab_file, sep=' ', header=None, names=['onset', 'offset', 'label'])
    return data

def load_source(data_path, item):
    source_file = data_path / 'source' / f'{item}.bin'
    return np.fromfile(source_file, dtype="float32").reshape((-1, 2))

def save(art_params, source, lab, cepstrum, wave, output_path, basename):
    # Write bin files
    out_file = output_path / 'art_params' / f'{basename}.bin'
    utils.mkdir(out_file.parent)
    art_params.astype("float32").tofile(out_file)

    out_file = output_path / 'source' / f'{basename}.bin'
    utils.mkdir(out_file.parent)
    source.astype("float32").tofile(out_file)

    out_file = output_path / 'cepstrum' / f'{basename}.bin'
    utils.mkdir(out_file.parent)
    cepstrum.astype("float32").tofile(out_file)

    # Write lab
    out_file = output_path / 'lab' / f'{basename}.lab'
    utils.mkdir(out_file.parent)
    labels = []
    for idx, row in lab.iterrows():
        labels.append({'start': row['onset'], 'end': row['offset'], 'name': row['label']})
    utils.save_lab_file(out_file, labels)

    # Write wav
    out_file = output_path / 'wav' / f'{basename}.wav'
    utils.mkdir(out_file.parent)
    wavfile.write(out_file, 16000, wave)

def interpolate_articulatory(art_params, source, lab, new_length):
    # 1) Interpolate articulatory trajectories
    old_length = art_params.shape[0]
    num_art_params = art_params.shape[1]
    new_indices = np.linspace(0, old_length - 1, num=new_length)
    new_art_params = np.zeros((new_length, num_art_params))
    xp = np.arange(old_length)
    for i in range(num_art_params):
        new_art_params[:, i] = np.interp(new_indices, xp, art_params[:, i])

    # 2) Interpolate source
    num_source = 2
    new_source = np.zeros((new_length, num_source))
    for i in range(num_source):
        new_source[:, i] = np.interp(new_indices, xp, source[:, i])

    new_lab = lab.copy()
    new_lab['onset'] *= new_length / old_length
    new_lab['offset'] *= new_length / old_length
    return new_art_params, new_source, new_lab


def generate_data(art_params, source, lab, synthesizer, output_path, idx_category,
                  type, NB_SEQUENCES=30, MIN_DUR=.5, MAX_DUR=1.5):
    old_length = art_params.shape[0]
    for idx_seq in range(NB_SEQUENCES):
        new_length = np.random.randint(low=MIN_DUR*old_length, high=MAX_DUR*old_length)
        new_art_params, new_source, new_lab = interpolate_articulatory(art_params, source, lab, new_length)
        new_cepstrum = synthesizer.synthesize(new_art_params)
        new_sound = np.concatenate((new_cepstrum, new_source), axis=1)
        new_wave = lpcynet.synthesize_frames(new_sound)
        save(new_art_params, new_source, new_lab, new_cepstrum, new_wave, output_path, f'{type}_{idx_category}_seq_{idx_seq}')

def generate_all_isolated_vowels(data_path, synthesizer, output_path, ISOLATED_VOWELS):
    print("Generate isolated vowels:")
    for idx_vowel, item_file in tqdm(enumerate(ISOLATED_VOWELS)):
        art_params = load_art_params(data_path, item_file)
        lab = load_lab(data_path, item_file)
        source = load_source(data_path, item_file)
        min_len = np.min([art_params.shape[0], source.shape[0]])
        art_params, source = art_params[:min_len,:], source[:min_len,:]
        generate_data(art_params, source, lab, synthesizer, output_path, idx_vowel, type='vowel',
                      NB_SEQUENCES=30, MIN_DUR=.5, MAX_DUR=1.5)

def generate_with_no_aug(data_path, synthesizer, output_path, ISOLATED_VOWELS, TRIPHONES):
    for idx_vowel, item_file in tqdm(enumerate(ISOLATED_VOWELS)):
        art_params = load_art_params(data_path, item_file)
        lab = load_lab(data_path, item_file)
        source = load_source(data_path, item_file)
        min_len = np.min([art_params.shape[0], source.shape[0]])
        art_params, source = art_params[:min_len, :], source[:min_len, :]
        cepstrum = synthesizer.synthesize(art_params)
        sound = np.concatenate((cepstrum, source), axis=1)
        wave = lpcynet.synthesize_frames(sound)
        save(art_params, source, lab, cepstrum, wave,
             output_path, f'vowel_{idx_vowel}_seq_0')

    for idx_pair, item_file in tqdm(enumerate(TRIPHONES)):
        # Load data
        art_params = load_art_params(data_path, item_file)
        lab = load_lab(data_path, item_file)
        source = load_source(data_path, item_file)
        min_len = np.min([art_params.shape[0], source.shape[0]])
        art_params, source = art_params[:min_len, :], source[:min_len, :]
        # Cut off first vowel
        onset_consonant = lab.iloc[2]['onset']
        art_params = art_params[onset_consonant:, ]
        source = source[onset_consonant:, ]
        lab = lab[2:].reset_index(drop=True)
        lab['onset'] -= onset_consonant
        lab['offset'] -= onset_consonant
        # Generate new cepstrum
        cepstrum = synthesizer.synthesize(art_params)
        sound = np.concatenate((cepstrum, source), axis=1)
        wave = lpcynet.synthesize_frames(sound)
        save(art_params, source, lab, cepstrum, wave,
             output_path, f'biphone_{idx_pair}_seq_0')

def generate_all_biphones(data_path, synthesizer, output_path, TRIPHONES):
    print("Generate biphones:")
    for idx_pair, item_file in tqdm(enumerate(TRIPHONES)):
        art_params = load_art_params(data_path, item_file)
        lab = load_lab(data_path, item_file)
        source = load_source(data_path, item_file)
        min_len = np.min([art_params.shape[0], source.shape[0]])
        art_params, source = art_params[:min_len, :], source[:min_len, :]


        # Cut off first vowel
        onset_consonant = lab.iloc[2]['onset']
        art_params = art_params[onset_consonant:,]
        source = source[onset_consonant:,]
        lab = lab[2:].reset_index(drop=True)
        lab['onset'] -= onset_consonant
        lab['offset'] -= onset_consonant
        generate_data(art_params, source, lab, synthesizer, output_path, idx_pair, type='biphone',
                      NB_SEQUENCES=30, MIN_DUR=.5, MAX_DUR=1.5)

def generate_random(data_path):
    # we sample the same number of sequences as we rely on during real sampling
    nb_vowels = 25
    nb_triphones = 37
    nb_phones = {item_file.stem: len(load_lab(data_path, item_file.stem))-2
                  for item_file in (data_path / 'lab').glob('*.lab')}
    vowels_to_sample = [item_file for item_file in nb_phones.keys() if nb_phones[item_file] == 1]
    triphones_to_sample = [item_file for item_file in nb_phones.keys() if nb_phones[item_file] == 3]

    sampled_vowels = random.sample(vowels_to_sample, nb_vowels)
    sampled_triphones = random.sample(triphones_to_sample, nb_triphones)
    return sampled_vowels, sampled_triphones

def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/home/engaclew/agent/datasets/pb2007',
                        help='Path to PB2007')
    parser.add_argument('--synthesizer', type=str, default='/home/engaclew/agent/out/synthesizer/ea587b76c95fecef01cfd16c7f5f289d-3',
                        help='Path to synthesizer used to generate babbling')
    parser.add_argument('--out_path', type=str,
                        default='/home/engaclew/agent/datasets',
                        help='Output path')
    parser.add_argument('--skip_aug', action='store_true',
                        help='If activated, will skip the data augmentation and only consider original files')
    parser.add_argument('--random', action='store_true',
                        help='If activated, will generate babbling based on random sampling of sequences')
    args = parser.parse_args(argv)
    data_path = Path(args.data_path)
    output_path = Path(args.out_path)
    synthesizer = Synthesizer.reload(args.synthesizer)

    # From Davis, Barbara L.; MacNeilage, Peter F.  (1995). The Articulatory Basis of Babbling. Journal of Speech Language and Hearing Research, 38(6), 1199–. doi:10.1044/jshr.3806.1199 
    # 1) central vowels (eu (peur),  eu (creux), u, un) comes predominantly with labial consonants (p, b, m, f, v)
    # 2) front vowels (i, é, è, a, an) comes with alveolar consonants (s, z, l, d, t, n)
    # 3) back vowels (ou, o (mot), o (botte), on) comes with velar consonants (k, g)

    # Consonantal phones most frequently reported are:
    # [b], [d] and sometimes [g]; nasals [m] and [n]; and glides [j], [w] and [h]
    # Mid and low front and central vowels are most often reported:
    # [ɛ] (bête), [e] (beauté), [a], [ae] (bain), [ə] (bird; doesn't exist in FR), [ʌ] (sotte)

    ISOLATED_VOWELS = [
        'item_0000',  # 0. [a] a
        'item_0310',  # 1. [a] a
        'item_0001',  # 2. [ɛ] (bête) e^
        'item_0009',  # 3. [ɛ] (bête) e^
        'item_0311',  # 4. [ɛ] (bête) e^
        'item_0002',  # 5. [e] (beauté) e
        'item_0312',  # 6. [e] (beauté) e
        'item_0007',  # 7. [ø] (creux) x
        'item_0008',  # 8. [ø] (creux) x
        'item_0317',  # 9. [ø] (creux) x
        'item_0010',  # 10. [ə] (bird, doesn't exist in FR) x^
        'item_0017',  # 11. [ə] (bird, doesn't exist in FR) x^
        'item_0320',  # 12. [ə] (bird, doesn't exist in FR) x^
        'item_0013',  # 13. [œ̃] (bain) e~
        'item_0323',  # 14. [œ̃] (bain) e~
        'item_0004',  # 15. [y] (u) y
        'item_0314',  # 16. [y] (u) y
        'item_0005',  # 17. [u] (ou) u
        'item_0315',  # 18. [u] (ou) u
        'item_1088',  # 19. [u] (ou) u
        'item_1092',  # 20. [u] (ou) u
        'item_0015',  # 21. [ɔ̃] (on) o~
        'item_0324',  # 22. [ɔ̃] (on) o~
        'item_0012',  # 23. [ɑ̃] (an) a~
        'item_0321',  # 24. [ɑ̃] (an) a~
    ]

    TRIPHONES = [
        # b consonant with central vowels
        'item_0177',  # 0. xbx [øbø]
        'item_0469',  # 1. xbx [øbø]
        'item_0217',  # 2. x^bx^ [əbə] (eubeu)
        'item_0508',  # 3. x^bx^ [əbə]
        'item_0257',  # 4. e~be~ [œ̃bœ̃] (un bain)
        'item_0548',  # 5. e~be~ [œ̃bœ̃]
        'item_0110',  # 6. yby [yby] (ubu) /!\ [y] sound only come later, may remove it
        'item_0411',  # 7. yby [yby] (ubu) /!\ [y] sound only come later, may remove it

        # m consonant with central vowels
        'item_0184',  # 8. xmx [ømø]
        'item_0475',  # 9. xmx [ømø]
        'item_0223',  # 10. x^mx^ [əmə]
        'item_0515',  # 11. x^mx^ [əmə]
        'item_0256',  # 12. e~me~ [œ̃mœ̃]
        'item_0263',  # 13. e~me~ [œ̃mœ̃]
        'item_0554',  # 14. e~me~ [œ̃mœ̃]
        'item_0116',  # 15. ymy [ymy] (umu) /!\ [y] sound only come later, may remove it
        'item_0417',  # 16. ymy [ymy] (umu) /!\ [y] sound only come later, may remove it

        # d consonant with front vowels
        'item_0025',  # 17. ada [ada]
        'item_0332',  # 18. ada [ada]
        'item_0054',  # 19. e^de^ [ɛdɛ] aidai
        'item_0353',  # 20. e^de^ [ɛdɛ]
        'item_0073',  # 21. ede [ede] édé
        'item_0373',  # 22. ede [ede]
        'item_0237',  # 23. a~da~ [ɑ̃dɑ̃] andan
        'item_0529',  # 24. a~da~ [ɑ̃dɑ̃] andan

        # n consonant with front vowels
        'item_0031',  # 25. ana [ana]
        'item_0040',  # 26. ana [ana]
        'item_0060',  # 27. e^ne^ [ɛnɛ]
        'item_0360',  # 28. e^ne^ [ɛnɛ]
        'item_0079',  # 29. ene [ene] éné
        'item_0380',  # 30. ene [ene] éné
        'item_0244',  # 31. a~na~ [ɑ̃nɑ̃] annan
        'item_0536',  # 32. a~na~ [ɑ̃nɑ̃] annan

        # g consonant with back vowels
        'item_0140',  # 33. ugu [ugu] ougou /!\ [u] comes only later, may remove it
        'item_0429',  # 34. ugu [ugu] ougou /!\ [u] comes only later, may remove it
        'item_0298',  # 35. o~go~ [ɔ̃gɔ̃] ongon /!\ [ɔ̃] comes only later, may remove it
        'item_0589',  # 36. o~go~ [ɔ̃gɔ̃] ongon /!\ [ɔ̃] comes only later, may remove it
    ]

    if args.random:
        ISOLATED_VOWELS, TRIPHONES = generate_random(data_path)

    if args.skip_aug:
        out_folder = f'pb2007_babbling_no_aug_random_{SEED}' if args.random else 'pb2007_babbling_no_aug'
        generate_with_no_aug(data_path, synthesizer, output_path / out_folder, ISOLATED_VOWELS, TRIPHONES)
    else:
        out_folder = f'pb2007_babbling_phase_random_{SEED}' if args.random else 'pb2007_babbling_phase'
        generate_all_isolated_vowels(data_path, synthesizer, output_path / out_folder, ISOLATED_VOWELS)
        generate_all_biphones(data_path, synthesizer, output_path / out_folder, TRIPHONES)
    print('Done.')

if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)