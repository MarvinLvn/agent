from pathlib import Path
from imitative_agent import ImitativeAgent
import pickle
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from lib.dataset_wrapper import Dataset
import numpy as np
from pathlib import Path
from lib import utils
from lib import abx_utils

ABX_NB_SAMPLES = 50

distance = {
        "art_estimated": {
            "metric": "cosine",
            "weight": 1,
        },
    }

model_path = Path('out/imitative_agent/babbling_v2')
basenames = ['classic', 'classic_with_babblings_no_aug', 'classic_with_babblings']

agents_abx_matrices = utils.pickle_load(model_path / 'abx_cache.pickle', {})
for basename in tqdm(basenames):
    agent_path = model_path / basename
    if agent_path not in agents_abx_matrices:
        agents_abx_matrices[agent_path] = {}
    agent_abx_matrices = agents_abx_matrices[agent_path]

    agent = ImitativeAgent.reload(str(agent_path))
    # Load vowels and consonants info
    main_dataset = agent.get_main_dataset()
    consonants = main_dataset.phones_infos["consonants"]
    in_consonants = ['b', 'm', 'd', 'n', 'g']
    out_consonants = list(set(consonants) - set(in_consonants))
    vowels = main_dataset.phones_infos["vowels"]
    in_vowels = ['a', 'e^', 'e', 'x', 'x^', 'e~', 'y', 'u', 'o~', 'a~']
    out_vowels = list(set(vowels) - set(in_vowels))

    consonant_babbling_groups = {
        'in_consonant_babbling': [in_consonants],
        'out_consonant_babbling': [out_consonants]
    }

    # Compute ABX on VCV triplets
    distance_signature = abx_utils.get_distance_signature(distance)
    #if distance_signature in agent_abx_matrices: continue
    agent_lab = agent.get_datasplit_lab(2)
    agent_features = agent.repeat_datasplit(2)
    consonants_indexes = abx_utils.get_datasets_phones_indexes(agent_lab, consonants, vowels)
    abx_matrix = abx_utils.get_abx_matrix(consonants, consonants_indexes, agent_features, distance, ABX_NB_SAMPLES,
                                          seed=43)
    agent_abx_matrices[distance_signature] = abx_matrix
    utils.pickle_dump(model_path / 'abx_cache.pickle', agents_abx_matrices)

for basename in basenames:
    agent_path = model_path / basename
    distance_signature = abx_utils.get_distance_signature(distance)
    agent_abx_matrix = agents_abx_matrices[agent_path][distance_signature]
    in_out_score = abx_utils.get_groups_score(consonants, agent_abx_matrix, consonant_babbling_groups)
    groups_score = abx_utils.get_groups_score(consonants, agent_abx_matrix,
                                              main_dataset.phones_infos["consonant_groups"])
    global_score = abx_utils.get_global_score(agent_abx_matrix)
    print(basename, in_out_score)
    print(basename, groups_score)
    print(basename, global_score)