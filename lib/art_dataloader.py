import numpy as np
import random
import torch
from torch.nn.utils.rnn import pad_sequence

from lib import utils
from lib.dataset_wrapper import Dataset


def pad_collate(batch):
    art_seqs, seqs_len, seqs_mask = zip(*batch)

    seqs_len = torch.LongTensor(seqs_len)
    sorted_indices = seqs_len.argsort(descending=True)

    art_seqs_padded = pad_sequence(art_seqs, batch_first=True, padding_value=0)
    seqs_mask = pad_sequence(seqs_mask, batch_first=True, padding_value=0)

    return (
        art_seqs_padded[sorted_indices],
        seqs_len[sorted_indices],
        seqs_mask[sorted_indices],
    )


class ArtDataset(torch.utils.data.Dataset):
    def __init__(self, art_seqs):
        self.art_seqs = art_seqs
        self.seqs_len = [len(art_seq) for art_seq in art_seqs]
        self.seqs_mask = [torch.BoolTensor([1] * seq_len) for seq_len in self.seqs_len]
        self.len = len(art_seqs)

    def __getitem__(self, idx):
        art_seq = self.art_seqs[idx]
        seq_len = self.seqs_len[idx]
        seq_mask = self.seqs_mask[idx]
        return art_seq, seq_len, seq_mask

    def __len__(self):
        return self.len

def get_dataloaders(dataset_config, art_scaler, datasplits):
    datasets_art_data = {}
    if datasplits is None:
        datasplits = {}

    for dataset_name in dataset_config["names"]:
        dataset = Dataset(dataset_name)
        dataset_art_data = dataset.get_items_data(
            dataset_config["art_type"], cut_silences=True
        )
        dataset_items_name = list(dataset_art_data.keys())
        if dataset_name not in datasplits:
            dataset_datasplits = utils.shuffle_and_split(
                dataset_items_name, dataset_config["datasplits_size"]
            )
            datasplits[dataset_name] = dataset_datasplits

        datasets_art_data[dataset_name] = dataset_art_data

    dataloaders = []

    for i_datasplit in range(3):
        split_art_seqs = []
        for dataset_name, dataset_art_data in datasets_art_data.items():
            dataset_split_items = datasplits[dataset_name][i_datasplit]
            dataset_art_data = datasets_art_data[dataset_name]
            split_art_seqs += [
                dataset_art_data[split_item] for split_item in dataset_split_items
            ]

        # We don't rescale if the art_scaler has already been fit
        # ML noticed scaling on the training set can yield quite different
        # mean values depending how it has been split
        if i_datasplit == 0 and not hasattr(art_scaler, 'mean_'):
            split_art_concat = np.concatenate(split_art_seqs)
            art_scaler.fit(split_art_concat)

        split_art_seqs = [
            torch.FloatTensor(art_scaler.transform(split_art_seq))
            for split_art_seq in split_art_seqs
        ]
        
        split_dataloader = torch.utils.data.DataLoader(
            ArtDataset(split_art_seqs),
            batch_size=dataset_config["batch_size"],
            shuffle=dataset_config["shuffle_between_epochs"],
            num_workers=dataset_config["num_workers"],
            collate_fn=pad_collate,
        )
        dataloaders.append(split_dataloader)
    return datasplits, dataloaders

class ArtFrameDataset(torch.utils.data.Dataset):
    def __init__(self, art_locs):
        self.art_locs = art_locs
        self.len = len(art_locs)

    def __getitem__(self, idx):
        art_locs = self.art_locs[idx]
        return art_locs

    def __len__(self):
        return self.len

def get_dataloaders_frame_level(dataset_config, art_scaler, datasplits):
    datasets_art_data = {}
    if datasplits is None:
        datasplits = {}

    # 1) Split sequences
    for dataset_name in dataset_config["names"]:
        dataset = Dataset(dataset_name)
        dataset_art_data = dataset.get_items_data(
            dataset_config["art_type"], cut_silences=True
        )
        dataset_items_name = list(dataset_art_data.keys())
        if dataset_name not in datasplits:
            dataset_datasplits = utils.shuffle_and_split(
                dataset_items_name, dataset_config["datasplits_size"], dataset_config["datasplit_seed"]
            )
            datasplits[dataset_name] = dataset_datasplits

        datasets_art_data[dataset_name] = dataset_art_data

    dataloaders = []

    # Loop through train, val, test
    for i_datasplit in range(3):
        split_art_seqs = []
        for dataset_name, dataset_art_data in datasets_art_data.items():
            # Get list of sequences
            dataset_split_items = datasplits[dataset_name][i_datasplit]
            # Get list of articulatory trajectories
            dataset_art_data = datasets_art_data[dataset_name]
            split_art_seqs += [
                dataset_art_data[split_item] for split_item in dataset_split_items
            ]

        split_art_frames = np.concatenate(split_art_seqs)
        # We don't rescale if the art_scaler has already been fit
        # ML noticed scaling on the training set can yield quite different
        # mean values depending how it has been split
        if i_datasplit == 0 and not hasattr(art_scaler, 'mean_'):
            art_scaler.fit(split_art_frames)

        split_art_frames = art_scaler.transform(split_art_frames)

        # Shuffle frames
        if 'datasplit_seed' in dataset_config:
            np.random.seed(dataset_config['datasplit_seed'])
        np.random.shuffle(split_art_frames)

        split_dataloader = torch.utils.data.DataLoader(
            ArtFrameDataset(split_art_frames),
            batch_size=dataset_config["batch_size"],
            shuffle=dataset_config["shuffle_between_epochs"],
            num_workers=dataset_config["num_workers"],
        )
        dataloaders.append(split_dataloader)
    return datasplits, dataloaders