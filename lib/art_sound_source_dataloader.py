import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from lib import utils
from lib.dataset_wrapper import Dataset


def pad_collate(batch):
    art_seqs, sound_seqs, source_seqs, seqs_len, seqs_mask = zip(*batch)

    seqs_len = torch.LongTensor(seqs_len)
    sorted_indices = seqs_len.argsort(descending=True)

    art_seqs_padded = pad_sequence(art_seqs, batch_first=True, padding_value=0)
    sound_seqs_padded = pad_sequence(sound_seqs, batch_first=True, padding_value=0)

    source_seqs_padded = pad_sequence(source_seqs, batch_first=True, padding_value=0)
    seqs_mask = pad_sequence(seqs_mask, batch_first=True, padding_value=0)

    return (
        art_seqs_padded[sorted_indices],
        sound_seqs_padded[sorted_indices],
        source_seqs_padded[sorted_indices],
        seqs_len[sorted_indices],
        seqs_mask[sorted_indices],
    )


class ArtSoundSourceDataset(torch.utils.data.Dataset):
    def __init__(self, art_seqs, sound_seqs, source_seqs):
        self.art_seqs = art_seqs
        self.sound_seqs = sound_seqs
        self.source_seqs = source_seqs
        self.seqs_len = [len(art_seq) for art_seq in art_seqs]
        self.seqs_mask = [torch.BoolTensor([1] * seq_len) for seq_len in self.seqs_len]
        self.len = len(art_seqs)

        for i in range(self.len):
            assert len(art_seqs[i]) == len(sound_seqs[i])
            assert len(art_seqs[i]) == len(source_seqs[i])

    def __getitem__(self, idx):
        art_seq = self.art_seqs[idx]
        sound_seq = self.sound_seqs[idx]
        source_seq = self.source_seqs[idx]
        seq_len = self.seqs_len[idx]
        seq_mask = self.seqs_mask[idx]
        return art_seq, sound_seq, source_seq, seq_len, seq_mask

    def __len__(self):
        return self.len


def get_dataloaders(dataset_config, art_scaler, sound_scaler, datasplits, cut_silences=True, fit=True, transform=True, format='.bin'):
    dataset = Dataset(dataset_config["name"])
    art_data = dataset.get_items_data(dataset_config["art_type"], cut_silences=cut_silences, format=format)
    sound_data = dataset.get_items_data(dataset_config["sound_type"], cut_silences=cut_silences, format=format)
    source_data = dataset.get_items_data(dataset_config["source_type"], cut_silences=cut_silences, format=format)
    items_name = list(sound_data.keys())

    if datasplits is None:
        datasplits = utils.shuffle_and_split(
            items_name,
            dataset_config["datasplits_size"],
            dataset_config["datasplit_seed"],
        )
    dataloaders = []

    for i_datasplit, split_items in enumerate(datasplits):
        if len(split_items) == 0:
            dataloaders.append(None)
            continue
        split_art_seqs = [art_data[split_item] for split_item in split_items]
        split_sound_seqs = [sound_data[split_item] for split_item in split_items]
        split_source_seqs = [source_data[split_item] for split_item in split_items]

        if i_datasplit == 0:
            split_art_concat = np.concatenate(split_art_seqs)
            split_sound_concat = np.concatenate(split_sound_seqs)
            if fit:
                art_scaler.fit(split_art_concat)
                sound_scaler.fit(split_sound_concat)

        if transform:
            split_art_seqs = [
                art_scaler.transform(split_art_seq)
                for split_art_seq in split_art_seqs
            ]
            split_sound_seqs = [
                sound_scaler.transform(split_sound_seq)
                for split_sound_seq in split_sound_seqs
            ]
        split_art_seqs = [
            torch.FloatTensor(split_art_seq)
            for split_art_seq in split_art_seqs
        ]
        split_sound_seqs = [
            torch.FloatTensor(split_sound_seq)
            for split_sound_seq in split_sound_seqs
        ]
        split_source_seqs = [
            torch.FloatTensor(split_source_seq)
            for split_source_seq in split_source_seqs
        ]

        split_dataloader = torch.utils.data.DataLoader(

            ArtSoundSourceDataset(split_art_seqs, split_sound_seqs, split_source_seqs),
            batch_size=dataset_config["batch_size"],
            shuffle=dataset_config["shuffle_between_epochs"],
            num_workers=dataset_config["num_workers"],
            collate_fn=pad_collate,
        )
        dataloaders.append(split_dataloader)
    return datasplits, dataloaders
