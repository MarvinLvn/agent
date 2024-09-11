import torch
from torch.nn.utils.rnn import pad_sequence

from lib import utils
from lib.dataset_wrapper import Dataset


def pad_collate(batch):
    sound_seqs, sound_seqs_len, sound_seqs_mask, source_seqs, source_seqs_len, source_seqs_mask = zip(*batch)

    sound_seqs_len = torch.LongTensor(sound_seqs_len)
    sorted_indices = sound_seqs_len.argsort(descending=True)
    sound_seqs_padded = pad_sequence(sound_seqs, batch_first=True, padding_value=0)
    sound_seqs_mask = pad_sequence(sound_seqs_mask, batch_first=True, padding_value=0)

    source_seqs_len = torch.LongTensor(source_seqs_len)
    source_seqs_padded = pad_sequence(source_seqs, batch_first=True, padding_value=0)
    source_seqs_mask = pad_sequence(source_seqs_mask, batch_first=True, padding_value=0)

    return (
        sound_seqs_padded[sorted_indices],
        sound_seqs_len[sorted_indices],
        sound_seqs_mask[sorted_indices],
        source_seqs_padded[sorted_indices],
        source_seqs_len[sorted_indices],
        source_seqs_mask[sorted_indices],
    )


class SoundSourceDataset(torch.utils.data.Dataset):
    def __init__(self, sound_seqs, source_seqs):
        self.sound_seqs = sound_seqs
        self.sound_seqs_len = [len(sound_seq) for sound_seq in sound_seqs]
        self.sound_seqs_mask = [torch.BoolTensor([1] * seq_len) for seq_len in self.sound_seqs_len]

        self.source_seqs = source_seqs
        self.source_seqs_len = [len(source_seq) for source_seq in source_seqs]
        self.source_seqs_mask = [torch.BoolTensor([1] * source_seq_len) for source_seq_len in self.source_seqs_len]

        self.len = len(sound_seqs)

    def __getitem__(self, idx):
        sound_seq = self.sound_seqs[idx]
        sound_len = self.sound_seqs_len[idx]
        sound_mask = self.sound_seqs_mask[idx]
        source_seq = self.source_seqs[idx]
        source_len = self.source_seqs_len[idx]
        source_mask = self.source_seqs_mask[idx]
        return sound_seq, sound_len, sound_mask, source_seq, source_len, source_mask

    def __len__(self):
        return self.len


def get_dataloaders(dataset_config, datasplits, cut_silences=True, max_len=None):
    dataset = Dataset(dataset_config['name'])
    sound_data = dataset.get_items_data(dataset_config["sound_type"], cut_silences=cut_silences, format='.wav')
    source_data = dataset.get_items_data(dataset_config["source_type"], cut_silences=cut_silences, format='.npy')
    items_name = list(sound_data.keys())

    if datasplits is None:
        datasplits = utils.shuffle_and_split(items_name,
                                             dataset_config["datasplits_size"],
                                             dataset_config["datasplit_seed"])

    dataloaders = []

    for i_datasplit, split_items in enumerate(datasplits):
        if len(split_items) == 0:
            dataloaders.append(None)
            continue

        split_sound_seqs = [sound_data[split_item] for split_item in split_items]
        split_source_seqs = [source_data[split_item] for split_item in split_items]

        split_sound_seqs = [torch.FloatTensor(split_sound_seq) for split_sound_seq in split_sound_seqs]
        split_source_seqs = [torch.FloatTensor(split_source_seq) for split_source_seq in split_source_seqs]

        if max_len is not None:
            split_sound_seqs = [sub_seg for seg in split_sound_seqs for sub_seg in torch.split(seg, max_len)]
            split_source_seqs = [sub_seg for seg in split_source_seqs for sub_seg in torch.split(seg, max_len//160)]

        split_dataloader = torch.utils.data.DataLoader(
            SoundSourceDataset(split_sound_seqs, split_source_seqs),
            batch_size=dataset_config["batch_size"],
            shuffle=dataset_config["shuffle_between_epochs"],
            num_workers=dataset_config["num_workers"],
            collate_fn=pad_collate,
        )
        dataloaders.append(split_dataloader)
    return datasplits, dataloaders
