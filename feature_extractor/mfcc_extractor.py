import torch
from torchaudio.transforms import MFCC, ComputeDeltas
from torch import nn

class MFCCExtractor(nn.Module):
    def __init__(self, n_mfcc=13, n_fft=640, hop_length=320,
                 n_mels=26, add_delta=False, sampling_rate=16000):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.add_delta = add_delta
        self.feat_dim = n_mfcc*3 if add_delta else n_mfcc

        self.mfcc_transform = MFCC(
            sample_rate=sampling_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )
        if add_delta:
            self.delta_transform = ComputeDeltas()

    def forward(self, wav_seqs, lengths=None):
        features = self.mfcc_transform(wav_seqs)
        if self.add_delta:
            delta1 = self.delta_transform(features)
            delta2 = self.delta_transform(delta1)
            features = torch.cat([features, delta1, delta2], dim=1)

        features = features.transpose(2, 1)
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True)
        features = (features - mean) / (std + 1e-8)

        adjusted_lengths, seqs_mask = None, None
        if lengths is not None:
            # Mask features that are out of boundaries
            max_len = features.shape[1]
            ratio = lengths[0] / max_len
            adjusted_lengths = torch.floor(lengths / ratio).long().to('cpu')
            seqs_mask = torch.arange(max_len, device='cpu')[None, :] < adjusted_lengths[:, None]
            seqs_mask = seqs_mask.to(features.device)
            features = features[:, :max_len, :]
            features = features * seqs_mask.unsqueeze(-1)
        return features, adjusted_lengths, seqs_mask
