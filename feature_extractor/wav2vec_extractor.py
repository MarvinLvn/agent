from pathlib import Path

import torch
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining

# Need to:
# pip install transformers
# pip install -U flash-attn --no-build-isolation

# Available wav2vec models can be found there:
# https://huggingface.co/models?other=wav2vec2
# And documentation:
# https://huggingface.co/docs/transformers/main/model_doc/wav2vec2

# Pretrained on 10k hours:
# https://huggingface.co/facebook/wav2vec2-base-10k-voxpopuli

# Pretrained on 10k hours and fine-tuned on french
#  https://huggingface.co/facebook/wav2vec2-base-10k-voxpopuli-ft-fr

EXTRACTORS_PATH = Path(__file__).parent.resolve() / "../out/feature_extractor"


class Wav2Vec2Extractor(nn.Module):
    def __init__(self, model_name, num_layers, sampling_rate):
        super().__init__()
        self.model = self._load_model(Wav2Vec2ForPreTraining, model_name)
        self.model.eval()

        self.num_layers = num_layers
        self.sampling_rate = sampling_rate
        self.feat_dim = self.model.config.output_hidden_size
        if self.num_layers < 0 or self.num_layers > self.model.config.num_hidden_layers:
            raise ValueError(f"num_layers should be between 0 and {self.model.config.num_hidden_layers}")

    def _load_model(self, model_class, model_name):
        # We avoid checking for updates if the model is already present in cache
        # So that one can train the model without internet access
        try:
            return model_class.from_pretrained(
                model_name,
                cache_dir=EXTRACTORS_PATH,
                output_hidden_states=True,
                local_files_only=True
            )
        except Exception as e:
            print(f"Couldn't load model locally: {e}")
            print("Attempting to download...")
            return model_class.from_pretrained(
                model_name,
                cache_dir=EXTRACTORS_PATH,
                output_hidden_states=True
            )

    def forward(self, wav_seqs, lengths=None):
        outputs = self.model(wav_seqs)
        features = outputs.hidden_states[self.num_layers]

        adjusted_lengths, seqs_mask = None, None
        if lengths is not None:
            # Mask features that are out of boundaries
            adjusted_lengths = self.get_output_lengths(lengths).to('cpu')
            seqs_mask = torch.arange(features.shape[1])[None, :] < adjusted_lengths[:, None]
            seqs_mask = seqs_mask.to(self.model.device)
            features = features * seqs_mask.unsqueeze(-1)
        return features, adjusted_lengths, seqs_mask

    def get_output_lengths(self, input_lengths, add_adapter=None):
        # Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py
        add_adapter = self.model.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.floor_divide(input_length - kernel_size, stride) + 1

        for kernel_size, stride in zip(self.model.config.conv_kernel, self.model.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.model.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.model.adapter_stride)

        return input_lengths



