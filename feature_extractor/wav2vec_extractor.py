from pathlib import Path

import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

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


class Wav2Vec2Extractor:
    def __init__(self, model_name, num_layers, sampling_rate):

        self.model = self._load_model(Wav2Vec2Model, model_name)
        self.model.eval()
        self.feature_extractor = self._load_feature_extractor(Wav2Vec2FeatureExtractor, model_name)
        #Wav2Vec2FeatureExtractor.from_pretrained(model_name)

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

    def _load_feature_extractor(self, extractor_class, extractor_name):
        try:
            return extractor_class.from_pretrained(extractor_name, cache_dir=EXTRACTORS_PATH, local_files_only=True)
        except Exception as e:
            print(f"Couldn't load feature extractor locally: {e}")
            print("Attempting to download...")
            return extractor_class.from_pretrained(extractor_name, cache_dir=EXTRACTORS_PATH)


    def extract_features(self, wav_seqs, lengths=None):
        inputs = self.feature_extractor(wav_seqs, sampling_rate=self.sampling_rate, return_tensors='pt')
        inputs = inputs['input_values'].squeeze(0).to(self.model.device)
        outputs = self.model(inputs)
        features = outputs.hidden_states[self.num_layers]

        adjusted_lengths, seqs_mask = None, None
        if lengths is not None:
            # Mask features that are out of boundaries
            length_ratio = features.shape[1] / inputs.shape[1]
            adjusted_lengths = torch.ceil(lengths.float() * length_ratio).long().to('cpu')
            seqs_mask = torch.arange(features.shape[1])[None, :] < adjusted_lengths[:, None]
            seqs_mask = seqs_mask.to(self.model.device)
            features = features * seqs_mask.unsqueeze(-1)

        return features, adjusted_lengths, seqs_mask



