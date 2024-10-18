import torch
import pickle
import yaml
import numpy as np
from lib.base_agent import BaseAgent
from lib.wav_source_dataloader import get_dataloaders as get_sound_source_loaders
from lib.nn.simple_lstm import LSTM_FF
from lib.nn.feedforward import FeedForward
from lib.nn.data_scaler import DataScaler
from lib.dataset_wrapper import Dataset

from inverse_model.inverse_model import InverseModel
from ssl_agent.ssl_agent_nn import SSLAgentNN
from synthesizer.synthesizer import Synthesizer
from feature_extractor.wav2vec_extractor import Wav2Vec2Extractor
from feature_extractor.mfcc_extractor import MFCCExtractor
from vocoder.hifigan_vocoder import HifiGAN
from pathlib import Path

VOCODERS_PATH = Path(__file__).parent.resolve() / "../out/vocoder"
SYNTHESIZERS_PATH = Path(__file__).parent.resolve() / "../out/synthesizer"
EXTRACTORS_PATH = Path(__file__).parent.resolve() / "../out/feature_extractor"


class SSLAgent(BaseAgent):
    def __init__(self, config, device='cuda'):
        self.config = config
        self.datasplits = None

        self.cut_silences = self.config['dataset'].get('cut_silences', False)
        self.max_len = self.config['dataset'].get('max_len', None)
        # 1) We load the feature extractor (wav -> feat)
        if self.config['feature_extractor']['name'] == 'mfcc':
            self.feature_extractor = MFCCExtractor(n_mfcc=self.config['feature_extractor']['n_mfcc'],
                                                   n_fft=self.config['feature_extractor']['n_fft'],
                                                   hop_length=self.config['feature_extractor']['hop_length'],
                                                   n_mels=self.config['feature_extractor']['n_mels'],
                                                   add_delta=self.config['feature_extractor']['add_delta'],
                                                   sampling_rate=self.config['feature_extractor']['sampling_rate'])
            self.feature_extractor.mfcc_transform.to(device)
            self.feature_extractor.delta_transform.to(device)
        else:
            self.feature_extractor = Wav2Vec2Extractor(model_name=self.config['feature_extractor']['name'],
                                                       num_layers=self.config['feature_extractor']['layer'],
                                                       sampling_rate=self.config['feature_extractor']['sampling_rate'])
            self.feature_extractor.model.to(device)

        # 2) We load the synthesizer (art_params -> mel_spectro)
        self.synthesizer = Synthesizer.reload("%s/%s" % (SYNTHESIZERS_PATH, config["synthesizer"]["name"]), load_nn=True)
        self.synthesizer.nn.to(device)
        self.synthesizer.sound_scaler_diff = DataScaler.from_standard_scaler(self.synthesizer.sound_scaler).to(device)

        # 3) We load the HiFi-GAN vocoder (mel_spectro -> wav)
        self.vocoder = HifiGAN(self.config['vocoder']['name'])
        self.vocoder.generator.to(device)

        # 4) We build the inverse model (+ discriminator model)
        self._build_nn(self.config["model"])
        self.nn.eval()

        # 5) We move everything to the right device
        self.nn.to(device)
        self.MAX_LEN = 16_000 * 2 # to avoid OOM during inference
        self.FRAME_DUR = 320 # number of audio frames in a source or wav2vec 2.0 frame


    def _build_nn(self, model_config):
        self.art_dim = self.synthesizer.art_dim
        self.feat_dim = self.feature_extractor.feat_dim

        inverse_model = InverseModel.build_lstm(
            self.feat_dim,
            self.art_dim,
            model_config["inverse_model"]["hidden_size"],
            model_config["inverse_model"]["num_layers"],
            model_config["inverse_model"]["dropout_p"],
            model_config["inverse_model"]["bidirectional"])

        discriminator_model = None
        if 'discriminator_model' in self.config['model']:
            if 'ff' in self.config['model']['discriminator_model']:
                discriminator_model = FeedForward(
                    self.art_dim,
                    1,
                    model_config["discriminator_model"]['ff']["hidden_layers"],
                    model_config["discriminator_model"]['ff']["activation"],
                    model_config["discriminator_model"]['ff']["dropout_p"],
                    model_config["discriminator_model"]['ff']["batch_norm"],
                    add_sigmoid=True,
                )
            elif 'rnn' in self.config['model']['discriminator_model']:
                discriminator_model = LSTM_FF(
                    self.art_dim,
                    1,
                    model_config["discriminator_model"]['rnn']['lstm']["hidden_size"],
                    model_config["discriminator_model"]['rnn']['lstm']["num_layers"],
                    model_config["discriminator_model"]['rnn']['lstm']["dropout_p"],
                    model_config["discriminator_model"]['rnn']['lstm']["bidirectional"],
                )
            else:
                raise ValueError('Discriminator should either be rnn or ff.')
        self.nn = SSLAgentNN(inverse_model, self.feature_extractor,
                             self.synthesizer, self.vocoder, discriminator_model).to("cuda")

    def get_dataloaders(self):
        datasplits, dataloaders = get_sound_source_loaders(self.config["dataset"], self.datasplits,
                                                           cut_silences=self.cut_silences,
                                                           max_len=self.max_len)
        self.datasplits = datasplits
        return dataloaders

    def get_optimizers(self):
        optimizers = {}
        optimizers["inverse_model"] = torch.optim.Adam(
            self.nn.inverse_model.parameters(),
            lr=self.config["training"]["learning_rate"]
        )

        if "discriminator_model" in self.config["model"]:
            optimizers["discriminator_model"] = torch.optim.Adam(
                self.nn.discriminator_model.parameters(),
                lr=self.config["training"]["discriminator_learning_rate"]
            )
        return optimizers

    def get_losses_fn(self):
        bce_loss = torch.nn.BCELoss()

        def inverse_model_loss(sound_seqs_pred, sound_seqs, seqs_mask, predicted_labels):
            reconstruction_loss = cosine_distance(sound_seqs_pred, sound_seqs, seqs_mask)

            total_loss = reconstruction_loss
            # Fake labels are real for the generator
            fool_discrimination_loss = torch.FloatTensor([0.0])
            if predicted_labels is not None:
                real_labels = torch.full(predicted_labels.shape, 1, dtype=torch.float).to(predicted_labels.device)
                fool_discrimination_loss = bce_loss(predicted_labels, real_labels)
                total_loss += fool_discrimination_loss * self.config["training"]["discriminator_loss_weight"]
            return total_loss, reconstruction_loss, fool_discrimination_loss

        def bce(predicted_labels, real_labels):
            return bce_loss(predicted_labels, real_labels)

        def cosine_distance(feat_seqs, feat_seqs_estimated, mask):
            similarity = torch.nn.functional.cosine_similarity(feat_seqs * mask.unsqueeze(-1),
                                                               feat_seqs_estimated * mask.unsqueeze(-1),
                                                               dim=-1)
            distance = (1 - similarity) * mask
            total_distance = distance.sum()
            total_mask = mask.sum()

            if total_mask > 0:
                mean_distance = total_distance / total_mask
            else:
                mean_distance = torch.tensor(0.0, device=feat_seqs.device)
            return mean_distance

        return {"inverse_loss": inverse_model_loss, "bce": bce, "cosine": cosine_distance}

    def save(self, save_path):
        if isinstance(save_path, str):
            save_path = Path(save_path)
        with open(save_path / "config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        with open(save_path / "datasplits.pickle", "wb") as f:
            pickle.dump(self.datasplits, f)
        torch.save(self.nn.state_dict(), save_path / "nn_weights.pt")

    @staticmethod
    def reload(save_path, device='cuda'):
        if isinstance(save_path, str):
            save_path = Path(save_path)
        with open(save_path / "config.yaml", "r") as f:
            config = yaml.safe_load(f)
        agent = SSLAgent(config, device=device)

        with open(save_path / "datasplits.pickle", "rb") as f:
            agent.datasplits = pickle.load(f)

        agent.nn.load_state_dict(torch.load(save_path / "nn_weights.pt", map_location=device))
        agent.nn.eval()

        return agent

    def repeat(self, sound_seq, source_seq, device='cuda', lightweight=False):
        sound_seq = torch.FloatTensor(sound_seq).to(device)
        sound_seq = sound_seq[None, :]
        source_seq = torch.FloatTensor(source_seq).to(device)
        source_seq = source_seq[None, :]
        with torch.no_grad():
            # 1) Extract representations
            feat_seq, _, _ = self.feature_extractor(sound_seq)
            # 2) Run inverse model
            art_seq_estimated = self.nn(feat_seq)

            if lightweight:
                return {
                    "feat_seq": feat_seq[0].cpu().numpy(),
                    "art_estimated": art_seq_estimated[0].cpu().numpy(),
                }

            # 3) Run synthesizer
            source_seq = torch.nn.functional.interpolate(source_seq.permute(0, 2, 1),
                                                         size=art_seq_estimated.shape[1]).permute(0, 2, 1)
            art_source_seq_estimated = torch.cat((art_seq_estimated, source_seq), dim=2).to(device)
            mel_spec_repeated = self.synthesizer.nn(art_source_seq_estimated)
            mel_spec_repeated = self.synthesizer.sound_scaler_diff.inverse_transform(mel_spec_repeated)

            # 4) Run vocoder
            audio_seq_repeated = self.vocoder.resynth(mel_spec_repeated.permute(0, 2, 1))

            # 5) Re-extract representations
            feat_seq_repeated, _, _ = self.feature_extractor(audio_seq_repeated)

            min_len = min(feat_seq.shape[1], feat_seq_repeated.shape[1])
            feat_seq = feat_seq[:, :min_len, :]
            feat_seq_repeated = feat_seq_repeated[:, :min_len, :]

            return {
                "feat_seq": feat_seq[0].cpu().numpy(),
                "art_estimated": art_seq_estimated[0].cpu().numpy(),
                "feat_seq_repeated": feat_seq_repeated[0].cpu().numpy(),
                "mel_spec_repeated": mel_spec_repeated[0].cpu().numpy(),
                "audio_seq_repeated": audio_seq_repeated[0].cpu().numpy()
            }

    # Should be moved to BaseAgent once Imitative and Communicative Agent will be removed
    def get_main_dataset(self):
        return Dataset(self.config["dataset"]["name"])

    def get_datasplit_lab(self, datasplit_index=None):
        datasplit_lab = {}

        dataset = self.get_main_dataset()
        dataset_name = self.config["dataset"]["name"]

        if datasplit_index is None:
            dataset_lab = dataset.lab
        else:
            dataset_split = self.datasplits[datasplit_index]
            dataset_lab = {
                item_name: dataset.lab[item_name] for item_name in dataset_split
            }
        datasplit_lab[dataset_name] = dataset_lab
        return datasplit_lab

    def repeat_datasplit(self, datasplit_index=None, cut_long=False, device='cuda'):
        agent_features = {}
        sound_type = self.config["dataset"]["sound_type"]
        dataset_name = self.config["dataset"]["name"]
        dataset_features = {}

        dataset = Dataset(dataset_name)
        if datasplit_index is None:
            items_name = dataset.get_items_name(sound_type, format='.wav')
        else:
            items_name = self.datasplits[datasplit_index]

        fmt = '.npy'
        if self.config["dataset"]["sound_type"] == 'wav':
            fmt = '.wav'
        items_sound = dataset.get_items_data(self.config["dataset"]["sound_type"], format=fmt)
        items_source = dataset.get_items_data(self.config["dataset"]["source_type"], format='.npy')
        for item_name in items_name:
            item_sound = items_sound[item_name]
            item_source = items_source[item_name]
            if cut_long:
                repetition = self.repeat_lightweight(item_sound, item_source, device=device)
            else:
                repetition = self.repeat(item_sound, item_source, device=device)
            for repetition_type, repetition_data in repetition.items():
                if repetition_type not in dataset_features:
                    dataset_features[repetition_type] = {}
                dataset_features[repetition_type][item_name] = repetition_data

        agent_features[dataset_name] = dataset_features
        return agent_features

    def repeat_datasplit_debug(self, datasplit_index=None):
        agent_features = {}
        sound_type = self.config["dataset"]["sound_type"]
        dataset_name = self.config["dataset"]["name"]
        dataset_features = {}

        dataset = Dataset(dataset_name)
        if datasplit_index is None:
            items_name = dataset.get_items_name(sound_type, format='.wav')[:100]
        else:
            items_name = self.datasplits[datasplit_index][:100]

        fmt = '.npy'
        if self.config["dataset"]["sound_type"] == 'wav':
            fmt = '.wav'
        items_sound = dataset.get_items_data(self.config["dataset"]["sound_type"], format=fmt)
        items_source = dataset.get_items_data(self.config["dataset"]["source_type"], format='.npy')
        for item_name in items_name:
            item_sound = items_sound[item_name]
            item_source = items_source[item_name]
            repetition = self.repeat(item_sound, item_source)
            for repetition_type, repetition_data in repetition.items():
                if repetition_type not in dataset_features:
                    dataset_features[repetition_type] = {}
                dataset_features[repetition_type][item_name] = repetition_data

        agent_features[dataset_name] = dataset_features
        return agent_features