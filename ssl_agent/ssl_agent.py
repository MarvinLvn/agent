import torch
import pickle
import yaml
import os
from lib.base_agent import BaseAgent
from lib.wav_source_dataloader import get_dataloaders as get_sound_source_loaders

from lib.nn.simple_lstm import LSTM_FF
from lib.nn.feedforward import FeedForward
from lib.nn.loss import compute_jerk_loss

from inverse_model.inverse_model import InverseModel
from ssl_agent_nn import SSLAgentNN
from synthesizer.synthesizer import Synthesizer
from feature_extractor.wav2vec_extractor import Wav2Vec2Extractor
from vocoder.hifigan_vocoder import HifiGAN
from pathlib import Path

VOCODERS_PATH = Path(__file__).parent.resolve() / "../out/vocoder"
SYNTHESIZERS_PATH = Path(__file__).parent.resolve() / "../out/synthesizer"
EXTRACTORS_PATH = Path(__file__).parent.resolve() / "../out/feature_extractor"


class SSLAgent(BaseAgent):
    def __init__(self, config):
        self.config = config
        self.datasplits = None

        self.cut_silences = self.config['dataset'].get('cut_silences', False)
        self.max_len = self.config['dataset'].get('max_len', None)
        # 1) We load the feature extractor (wav -> feat)
        self.feature_extractor = Wav2Vec2Extractor(model_name=self.config['feature_extractor']['name'],
                                                   num_layers=self.config['feature_extractor']['layer'],
                                                   sampling_rate=self.config['feature_extractor']['sampling_rate'])

        # 2) We load the synthesizer (art_params -> mel_spectro)
        self.synthesizer = Synthesizer.reload("%s/%s" % (SYNTHESIZERS_PATH, config["synthesizer"]["name"]), load_nn=True)

        # 3) We load the HiFi-GAN vocoder (mel_spectro -> wav)
        self.vocoder = HifiGAN(self.config['vocoder']['name'])

        # 4) We build the inverse model (+ discriminator model)
        self._build_nn(self.config["model"])
        self.nn.eval()

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
                lr=self.config["training"]["discriminator_model_learning_rate"]
            )
        return optimizers

    def get_losses_fn(self):
        art_scaler_var = torch.FloatTensor(self.synthesizer.art_scaler.var_).to("cuda")
        bce_loss = torch.nn.BCELoss()

        def inverse_model_loss(art_seqs_pred, sound_seqs_pred, sound_seqs, seqs_mask,
                               predicted_labels):
            reconstruction_loss = mse(sound_seqs_pred, sound_seqs, seqs_mask)

            art_seqs_pred = art_seqs_pred * art_scaler_var
            jerk_loss = compute_jerk_loss(art_seqs_pred, seqs_mask)

            # Fake labels are real for the generator
            real_labels = torch.full(predicted_labels.shape, 1, dtype=torch.float).to(predicted_labels.device)
            fool_discrimination_loss = bce_loss(predicted_labels, real_labels)

            total_loss = (
                    reconstruction_loss +
                    jerk_loss * self.config["training"]["jerk_loss_weight"] +
                    fool_discrimination_loss * self.config["training"]["discriminator_loss_weight"]
            )

            return total_loss, reconstruction_loss, jerk_loss, fool_discrimination_loss

        def mse(seqs_pred, seqs, seqs_mask):
            reconstruction_error = (seqs_pred - seqs) ** 2
            reconstruction_loss = reconstruction_error[seqs_mask].mean()
            return reconstruction_loss

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

        return {"inverse_model": inverse_model_loss, "mse": mse, "bce": bce, "cosine": cosine_distance}

    def save(self, save_path):
        with open(save_path + "/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        with open(save_path + "/datasplits.pickle", "wb") as f:
            pickle.dump(self.datasplits, f)
        torch.save(self.nn.state_dict(), save_path + "/nn_weights.pt")

    @staticmethod
    def reload(save_path, load_nn=True):
        with open(save_path + "/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        agent = SSLAgent(config)

        with open(save_path + "/datasplits.pickle", "rb") as f:
            agent.datasplits = pickle.load(f)

        if load_nn:
            agent.nn.load_state_dict(torch.load(save_path + "/nn_weights.pt"))
            agent.nn.eval()

        return agent

    def repeat(self, sound_seq):
        nn_input = torch.FloatTensor(sound_seq).to("cuda")
        with torch.no_grad():
            sound_seq_estimated, art_seq_estimated_unscaled = self.nn(
                nn_input[None, :, :]
            )
            if hasattr(self.nn, 'art_quantizer'):
                _, art_unit_seq, _, _ = self.nn.art_quantizer.encode(
                    art_seq_estimated_unscaled,
                )
        sound_seq_estimated = sound_seq_estimated[0].cpu().numpy()
        art_seq_estimated_unscaled = art_seq_estimated_unscaled[0].cpu().numpy()
        art_seq_estimated = self.synthesizer.art_scaler.inverse_transform(
            art_seq_estimated_unscaled
        )

        sound_seq_repeated = self.synthesizer.synthesize(art_seq_estimated)
        if hasattr(self.nn, 'art_quantizer'):
            art_unit_seq = art_unit_seq[0].cpu().numpy()
            return {
                "sound_repeated": sound_seq_repeated,
                "sound_estimated": sound_seq_estimated,
                "art_estimated": art_seq_estimated,
                "art_units": art_unit_seq,
            }
        return {
            "sound_repeated": sound_seq_repeated,
            "sound_estimated": sound_seq_estimated,
            "art_estimated": art_seq_estimated,
        }

