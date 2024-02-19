import torch
import yaml
import os

from lib.base_agent import BaseAgent
from lib.nn.simple_lstm import SimpleLSTM, LSTM_FF
from lib.nn.feedforward import FeedForward
from lib.nn.loss import compute_jerk_loss

from communicative_agent_nn import CommunicativeAgentNN
from synthesizer.synthesizer import Synthesizer
from quantizer.quantizer import Quantizer

SYNTHESIZERS_PATH = os.path.join(os.path.dirname(__file__), "../out/synthesizer")
QUANTIZERS_PATH = os.path.join(os.path.dirname(__file__), "../out/quantizer")

class CommunicativeAgent(BaseAgent):
    def __init__(self, config, load_nn=True):
        self.config = config
        if "use_synth_as_direct_model" in self.config["model"]:
            if self.config["model"]["use_synth_as_direct_model"]:
                if "direct_model" in self.config["model"]:
                    del self.config["model"]["direct_model"]
            else:
                del self.config["model"]["use_synth_as_direct_model"]

        self.synthesizer = Synthesizer.reload(
            "%s/%s" % (SYNTHESIZERS_PATH, config["synthesizer"]["name"]),
            load_nn=load_nn,
        )
        self.sound_quantizer = Quantizer.reload(
            "%s/%s" % (QUANTIZERS_PATH, config["sound_quantizer"]["name"]),
            load_nn=load_nn,
        )
        self.nb_speakers = len(self.sound_quantizer.config["dataset"]["names"])
        self.sound_scaler = self.sound_quantizer.data_scaler
        self.datasplits = self.sound_quantizer.datasplits
        if load_nn:
            self._build_nn(self.config["model"])
            self.nn.eval()

    def _build_nn(self, model_config):
        self.art_dim = self.synthesizer.art_dim
        self.sound_dim = self.synthesizer.sound_dim
        embedding_dim = self.sound_quantizer.config["model"]["embedding_dim"]

        inverse_model = SimpleLSTM(
            embedding_dim,
            self.art_dim,
            model_config["inverse_model"]["hidden_size"],
            model_config["inverse_model"]["num_layers"],
            model_config["inverse_model"]["dropout_p"],
            model_config["inverse_model"]["bidirectional"],
        )

        if "use_synth_as_direct_model" not in self.config["model"]:
            direct_model = FeedForward(
                self.art_dim,
                self.sound_dim,
                model_config["direct_model"]["hidden_layers"],
                model_config["direct_model"]["activation"],
                model_config["direct_model"]["dropout_p"],
                model_config["direct_model"]["batch_norm"],
            )
        else:
            direct_model = self.synthesizer.nn

        discriminator_model = None
        if 'discriminator_model' in self.config["model"]:
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

        self.nn = CommunicativeAgentNN(
            inverse_model, direct_model, self.sound_quantizer.nn, discriminator_model
        ).to("cuda")

    def get_dataloaders(self):
        return self.sound_quantizer.get_dataloaders()

    def get_artloaders(self):
        return self.synthesizer.get_dataloaders()

    def get_optimizers(self):
        optimizers = {}
        optimizers["inverse_model"] = torch.optim.Adam(
            self.nn.inverse_model.parameters(),
            lr=self.config["training"]["inverse_model_learning_rate"],
        )
        if "use_synth_as_direct_model" not in self.config["model"]:
            optimizers["direct_model"] = torch.optim.Adam(
                self.nn.direct_model.parameters(),
                lr=self.config["training"]["direct_model_learning_rate"],
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

        def inverse_model_loss(art_seqs_pred, sound_seqs_pred, sound_seqs, seqs_mask, predicted_labels):
            reconstruction_error = (sound_seqs_pred - sound_seqs) ** 2
            reconstruction_loss = reconstruction_error[seqs_mask].mean()

            art_seqs_pred = art_seqs_pred * art_scaler_var
            jerk_loss = compute_jerk_loss(art_seqs_pred, seqs_mask)

            # Fake labels are real for the generator
            real_labels = torch.full(predicted_labels.shape, 1, dtype=torch.float).to(predicted_labels.device)
            fool_discrimination_loss = bce_loss(predicted_labels, real_labels)
            total_loss = (
                    reconstruction_loss
                    + jerk_loss * self.config["training"]["jerk_loss_weight"]
                    + fool_discrimination_loss * self.config["training"]["discriminator_loss_weight"]
            )

            return total_loss, reconstruction_loss, jerk_loss, fool_discrimination_loss

        def mse(seqs_pred, seqs, seqs_mask):
            reconstruction_error = (seqs_pred - seqs) ** 2
            reconstruction_loss = reconstruction_error[seqs_mask].mean()
            return reconstruction_loss

        def bce(predicted_labels, real_labels):
            return bce_loss(predicted_labels, real_labels)

        return {"inverse_model": inverse_model_loss, "mse": mse, "bce": bce}

    def save(self, save_path):
        with open(save_path + "/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        torch.save(self.nn.state_dict(), save_path + "/nn_weights.pt")

    @staticmethod
    def reload(save_path, load_nn=True):
        with open(save_path + "/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        agent = CommunicativeAgent(config, load_nn=load_nn)

        if load_nn:
            agent.nn.load_state_dict(torch.load(save_path + "/nn_weights.pt"))
            agent.nn.eval()

        return agent

    def repeat(self, sound_seq):
        nn_input = torch.FloatTensor(self.sound_scaler.transform(sound_seq)).to("cuda")[
            None, :, :
        ]
        with torch.no_grad():
            _, sound_unit_seq, _, _ = self.nn.sound_quantizer.encode(nn_input)
            art_seq_estimated_unscaled = self.nn.inverse_model(sound_unit_seq)
            sound_seq_estimated_unscaled = self.nn.direct_model(
                art_seq_estimated_unscaled
            )
            if hasattr(self.nn, 'art_quantizer'):
                _, art_unit_seq, _, _ = self.nn.art_quantizer.encode(
                    art_seq_estimated_unscaled,
                )

        sound_seq_estimated_unscaled = sound_seq_estimated_unscaled[0].cpu().numpy()
        art_seq_estimated_unscaled = art_seq_estimated_unscaled[0].cpu().numpy()
        art_seq_estimated = self.synthesizer.art_scaler.inverse_transform(
            art_seq_estimated_unscaled
        )
        sound_seq_estimated = self.sound_scaler.inverse_transform(
            sound_seq_estimated_unscaled
        )
        sound_unit_seq = sound_unit_seq[0].cpu().numpy()
        sound_seq_repeated = self.synthesizer.synthesize(art_seq_estimated)
        if hasattr(self.nn, 'art_quantizer'):
            art_unit_seq = art_unit_seq[0].cpu().numpy()
            return {
                "sound_units": sound_unit_seq,
                "sound_repeated": sound_seq_repeated,
                "sound_estimated": sound_seq_estimated,
                "art_estimated": art_seq_estimated,
                "art_units": art_unit_seq,
            }
        return {
            "sound_units": sound_unit_seq,
            "sound_repeated": sound_seq_repeated,
            "sound_estimated": sound_seq_estimated,
            "art_estimated": art_seq_estimated,
        }
