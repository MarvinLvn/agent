import torch
import pickle
import yaml
import os
from sklearn.preprocessing import StandardScaler

from lib.base_agent import BaseAgent
from lib.sound_dataloader import get_dataloaders
from lib.nn.simple_lstm import SimpleLSTM, LSTM_FF
from lib.nn.feedforward import FeedForward
from lib.nn.loss import compute_jerk_loss

from imitative_agent_nn import ImitativeAgentNN
from synthesizer.synthesizer import Synthesizer

SYNTHESIZERS_PATH = os.path.join(os.path.dirname(__file__), "../out/synthesizer")


class ImitativeAgent(BaseAgent):
    def __init__(self, config, load_nn=True):
        self.config = config
        self.sound_scaler = StandardScaler()
        self.datasplits = None
        if load_nn:
            self.synthesizer = Synthesizer.reload(
                "%s/%s" % (SYNTHESIZERS_PATH, config["synthesizer"]["name"])
            )
            self._build_nn(self.config["model"])
            self.nn.eval()

    def _build_nn(self, model_config):
        self.art_dim = self.synthesizer.art_dim
        self.sound_dim = self.synthesizer.sound_dim

        inverse_model = SimpleLSTM(
            self.sound_dim,
            self.art_dim,
            model_config["inverse_model"]["hidden_size"],
            model_config["inverse_model"]["num_layers"],
            model_config["inverse_model"]["dropout_p"],
            model_config["inverse_model"]["bidirectional"],
        )

        direct_model = FeedForward(
            self.art_dim,
            self.sound_dim,
            model_config["direct_model"]["hidden_layers"],
            model_config["direct_model"]["activation"],
            model_config["direct_model"]["dropout_p"],
            model_config["direct_model"]["batch_norm"],
        )

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
        self.nn = ImitativeAgentNN(inverse_model, direct_model, discriminator_model).to("cuda")

    def get_dataloaders(self):
        datasplits, dataloaders = get_dataloaders(
            self.config["dataset"], self.sound_scaler, self.datasplits
        )
        self.datasplits = datasplits
        return dataloaders

    def get_optimizers(self):
        optimizers = {}
        optimizers["inverse_model"] = torch.optim.Adam(
            self.nn.inverse_model.parameters(),
            lr=self.config["training"]["learning_rate"]
        )
        optimizers["direct_model"] = torch.optim.Adam(
                self.nn.direct_model.parameters(),
                lr=self.config["training"]["learning_rate"],
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
                    reconstruction_loss +
                    jerk_loss * self.config["training"]["jerk_loss_weight"] +
                    fool_discrimination_loss * self.config["training"]["discriminator_loss_weight"]
            )

            return total_loss, reconstruction_loss, jerk_loss, fool_discrimination_loss

        def mse(sound_seqs_pred, sound_seqs, seqs_mask):
            reconstruction_error = (sound_seqs_pred - sound_seqs) ** 2
            reconstruction_loss = reconstruction_error[seqs_mask].mean()
            return reconstruction_loss

        def bce(predicted_labels, real_labels):
            return bce_loss(predicted_labels, real_labels)

        return {"inverse_model": inverse_model_loss, "mse": mse, "bce": bce}

    def save(self, save_path):
        with open(save_path + "/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        with open(save_path + "/sound_scaler.pickle", "wb") as f:
            pickle.dump(self.sound_scaler, f)
        with open(save_path + "/datasplits.pickle", "wb") as f:
            pickle.dump(self.datasplits, f)
        torch.save(self.nn.state_dict(), save_path + "/nn_weights.pt")

    @staticmethod
    def reload(save_path, load_nn=True):
        with open(save_path + "/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        agent = ImitativeAgent(config)

        with open(save_path + "/sound_scaler.pickle", "rb") as f:
            agent.sound_scaler = pickle.load(f)
        with open(save_path + "/datasplits.pickle", "rb") as f:
            agent.datasplits = pickle.load(f)
        if load_nn:
            agent.nn.load_state_dict(torch.load(save_path + "/nn_weights.pt"))
            agent.nn.eval()

        return agent

    def repeat(self, sound_seq):
        nn_input = torch.FloatTensor(self.sound_scaler.transform(sound_seq)).to("cuda")
        with torch.no_grad():
            sound_seq_estimated_unscaled, art_seq_estimated_unscaled = self.nn(
                nn_input[None, :, :]
            )
        sound_seq_estimated_unscaled = sound_seq_estimated_unscaled[0].cpu().numpy()
        art_seq_estimated_unscaled = art_seq_estimated_unscaled[0].cpu().numpy()
        art_seq_estimated = self.synthesizer.art_scaler.inverse_transform(
            art_seq_estimated_unscaled
        )
        sound_seq_estimated = self.sound_scaler.inverse_transform(
            sound_seq_estimated_unscaled
        )
        sound_seq_repeated = self.synthesizer.synthesize(art_seq_estimated)
        return {
            "sound_repeated": sound_seq_repeated,
            "sound_estimated": sound_seq_estimated,
            "art_estimated": art_seq_estimated,
        }

    # def invert_art(self, sound_seq):
    #     nn_input = torch.FloatTensor(self.sound_scaler.transform(sound_seq)).to("cuda")
    #     with torch.no_grad():
    #         art_seq_estimated_unscaled = self.nn.inverse_model(
    #             nn_input[None, :, :]
    #         )
    #     art_seq_estimated_unscaled = art_seq_estimated_unscaled[0].cpu().numpy()
    #     art_seq_estimated = self.synthesizer.art_scaler.inverse_transform(
    #         art_seq_estimated_unscaled
    #     )
    #     return art_seq_estimated
