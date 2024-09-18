import os
import pickle
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from lib.art_sound_dataloader import get_dataloaders
from lib.base_agent import BaseAgent
from lib.dataset_wrapper import Dataset
from lib.nn.feedforward import FeedForward
from lib.nn.simple_lstm import SimpleLSTM


class InverseModel(BaseAgent):
    def __init__(self, config, load_nn=True):
        self.config = config
        self.sound_scaler = StandardScaler()
        self.art_scaler = StandardScaler()
        self.datasplits = None
        self.dataset = None
        if 'dataset' in config:
            self.dataset = Dataset(config["dataset"]["name"])
        if load_nn:
            self._build_nn(self.config["model"])
            self.nn.eval()

    def _build_nn(self, model_config):
        self.sound_dim = self.dataset.get_modality_dim(self.config["dataset"]["sound_type"])
        self.art_dim = self.dataset.get_modality_dim(self.config["dataset"]["art_type"])

        self.nn = InverseModel.build_lstm(self.sound_dim, self.art_dim,
                                          model_config["inverse_model"]["hidden_size"],
                                          model_config["inverse_model"]["num_layers"],
                                          model_config["inverse_model"]["dropout_p"],
                                          model_config["inverse_model"]["bidirectional"])

    @staticmethod
    def build_lstm(sound_dim, art_dim, hidden_size, num_layers, dropout_p, bidirectional):
        return SimpleLSTM(
            sound_dim,
            art_dim,
            hidden_size,
            num_layers,
            dropout_p,
            bidirectional,
        ).to("cuda")

    def get_dataloaders(self):
        datasplits, dataloaders = get_dataloaders(
            self.config["dataset"], self.art_scaler, self.sound_scaler, self.datasplits
        )
        self.datasplits = datasplits
        return dataloaders

    def get_optimizer(self):
        return torch.optim.Adam(
            self.nn.parameters(),
            lr=self.config["training"]["inverse_model_learning_rate"],
        )

    def get_loss_fn(self):
        def loss_fn(art_seqs_pred, art_seqs, seqs_mask):
            # Supervised MSE loss for articulatory trajectories
            reconstruction_error = (art_seqs_pred - art_seqs) ** 2
            reconstruction_loss = reconstruction_error[seqs_mask].mean()
            return reconstruction_loss
        return loss_fn

    def save(self, save_path):
        with open(save_path + "/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        with open(save_path + "/art_scaler.pickle", "wb") as f:
            pickle.dump(self.art_scaler, f)
        with open(save_path + "/sound_scaler.pickle", "wb") as f:
            pickle.dump(self.sound_scaler, f)
        with open(save_path + "/datasplits.pickle", "wb") as f:
            pickle.dump(self.datasplits, f)
        torch.save(self.nn.state_dict(), save_path + "/nn_weights.pt")

    @staticmethod
    def reload(save_path, load_nn=True):
        with open(save_path + "/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        agent = InverseModel(config, load_nn=load_nn)

        with open(save_path + "/art_scaler.pickle", "rb") as f:
            agent.art_scaler = pickle.load(f)
        with open(save_path + "/sound_scaler.pickle", "rb") as f:
            agent.sound_scaler = pickle.load(f)
        with open(save_path + "/datasplits.pickle", "rb") as f:
            agent.datasplits = pickle.load(f)

        if load_nn:
            agent.nn.load_state_dict(torch.load(save_path + "/nn_weights.pt"))
            agent.nn.eval()
        return agent

    def predict_art(self, sound_seq, device='cuda'):
        nn_input = torch.FloatTensor(self.sound_scaler.transform(sound_seq)).to(device)
        with torch.no_grad():
            nn_output = self.nn(nn_input).cpu().numpy()
        art_seq_pred = self.art_scaler.inverse_transform(nn_output)
        return art_seq_pred

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

    def predict_datasplit(self, datasplit_index=None):
        agent_features = {}
        sound_type = self.config["dataset"]["sound_type"]

        dataset_features = {}
        dataset_name = self.config["dataset"]["name"]
        dataset = Dataset(dataset_name)
        if datasplit_index is None:
            items_name = dataset.get_items_name(sound_type)
        else:
            items_name = self.datasplits[datasplit_index]
        items_sound = dataset.get_items_data(self.config["dataset"]["sound_type"])
        dataset_features['art_estimated'] = {}
        for item_name in items_name:
            item_sound = items_sound[item_name]
            art_estimated = self.predict_art(item_sound)
            dataset_features['art_estimated'][item_name] = art_estimated

        agent_features[dataset_name] = dataset_features
        return agent_features