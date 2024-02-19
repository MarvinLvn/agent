import torch
import yaml
import os
import pickle
from lib import utils

from torch import nn
from sklearn.preprocessing import StandardScaler
from lib.art_dataloader import get_dataloaders_frame_level
from quantizer.quantizer import Quantizer
from lib.nn.simple_lstm import SimpleLSTM, LSTM_FF
from lib.nn.feedforward import FeedForward
from lib.dataset_wrapper import Dataset
from discriminator_nn import ArticuloGANNN
from synthesizer.synthesizer import Synthesizer


SYNTHESIZERS_PATH = os.path.join(os.path.dirname(__file__), "../out/synthesizer")
class ArticuloGAN:
    def __init__(self, config, generator, load_nn=True):
        self.config = config
        self.datasplits = None

        synthesizer = Synthesizer.reload(
            "%s/%s" % (SYNTHESIZERS_PATH, config["synthesizer"]["name"]),
            load_nn=load_nn,
        )
        self.art_scaler = synthesizer.art_scaler
        self.dataset = synthesizer.dataset
        self.datasplits = synthesizer.datasplits
        self.generator = generator

        if load_nn:
            self._build_nn(self.config["model"])
            self.nn.eval()

    def _build_nn(self, model_config):
        self.art_dim = self.dataset.get_modality_dim("art_params")

        discriminator = FeedForward(
            self.art_dim,
            1,
            model_config["discriminator"]["hidden_layers"],
            model_config["discriminator"]["activation"],
            model_config["discriminator"]["dropout_p"],
            model_config["discriminator"]["batch_norm"],
            add_sigmoid = True,
        )

        self.nn = ArticuloGANNN(discriminator, generator).to("cuda")

    def get_dataloaders(self):
        # We enforce loading articulatory parameters
        self.config["dataset"]["art_type"] = 'art_params'
        datasplits, dataloaders = get_dataloaders_frame_level(
            self.config["dataset"], self.art_scaler, self.datasplits
        )
        self.datasplits = datasplits
        return dataloaders

    def get_optimizers(self):
        return {
            "discriminator": torch.optim.Adam(
            self.nn.discriminator.parameters(),
            lr=self.config["training"]["discriminator_learning_rate"],
            betas=(0.5, 0.999),
        )
        }

    def get_losses_fn(self):
        return nn.BCELoss()


    def save(self, save_path, basename=None):
        with open(save_path + "/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        with open(save_path + "/datasplits.pickle", "wb") as f:
            pickle.dump(self.datasplits, f)
        if basename is None:
            basename = "nn_weights.pt"
        basename = "/" + basename
        torch.save(self.nn.state_dict(), save_path + basename)

    def reload(save_path, basename=None, load_nn=True):
        print(save_path + "/config.yaml")
        with open(save_path + "/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        model = ArticuloGAN(config, load_nn=load_nn)
        with open(save_path + "/datasplits.pickle", "rb") as f:
            model.datasplits = pickle.load(f)

        if load_nn:
            if basename is None:
                basename = "nn_weights.pt"
            basename = "/" + basename
            model.nn.load_state_dict(torch.load(save_path + basename))
            model.nn.eval()
        return model