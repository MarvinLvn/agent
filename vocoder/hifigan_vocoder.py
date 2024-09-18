import json
from pathlib import Path
from attrdict import AttrDict
import torch
from models import Generator
from torch import nn
import torchaudio
import matplotlib.pyplot as plt
VOCODERS_PATH = Path(__file__).parent.resolve() / "../out/vocoder"

class HifiGAN(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        model_path = VOCODERS_PATH / model_name
        assert (model_path.parent / 'config.json').is_file()
        assert model_path.is_file()

        # 1) Load config
        config_path = model_path.parent / 'config.json'
        with open(config_path, 'r') as fin:
            data = fin.read()
        config = json.loads(data)
        h = AttrDict(config)

        # 2) Load model
        state_dict = torch.load(model_path)
        self.generator = Generator(h)
        self.generator.load_state_dict(state_dict['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()
        self.frame_size = self.generator.h['hop_size']
        self.counter = 0

    def resynth(self, mel_specs, save_path=None):
        y_g_hat = self.generator(mel_specs)
        resynth_sounds = y_g_hat.squeeze(1)
        if self.counter % 400 == 0:
            sample = resynth_sounds[0, :].clone().detach().cpu()
            torchaudio.save(str(save_path / f'{self.counter}.wav'),
                            sample.unsqueeze(0),
                            sample_rate=16000)
            plt.figure(figsize=(12, 4))
            plt.plot(sample.numpy())
            plt.savefig(str(save_path / f'{self.counter}.png'), dpi=200)
            plt.close()
        self.counter += 1
        return resynth_sounds


