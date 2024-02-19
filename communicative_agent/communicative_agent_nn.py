from torch import nn


class CommunicativeAgentNN(nn.Module):
    def __init__(self, inverse_model, direct_model, sound_quantizer, discriminator_model=None):
        super(CommunicativeAgentNN, self).__init__()
        self.inverse_model = inverse_model
        self.direct_model = direct_model
        self.sound_quantizer = sound_quantizer
        self.discriminator_model = discriminator_model
