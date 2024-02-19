from torch import nn

class ArticuloGANNN(nn.Module):
    def __init__(self, discriminator, generator):
        super(ArticuloGANNN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
